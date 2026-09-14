"""操作空间控制 (Operational Space Control, Khatib 1987)：力矩级任务空间动力学控制。

与 dls.py 的速度级方案的本质区别：速度级只回答"末端想要什么速度"，关节力矩
仍由执行器内部的位置环消化；本模块直接计算关节力矩，把机器人动力学
M(q) q̈ + h(q, q̇) = τ 显式纳入控制律，任务空间的惯性、重力、科氏力全部前馈补偿。

每控制周期 (1 kHz):
    ẍ_cmd = a_ff + Kp·e + Kd·(ẋ_d − ẋ)          6D 任务空间参考加速度
    Λ   = (J M⁻¹ Jᵀ + λ²I)⁻¹                     任务空间惯量 (接近奇异时自适应加阻尼,
                                                 与 DLS 同一套 λ(σ_min) 调度)
    F   = Λ·(ẍ_cmd − J̇q̇) + Λ J M⁻¹ h             任务力 = 惯量×参考加速度 + 任务空间重力/科氏
    τ   = Jᵀ F  +  (I − Jᵀ J̄) τ₀  +  D q̇        关节力矩 = 任务力 + 动力学一致零空间姿态项 + 阻尼补偿
    J̄   = M⁻¹ Jᵀ Λ                               动力学一致伪逆 (Khatib)

数值来源:
    M      mj_fullM(data.qM) —— 含 armature, 是真实的关节空间惯量;
    h      data.qfrc_bias = C(q,q̇)q̇ + g(q), MuJoCo 递推动力学结果;
    J̇q̇    方向中心差分: (J(q+εq̇) − J(q−εq̇))/(2ε) · q̇ —— 对 ẋ = Jq̇ 求导的
           直接数值实现, O(ε²) 精度, 每步只多 2 次 FK+雅可比;
    D q̇   关节粘性阻尼补偿 (模型 damping=1 N·m·s/rad), 摩擦力矩用平滑 sign 补偿。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import mujoco

from . import kinematics as kin
from .control import DEMO_Q0, PandaSim, MODEL_PATH, _HERE
from .dls import compute_lambda
from .quaternion import rot_log

#: panda_motor 场景 (力矩执行器)
SCENE_MOTOR_PATH = os.path.join(_HERE, "..", "models", "franka_emika_panda", "scene_motor.xml")

#: Panda 官方关节力矩上限 [N·m]
PANDA_TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

Q_LO = kin.JOINT_LIMITS[:, 0]
Q_HI = kin.JOINT_LIMITS[:, 1]
Q_MID = kin.JOINT_LIMITS.mean(axis=1)
Q_HALF = (kin.JOINT_LIMITS[:, 1] - kin.JOINT_LIMITS[:, 0]) / 2.0


@dataclass
class OSCConfig:
    kp: float = 400.0               # 位置刚度 [1/s²]
    ko: float = 260.0               # 姿态刚度 [1/s²]
    zeta: float = 1.0               # 阻尼比: Kd = 2ζ√K (1 = 临界阻尼)
    lam_mode: str = "adaptive"      # 任务空间惯量的奇异调度 (与 DLSConfig 同名字段,
    lambda0: float = 0.05           #   compute_lambda 按鸭子类型直接复用)
    sigma_eps: float = 2e-3         # σ_min(J M⁻¹ Jᵀ) 激活阈值 (惯量矩阵量纲)
    k_posture: float = 30.0         # 零空间姿态刚度 [N·m] (限位中值吸引)
    d_posture: float = 8.0          # 零空间关节阻尼 [N·m·s/rad]


def _new_osc_log():
    return {k: [] for k in (
        "t", "q", "qd", "p", "p_d", "ep_mm", "eo_deg", "eo_rad",
        "smin", "cond", "manip", "lam",
        "tau", "tau_task", "tau_null", "tau_inf", "qd_inf")}


def _osc_log_step(log, t, q, qd, p, p_d, e, info):
    log["t"].append(t)
    log["q"].append(q.copy())
    log["qd"].append(qd.copy())
    log["p"].append(p.copy())
    log["p_d"].append(np.asarray(p_d).copy())
    log["ep_mm"].append(1000.0 * np.linalg.norm(e[:3]))
    log["eo_deg"].append(np.degrees(np.linalg.norm(e[3:])))
    log["eo_rad"].append(float(np.linalg.norm(e[3:])))
    log["smin"].append(info["smin"])
    log["cond"].append(info["cond"])
    log["manip"].append(info["manip"])
    log["lam"].append(info["lam"])
    log["tau"].append(info["tau"])
    log["tau_task"].append(info["tau_task"])
    log["tau_null"].append(info["tau_null"])
    log["tau_inf"].append(float(np.max(np.abs(info["tau"]))))
    log["qd_inf"].append(float(np.max(np.abs(qd))))


def run_osc_tracking(traj, cfg: OSCConfig, q0: np.ndarray = DEMO_Q0,
                     dt: float = 0.001, model_path: str = SCENE_MOTOR_PATH) -> dict:
    """力矩级闭环: 操作空间控制沿轨迹跟踪, 返回逐周期指标日志。

    每步先 mj_forward 刷新当前状态的 qM / qfrc_bias (mj_step 后这些量属于
    上一步状态), 再计算力矩写入 ctrl, 最后 mj_step 积分。
    """
    sim = PandaSim(model_path)
    model, data = sim.model, sim.data
    n_steps = int(np.ceil(traj.duration / dt))

    q = np.clip(np.asarray(q0, float), Q_LO, Q_HI)
    sim.set_q(q)

    M_full = np.zeros((model.nv, model.nv))
    D = np.diag(model.dof_damping[:7])                    # 粘性阻尼
    fl = model.dof_frictionloss[:7]                       # 库仑摩擦 (本模型为 0)
    log = _new_osc_log()

    kd_p = 2.0 * cfg.zeta * np.sqrt(cfg.kp)
    kd_o = 2.0 * cfg.zeta * np.sqrt(cfg.ko)
    eps = 1e-5                                            # J̇q̇ 方向差分布长

    for k in range(n_steps):
        t = k * dt
        p_d, R_d, v_ff, w_ff, a_ff, alpha_ff = traj.sample(t)

        q = data.qpos[:7].copy()
        qd = data.qvel[:7].copy()
        mujoco.mj_forward(model, data)                    # 刷新 qM / qfrc_bias
        if hasattr(data, "M"):                            # mujoco >= 3.11: qM 改名 M,
            mujoco.mj_fullM(model, data, M_full)          #   mj_fullM(m, d, dst) 直读 data.M
        else:                                             # 旧版: mj_fullM(m, dst, qM)
            mujoco.mj_fullM(model, M_full, data.qM)
        M = M_full[:7, :7]
        h = data.qfrc_bias[:7].copy()

        Th = kin.fk_hand(q)
        p, R = Th[:3, 3], Th[:3, :3]
        J = kin.hand_jacobian(q)

        # ---- 任务空间: 误差 / 参考加速度 ----
        e_p = p_d - p
        e_o = rot_log(R_d @ R.T)
        xdot = J @ qd                                     # [v; w]
        xd_ff = np.concatenate([v_ff, w_ff])
        ed = xd_ff - xdot
        xdd_cmd = (np.concatenate([a_ff, alpha_ff])
                   + np.concatenate([cfg.kp * e_p, cfg.ko * e_o])
                   + np.concatenate([kd_p * ed[:3], kd_o * ed[3:]]))

        # ---- 动力学 ----
        X = np.linalg.solve(M, J.T)                       # M⁻¹ Jᵀ (7x6)
        A_t = J @ X                                       # J M⁻¹ Jᵀ (6x6)
        S_t = np.linalg.svd(A_t, compute_uv=False)
        smin = float(S_t[-1])
        lam = compute_lambda(smin, cfg)
        Lam = np.linalg.inv(A_t + (lam * lam + 1e-12) * np.eye(6))
        Jbar = X @ Lam                                    # 动力学一致伪逆

        h_task = J @ np.linalg.solve(M, h)                # J M⁻¹ h (任务空间重力+科氏)
        Jdot = (kin.hand_jacobian(q + qd * eps) - kin.hand_jacobian(q - qd * eps)) / (2.0 * eps)
        F = Lam @ (xdd_cmd - Jdot @ qd) + Lam @ h_task

        tau_task = J.T @ F
        tau0 = cfg.k_posture * (Q_MID - q) / Q_HALF - cfg.d_posture * qd
        # 动力学一致零空间投影 (作用于力矩): N^T = I − JᵀΛJM⁻¹ = (J̄J)ᵀ
        tau_null = (np.eye(7) - (Jbar @ J).T) @ tau0
        tau = tau_task + tau_null + D @ qd
        if np.any(fl > 0.0):
            tau = tau + fl * np.tanh(qd / 1e-3)           # 平滑符号的摩擦补偿
        tau = np.clip(tau, -PANDA_TAU_MAX, PANDA_TAU_MAX)

        data.ctrl[:7] = tau
        data.ctrl[7] = 255.0                              # 夹爪张开 (腱驱动执行器不变)
        mujoco.mj_step(model, data)

        U, S, _ = np.linalg.svd(J)
        info = dict(
            smin=float(S[-1]), smax=float(S[0]),
            cond=float(S[0] / max(S[-1], 1e-12)),
            manip=float(np.prod(S)), lam=lam,
            tau=tau, tau_task=tau_task, tau_null=tau_null)
        _osc_log_step(log, t, q, qd, p, p_d, np.concatenate([e_p, e_o]), info)

    return {k: np.asarray(v) for k, v in log.items()}
