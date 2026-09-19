"""MuJoCo 仿真闭环：运动学层 / 动力学层两种模式。

运动学层 (mode="kinematic"):
    q <- q + dq*dt 直接积分写 qpos, 无动力学干扰 —— 专门确认控制律与轨迹
    本身的正确性, 误差应收敛到数值精度级。

动力学层 (mode="dynamic"):
    q_des 发给 panda 的 position 舵机执行器 (gainprm=4500/3500/2000,
    biasprm 仿射 = kp 刚度 + kv 阻尼), 每 mj_step 1 kHz 下发, 跟踪误差
    由执行器带宽与重力决定 —— 与运动学层的差值即 "执行/动力学误差"。

两模式控制频率均为 1 kHz (Panda 真机控制频率口径)。位姿与雅可比统一用
自研 MDH 运动学计算（已与 MuJoCo 比对 <1e-9）, 避免每步调用 mj_forward。
"""
from __future__ import annotations

import os

import numpy as np
import mujoco

from . import kinematics as kin
from .dls import DLSConfig, dls_step, task_velocity
from .qp import QPConfig, QPController

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "models", "franka_emika_panda", "scene.xml")

#: 起始位形（Franka 风格 "ready" 姿态附近的肘抬起位形, 限位裕度充足）
DEMO_Q0 = np.array([0.0, -0.30, 0.0, -2.20, 0.0, 2.00, 0.785])

Q_LO = kin.JOINT_LIMITS[:, 0]
Q_HI = kin.JOINT_LIMITS[:, 1]


class PandaSim:
    """panda.xml 的 MuJoCo 封装。"""

    def __init__(self, model_path: str = MODEL_PATH, timestep: float = 0.001):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.hand_id = self.model.body("hand").id
        self.set_q(np.zeros(7))

    def set_q(self, q: np.ndarray):
        self.data.qpos[:7] = q
        self.data.qpos[7:] = 0.04          # 手指张开
        mujoco.mj_forward(self.model, self.data)

    def hand_pose(self):
        p = self.data.xpos[self.hand_id].copy()
        R = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        return p, R


def _new_log():
    return {k: [] for k in (
        "t", "q", "q_des", "p", "p_d", "ep_mm", "eo_deg", "eo_rad",
        "smin", "cond", "manip", "lam", "dq_inf_raw", "dq", "dq_task", "dq_null", "scale")}


def _log_step(log, t, q, q_des, p, p_d, e, info):
    log["t"].append(t)
    log["q"].append(q.copy())
    log["q_des"].append(q_des.copy())
    log["p"].append(p.copy())
    log["p_d"].append(np.asarray(p_d).copy())
    log["ep_mm"].append(1000.0 * np.linalg.norm(e[:3]))
    log["eo_deg"].append(np.degrees(np.linalg.norm(e[3:])))
    log["eo_rad"].append(float(np.linalg.norm(e[3:])))
    log["smin"].append(info["smin"])
    log["cond"].append(info["cond"])
    log["manip"].append(info["manip"])
    log["lam"].append(info["lam"])
    log["dq_inf_raw"].append(float(np.max(np.abs(info["dq_raw"]))))
    log["dq"].append(info["dq_raw"] * info["scale"])
    log["dq_task"].append(info["dq_task"])
    log["dq_null"].append(info["dq_null"])
    log["scale"].append(info["scale"])


def run_tracking(traj, cfg: DLSConfig | QPConfig, mode: str = "dynamic", q0: np.ndarray = DEMO_Q0,
                 dt: float | None = None, model_path: str = MODEL_PATH) -> dict:
    """沿轨迹做闭环跟踪, 返回逐周期指标日志（dict of np.ndarray）。

    cfg 传 DLSConfig 用解析 DLS, 传 QPConfig 用 OSQP 约束求解
    (速度级控制律同构: xd = K·e + 前馈 → dq)。
    dt: 运动学层默认 0.002; 动力学层强制 = 模型步长 (0.001, Panda 真机口径)。
    """
    dt = dt or (0.001 if mode == "dynamic" else 0.002)
    qp = QPController(cfg, dt) if isinstance(cfg, QPConfig) else None
    sim = PandaSim(model_path)
    n_steps = int(np.ceil(traj.duration / dt))

    q = np.clip(np.asarray(q0, float), Q_LO, Q_HI)
    q_des = q.copy()
    if mode == "dynamic":
        sim.set_q(q)
    log = _new_log()

    for k in range(n_steps):
        t = k * dt
        p_d, R_d, v_ff, w_ff, *_ = traj.sample(t)

        if mode == "kinematic":
            Th = kin.fk_hand(q)
            p, R = Th[:3, 3], Th[:3, :3]
            J = kin.hand_jacobian(q)
        else:
            q = sim.data.qpos[:7].copy()          # 测量值驱动控制器
            Th = kin.fk_hand(q)
            p, R = Th[:3, 3], Th[:3, :3]
            J = kin.hand_jacobian(q)

        xd, e = task_velocity(p, R, p_d, R_d, v_ff, w_ff, cfg)
        if qp is not None:
            dq, info = qp.step(q, p, J, xd)
        else:
            dq, info = dls_step(q, xd, J, cfg)

        if mode == "kinematic":
            q = np.clip(q + dq * dt, Q_LO, Q_HI)
            q_des = q.copy()
        else:
            q_des = np.clip(q_des + dq * dt, Q_LO, Q_HI)
            sim.data.ctrl[:7] = q_des
            sim.data.ctrl[7] = 255.0              # 夹爪张开
            mujoco.mj_step(sim.model, sim.data)

        _log_step(log, t, q, q_des, p, p_d, e, info)

    return {k: np.asarray(v) for k, v in log.items()}


def run_with_viewer(traj, cfg: DLSConfig, q0: np.ndarray = DEMO_Q0, model_path: str = MODEL_PATH):
    """实时 viewer 里的闭环跟踪（不落盘）。

    场景内叠加两条末端轨迹: 期望轨迹（蓝）与实测末端轨迹（橙, 随运行增长）。
    """
    import mujoco.viewer

    from .trail import draw_user_trails, ref_tip, tip_from_pose

    sim = PandaSim(model_path)
    q = np.clip(np.asarray(q0, float), Q_LO, Q_HI)
    q_des = q.copy()
    sim.set_q(q)

    ref_pts = np.stack([ref_tip(*traj.sample(traj.duration * i / 240)[:2])
                        for i in range(241)])
    trail_pts = []

    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        for k in range(int(np.ceil(traj.duration / 0.001))):
            t = k * 0.001
            p_d, R_d, v_ff, w_ff, *_ = traj.sample(t)
            q = sim.data.qpos[:7].copy()
            Th = kin.fk_hand(q)
            if k % 8 == 0:
                trail_pts.append(tip_from_pose(Th[:3, 3], Th[:3, :3]))
            xd, _ = task_velocity(Th[:3, 3], Th[:3, :3], p_d, R_d, v_ff, w_ff, cfg)
            dq, _ = dls_step(q, xd, kin.hand_jacobian(q), cfg)
            q_des = np.clip(q_des + dq * 0.001, Q_LO, Q_HI)
            sim.data.ctrl[:7] = q_des
            sim.data.ctrl[7] = 255.0
            mujoco.mj_step(sim.model, sim.data)
            if k % 10 == 0:
                draw_user_trails(v.user_scn, ref_pts, trail_pts)
                v.sync()
    print("viewer 运行结束")
