"""DLS（阻尼最小二乘）分辨率速度控制器 + 零空间次任务，纯 NumPy。

核心公式（每控制周期一步, 误差驱动, 无需迭代收敛）:
    dq = J^T (J J^T + lam^2 I)^-1 · (K e + xd_ff)
         + (I - J^T (J J^T + lam^2 I)^-1 J) · k_n · z(q)

三个组成部分:
    主任务   DLS 伪逆把 6D 任务速度映射到关节速度, lam 把 (J J^T) 谱半径
             抬离 0, 用可控偏差换奇异附近的数值连续性（Tikhonov 正则视角）;
    自适应   lam^2 = lam0^2 (1 - (smin/eps)^2) 当 smin < eps, 否则 0:
             远离奇异退化为伪逆不牺牲精度, 接近奇异平滑加阻尼;
    零空间   N(q) = I - J^+ J 把次任务投影到不影响主任务的方向:
             关节限位中值吸引 (k_limit) 与可操作度梯度上升 (k_manip)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kinematics as kin
from .quaternion import rot_log

#: Panda 官方关节速度上限 (rad/s)
PANDA_QD_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])


@dataclass
class DLSConfig:
    kp: float = 1.0                  # 位置误差增益 [1/s]
    ko: float = 1.0                  # 姿态误差增益 [1/s]
    use_ff: bool = True              # 是否叠加轨迹前馈速度
    lam_mode: str = "adaptive"       # "adaptive" | "fixed" | "none"(纯伪逆)
    lambda0: float = 0.10            # 自适应模式的峰值阻尼 / 固定模式的阻尼值
    sigma_eps: float = 0.03          # 最小奇异值激活阈值 (J 以 m 计, Panda 灵活工作区
                                     # σ_min 约 0.04~0.1, 0.03 只在真接近奇异时介入)
    k_limit: float = 0.5             # 零空间限位中值吸引增益, 0 = 关闭
    k_manip: float = 0.0             # 零空间可操作度梯度增益, 0 = 关闭
    dq_max: np.ndarray = field(default_factory=lambda: PANDA_QD_MAX.copy())
    clamp_dq: bool = True            # 关节速度限幅（奇异实验里关掉以暴露原始需求）


def compute_lambda(smin: float, cfg: DLSConfig) -> float:
    """按最小奇异值 smin 计算当前阻尼 lam。"""
    if cfg.lam_mode == "none":
        return 0.0
    if cfg.lam_mode == "fixed":
        return cfg.lambda0
    if smin >= cfg.sigma_eps:
        return 0.0
    r = smin / cfg.sigma_eps
    return cfg.lambda0 * np.sqrt(max(0.0, 1.0 - r * r))


def task_velocity(p, R, p_d, R_d, v_ff, w_ff, cfg: DLSConfig) -> tuple[np.ndarray, np.ndarray]:
    """6D 任务速度 = 增益·误差 + 前馈。误差向量一并返回供记录。

    姿态误差用 rot_log(R_d R^T): 世界系轴角向量, 与 J 的角速度行(世界系)同空间。
    """
    e_p = p_d - p
    e_o = rot_log(R_d @ R.T)
    v = cfg.kp * e_p + (v_ff if cfg.use_ff else 0.0)
    w = cfg.ko * e_o + (w_ff if cfg.use_ff else 0.0)
    return np.concatenate([v, w]), np.concatenate([e_p, e_o])


def manipulability(q: np.ndarray) -> float:
    """可操作度 m(q) = sqrt(det(J J^T)), Yoshikawa。"""
    J = kin.hand_jacobian(q)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def manip_gradient(q: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """可操作度数值梯度（中心差分, 7 维 14 次 FK, 足够快; 解析式留作扩展）。"""
    g = np.zeros(7)
    for i in range(7):
        qp, qm = q.copy(), q.copy()
        qp[i] += h
        qm[i] -= h
        g[i] = (manipulability(qp) - manipulability(qm)) / (2.0 * h)
    return g


def secondary_task(q: np.ndarray, cfg: DLSConfig) -> np.ndarray:
    """零空间次任务方向 z(q): 限位中值吸引 + 可操作度梯度。"""
    z = np.zeros(7)
    if cfg.k_limit > 0.0:
        mid = kin.JOINT_LIMITS.mean(axis=1)
        half = (kin.JOINT_LIMITS[:, 1] - kin.JOINT_LIMITS[:, 0]) / 2.0
        z += cfg.k_limit * (mid - q) / half          # 归一成无量纲 O(1)
    if cfg.k_manip > 0.0:
        z += cfg.k_manip * manip_gradient(q)
    return z


def dls_step(q: np.ndarray, xd: np.ndarray, J: np.ndarray, cfg: DLSConfig):
    """单步 DLS 求解。

    参数:
        q   当前关节角 (7,)
        xd  6D 任务速度 [v; w] 世界系
        J   6x7 雅可比 (hand 参考点, 世界系)
    返回:
        dq    实际使用的关节速度（可能被限幅缩放）
        info  诊断字典: smin/smax/cond/lam/manip/dq_task/dq_null/dq_raw
    """
    U, S, Vt = np.linalg.svd(J)
    smin, smax = float(S[-1]), float(S[0])
    lam = compute_lambda(smin, cfg)

    JJt = J @ J.T + (lam * lam + 1e-12) * np.eye(6)   # 1e-12: 纯伪逆模式防奇异异常
    y = np.linalg.solve(JJt, xd)                       # 不显式求逆: 解方程组数值更稳
    dq_task = J.T @ y

    dq_null = np.zeros(7)
    if cfg.k_limit > 0.0 or cfg.k_manip > 0.0:
        N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)  # 零空间投影 N = I - J^+ J
        dq_null = N @ secondary_task(q, cfg)

    dq = dq_task + dq_null
    dq_raw = dq.copy()
    scale = 1.0
    if cfg.clamp_dq:
        scale = float(np.min(cfg.dq_max / np.maximum(np.abs(dq), 1e-12)))
        scale = min(1.0, scale)
        dq = dq * scale

    info = dict(
        smin=smin, smax=smax, cond=smax / max(smin, 1e-12), lam=lam,
        manip=float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0))),
        dq_task=dq_task, dq_null=dq_null, dq_raw=dq_raw, scale=scale,
        err_norm=float(np.linalg.norm(xd)),
    )
    return dq, info


def _solve_ik_once(target_p, target_R, q_init, base: dict, max_iter: int,
                   tol_p: float, tol_o: float):
    """单起点两阶段迭代。阶段1 自适应阻尼稳健接近; 阶段2 lam=0 纯伪逆
    抛光(零偏置)到公差。返回 (q, 收敛步数/-1)。"""
    stages = [(DLSConfig(**base), int(max_iter * 0.7)), (DLSConfig(**{**base, "lam_mode": "none"}), None)]
    q = np.clip(np.asarray(q_init, float), kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1])
    it_total = 0
    for stage_cfg, it_max in stages:
        it_cap = it_max if it_max is not None else max_iter
        for _ in range(it_cap):
            Th = kin.fk_hand(q)
            xd, e = task_velocity(Th[:3, 3], Th[:3, :3], target_p, target_R, np.zeros(3), np.zeros(3), stage_cfg)
            if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_o:
                return q, it_total
            if it_total >= max_iter:
                return q, -1
            dq, _ = dls_step(q, xd, kin.hand_jacobian(q), stage_cfg)
            q = np.clip(q + dq, kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1])
            it_total += 1
    return q, -1


def solve_ik(target_p, target_R, q_init, cfg: DLSConfig | None = None,
             max_iter: int = 300, tol_p: float = 1e-4, tol_o: float = 1e-3,
             rng: np.random.Generator | None = None, restarts: int = 3):
    """静态 IK: 同一 DLS 公式按迭代法使用, 带多起点重启。

    为什么重启: 任务误差方向指向关节限位时, 零空间无能为力、硬裁剪停摆,
    该盆地内无法收敛 —— 换个起点换条路径是数值 IK 的标准工程实践。
    返回 (q, 收敛总步数); 全部失败记 -1。"""
    cfg = cfg or DLSConfig()
    base = {**cfg.__dict__, "clamp_dq": False, "k_limit": min(cfg.k_limit, 0.3)}
    rng = rng or np.random.default_rng()
    best_q, best_err = np.asarray(q_init, float), np.inf
    for r in range(max(1, restarts)):
        if r == 0:
            q0 = q_init
        else:   # 重启点: 限位区间内随机撒点, 远离上一次卡死的盆地
            span = kin.JOINT_LIMITS[:, 1] - kin.JOINT_LIMITS[:, 0]
            q0 = kin.JOINT_LIMITS[:, 0] + rng.uniform(0.1, 0.9, 7) * span
        q, it = _solve_ik_once(target_p, target_R, q0, base, max_iter, tol_p, tol_o)
        if it >= 0:
            return q, it + r * max_iter
        Th = kin.fk_hand(q)
        _, e = task_velocity(Th[:3, 3], Th[:3, :3], target_p, target_R, np.zeros(3), np.zeros(3), cfg)
        err = np.linalg.norm(e)
        if err < best_err:
            best_q, best_err = q, err
    return best_q, -1
