"""QP 控制器 (OSQP) 正确性测试 + OSC 冒烟测试。

运行: cd demo && python -m pytest tests/test_qp.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from panda_dls import kinematics as kin
from panda_dls.osc import OSCConfig, PANDA_TAU_MAX, run_osc_tracking
from panda_dls.qp import QPConfig, QPController
from panda_dls.traj import CircleTraj

DT = 0.002


def _make_case(rng, near_limit=False):
    """随机生成 (q, J, p, xd)。near_limit=True 时把 q 推到限位附近。"""
    lo, hi = kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1]
    if near_limit:
        j = int(rng.integers(0, 7))
        q = rng.uniform(lo + 0.5, hi - 0.5, 7)
        q[j] = lo[j] + 0.05 if rng.random() < 0.5 else hi[j] - 0.05
    else:
        q = rng.uniform(lo + 0.2, hi - 0.2, 7)
    J = kin.hand_jacobian(q)
    p = kin.fk_hand(q)[:3, 3]
    xd = rng.normal(0, 0.3, 6)
    return q, J, p, xd


def test_qp_unconstrained_matches_closed_form():
    """无约束 (限位远、速度上限放大、无加速度/平面) 时应退化为加权最小二乘闭式解。

    rho 取 1e-2 保持 QP 良态: rho→0 时 P 在 J 零空间方向曲率趋 0,
    ADMM 尾部收敛慢, 对拍的是建模正确性而非病态问题的求解精度。
    """
    rng = np.random.default_rng(0)
    cfg = QPConfig(rho=1e-2, w_ori=0.5)
    lo, hi = kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1]
    for _ in range(20):
        q, J, p, xd = _make_case(rng)
        W = np.diag(np.concatenate([np.ones(3), np.full(3, cfg.w_ori)]))
        mid = kin.JOINT_LIMITS.mean(axis=1)
        half = (hi - lo) / 2.0
        dq_ref = cfg.k_posture * (mid - q) / half
        ref = np.linalg.solve(J.T @ W @ J + cfg.rho * np.eye(7), J.T @ W @ xd + cfg.rho * dq_ref)
        ctrl = QPController(cfg, DT)
        ctrl.cfg.dq_max = np.full(7, 1e3)
        dq, info = ctrl.step(q, p, J, xd)
        assert info["qp_status"] == "solved"
        assert not info["fallback"]
        assert np.max(np.abs(dq - ref)) < 1e-5


def test_qp_respects_all_hard_constraints():
    """贴限位 + 小速度上限: 位置限位 / 速度盒约束必须严格成立。"""
    rng = np.random.default_rng(1)
    lo, hi = kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1]
    for _ in range(30):
        q, J, p, xd = _make_case(rng, near_limit=True)
        cfg = QPConfig(dq_max=np.full(7, 0.3), limit_margin=0.02)
        ctrl = QPController(cfg, DT)
        dq, info = ctrl.step(q, p, J, xd)
        assert np.all(np.abs(dq) <= cfg.dq_max + 1e-9)
        assert np.all(q + DT * dq >= lo + cfg.limit_margin - 1e-9)
        assert np.all(q + DT * dq <= hi - cfg.limit_margin + 1e-9)


def test_qp_slides_along_limit_boundary():
    """任务需求指向限位外时, 解应贴在限位边界内侧滑动而不是越过/停摆。

    速度级每步位移 = dq·dt, 用大 dt (0.05) 让单步需求越过受限 room,
    约束才会激活并贴边。
    """
    rng = np.random.default_rng(2)
    lo, hi = kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1]
    q = rng.uniform(lo + 0.5, hi - 0.5, 7)
    q[6] = hi[6] - 0.06                                   # j7 距上限仅 0.06 rad
    J = kin.hand_jacobian(q)
    p = kin.fk_hand(q)[:3, 3]
    cfg = QPConfig(rho=0.0, k_posture=0.0)
    ctrl = QPController(cfg, dt=0.05)
    ctrl.cfg.dq_max = np.full(7, 5.0)
    xd = J @ np.concatenate([np.zeros(6), [2.0]])          # 任务速度恰好需要 q7 以 2 rad/s 运动
    dq, info = ctrl.step(q, p, J, xd)
    # 单步需求 0.1 rad > 剩余 room 0.04 rad → 贴在 hi − margin
    assert q[6] + 0.05 * dq[6] <= hi[6] - cfg.limit_margin + 1e-9
    assert q[6] + 0.05 * dq[6] >= hi[6] - cfg.limit_margin - 1e-9   # 投影后精确贴边
    assert info["n_active"] >= 1


def test_qp_respects_safety_plane():
    """安全平面: 无论任务怎么要求, 下一周期末端 z 不得低于平面。"""
    rng = np.random.default_rng(3)
    z_plane = 0.3
    for _ in range(10):
        q, J, p, xd = _make_case(rng)
        if p[2] < z_plane + 0.1:
            continue
        xd[2] = -1.0                                       # 强烈向下需求
        cfg = QPConfig(safe_plane_z=z_plane, dq_max=np.full(7, 2.0))
        ctrl = QPController(cfg, DT)
        dq, info = ctrl.step(q, p, J, xd)
        assert p[2] + DT * float(J[2, :] @ dq) >= z_plane - 1e-9


def test_qp_warm_start_converges_fast():
    """相邻周期的相似问题: warm start 后迭代次数应远小于冷启动。"""
    rng = np.random.default_rng(4)
    q, J, p, xd = _make_case(rng)
    ctrl = QPController(QPConfig(), DT)
    _, i1 = ctrl.step(q, p, J, xd)
    _, i2 = ctrl.step(q + 1e-4, p, J, xd + 1e-4)
    assert i2["qp_iters"] <= max(10, i1["qp_iters"] // 2)


def test_osc_smoke_tracking_and_torque_limits():
    """OSC 冒烟: 2 s 圆跟踪 RMS < 1 mm, 力矩不超限, 关节速度不超限。"""
    from panda_dls.dls import PANDA_QD_MAX
    q0 = np.array([0.0, -0.30, 0.0, -2.20, 0.0, 2.00, 0.785])
    Th = kin.fk_hand(q0)
    traj = CircleTraj(Th[:3, 3] - np.array([0.10, 0, 0]), 0.10, Th[:3, :3],
                      2.0, laps=1.0, ori_amplitude=np.deg2rad(30))
    log = run_osc_tracking(traj, OSCConfig(), q0=q0)
    n = len(log["ep_mm"])
    rms = float(np.sqrt(np.mean(log["ep_mm"][n // 4:] ** 2)))
    assert rms < 1.0, f"OSC RMS {rms:.3f} mm 超过 1 mm 冒烟阈值"
    assert np.all(np.abs(log["tau"]) <= PANDA_TAU_MAX + 1e-9)
    assert np.all(np.abs(log["qd"]) <= PANDA_QD_MAX + 1e-6)
