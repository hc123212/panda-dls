"""Panda 机械臂 MDH（Craig 改进 DH）运动学：FK + 几何雅可比，纯 NumPy。

变换约定（Craig MDH）:
    A_i = Rx(alpha_{i-1}) · Tx(a_{i-1}) · Rz(theta_i) · Tz(d_i)
    theta_i = q_i + theta_offset_i

参数表推导过程（可由 scripts/run_verify.py 复现）:
    1) mj_forward 后读出 7 个关节轴线的世界原点/方向（铰链轴过 body 原点，
       方向 = body 旋转矩阵第三列）；
    2) 相邻两轴做公垂线 => alpha_{i-1}（轴间角）与 a_{i-1}（公垂线长），
       沿 z_i 轴的滑移量 => d_i；
    3) 随机 500 组 q 与 MuJoCo FK 对拍 < 1e-9。

theta_offset 全为 0：menagerie 模型用 body quat（link2 为 Rx(-90°) 等）把
"关节零位"直接做成了 MDH 零位，因此 MDH 表里不需要关节偏置；文献常见的
SDH 表则带有偏置与符号折叠（如 joint4 轴为 -y）。同一机器人可以有多张
合法的 DH 表，差别只在连杆坐标系约定。
"""
from __future__ import annotations

import numpy as np

from .quaternion import R_from_q

PI = np.pi

#: MDH 参数表, 每行 = [alpha_{i-1}, a_{i-1}, d_i, theta_offset_i], 单位 rad / m
MDH = np.array([
    [0.0,       0.0,      0.333, 0.0],  # joint 1: 腰部 z 轴, 肩高 0.333
    [-PI / 2,   0.0,      0.0,   0.0],  # joint 2: 肩部 y 轴（与 joint1 共点）
    [PI / 2,    0.0,      0.316, 0.0],  # joint 3: 肘部 z 轴, 上臂长 0.316
    [PI / 2,    0.0825,   0.0,   0.0],  # joint 4: 前臂 -y 轴, 肘偏置 +0.0825
    [-PI / 2,  -0.0825,   0.384, 0.0],  # joint 5: 腕 z 轴, 偏置 -0.0825, 前臂长 0.384
    [PI / 2,    0.0,      0.0,   0.0],  # joint 6: 腕 -y 轴（与 joint5 共点）
    [PI / 2,    0.088,    0.0,   0.0],  # joint 7: 末端 -z 轴, 法兰横向偏置 0.088
])

#: 关节限位 [lower, upper] (rad), 与 models/franka_emika_panda/panda.xml 一致
JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])

#: link7 -> hand 固定变换: 平移 Tz(0.107) (法兰到手掌), 旋转 Rz(-45°) (hand 坐标系定义)
TOOL_P = np.array([0.0, 0.0, 0.107])
#: XML 原文四元数只写 7 位小数, 按原值使用才能与 MuJoCo 对拍到机器精度
TOOL_QUAT = np.array([0.9238795, 0.0, 0.0, -0.3826834])
TOOL_R = R_from_q(TOOL_QUAT)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def mdh_A(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """单个 MDH 连杆变换 4x4: Rx(alpha) Tx(a) Rz(theta) Tz(d)。"""
    Rx, Rz = _rx(alpha), _rz(theta)
    T = np.eye(4)
    T[:3, :3] = Rx @ Rz
    T[:3, 3] = Rx @ np.array([a, 0.0, 0.0]) + Rx @ Rz @ np.array([0.0, 0.0, d])
    return T


def fk_frames(q: np.ndarray):
    """正向运动学逐关节累乘。

    返回 (frames, pivots, axes):
        frames[k]  -- 第 k 关节建立后的 4x4 位姿, frames[0] = I(基座), 共 8 项
        pivots[i]  -- 第 i+1 关节轴上的锚点（公垂线与轴的交点）, 世界系
        axes[i]    -- 第 i+1 关节轴单位方向, 世界系
    """
    T = np.eye(4)
    frames = [T.copy()]
    pivots, axes = [], []
    for i in range(7):
        alpha, a, d, off = MDH[i]
        R_pre = T[:3, :3] @ _rx(alpha)                    # Rx 之后、Rz 之前: z 轴已对齐关节 i+1
        p_pre = T[:3, 3] + T[:3, :3] @ np.array([a, 0.0, 0.0])
        pivots.append(p_pre.copy())
        axes.append(R_pre[:, 2].copy())                   # Rz/Tz 均不动关节轴
        T = T @ mdh_A(alpha, a, q[i] + off, d)
        frames.append(T.copy())
    return frames, pivots, axes


def fk(q: np.ndarray) -> np.ndarray:
    """link7 (法兰前最后一杆) 位姿 4x4。"""
    return fk_frames(q)[0][7]


def fk_hand(q: np.ndarray) -> np.ndarray:
    """hand (跟踪点) 位姿 4x4 = fk(q) 加固定工具变换。"""
    T7 = fk(q)
    T = np.eye(4)
    T[:3, :3] = T7[:3, :3] @ TOOL_R
    T[:3, 3] = T7[:3, 3] + T7[:3, :3] @ TOOL_P
    return T


def hand_jacobian(q: np.ndarray) -> np.ndarray:
    """hand 参考点处的几何雅可比 J (6x7), 世界系。

    J_v,i = z_i x (p_hand - O_i),  J_w,i = z_i
    与 MuJoCo mj_jac 在 hand body 原点处的输出逐列一致（可对拍）。
    """
    frames, pivots, axes = fk_frames(q)
    T7 = frames[7]
    p_hand = T7[:3, 3] + T7[:3, :3] @ TOOL_P
    J = np.zeros((6, 7))
    for i in range(7):
        J[:3, i] = np.cross(axes[i], p_hand - pivots[i])
        J[3:, i] = axes[i]
    return J
