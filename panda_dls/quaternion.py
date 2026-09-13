"""四元数与旋转工具（w, x, y, z 约定, 与 MuJoCo 一致），纯 NumPy。

设计要点:
    - q_from_R 用 Shepperd 分支法, 按旋转矩阵对角元最大者选公式, 数值稳定;
    - rot_log 是 so(3) 对数映射 R -> 轴角向量, 是姿态误差进入 6D twist 的
      正确形态（雅可比角速度行所在的空间）, theta 被限定在 [0, pi]（短弧）,
      天然规避四元数双覆盖 q/-q 问题;
    - chordal/chordal_to_angle 用于评估口径: ||R_d^T R - I||_F = 2*sqrt(2)*sin(th/2)。
"""
from __future__ import annotations

import numpy as np

PI = np.pi


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 乘积 a (x) b。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnormalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def R_from_q(q: np.ndarray) -> np.ndarray:
    w, x, y, z = qnormalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def q_from_R(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 四元数（Shepperd 分支, 避免除小数）。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return qnormalize(np.array(q))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """球面线性插值。dot<0 时翻转 q1 消除双覆盖; 角度极小退化为线性插值。"""
    q0 = qnormalize(np.asarray(q0, float))
    q1 = qnormalize(np.asarray(q1, float))
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 1.0 - 1e-10:
        return qnormalize(q0 + t * (q1 - q0))
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    return (np.sin((1.0 - t) * th) / s) * q0 + (np.sin(t * th) / s) * q1


def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.concatenate([[np.cos(angle / 2.0)], axis * s])


def rot_log(R: np.ndarray) -> np.ndarray:
    """so(3) 对数映射: R -> 轴角向量 u*theta, theta ∈ [0, pi]。

    三个分支: 小角度(反对称部分直接近似, 避免除以 sin)、常规、theta≈pi
    (反对称部分趋于 0, 从 (R+I)/2 = u u^T 恢复转轴)。
    """
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * w
    if th > PI - 1e-6:
        A = (R + np.eye(3)) / 2.0
        u = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(u))
        u = u * np.where(A[k] >= 0.0, 1.0, -1.0)   # 用绝对值最大行的符号校正整列
        return (u / np.linalg.norm(u)) * th
    return w / (2.0 * np.sin(th)) * th


def chordal(R_d: np.ndarray, R: np.ndarray) -> float:
    """旋转矩阵 Frobenius chordal 距离: ||R_d^T R - I||_F ∈ [0, 2*sqrt(3)]。

    对旋转其上界为 2*sqrt(2)（相对转角 pi 时）, 用作 PASS/FAIL 判据口径。
    """
    return float(np.linalg.norm(R_d.T @ R - np.eye(3), "fro"))


def chordal_to_angle(c: float) -> float:
    """chordal 距离 -> 相对转角 (rad): th = 2*arcsin(c / (2*sqrt(2)))。"""
    return 2.0 * np.arcsin(min(c, 2.0 * np.sqrt(2.0)) / (2.0 * np.sqrt(2.0)))
