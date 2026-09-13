"""笛卡尔空间参考轨迹：五次多项式时间律 + 直线 / 空间圆 + 同步姿态旋转。

统一接口 sample(t) -> (p, R, v, w):
    p, R    期望位姿;  v, w 期望线/角速度, 用作 DLS 的前馈项。

时间律用归一化五次多项式 s(τ) = 10τ³ - 15τ⁴ + 6τ⁵: 两端速度、加速度同时
为零, 比梯形律更平滑; 姿态"旋转角"同样用它驱动, 于是角速度在两端自然
归零, 不需要单独的姿态梯形规划。
"""
from __future__ import annotations

import numpy as np


def quintic(t: float, T: float):
    """归一化五次多项式时间律, 返回 (s, ds/dt, d²s/dt²), s ∈ [0,1]。"""
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    sd = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    sdd = (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / T ** 2
    return s, sd, sdd


def _rz(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class CircleTraj:
    """x-y 平面空间圆 + 绕世界 z 轴的同步姿态摆动（"搅拌"式跟踪场景）。

    p(t) = c + r [cos φ, sin φ, 0];   R(t) = Rz(θ) R0
    φ(t) = Φ · s(t/T)  (圆弧角);      θ(t) = A · sin(π · s(t/T))  (姿态摆角)

    姿态角用正弦包络: 中途摆到 +A、末端回到 0, 两端角速度自然为零。
    A 的行程预算: 摆动主要由腕部滚转关节 j7 吸收, |A| 必须远小于其
    ±166° 行程 —— 若让姿态连续转 2π×n 圈, j1+j5+j7 总行程(约 500°)也
    不够 720°, 必然撞限位（实测教训, 详见讲解文档）。
    """

    def __init__(self, center, radius, R0, duration, laps=1.0, ori_amplitude=0.0, phi0=0.0):
        self.c = np.asarray(center, float)
        self.r = float(radius)
        self.R0 = np.asarray(R0, float)
        self.Phi = 2.0 * np.pi * float(laps)      # 位置圈数: 圆弧角可缠绕, 无行程问题
        self.phi0 = float(phi0)
        self.A = float(ori_amplitude)
        self.duration = float(duration)

    def sample(self, t: float):
        s, sd, _ = quintic(t, self.duration)
        phi = self.phi0 + self.Phi * s
        phid = self.Phi * sd
        th = self.A * np.sin(np.pi * s)
        thd = self.A * np.pi * np.cos(np.pi * s) * sd
        p = self.c + self.r * np.array([np.cos(phi), np.sin(phi), 0.0])
        R = _rz(th) @ self.R0
        v = self.r * phid * np.array([-np.sin(phi), np.cos(phi), 0.0])
        w = thd * np.array([0.0, 0.0, 1.0])
        return p, R, v, w


class LineTraj:
    """直线 p0 -> p1, 姿态恒定 R0。五次律起停; 用于奇异实验伸向边界。"""

    def __init__(self, p0, p1, R0, duration):
        self.p0 = np.asarray(p0, float)
        self.dp = np.asarray(p1, float) - self.p0
        self.R0 = np.asarray(R0, float)
        self.duration = float(duration)

    def sample(self, t: float):
        s, sd, _ = quintic(t, self.duration)
        p = self.p0 + s * self.dp
        v = sd * self.dp
        return p, self.R0, v, np.zeros(3)
