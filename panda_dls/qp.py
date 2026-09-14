"""QP 约束感知分辨率速度控制器，OSQP 求解（速度级，与 dls_step 同接口）。

每个控制周期求解一个小规模 QP（决策变量 dq ∈ R^7）:

    min_dq   ‖W^{1/2} (J dq − ẋ_d)‖²  +  ρ‖dq − dq_ref‖²  +  σ‖dq − dq_prev‖²
    s.t.     q_lo + m  ≤  q + dq·dt  ≤  q_hi − m        关节位置硬限位
             |dq|      ≤  dq_max                        关节速度硬限制
             |dq − dq_prev| ≤ ddq_max·dt                 关节加速度限制(可选)
             p_z + dt·(J_v dq)_z  ≥  z_plane             末端安全平面(可选)

设计要点:
    - 代价三项: 任务跟踪(加权最小二乘) + 姿态正则目标(零空间次任务以软方式进入)
      + 平滑项(抑制相邻周期解跳变); DLS 的"解析投影 + clip"在这里统一为
      "加权优化 + 硬约束", 限位/限速不再靠事后裁剪(裁剪会停摆丢任务),
      而是让解沿着可行域边界滑动;
    - OSQP 稀疏模式固定: 每周期只 update P 上三角 / q / l / u / A 的安全平面行,
      warm start 复用上一步解, 实测单步 p50 ≈ 0.1 ms, 满足 1 kHz 预算;
    - 求解失败时的降级: 回退到无约束闭式解再投影到约束盒(安全可用);
    - OSQP 容差内可能轻微越界, 最后做一次逐关节投影, 保证约束严格成立。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import osqp

from . import kinematics as kin
from .dls import PANDA_QD_MAX

_NQ = 7
#: 约束行布局 (43 行, 全部写成 A dq ≤ b 单边形式):
#:   [0:7]   关节下限   −dt·e_i,  b = q − q_lo − m
#:   [7:14]  关节上限   +dt·e_i,  b = q_hi − m − q
#:   [14:21] 速度上限   +e_i,     b = dq_max
#:   [21:28] 速度下限   −e_i,     b = dq_max
#:   [28:35] 加速度上限 +e_i,     b = dq_prev + ddq_max·dt   (关闭则 b=+inf)
#:   [35:42] 加速度下限 −e_i,     b = −dq_prev + ddq_max·dt  (关闭则 b=+inf)
#:   [42]    安全平面   −dt·J_v[z,:],  b = p_z − z_plane      (关闭则 b=+inf)
_N_CON = 43


@dataclass
class QPConfig:
    kp: float = 1.0                  # 位置误差增益 [1/s] (任务速度构造, 同 DLSConfig)
    ko: float = 1.0                  # 姿态误差增益 [1/s]
    use_ff: bool = True              # 是否叠加轨迹前馈速度
    w_ori: float = 0.3               # 姿态任务相对位置任务的权重 (位置 m vs 姿态 rad 量纲折中)
    rho: float = 1e-4                # ‖dq − dq_ref‖² 正则增益: 只需足够小以保持在
                                     #   任务子空间内的偏置可忽略; J 零空间内的预拉强度
                                     #   与 rho 无关 (该子空间任务项无曲率)
    sigma_smooth: float = 0.0        # ‖dq − dq_prev‖² 平滑增益 (0 = 关闭)
    k_posture: float = 0.3           # 次任务: 关节限位中值吸引增益 (归一化方向)
    limit_margin: float = 0.02       # 关节限位安全裕度 [rad]
    ddq_max: np.ndarray | None = None    # 关节加速度上限 [rad/s²], None = 不约束
    safe_plane_z: float | None = None    # 末端 z ≥ 该值 [m], None = 不约束
    dq_max: np.ndarray = field(default_factory=lambda: PANDA_QD_MAX.copy())


class QPController:
    """跨周期持有 OSQP 实例的控制器。用法: 每周期调用 step()。"""

    def __init__(self, cfg: QPConfig, dt: float):
        self.cfg = cfg
        self.dt = float(dt)
        self.dq_prev = np.zeros(_NQ)
        self.n_fallback = 0
        self.n_solve = 0
        self.solve_ms = []

        # ---- 固定稀疏结构: A (43x7) 与 P (7x7 上三角) ----
        # CSC 列主序: 第 j 列的非零行 = {j, 7+j, 14+j, 21+j, 28+j, 35+j, 42}
        rows = np.array([[j, 7 + j, 14 + j, 21 + j, 28 + j, 35 + j, 42] for j in range(_NQ)]).ravel()
        cols = np.repeat(np.arange(_NQ), 7)
        self._a_rows, self._a_cols = rows, cols
        A0 = np.zeros((_N_CON, _NQ))
        A0[np.arange(_NQ), np.arange(_NQ)] = -dt                # [0:7]
        A0[7 + np.arange(_NQ), np.arange(_NQ)] = dt              # [7:14]
        A0[14 + np.arange(_NQ), np.arange(_NQ)] = 1.0            # [14:21]
        A0[21 + np.arange(_NQ), np.arange(_NQ)] = -1.0           # [21:28]
        A0[28 + np.arange(_NQ), np.arange(_NQ)] = 1.0            # [28:35]
        A0[35 + np.arange(_NQ), np.arange(_NQ)] = -1.0           # [35:42]
        A0[42, :] = -dt * np.ones(_NQ)                           # 占位, 每步覆盖
        self._A = A0
        A_sp = sp.csc_matrix((A0[rows, cols], (rows, cols)), shape=(_N_CON, _NQ))

        # P 上三角 (含对角) 的 CSC 顺序索引: 列 j 的行 0..j
        tri = [(i, j) for j in range(_NQ) for i in range(j + 1)]
        self._p_rows = np.array([i for i, _ in tri])
        self._p_cols = np.array([j for _, j in tri])
        P0 = np.eye(_NQ)
        P_sp = sp.csc_matrix((P0[self._p_rows, self._p_cols],
                              (self._p_rows, self._p_cols)), shape=(_NQ, _NQ))

        self._solver = osqp.OSQP()
        # polishing 关闭: C 层 "Polishing not needed" 提示无法静音, 且容差 1e-9
        # + 求解后逐关节投影已足够保证边界精度
        self._solver.setup(P=P_sp, q=np.zeros(_NQ), A=A_sp,
                           l=-np.inf * np.ones(_N_CON), u=np.ones(_N_CON),
                           verbose=False, eps_abs=1e-9, eps_rel=1e-9,
                           max_iter=4000, polishing=False,
                           check_termination=5, warm_starting=True)
        self._l_vec = -np.inf * np.ones(_N_CON)

    # ------------------------------------------------------------------
    def step(self, q: np.ndarray, p: np.ndarray, J: np.ndarray, xd: np.ndarray,
             dt: float | None = None):
        """求解当前周期的 QP。

        参数: q 当前关节角, p 末端位置(world), J 6x7 雅可比, xd 6D 期望任务速度。
        返回: (dq, info), info 兼容 control._log_step 的键。
        """
        cfg = self.cfg
        dt = self.dt if dt is None else dt
        lo, hi = kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1]
        mid = kin.JOINT_LIMITS.mean(axis=1)
        half = (hi - lo) / 2.0

        # ---- 代价: P dq/2 + q_vec ----
        W = np.diag(np.concatenate([np.ones(3), np.full(3, cfg.w_ori)]))
        JW = J.T @ W
        dq_ref = cfg.k_posture * (mid - q) / half          # 次任务软目标
        P = JW @ J + (cfg.rho + cfg.sigma_smooth) * np.eye(_NQ)
        q_vec = -(JW @ xd + cfg.rho * dq_ref + cfg.sigma_smooth * self.dq_prev)

        # ---- 约束右端 ----
        b = np.empty(_N_CON)
        b[0:_NQ] = q - lo - cfg.limit_margin
        b[_NQ:2 * _NQ] = hi - cfg.limit_margin - q
        b[2 * _NQ:3 * _NQ] = cfg.dq_max
        b[3 * _NQ:4 * _NQ] = cfg.dq_max
        if cfg.ddq_max is not None:
            b[4 * _NQ:5 * _NQ] = self.dq_prev + cfg.ddq_max * dt
            b[5 * _NQ:6 * _NQ] = -self.dq_prev + cfg.ddq_max * dt
        else:
            b[4 * _NQ:6 * _NQ] = np.inf
        if cfg.safe_plane_z is not None:
            self._A[42, :] = -dt * J[2, :]                  # p_z + dt J_z dq ≥ z_plane
            b[42] = p[2] - cfg.safe_plane_z
        else:
            self._A[42, :] = 0.0
            b[42] = np.inf

        Ax = self._A[self._a_rows, self._a_cols]
        Px = P[self._p_rows, self._p_cols]

        # ---- OSQP: update + warm start solve ----
        import time
        t0 = time.perf_counter()
        self._solver.update(Px=Px, q=q_vec, l=self._l_vec, u=b, Ax=Ax)
        res = self._solver.solve()
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.n_solve += 1

        ok = res.info.status in ("solved", "solved inaccurate")
        if ok and res.x is not None and np.all(np.isfinite(res.x)):
            dq = np.asarray(res.x, float)
            fallback = False
        else:                                               # 降级: 无约束闭式解 + 投影
            dq = np.linalg.solve(P, -q_vec)
            fallback = True
            self.n_fallback += 1

        # ---- 严格可行性投影 (OSQP 容差内的小越界在此消除) ----
        dq = np.clip(dq, -cfg.dq_max, cfg.dq_max)
        dq = np.clip(dq, (lo + cfg.limit_margin - q) / dt, (hi - cfg.limit_margin - q) / dt)
        if cfg.ddq_max is not None:
            dq = np.clip(dq, self.dq_prev - cfg.ddq_max * dt, self.dq_prev + cfg.ddq_max * dt)
        if cfg.safe_plane_z is not None:
            # 若仍轻微越平面(理论上不会): 沿 J_z 方向最小修正
            viol = cfg.safe_plane_z - (p[2] + dt * float(J[2, :] @ dq))
            if viol > 0.0:
                jz = J[2, :]
                nn = jz @ jz
                if nn > 1e-12:
                    dq = dq + (viol / (dt * nn)) * jz

        self.dq_prev = dq.copy()

        # ---- info (兼容 _log_step) ----
        U, S, Vt = np.linalg.svd(J)
        smin, smax = float(S[-1]), float(S[0])
        n_active = int(np.sum(b - (self._A @ dq) < 1e-6)) if np.isfinite(b).any() else 0
        info = dict(
            smin=smin, smax=smax, cond=smax / max(smin, 1e-12), lam=0.0,
            manip=float(np.prod(S)),
            dq_task=dq, dq_null=np.zeros(_NQ), dq_raw=dq, scale=1.0,
            err_norm=float(np.linalg.norm(xd)),
            qp_iters=int(res.info.iter) if ok else -1,
            qp_status=str(res.info.status),
            qp_ms=self.solve_ms[-1],
            n_active=n_active, fallback=fallback,
        )
        return dq, info
