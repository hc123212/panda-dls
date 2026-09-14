"""实验 6 (T0): QP 约束控制器对比 —— 硬限位救援 + 末端安全平面。

用法:
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_qp.py

实验 A 贴限位起始 (q7 = −1.8, 距限位 1.1 rad, 空间圆 1 圈):
    DLS+clip     限位靠事后裁剪: 撞限位后关节被钉住, 任务误差持续增长
    DLS+零空间   中值吸引提前拉离限位 (已有结论), 但约束是软的、无保证
    QP           限位作为硬约束进入求解: 绝不越界 (含安全裕度), 任务损失最小化
实验 B 末端安全平面: 直线轨迹向下穿过 z = 0.35 m 平面, QP 把末端钳在平面上。

输出: results/qp_limit.png, results/qp_plane.png + log_qp_*.npz
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from panda_dls import kinematics as kin
from panda_dls.control import DEMO_Q0, run_tracking
from panda_dls.dls import DLSConfig
from panda_dls.qp import QPConfig
from panda_dls.traj import CircleTraj, LineTraj

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def circle_traj(q0, duration=13.0):
    Th0 = kin.fk_hand(q0)
    p0, R0 = Th0[:3, 3], Th0[:3, :3]
    return CircleTraj(p0 - np.array([0.10, 0, 0]), 0.10, R0, duration,
                      laps=1.0, ori_amplitude=np.deg2rad(75))


def save(log, name):
    np.savez_compressed(os.path.join(RESULTS, f"log_{name}.npz"), **log)


def exp_limit():
    print("== 实验 6a: 贴限位起始 (q7=-1.8) 空间圆 1 圈: DLS+clip vs DLS零空间 vs QP ==")
    q0 = DEMO_Q0.copy()
    q0[6] = -1.8
    traj = circle_traj(q0)
    logs = {
        "DLS+clip": run_tracking(traj, DLSConfig(k_limit=0.0), mode="kinematic", q0=q0),
        "DLS+零空间": run_tracking(traj, DLSConfig(k_limit=0.5), mode="kinematic", q0=q0),
        "QP 硬约束": run_tracking(traj, QPConfig(), mode="kinematic", q0=q0),
    }
    lo, hi = kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1]
    margin_cfg = 0.02
    for name, log in logs.items():
        ep = log["ep_mm"]
        q7 = log["q"][:, 6]
        worst = float(np.min(np.minimum(q7 - lo[6], hi[6] - q7)))
        print(f"  [{name:<8}] 位置RMS {np.sqrt(np.mean(ep**2)):7.3f} mm | 姿态max {log['eo_deg'].max():6.2f} deg"
              f" | q7 最小裕度 {worst:6.3f} rad")
        save(log, f"qp_limit_{name}")

    # 违限统计: DLS+clip 是否真的撞进限位 (q + 0 直接对比)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, log in logs.items():
        ax1.plot(log["t"], log["ep_mm"], lw=1.2,
                 label=f"{name} (RMS {np.sqrt(np.mean(log['ep_mm']**2)):.2f} mm)")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("位置误差 [mm]")
    ax1.set_title("主任务误差")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    for name, log in logs.items():
        ax2.plot(log["t"], log["q"][:, 6], lw=1.2, label=name)
    ax2.axhline(lo[6], color="r", lw=1.0, ls="--", label="j7 下限")
    ax2.axhspan(lo[6], lo[6] + margin_cfg, color="r", alpha=0.2)   # QP 安全裕度带
    ax2.axhspan(hi[6] - margin_cfg, hi[6], color="r", alpha=0.1)
    ax2.set_ylim(lo[6] - 0.12, -1.55)                  # 聚焦到运动发生的一段
    ax2.set_xlim(log["t"][0], log["t"][-1])
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("q7 [rad]")
    ax2.set_title("q7 关节轨迹 (深色阴影 = QP 安全裕度带, 解贴边滑动不越界)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)
    fig.suptitle("关节限位: 事后裁剪 vs 零空间软约束 vs QP 硬约束", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "qp_limit.png"), dpi=150)
    plt.close(fig)


def exp_plane():
    print("== 实验 6b: 末端安全平面 (直线轨迹向下穿平面, 运动学层) ==")
    Th0 = kin.fk_hand(DEMO_Q0)
    p0, R0 = Th0[:3, 3], Th0[:3, :3]
    z_plane = 0.35
    target = p0 + np.array([0.05, 0.0, -0.22])
    print(f"  起始 z = {p0[2]:.3f} m, 目标 z = {target[2]:.3f} m, 平面 z = {z_plane} m")
    traj = LineTraj(p0, target, R0, duration=6.0)
    logs = {
        "DLS+clip": run_tracking(traj, DLSConfig(), mode="kinematic", q0=DEMO_Q0),
        "QP+安全平面": run_tracking(traj, QPConfig(safe_plane_z=z_plane), mode="kinematic", q0=DEMO_Q0),
    }
    for name, log in logs.items():
        pz = log["p"][:, 2]
        n_below = int(np.sum(pz < z_plane - 1e-4))     # 容差: 贴面时允许 0.1 mm 级数值抖动
        print(f"  [{name:<8}] 末端 z 终值 {pz[-1]:.4f} m | 明显低于平面步数 "
              f"{n_below} / {len(pz)} | 位置误差终值 {log['ep_mm'][-1]:6.1f} mm")
        save(log, f"qp_plane_{name}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, log in logs.items():
        ax1.plot(log["t"], log["p"][:, 2], lw=1.4, label=name)
        ax2.plot(log["t"], log["ep_mm"], lw=1.2, label=name)
    ax1.axhline(z_plane, color="r", lw=1.2, ls="--", label="安全平面 z=0.35 m")
    ax1.plot(logs["DLS+clip"]["t"], logs["DLS+clip"]["p_d"][:, 2], lw=0.9, ls=":",
             color="0.4", label="参考轨迹")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("末端 z [m]")
    ax1.set_title("末端高度: QP 把末端钳在平面上")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("位置误差 [mm]")
    ax2.set_title("跟踪误差: 平面处的误差 = 参考轨迹的越界量 (有界, 回升后收敛)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig.suptitle("末端安全平面约束: 硬约束 vs 无约束跟踪", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "qp_plane.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    exp_limit()
    exp_plane()
    print("\n输出: results/qp_limit.png, results/qp_plane.png + log_qp_*.npz")


if __name__ == "__main__":
    main()
