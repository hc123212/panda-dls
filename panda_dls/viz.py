"""指标可视化：跟踪误差曲线 / 关节曲线 / 奇异对比 / 实验对比图 / GIF 渲染。"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
import mujoco

from . import kinematics as kinematics

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_tracking(log: dict, path: str, title: str = "笛卡尔空间轨迹跟踪误差"):
    """位置误差 (mm) 与姿态误差 (deg) 双纵轴曲线。"""
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(log["t"], log["ep_mm"], color="#1f77b4", lw=1.2, label="位置误差")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("位置误差 [mm]", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(log["t"], log["eo_deg"], color="#d62728", lw=1.2, label="姿态误差")
    ax2.set_ylabel("姿态误差 [deg]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    rms = float(np.sqrt(np.mean(log["ep_mm"] ** 2)))
    ax1.set_title(f"{title}\n位置误差 RMS {rms:.2f} mm / max {log['ep_mm'].max():.2f} mm, "
                  f"姿态误差 max {log['eo_deg'].max():.3f} deg")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_joints(log: dict, path: str, title: str = "关节轨迹与限位"):
    """7 个关节角轨迹 + 限位带。有 q_des 时叠加虚线（速度级），力矩层日志无此项。"""
    from . import kinematics as kin
    has_des = "q_des" in log
    fig, ax = plt.subplots(figsize=(9, 5))
    t = log["t"]
    lim = kin.JOINT_LIMITS
    for j in range(7):
        ax.fill_between([t[0], t[-1]], lim[j, 0], lim[j, 1], color="0.9", zorder=0)
        ax.plot(t, log["q"][:, j], lw=1.1, label=f"q{j + 1}")
        if has_des:
            ax.plot(t, log["q_des"][:, j], lw=0.7, ls="--", color=f"C{j}", alpha=0.6)
    for j in range(7):
        for b in lim[j]:
            ax.hlines(b, t[0], t[-1], color="r", lw=0.5, alpha=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("关节角 [rad]")
    suffix = "实线=实际, 虚线=期望" if has_des else "实线=实际 (力矩层无期望位置指令)"
    ax.set_title(title + f"\n（灰带=限位区间, 红线=限位边界, {suffix}）")
    ax.legend(ncol=7, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lambda_comparison(logs: dict, path: str):
    """奇异鲁棒性对比: 不同 λ 策略下 ||dq_raw||∞ (对数轴) 与 σ_min。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, log in logs.items():
        ax1.semilogy(log["t"], np.maximum(log["dq_inf_raw"], 1e-6), lw=1.1, label=name)
        ax2.plot(log["t"], log["smin"], lw=1.1)
    ax1.set_ylabel("‖dq_raw‖∞ [rad/s]（对数轴）")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    ax2.set_ylabel(r"最小奇异值 $\sigma_{min}$ [m]")
    ax2.set_xlabel("t [s]")
    ax2.grid(alpha=0.3)
    ax1.set_title("奇异鲁棒性对比: 伪逆(λ=0) vs 固定阻尼 vs 自适应阻尼\n"
                  "（末端伸向工作空间边界, σ_min → 0）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_comparison(logs: dict, path: str, title: str):
    """多组 run 的位置/姿态误差对比（零空间开关、前馈开关、运动学 vs 动力学）。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, log in logs.items():
        ax1.plot(log["t"], log["ep_mm"], lw=1.1, label=f"{name} (RMS {np.sqrt(np.mean(log['ep_mm']**2)):.2f} mm)")
        ax2.plot(log["t"], log["eo_deg"], lw=1.1)
    ax1.set_ylabel("位置误差 [mm]")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("姿态误差 [deg]")
    ax2.set_xlabel("t [s]")
    ax2.grid(alpha=0.3)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_manip_experiment(logs: dict, path: str, title: str = "零空间可操作度最大化（边界直线轨迹）"):
    """1x2: 可操作度 m(t) 对比 + 主任务误差对比（证明次任务不伤主任务）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, log in logs.items():
        ax1.plot(log["t"], log["manip"], lw=1.3, label=n)
        ax2.plot(log["t"], log["ep_mm"], lw=1.3, label=f"{n} (末端 {log['ep_mm'][-1]:.0f} mm)")
    ax1.set_ylabel(r"可操作度 $\sqrt{\det JJ^T}$")
    ax1.set_xlabel("t [s]")
    ax1.set_title("可操作度: 零空间梯度项让肘部保持折叠")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("位置误差 [mm]")
    ax2.set_xlabel("t [s]")
    ax2.set_title("主任务误差: 冗余优化不以牺牲跟踪为代价")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_joint_posture(logs: dict, path: str, joints=(5, 7), title: str = "同一末端轨迹的关节姿态选择"):
    """同轨迹下不同零空间策略的关节姿态对比（实线=q5, 虚线=q7）。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, log in logs.items():
        for j in joints:
            ax.plot(log["t"], log["q"][:, j], lw=1.2, ls="-" if j == joints[0] else "--",
                    label=f"{n}  q{j + 1}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("关节角 [rad]")
    ax.set_title(title + "\n（末端轨迹完全相同, 零空间决定冗余关节走哪条路）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_video(model_path: str, q_seq: np.ndarray, path: str, fps: int = 10,
                 width: int = 960, height: int = 600,
                 ref_points: np.ndarray | None = None,
                 actual_points: np.ndarray | None = None):
    """渲染关节序列 -> GIF 或 MP4 (按扩展名自动选择, 流式写入控制内存)。

    ref_points / actual_points: 末端期望与实际轨迹点 (N,3), 传入则在场景中
    叠加绘制（蓝=期望, 橙=实际, 后者随帧数增长, 与 q_seq 逐帧对齐）。
    """
    from .trail import (ACTUAL_RGBA, REF_RADIUS, REF_RGBA, ACTUAL_RADIUS,
                        REF_EMISSION, ACTUAL_EMISSION, add_trail, decimate)

    model = mujoco.MjModel.from_xml_path(model_path)
    model.vis.global_.offwidth = width        # 默认离屏缓冲 640x480, 大分辨率需先扩
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.32, 0.0, 0.52]
    cam.distance = 1.25
    cam.elevation = -18.0
    cam.azimuth = 140.0

    ref_all = actual_all = None
    n_ref = n_act = 0
    if ref_points is not None or actual_points is not None:
        data.qpos[:7] = q_seq[0] if len(q_seq) else 0.0
        data.qpos[7:] = 0.04
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)   # 探测一次模型 geom 数以分配容量
        budget = renderer.scene.maxgeom - renderer.scene.ngeom
        n_ref = min(0 if ref_points is None else len(ref_points), int(budget * 0.45))
        n_act = min(0 if actual_points is None else len(actual_points), max(0, budget - n_ref))
        ref_all = decimate(ref_points, n_ref) if n_ref else None
        act_len = min(len(actual_points), n_act) if actual_points is not None else 0
        actual_all = np.asarray(actual_points, float)[:act_len] if act_len else None

    n = 0
    if path.endswith(".mp4"):
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=6,
                                    pixelformat="yuv420p")
    else:
        writer = imageio.get_writer(path, mode="I", fps=fps, loop=0)
    with writer as w:
        for i, q in enumerate(q_seq):
            data.qpos[:7] = q
            data.qpos[7:] = 0.04
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            if ref_all is not None:
                add_trail(renderer.scene, ref_all, REF_RGBA, REF_RADIUS, REF_EMISSION)
            if actual_all is not None:
                add_trail(renderer.scene,
                          decimate(actual_all[: i + 1], n_act),
                          ACTUAL_RGBA, ACTUAL_RADIUS, ACTUAL_EMISSION)
            w.append_data(renderer.render())
            n += 1
    renderer.close()
    return n
