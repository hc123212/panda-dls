"""对比实验套件（M7/M8）: 奇异鲁棒性 / 零空间开关 / 前馈开关 / 运动学vs动力学。

用法:
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_experiments.py

输出到 results/: singularity.png, nullspace.png, feedforward.png, stage_ab.png
以及对应 npz 日志。全部为运动学层闭环（考察算法本身, 隔离执行器因素）,
仅 stage_ab 用动力学层。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from panda_dls import kinematics as kin
from panda_dls import viz
from panda_dls.control import DEMO_Q0, MODEL_PATH, run_tracking
from panda_dls.dls import DLSConfig
from panda_dls.traj import CircleTraj, LineTraj

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def circle_traj(q0=DEMO_Q0, duration=13.0):
    Th0 = kin.fk_hand(q0)
    p0, R0 = Th0[:3, 3], Th0[:3, :3]
    return CircleTraj(p0 - np.array([0.10, 0, 0]), 0.10, R0, duration,
                      laps=1.0, ori_amplitude=np.deg2rad(75))


def line_traj(length=0.32, duration=8.0):
    """从起始点沿 +x 伸向工作空间边界的直线（恒姿态）——喂给奇异实验。"""
    Th0 = kin.fk_hand(DEMO_Q0)
    p0, R0 = Th0[:3, 3], Th0[:3, :3]
    return LineTraj(p0, p0 + np.array([length, 0.0, 0.0]), R0, duration)


def save(log, name):
    np.savez_compressed(os.path.join(RESULTS, f"log_{name}.npz"), **log)


def exp_singularity():
    print("== 实验 1: 奇异鲁棒性 (伸向边界直线, 关速度限幅以暴露原始需求) ==")
    logs = {}
    for name, lam in (("伪逆 λ=0", "none"), ("固定 λ=0.15", "fixed"), ("自适应 λ", "adaptive")):
        cfg = DLSConfig(lam_mode=lam, lambda0=0.15, sigma_eps=0.03, clamp_dq=False, k_limit=0.0)
        logs[name] = run_tracking(line_traj(), cfg, mode="kinematic", q0=DEMO_Q0)
        log = logs[name]
        peak = log["dq_inf_raw"].max()
        print(f"  [{name:<10}] ‖dq_raw‖∞ 峰值 {peak:9.1f} rad/s | "
              f"σ_min 最小 {log['smin'].min():.4f} | 末端位置误差 {log['ep_mm'][-1]:7.1f} mm")
        save(log, f"sing_{lam}")
    viz.plot_lambda_comparison(logs, os.path.join(RESULTS, "singularity.png"))


def exp_nullspace_manip():
    print("== 实验 2a: 零空间可操作度最大化 (边界直线, 伸到 σ_min 掉 7 倍处) ==")
    logs = {}
    for name, km in (("零空间关", 0.0), ("可操作度最大化", 5.0)):
        cfg = DLSConfig(k_limit=0.0, k_manip=km, sigma_eps=0.03)
        logs[name] = run_tracking(line_traj(), cfg, mode="kinematic", q0=DEMO_Q0)
        log = logs[name]
        print(f"  [{name:<8}] 可操作度 起/末 {log['manip'][0]:.4f}/{log['manip'][-1]:.4f} | "
              f"全程最小 {log['manip'].min():.4f} | 末端位置误差 {log['ep_mm'][-1]:6.1f} mm")
        save(log, f"manip_{km}")
    viz.plot_manip_experiment(logs, os.path.join(RESULTS, "nullspace_manip.png"))


def exp_nullspace_limit():
    """起始 q7=-2.0 (距限位仅 0.9 rad): 姿态需求沿 j7 轴为负方向, 直接把 q7
    往限位推 —— 零空间中值吸引提前把 q7 拉回中段, 给摆动腾出行程。"""
    print("== 实验 2b: 零空间限位中值吸引 (q7 贴限位起始 + 空间圆 1 圈) ==")
    q0 = DEMO_Q0.copy()
    q0[6] = -1.8
    logs = {}
    for name, kl in (("限位吸引关", 0.0), ("限位吸引开", 0.5)):
        cfg = DLSConfig(k_limit=kl)
        logs[name] = run_tracking(circle_traj(q0), cfg, mode="kinematic", q0=q0)
        log = logs[name]
        margin_t = np.minimum(
            log["q"] - kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1] - log["q"])
        j = int(margin_t.min(axis=0).argmin())
        print(f"  [{name}] 最小限位裕度 {margin_t.min():.3f} rad (j{j + 1}) | "
              f"姿态误差 max {log['eo_deg'].max():.2f} deg | "
              f"RMS 位置误差 {np.sqrt(np.mean(log['ep_mm']**2)):.3f} mm")
        save(log, f"nslimit_{kl}")
    viz.plot_joint_posture(logs, os.path.join(RESULTS, "nullspace_limit.png"), joints=(4, 6))


def exp_feedforward():
    print("== 实验 3: 前馈 开/关 (空间圆 1 圈, 运动学层) ==")
    logs = {}
    for name, ff in (("有前馈", True), ("无前馈", False)):
        cfg = DLSConfig(use_ff=ff)
        logs[name] = run_tracking(circle_traj(), cfg, mode="kinematic", q0=DEMO_Q0)
        log = logs[name]
        print(f"  [{name}] RMS 位置误差 {np.sqrt(np.mean(log['ep_mm']**2)):.3f} mm / "
              f"max {log['ep_mm'].max():.3f} mm")
        save(log, f"ff_{ff}")
    viz.plot_error_comparison(logs, os.path.join(RESULTS, "feedforward.png"),
                              title="轨迹前馈 开/关 对比: 无前馈时纯反馈跟踪存在与速度成正比的滞后")


def exp_stage_ab():
    print("== 实验 4: 运动学层 vs 动力学层 (空间圆 1 圈) ==")
    logs = {}
    for name, mode in (("运动学层", "kinematic"), ("动力学层", "dynamic")):
        logs[name] = run_tracking(circle_traj(), DLSConfig(), mode=mode, q0=DEMO_Q0)
        log = logs[name]
        print(f"  [{name}] RMS 位置误差 {np.sqrt(np.mean(log['ep_mm']**2)):.3f} mm / "
              f"max {log['ep_mm'].max():.3f} mm | 姿态 max {log['eo_deg'].max():.4f} deg")
        save(log, f"stage_{mode}")
    viz.plot_error_comparison(logs, os.path.join(RESULTS, "stage_ab.png"),
                              title="运动学层 vs 动力学层: 差值即执行器带宽/重力带来的跟踪误差")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    exp_singularity()
    exp_nullspace_manip()
    exp_nullspace_limit()
    exp_feedforward()
    exp_stage_ab()
    print("全部实验完成, 图表见 results/")


if __name__ == "__main__":
    main()
