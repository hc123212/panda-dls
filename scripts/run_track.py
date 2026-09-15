"""主 demo: Panda 空间圆轨迹 + 姿态同步旋转, 闭环跟踪（默认力矩级操作空间控制）。

用法:
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_track.py            # 力矩级 OSC（默认）
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_track.py --servo    # PD 位置舵机基线
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_track.py --kin      # 运动学层
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_track.py --laps 2 --dur 26

输出 (results/):
    track_{tag}.png   误差曲线;  joints_{tag}.png  关节轨迹+限位带
    log_{tag}.npz     原始日志;  demo.gif / demo.mp4  轨迹动画
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from panda_dls import kinematics as kin
from panda_dls import viz
from panda_dls.control import DEMO_Q0, MODEL_PATH, run_tracking, run_with_viewer
from panda_dls.dls import DLSConfig
from panda_dls.osc import OSCConfig, SCENE_MOTOR_PATH, run_osc_tracking, run_osc_viewer
from panda_dls.traj import CircleTraj

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def make_circle_traj(laps: float, duration: float):
    """从 DEMO_Q0 的实际位姿生成空间圆: 圆心在起始点 -x 方向 r 处, φ(0)=0。

    姿态摆幅 ±75°: 行程预算内 (j7 行程 ±166°) 且能明显展示位姿双任务跟踪。
    """
    Th0 = kin.fk_hand(DEMO_Q0)
    p0, R0 = Th0[:3, 3], Th0[:3, :3]
    r = 0.10
    center = p0 - np.array([r, 0.0, 0.0])
    return CircleTraj(center, r, R0, duration, laps=laps, ori_amplitude=np.deg2rad(75))


def summarize(tag: str, log: dict):
    ep, eo = log["ep_mm"], log["eo_deg"]
    smin, lam = log["smin"], log["lam"]
    margin = np.minimum(
        log["q"] - np.array([kin.JOINT_LIMITS[:, 0]]), np.array([kin.JOINT_LIMITS[:, 1]]) - log["q"])
    extra = ""
    if "tau" in log:
        extra = f" | 力矩 max {np.abs(log['tau']).max(axis=0).round(1)} N·m"
    print(f"[{tag}] 位置误差 RMS {np.sqrt(np.mean(ep**2)):.3f} mm / max {ep.max():.3f} mm | "
          f"姿态误差 max {eo.max():.4f} deg | "
          f"sigma_min [{smin.min():.4f}, {smin.max():.4f}] | "
          f"阻尼激活 {(lam > 0).mean() * 100:.1f}% 时间 | "
          f"限位最小裕度 {margin.min():.2f} rad{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kin", action="store_true", help="运动学层(默认力矩级 OSC)")
    ap.add_argument("--servo", action="store_true", help="PD 位置舵机基线(默认力矩级 OSC)")
    ap.add_argument("--laps", type=float, default=2.0)
    ap.add_argument("--dur", type=float, default=26.0)
    ap.add_argument("--view", action="store_true", help="实时 viewer 查看(不落盘)")
    ap.add_argument("--gif", action="store_true", default=True)
    args = ap.parse_args()

    mode = "kin" if args.kin else ("servo" if args.servo else "osc")
    label = {"kin": "运动学层", "servo": "位置舵机基线", "osc": "力矩级操作空间控制"}[mode]
    traj = make_circle_traj(args.laps, args.dur)

    if args.view:
        if mode == "osc":
            run_osc_viewer(traj, OSCConfig(), q0=DEMO_Q0, model_path=SCENE_MOTOR_PATH)
        else:
            # 运动学层无物理可看, viewer 一律走舵机基线
            run_with_viewer(traj, DLSConfig(), q0=DEMO_Q0, model_path=MODEL_PATH)
        return

    if mode == "osc":
        log = run_osc_tracking(traj, OSCConfig(), q0=DEMO_Q0, model_path=SCENE_MOTOR_PATH)
    elif mode == "kin":
        log = run_tracking(traj, DLSConfig(), mode="kinematic", q0=DEMO_Q0, model_path=MODEL_PATH)
    else:
        log = run_tracking(traj, DLSConfig(), mode="dynamic", q0=DEMO_Q0, model_path=MODEL_PATH)
    summarize(mode, log)

    os.makedirs(RESULTS, exist_ok=True)
    viz.plot_tracking(log, os.path.join(RESULTS, f"track_{mode}.png"),
                      title=f"{label}: 空间圆 {args.laps} 圈 + 姿态同步旋转 "
                            f"({args.dur:.0f} s, 五次多项式时间律)")
    viz.plot_joints(log, os.path.join(RESULTS, f"joints_{mode}.png"),
                    title=f"关节轨迹 ({label})")
    np.savez_compressed(os.path.join(RESULTS, f"log_{mode}.npz"), **log)

    if args.gif and mode != "kin":
        # GIF 只取第一圈(README 门面, 控制体积), 全程高清由 MP4 承担
        # 采样 20 Hz、写 20 fps, 回放速度与真实时间一致
        stride = int(0.05 / 0.001)
        q_all = log["q"][::stride]
        n_lap = int((args.dur / max(args.laps, 1e-9)) / 0.05) + 1
        viz.render_video(MODEL_PATH, q_all[:n_lap], os.path.join(RESULTS, "demo.gif"),
                         fps=20, width=640, height=400)
        viz.render_video(MODEL_PATH, q_all, os.path.join(RESULTS, "demo.mp4"),
                         fps=20, width=1280, height=800)
        print("媒体: results/demo.gif (首圈) / demo.mp4 (全程)")


if __name__ == "__main__":
    main()
