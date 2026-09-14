"""实验 5 (T0): 执行层级三方对比 —— 运动学层 / 位置舵机 / 操作空间控制。

用法:
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_osc.py

同一空间圆轨迹 (1 圈 13 s, 姿态摆幅 ±75°, 与 run_experiments.py 一致):
    kinematic  运动学层: q 直接积分, 只有算法误差 (≈0.04 mm)
    servo      位置舵机: q_des 发给 panda.xml 的 position 执行器 (≈4.2 mm)
    osc        操作空间控制: panda_motor.xml 力矩执行器 + Khatib 任务空间动力学

输出: results/osc_compare.png + log_osc_{kinematic,servo,osc}.npz
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from panda_dls import kinematics as kin
from panda_dls import viz
from panda_dls.control import DEMO_Q0, MODEL_PATH, run_tracking
from panda_dls.dls import DLSConfig
from panda_dls.osc import OSCConfig, run_osc_tracking
from panda_dls.traj import CircleTraj

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def circle_traj(q0=DEMO_Q0, duration=13.0):
    Th0 = kin.fk_hand(q0)
    p0, R0 = Th0[:3, 3], Th0[:3, :3]
    return CircleTraj(p0 - np.array([0.10, 0, 0]), 0.10, R0, duration,
                      laps=1.0, ori_amplitude=np.deg2rad(75))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    traj = circle_traj()
    logs = {}

    print("== 实验 5: 执行层级三方对比 (空间圆 1 圈 13 s) ==")
    logs["运动学层"] = run_tracking(traj, DLSConfig(), mode="kinematic", q0=DEMO_Q0)
    logs["位置舵机"] = run_tracking(traj, DLSConfig(), mode="dynamic", q0=DEMO_Q0,
                                    model_path=MODEL_PATH)
    logs["操作空间控制"] = run_osc_tracking(traj, OSCConfig(), q0=DEMO_Q0)

    print(f"\n{'层级':<10} {'位置RMS [mm]':>12} {'位置max [mm]':>12} {'姿态max [deg]':>13} {'力矩max [N·m]':>13}")
    for name, log in logs.items():
        ep, eo = log["ep_mm"], log["eo_deg"]
        tau_max = float(np.max(np.abs(log["tau"]))) if "tau" in log else float("nan")
        print(f"{name:<10} {np.sqrt(np.mean(ep**2)):>12.3f} {ep.max():>12.3f} "
              f"{eo.max():>13.4f} {tau_max:>13.2f}")

    for name, log in logs.items():
        tag = {"运动学层": "kinematic", "位置舵机": "servo", "操作空间控制": "osc"}[name]
        np.savez_compressed(os.path.join(RESULTS, f"log_osc_{tag}.npz"), **log)

    viz.plot_error_comparison(
        logs, os.path.join(RESULTS, "osc_compare.png"),
        title="执行层级三方对比: 运动学层 vs 位置舵机 vs 操作空间控制\n"
              "位置舵机的 4 mm 量级误差来自执行器带宽与无动力学补偿, OSC 把动力学显式纳入控制律")
    print("\n输出: results/osc_compare.png + log_osc_*.npz")


if __name__ == "__main__":
    main()
