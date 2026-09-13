"""静态 IK 收敛测试（M3 验收）: 随机可达位姿统计收敛率与迭代次数。

用法:
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_ik_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from panda_dls import kinematics as kin
from panda_dls.dls import DLSConfig, solve_ik


def main(n_targets: int = 50):
    rng = np.random.default_rng(42)
    cfg = DLSConfig(k_limit=0.3)                      # 轻微限位吸引, 提高可行性
    ok, iters = 0, []
    for _ in range(n_targets):
        q_true = rng.uniform(kin.JOINT_LIMITS[:, 0] + 0.1, kin.JOINT_LIMITS[:, 1] - 0.1)
        Th = kin.fk_hand(q_true)                      # 目标取自 FK => 必可达
        q0 = np.clip(q_true + rng.normal(0, 0.35, 7), kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1])
        q, it = solve_ik(Th[:3, 3], Th[:3, :3], q0, cfg, rng=rng)
        if it > 0:
            ok += 1
            iters.append(it)
    iters = np.array(iters)
    print(f"目标数 {n_targets}, 收敛 {ok} ({100 * ok / n_targets:.0f}%)")
    if iters.size:
        print(f"迭代次数: mean {iters.mean():.1f}, max {iters.max()}")
    rate = ok / n_targets
    print(f"[{'PASS' if rate >= 0.95 else 'FAIL'}] 收敛率要求 >= 95%")
    sys.exit(0 if rate >= 0.95 else 1)


if __name__ == "__main__":
    main()
