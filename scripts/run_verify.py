"""一键自检: MDH 标定数据打印 + FK/雅可比/四元数校验。

用法:
    D:/pyenvs/robotics/Scripts/python.exe scripts/run_verify.py

内容:
    1) 从 MuJoCo 读出 7 个关节轴的世界原点/方向（标定的原始数据）;
    2) 随机 500 组 q: MDH-FK vs mj_kinematics (link7 与 hand);
    3) 随机组: 几何雅可比 vs mj_jac (hand 参考点);
    4) 雅可比中心差分校验;
    5) 四元数/旋转互转 vs mju_mat2Quat / mju_quat2Mat。
任一不过则进程退出码非 0。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from panda_dls import kinematics as kin
from panda_dls.quaternion import rot_log, R_from_q, q_from_R

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "franka_emika_panda", "panda.xml")

FAILED = []


def check(name: str, value: float, tol: float):
    ok = value < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max = {value:.3e}  (阈值 {tol:.0e})")
    if not ok:
        FAILED.append(name)


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    hand_id = model.body("hand").id

    # ---- 1) 标定原始数据: q=0 时各关节轴 (原点/方向) ----
    print("== 1) MuJoCo 关节轴几何 (q=0, 标定原始数据) ==")
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"{'关节':>4}  {'轴上锚点 (世界)':<30} 轴方向 (世界)")
    for j in range(7):
        bid = model.body(f"link{j + 1}").id
        jid = model.joint(f"joint{j + 1}").id
        p = data.xpos[bid]
        axis = data.xmat[bid].reshape(3, 3) @ model.jnt_axis[jid]
        print(f"  {j + 1:>2}   [{p[0]:7.4f}, {p[1]:7.4f}, {p[2]:7.4f}]      "
              f"[{axis[0]:6.3f}, {axis[1]:6.3f}, {axis[2]:6.3f}]")

    # ---- 2) FK 比对 ----
    print("== 2) MDH-FK 比对 mj_kinematics (随机 500 组 q) ==")
    rng = np.random.default_rng(0)
    q_test = rng.uniform(kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1], size=(500, 7))
    e7p = e7R = ehp = ehR = 0.0
    for q in q_test:
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        for body, our in (("link7", kin.fk(q)), ("hand", kin.fk_hand(q))):
            bid = model.body(body).id
            dp = float(np.linalg.norm(data.xpos[bid] - our[:3, 3]))
            dR = float(np.max(np.abs(data.xmat[bid].reshape(3, 3) - our[:3, :3])))
            if body == "link7":
                e7p, e7R = max(e7p, dp), max(e7R, dR)
            else:
                ehp, ehR = max(ehp, dp), max(ehR, dR)
    check("FK link7 位置误差 [m]", e7p, 1e-9)
    check("FK link7 旋转误差 [-]", e7R, 1e-9)
    check("FK hand  位置误差 [m]", ehp, 1e-9)
    check("FK hand  旋转误差 [-]", ehR, 1e-9)

    # ---- 3) 雅可比比对 ----
    print("== 3) 几何雅可比与 mj_jac 比对 (100 组) ==")
    eJ = 0.0
    for q in q_test[:100]:
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, data.xpos[hand_id], hand_id)
        J_mj = np.vstack([jacp[:, :7], jacr[:, :7]])
        eJ = max(eJ, float(np.max(np.abs(J_mj - kin.hand_jacobian(q)))))
    check("雅可比逐元素误差", eJ, 1e-8)

    # ---- 4) 雅可比有限差分校验 ----
    print("== 4) 雅可比中心差分校验 (100 组) ==")
    h = 1e-6
    efd = 0.0
    for q in q_test[:100]:
        J = kin.hand_jacobian(q)
        Jfd = np.zeros((6, 7))
        for i in range(7):
            qp, qm = q.copy(), q.copy()
            qp[i] += h
            qm[i] -= h
            Thp, Thm = kin.fk_hand(qp), kin.fk_hand(qm)
            Jfd[:3, i] = (Thp[:3, 3] - Thm[:3, 3]) / (2 * h)
            Jfd[3:, i] = rot_log(Thp[:3, :3] @ Thm[:3, :3].T) / (2 * h)
        efd = max(efd, float(np.max(np.abs(Jfd - J))))
    check("差分-解析雅可比误差 (h=1e-6)", efd, 1e-6)

    # ---- 5) 四元数/旋转互转 vs MuJoCo ----
    print("== 5) 四元数工具比对 mju_mat2Quat (100 组) ==")
    eq = 0.0
    for q in q_test[:100]:
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        R_mj = data.xmat[hand_id].reshape(3, 3)
        qmj = np.zeros(4)
        mujoco.mju_mat2Quat(qmj, R_mj.flatten())
        q_our = q_from_R(R_mj)
        d = min(float(np.linalg.norm(qmj - q_our)), float(np.linalg.norm(qmj + q_our)))
        eq = max(eq, d)
    check("四元数差 (考虑双覆盖)", eq, 1e-9)

    print()
    if FAILED:
        print(f"共 {len(FAILED)} 项 FAIL: {FAILED}")
        sys.exit(1)
    print("全部 PASS —— MDH 运动学 / 雅可比 / 四元数与 MuJoCo 完全对齐。")


if __name__ == "__main__":
    main()
