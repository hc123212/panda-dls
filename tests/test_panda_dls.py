"""pytest 套件: FK/雅可比与 MuJoCo 比对、四元数工具、DLS 静态收敛。

运行: cd demo && D:/pyenvs/robotics/Scripts/python.exe -m pytest tests/ -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import mujoco

from panda_dls import kinematics as kin
from panda_dls import dls
from panda_dls.quaternion import rot_log, R_from_q, q_from_R, slerp, chordal, chordal_to_angle

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "franka_emika_panda", "panda.xml")


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    return model, data


def test_fk_against_mujoco(mj):
    model, data = mj
    rng = np.random.default_rng(1)
    for q in rng.uniform(kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1], size=(80, 7)):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        Th_mj = data.xmat[model.body("hand").id].reshape(3, 3)
        Th = kin.fk_hand(q)
        assert np.linalg.norm(data.xpos[model.body("hand").id] - Th[:3, 3]) < 1e-9
        assert np.max(np.abs(Th_mj - Th[:3, :3])) < 1e-9


def test_jacobian_against_mujoco(mj):
    model, data = mj
    hand = model.body("hand").id
    rng = np.random.default_rng(2)
    for q in rng.uniform(kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1], size=(30, 7)):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, data.xpos[hand], hand)
        J_mj = np.vstack([jacp[:, :7], jacr[:, :7]])
        assert np.max(np.abs(J_mj - kin.hand_jacobian(q))) < 1e-8


def test_quaternion_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        assert np.max(np.abs(R_from_q(q_from_R(R_from_q(v))) - R_from_q(v))) < 1e-12


def test_rot_log_properties():
    # 恒等旋转 -> 零; 已知 90° 绕 z -> (0,0,pi/2)
    assert np.linalg.norm(rot_log(np.eye(3))) < 1e-12
    Rz90 = R_from_q([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    e = rot_log(Rz90)
    assert np.allclose(e, [0, 0, np.pi / 2], atol=1e-9)
    # 负实数特征值分支 (theta≈pi): 绕 x 转 pi
    Rx_pi = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(rot_log(Rx_pi), [np.pi, 0, 0], atol=1e-6)


def test_slerp_endpoints_and_chordal():
    q0 = q_from_R(np.eye(3))
    q1 = q_from_R(R_from_q([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))
    assert np.linalg.norm(slerp(q0, q1, 0.0) * np.sign(slerp(q0, q1, 0.0)[0]) - q0) < 1e-12
    assert abs(chordal_to_angle(chordal(R_from_q(q1), np.eye(3))) - np.pi / 2) < 1e-9


def test_dls_static_convergence():
    rng = np.random.default_rng(4)
    cfg = dls.DLSConfig(k_limit=0.3)
    ok = 0
    n = 20
    for _ in range(n):
        q_true = rng.uniform(kin.JOINT_LIMITS[:, 0] + 0.1, kin.JOINT_LIMITS[:, 1] - 0.1)
        Th = kin.fk_hand(q_true)
        q0 = np.clip(q_true + rng.normal(0, 0.35, 7), kin.JOINT_LIMITS[:, 0], kin.JOINT_LIMITS[:, 1])
        q, it = dls.solve_ik(Th[:3, 3], Th[:3, :3], q0, cfg)
        ok += it > 0
    assert ok / n >= 0.95


def test_trail_overlay_and_decimate(mj):
    from panda_dls import trail

    model, _ = mj
    pts = np.linspace([0.0, 0.0, 0.5], [0.2, 0.0, 0.5], 30)

    # 等间隔抽稀: 首尾保留, 数量正确
    d = trail.decimate(pts, 7)
    assert len(d) == 7 and np.allclose(d[0], pts[0]) and np.allclose(d[-1], pts[-1])
    assert np.array_equal(trail.decimate(pts, 100), pts)

    # 场景追加与容量截断
    scn = mujoco.MjvScene(model=model, maxgeom=1000)
    scn.ngeom = 3
    assert trail.add_trail(scn, pts, trail.REF_RGBA, 0.0035) == 30
    assert trail.add_trail(scn, pts, trail.ACTUAL_RGBA, 0.005) == 30
    assert scn.ngeom == 63
    assert trail.add_trail(scn, np.zeros((5000, 3)), trail.REF_RGBA, 0.004) == 1000 - 63
    assert scn.ngeom == scn.maxgeom

    # user_scn 重绘: 每次清零重建
    uscn = mujoco.MjvScene(model=model, maxgeom=1000)
    counts = trail.draw_user_trails(uscn, ref_points=pts, actual_points=pts)
    assert counts == {"ref": 30, "actual": 30} and uscn.ngeom == 60
    counts = trail.draw_user_trails(uscn, ref_points=pts, actual_points=None)
    assert counts == {"ref": 30, "actual": 0} and uscn.ngeom == 30

    # 空输入不写入
    assert trail.add_trail(scn, np.zeros((0, 3)), trail.REF_RGBA, 0.004) == 0
