"""在 MuJoCo 场景中叠加绘制末端轨迹（期望轨迹 + 实际轨迹）。

实现走 MjvScene 的用户 geom 通道, 不改模型 XML: 用 mjv_initGeom 向场景
geom 缓冲区写入小球体。viewer 用 launch_passive 句柄的 user_scn（每次
sync 前清零重写）; 离屏渲染在 update_scene 之后向 renderer.scene 追加
（模型 geom 已装入, 追加不参与下一帧重建）。

geom 总数受 scene.maxgeom 上限约束, decimate 先等间隔抽稀再写入,
超预算的部分按比例丢弃, 不影响主场景渲染。用小球而非折线段的原因:
mjGEOM_LINE 的渲染宽度固定为 1 像素, 在 GIF 里几乎不可见。
"""
from __future__ import annotations

import mujoco
import numpy as np

#: 期望轨迹: 半透明蓝
REF_RGBA = np.array([0.35, 0.65, 1.0, 0.55], dtype=np.float32)
#: 实际轨迹: 橙
ACTUAL_RGBA = np.array([1.0, 0.62, 0.35, 0.85], dtype=np.float32)

REF_RADIUS = 0.006    # 期望轨迹小球半径 [m], 比实际轨迹大一号: 跟踪贴合时露出蓝色外环
ACTUAL_RADIUS = 0.004  # 实际轨迹小球半径 [m]


def decimate(points, n_max: int) -> np.ndarray:
    """等间隔抽稀到至多 n_max 个点（保序, 首尾保留）。None 返回空数组。"""
    if points is None:
        return np.zeros((0, 3))
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) <= n_max:
        return pts
    idx = np.unique(np.linspace(0, len(pts) - 1, n_max).round().astype(int))
    return pts[idx]


def add_trail(scene, points, rgba, radius: float) -> int:
    """向场景 geom 缓冲区追加一串小球体, 返回实际写入条数。

    容量不足时静默截断（调用方负责抽稀, 这里只做最后防线）。
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    cap = scene.maxgeom - scene.ngeom
    n = min(len(pts), cap)
    if n <= 0:
        return 0
    size = np.array([radius, 0.0, 0.0])
    mat = np.eye(3).ravel()
    rgba = np.asarray(rgba, np.float32)
    for i in range(n):
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom + i],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size, pts[i], mat, rgba,
        )
    scene.ngeom += n
    return n


def draw_user_trails(user_scn, ref_points=None, actual_points=None) -> dict:
    """重绘 viewer 的 user_scn 轨迹层。

    每次调用先清零 ngeom 再写入（user_scn 独立于主场景, 清零不影响
    机器人本体渲染）。期望轨迹占 1/3 预算, 剩余全部给实际轨迹。
    """
    user_scn.ngeom = 0
    ref_cap = user_scn.maxgeom // 3
    n_ref = add_trail(user_scn, decimate(ref_points, ref_cap), REF_RGBA, REF_RADIUS)
    n_act = add_trail(user_scn, decimate(actual_points, user_scn.maxgeom - user_scn.ngeom),
                      ACTUAL_RGBA, ACTUAL_RADIUS)
    return {"ref": n_ref, "actual": n_act}


def overlay_video_trails(scene, ref_points=None, actual_points=None) -> dict:
    """在 renderer.update_scene 之后向 scene 追加轨迹层。

    与 draw_user_trails 的区别: 不清零（模型 geom 由 update_scene 装入,
    清零会抹掉机器人本体）。期望/实际按 45%/55% 分配剩余容量。
    """
    budget = max(0, scene.maxgeom - scene.ngeom)
    ref_cap = int(budget * 0.45)
    n_ref = add_trail(scene, decimate(ref_points, ref_cap), REF_RGBA, REF_RADIUS)
    n_act = add_trail(scene, decimate(actual_points, max(0, budget - n_ref)),
                      ACTUAL_RGBA, ACTUAL_RADIUS)
    return {"ref": n_ref, "actual": n_act}
