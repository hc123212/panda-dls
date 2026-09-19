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

#: 期望轨迹: 浅蓝不透明虚线 (走 opaque 通道)。半透明方案会在与金环投影
#: 重叠处混出灰青色 (0.45 alpha), 腕部摆动让重叠弧段随时间变化 -> GIF 里
#: 金灰交替; 虚线化后重叠面积大减, 重叠处是干净的深度遮挡而非混色
REF_RGBA = np.array([0.35, 0.65, 1.0, 1.0], dtype=np.float32)
#: 实际轨迹: 金黄实心 (先于半透明渲染)
ACTUAL_RGBA = np.array([1.0, 0.72, 0.15, 1.0], dtype=np.float32)

REF_RADIUS = 0.007      # 期望轨迹小球半径 [m]
ACTUAL_RADIUS = 0.0045  # 实际轨迹小球半径 [m]
REF_DASH = 3            # 期望轨迹虚线化: 每 REF_DASH 个点画 1 个

#: 自发光系数: 实测轨迹设 1.0, 颜色不依赖光照——期望轨迹悬在实测正上方,
#: 平行光(dir=0,0,-1)会把蓝环的影子正落在金环上, 不加自发光金色会被影子洗成灰色
REF_EMISSION = 0.35
ACTUAL_EMISSION = 1.0

#: 期望轨迹整体抬升 [m] (世界系)。同心半透明壳不可行: 蓝球前后两层半透明面
#: 都会叠在黄球上, 两层 0.28 的 alpha 把实心层洗成背景色 (实测像素为证)。
#: 抬升 3.5 cm 后两条轨迹平行分离, 从相机俯视角下清晰可辨。
REF_LIFT = np.array([0.0, 0.0, 0.035])

#: hand 原点 -> 指尖 (hand 系 z 轴)。手指滑移关节固定张开 0.04,
#: 指尖距 hand 原点约 0.098 m, 取 0.15 m 让轨迹挂在手指前方, 避免圆环穿进夹爪
TIP_OFFSET = np.array([0.0, 0.0, 0.15])


def tip_from_pose(p, R) -> np.ndarray:
    """hand 原点 + 姿态 -> 指尖世界坐标。"""
    return np.asarray(p, float) + np.asarray(R, float) @ TIP_OFFSET


def ref_tip(p, R) -> np.ndarray:
    """期望轨迹绘制点: 指尖再抬升 REF_LIFT, 与实测轨迹平行。"""
    return tip_from_pose(p, R) + REF_LIFT


def dash_ref(points) -> np.ndarray:
    """期望轨迹虚线化: 每 REF_DASH 个点画 1 个, 保留末点。"""
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) <= REF_DASH:
        return pts
    idx = np.unique(np.concatenate([np.arange(0, len(pts), REF_DASH), [len(pts) - 1]]))
    return pts[idx]


def tips_from_joints(q_arr) -> np.ndarray:
    """从关节角序列重放 FK, 返回指尖轨迹 (N,3)。

    用于离屏渲染: 日志里只存了 hand 原点位置, 姿态从 q 重放更省日志体积。
    """
    from . import kinematics as kin

    q_arr = np.asarray(q_arr, float).reshape(-1, 7)
    out = np.zeros((len(q_arr), 3))
    for i, q in enumerate(q_arr):
        Th = kin.fk_hand(q)
        out[i] = Th[:3, 3] + Th[:3, :3] @ TIP_OFFSET
    return out


def decimate(points, n_max: int) -> np.ndarray:
    """等间隔抽稀到至多 n_max 个点（保序, 首尾保留）。None 返回空数组。"""
    if points is None:
        return np.zeros((0, 3))
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts) <= n_max:
        return pts
    idx = np.unique(np.linspace(0, len(pts) - 1, n_max).round().astype(int))
    return pts[idx]


def add_trail(scene, points, rgba, radius: float, emission: float = 0.0) -> int:
    """向场景 geom 缓冲区追加一串小球体, 返回实际写入条数。

    容量不足时静默截断（调用方负责抽稀, 这里只做最后防线）。
    emission: 自发光系数, 1.0 = 纯 rgba 颜色不依赖光照（影子遮不住）。
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
        g = scene.geoms[scene.ngeom + i]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            size, pts[i], mat, rgba,
        )
        g.emission = emission
    scene.ngeom += n
    return n


def draw_user_trails(user_scn, ref_points=None, actual_points=None) -> dict:
    """重绘 viewer 的 user_scn 轨迹层。

    每次调用先清零 ngeom 再写入（user_scn 独立于主场景, 清零不影响
    机器人本体渲染）。期望轨迹占 1/3 预算, 剩余全部给实际轨迹。
    """
    user_scn.ngeom = 0
    ref_cap = user_scn.maxgeom // 3
    n_ref = add_trail(user_scn, dash_ref(decimate(ref_points, ref_cap)),
                      REF_RGBA, REF_RADIUS, REF_EMISSION)
    n_act = add_trail(user_scn, decimate(actual_points, user_scn.maxgeom - user_scn.ngeom),
                      ACTUAL_RGBA, ACTUAL_RADIUS, ACTUAL_EMISSION)
    return {"ref": n_ref, "actual": n_act}


def overlay_video_trails(scene, ref_points=None, actual_points=None) -> dict:
    """在 renderer.update_scene 之后向 scene 追加轨迹层。

    与 draw_user_trails 的区别: 不清零（模型 geom 由 update_scene 装入,
    清零会抹掉机器人本体）。期望/实际按 45%/55% 分配剩余容量。
    """
    budget = max(0, scene.maxgeom - scene.ngeom)
    ref_cap = int(budget * 0.45)
    n_ref = add_trail(scene, dash_ref(decimate(ref_points, ref_cap)),
                      REF_RGBA, REF_RADIUS, REF_EMISSION)
    n_act = add_trail(scene, decimate(actual_points, max(0, budget - n_ref)),
                      ACTUAL_RGBA, ACTUAL_RADIUS, ACTUAL_EMISSION)
    return {"ref": n_ref, "actual": n_act}
