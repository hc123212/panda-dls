"""panda_dls: Panda 机械臂 MDH 运动学 + DLS 笛卡尔空间轨迹跟踪。

模块:
    kinematics  MDH 参数表、正向运动学 FK、几何雅可比      (纯 NumPy 手写)
    quaternion  四元数/旋转工具: 乘法、log 映射、slerp      (纯 NumPy 手写)
    dls         阻尼最小二乘分辨率速度控制 + 零空间次任务    (纯 NumPy 手写)
    traj        五次多项式时间律、直线/空间圆轨迹
    control     MuJoCo 仿真闭环 (运动学层 / 动力学层)
    viz         指标曲线绘制与 GIF 渲染
"""
__version__ = "1.0.0"
