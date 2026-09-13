<div align="center">

# panda-dls

**基于 MuJoCo 与阻尼最小二乘（DLS）的 7 自由度机械臂笛卡尔空间轨迹跟踪与姿态控制**

*Python 手写 MDH 运动学 · DLS 分辨率速度控制 · 自适应阻尼 · 零空间冗余优化 · MuJoCo 物理仿真闭环*

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.11-orange)
![核心依赖](https://img.shields.io/badge/核心数学-纯%20NumPy-informational)
![Tests](https://img.shields.io/badge/pytest-6%2F%206%20passed-brightgreen)
![平台](https://img.shields.io/badge/平台-Windows-0078D6?logo=windows)

![demo](results/demo.gif)

</div>

---

## 目录

- [特性](#特性)
- [结果](#结果)
- [安装](#安装)
- [快速开始](#快速开始)
- [原理](#原理)
- [目录结构](#目录结构)
- [实验](#实验)
- [文档](#文档)
- [定位与局限](#定位与局限)
- [致谢](#致谢)

## 特性

- 运动学全部手写：MDH（Craig 改进 DH）FK 与几何雅可比均为纯 NumPy 实现，随机 500 位形与 MuJoCo 官方实现（`mj_kinematics` / `mj_jac`）对拍到 $10^{-15}$ 量级；
- DLS 分辨率速度控制：阻尼 $\lambda$ 随最小奇异值 $\sigma_{\min}$ 在线调度，远离奇异时退化为伪逆（零偏置），接近奇异时平滑加阻尼；
- 位置与姿态双任务：姿态误差经 so(3) 对数映射进入 6D twist，参考姿态用四元数 slerp 生成，评估采用 chordal 距离换算物理转角；
- 冗余利用：零空间投影挂载可操作度最大化与关节限位中值吸引两个次任务，主任务误差不受影响；
- 结论可复现：阻尼策略、零空间、前馈、控制层级四组对照实验，一键出全部图表。

## 结果

| 指标 | 数值 |
|---|---|
| MDH-FK 对拍 `mj_kinematics`（500 随机位形） | 位置 / 旋转最大误差 $\sim 10^{-15}$ |
| 几何雅可比对拍 `mj_jac`（hand 参考点） | 逐元素误差 $3.7\times 10^{-15}$ |
| 静态 IK 收敛率（随机可达位姿，两阶段 + 多起点） | $96\%$（48/50） |
| 空间圆跟踪 · 运动学层（1 圈 13 s，1 kHz） | 位置 RMS $0.04$ mm，姿态 max $0.01^\circ$ |
| 空间圆跟踪 · 动力学层（PD 位置舵机） | 位置 RMS $4.2$ mm，姿态 max $1.28^\circ$ |
| 奇异实验（伸向工作空间边界，$\sigma_{\min}\to 0$） | 伪逆关节速度需求 $889$ rad/s vs 自适应 DLS $1.1$ rad/s |
| 零空间可操作度最大化 | 边界处可操作度保持 $3.5\times$，主任务误差几乎重合 |
| 零空间限位救援（$q_7$ 贴限位起始） | 关：撞限位 + 姿态误差 $6.58^\circ$；开：不贴限 + $0.01^\circ$ |
| 轨迹前馈 开 / 关 | 位置 RMS $0.04$ mm vs $46.2$ mm |

## 安装

**环境要求**：Windows、Python 3.12+、MuJoCo 3.11+（模型文件已随仓库放在 `models/`，无需额外下载）。

```bash
pip install mujoco numpy matplotlib imageio imageio-ffmpeg pytest
```

## 快速开始

```bash
# 按需把 python 换成你虚拟环境的解释器路径
cd demo
python scripts/run_verify.py        # 对拍自检：FK / 雅可比 / 四元数 vs MuJoCo（改代码后必跑）
python scripts/run_ik_test.py       # 静态 IK 收敛率统计
python -m pytest tests/ -q          # 单元测试（6 项）
python scripts/run_track.py         # 主 demo：动力学层闭环跟踪 + 图表 + GIF/MP4
python scripts/run_track.py --view  # 实时 viewer 演示
python scripts/run_experiments.py   # 四组对比实验 + 自动出图
```

`run_verify.py` 的预期输出以 5 个 `[PASS]` 结尾；`run_track.py` 结束后图表与媒体写入 `results/`。

## 原理

30 秒版本：参考轨迹给出期望位姿与前馈速度，手写 MDH 运动学算出当前位姿与雅可比，DLS 把 6D 误差映射为关节速度，MuJoCo 负责积分与渲染。

$$\Delta q \;=\; \underbrace{J^T\big(JJ^T+\lambda^2 I_6\big)^{-1}\big(K\,e+\dot{x}_d\big)}_{\text{主任务：DLS 分辨率速度控制}} \;+\; \underbrace{\big(I_7-J^T\big(JJ^T+\lambda^2 I_6\big)^{-1}J\big)\,k_n\,z(q)}_{\text{零空间次任务}}$$

- **误差** $e=[\,p_d-p,\;\log(R_d R^T)^\vee\,]$：姿态走 so(3) 对数映射（短弧、无双覆盖），与雅可比角速度行同处世界系；
- **自适应阻尼**：$\lambda^2=\lambda_0^2\big(1-(\sigma_{\min}/\varepsilon)^2\big)$（$\sigma_{\min}\ge\varepsilon$ 时取 0）——SVD 视角下每个奇异方向的增益由 $1/\sigma$ 压为 $\sigma/(\sigma^2+\lambda^2)$；
- **零空间**：$N(q)=I_7-J^+J$ 把次任务 $z(q)$（限位中值吸引 / 可操作度梯度）投影到不影响主任务的方向。

更完整的推导、逐模块实现细节与全部工程坑位见 [docs/讲解文档.md](docs/讲解文档.md)。

## 目录结构

```text
demo/
├── panda_dls/            # 核心包（除对拍外全部纯 NumPy 手写）
│   ├── kinematics.py     # MDH 参数表 / FK / 几何雅可比
│   ├── quaternion.py     # 四元数乘法 / so(3) 对数映射 / slerp / chordal 度量
│   ├── dls.py            # DLS 求解 / 自适应阻尼 / 零空间 / 两阶段静态 IK
│   ├── traj.py           # 五次多项式时间律 / 空间圆 / 直线轨迹
│   ├── control.py        # 运动学层 / 动力学层仿真闭环（1 kHz）
│   └── viz.py            # 指标曲线 / 实验图 / GIF + MP4 渲染
├── models/               # Franka Panda 模型（来自 mujoco_menagerie）
├── scripts/              # run_verify / run_ik_test / run_track / run_experiments
├── tests/                # pytest：对拍、四元数性质、DLS 收敛
├── results/              # 实验图表、demo.gif / demo.mp4、npz 日志
└── docs/                 # 技术方案 + 讲解文档
```

## 实验

| 图表 | 看什么 | 结论 |
|---|---|---|
| `results/singularity.png` | $\|\Delta q\|_\infty$（对数轴）与 $\sigma_{\min}$，$t\approx 6$ s 处到达边界 | 伪逆需求飙至 $889$ rad/s；固定阻尼平滑但滞后 $48$ mm；自适应阻尼 $1.1$ rad/s 且滞后最小 |
| `results/nullspace_manip.png` | 可操作度 $m(q)$ 与主任务误差 | 梯度项把边界处 $m$ 抬高 $3.5\times$，主任务不受损 |
| `results/nullspace_limit.png` | $q_5$ / $q_7$ 关节姿态与限位裕度 | 零空间提前把 $q_7$ 拉离限位，姿态误差 $6.58^\circ\to 0.01^\circ$ |
| `results/feedforward.png` | 前馈开/关的误差曲线 | 纯反馈滞后 $\propto$ 速度：$46$ mm $\to$ $0.04$ mm |
| `results/stage_ab.png` | 运动学层 vs 动力学层 | $0.04$ vs $4.2$ mm：算法误差与执行误差的分解 |
| `results/track_dyn.png` | 误差双轴曲线 | 误差峰与五次多项式速度包络同步，物理自洽 |

## 文档

| 文档 | 内容 |
|---|---|
| [docs/技术方案.md](docs/技术方案.md) | 立项时的完整技术方案（架构、数学施工图、里程碑与风险预案） |
| [docs/讲解文档.md](docs/讲解文档.md) | 开发全程实录、逐模块推导与实现细节、踩坑大全、实验解读、面试问答 |

## 定位与局限

本项目是教学与面试演示性质的运动学控制实现：运动学层已对拍到机器精度，动力学层采用现成 PD 位置舵机。以下事项明确不在范围内，扩展路线见讲解文档 §13：关节限位的加权最小二乘软约束、解析可操作度梯度、基于逆动力学的重力补偿力矩层、严格任务优先级 / QP 多任务框架、C++ / ROS2 移植。

## 致谢

- [MuJoCo](https://github.com/google-deepmind/mujoco) 与 [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0）——仿真引擎与 Franka Panda 模型均来自这两个项目；
- 姿态误差度量约定、DLS 参数整定方法参考了经典教材（Craig《机器人学导论》、Siciliano《Robotics: Modelling, Planning and Control》）与 Nakamura 的阻尼最小二乘文献。

---

*本项目为个人学习与求职演示用途，未配置开源许可证；如需引用其中实验数据请先联系作者。*
