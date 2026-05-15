# Text2Motion：从自然语言文本到人形机器人全身运动的端到端生成与部署系统
实际部署demo可见google drive: https://docs.google.com/videos/d/1yToQz0qyaa4KWoz-226hIxmNvIcGwF1jAWMcfbJhXKI/edit?usp=sharing
## 0. 项目全景

整个系统由三个子模块组成，全部由我深度参与：

| 子模块 | 角色 | 关键技术 |
|---|---|---|
| **RobotMDAR** | 文本条件动作生成 | MVAE 隐空间 + DDPM / Flow Matching 自回归生成 |
| **Tracker** | 跟踪策略 | RL（PPO / rsl_rl）训练关节级控制策略，跟住生成的参考轨迹 |
| **Deploy** | 仿真 + 真机部署链路 | Unitree Mujoco / G1 Edu 实机闭环，ONNX 推理，ROS2 |

**数据**：AMASS（SMPL 动作）+ BABEL-TEACH（文本标签）→ 通过 GMR retarget 到 Unitree G1 23-DOF / 智元 X2 31-DOF 骨架 → 重新 pack 为 50fps 训练集（约 8,285 条文本-动作对，105k 秒）。

**部署形态**：onboard 实时推理；Sim2Sim（Mujoco）与 Sim2Real（G1 Edu）一致性 ~95%。

---

## 1. 系统架构

```
                ┌─────────────────────┐
   文本指令 ──▶ │   CLIP Text Encoder │
                └─────────┬───────────┘
                          │ text_emb (512)
                          ▼
   历史帧 ─▶  ┌───────────────────────────────┐
              │   MVAE (Motion VAE)           │   动作隐空间
              │   Encoder / Decoder           │   z_t ∈ R^{T·D}
              └─────────┬─────────────────────┘
                        │ history latent
                        ▼
              ┌───────────────────────────────┐
              │  Conditional Generator        │
              │  ─ DAR (DDPM, 5 步)           │   ←─ 第一代
              │  ─ Flow Matching (10 步 Euler)│   ←─ 第二代
              │  ─ Flow A2A (1-2 步 NFE)      │   ←─ 自研第三代
              └─────────┬─────────────────────┘
                        │ 未来 N 帧 23/31-DOF 关节轨迹
                        ▼
              ┌───────────────────────────────┐
              │  Tracker (RL 控制策略)        │   ONNX export
              └─────────┬─────────────────────┘
                        ▼
              Unitree G1 Edu / 智元 X2 / Mujoco Sim
```

---

## 2. 我的具体工作

### 2.1 数据侧

**(a) 数据集蒸馏流水线（代码层）**
完整搭建可复现流水线：从已打包数据导出 feat-id 列表，回到 BABEL release 过滤，重新打包并算归一化统计量。便于做消融与快速迭代。

**(b) Retarget 后数据筛选**
SMPL（24 关节软体）→ G1/X2（23/31-DOF 刚体）retarget 后存在大量"非拟人"样本：
- 突变帧（IK 解算失败，手瞬移）
- 自穿模（胳膊穿胸腔、膝盖反关节）
- 整体姿态扭曲

把 retarget 结果**全部渲染成视频逐条审核**，人工剔除不合格样本。这一步直接决定了模型动作质量上限。

**(c) BABEL 脏标签清洗——`stand` 的定向修复**
BABEL 是人工标注的，`stand` 标签里混入了大量"挥手 / 扭身 / 踱步"。`stand` 在系统里语义等价于"待机"——这种污染直接导致模型在收到 `stand` 时表现不稳、用户体验崩坏。我针对 `stand` 类做了定向重清洗：用关键关节速度统计 + 重心位移阈值过滤，把假`stand` 样本从训练集中剔除。修复后模型 `stand` 行为显著稳定。

**(d) 骨架扩展**
从 G1 23-DOF 适配到 X2 31-DOF：编辑骨架描述（`description/robots/g1`）和 MJCF，验证 forward kinematics 与渲染一致性。

---

### 2.2 模型侧——三代生成器演进

#### 第一代：DAR（DDPM，5 步）
- 在 MVAE 隐空间中跑 5 步 DDPM 完成"text + history → 未来 N 帧 latent"。
- 5 步 NFE 是为了 onboard 实时推理而设的下限。
- 问题：5 步太少，**幻觉重**——同一文本反复采样会得到截然不同的动作；rollout 闭环 10 秒后明显漂移。

#### 第二代：Flow Matching（10 步 Euler）
- 同等推理时间预算下，**Flow Matching 在低 NFE 区间质量明显优于 DDPM**（直线流场更易蒸馏）。
- 我把 DDPM 替换成 Rectified Flow，10 步 Euler 在质量和速度上同时占优。
- 引入 **HighRoll Manager**：训练时让模型在长 rollout 上学会"自己接自己"，闭环稳定时间从 ~10s 提升到 30s+。

#### 第三代：Flow A2A（1-2 步 NFE，自研）
**问题**：
1. **站立死局**——数据集 `stand` 占比偏高，模型形成"站立吸引子"，发完 `stand` 再发新指令时有概率卡死不动。
2. 10 步 Euler 在 onboard 仍偏慢。

**思路来源**：A2A (Action-to-Action) 论文把 flow 起点从噪声换成"上一个 action latent"，流场学的是 latent → latent 而不是 noise → latent。我判断这能直接迁移到 text2motion：
- 起点和终点语义接近 → 流场近似直线 → **1-2 步 NFE 即可**
- 起点已含历史动作信息 → smoothness 自然变好
- 不再"从纯噪声出发" → 更容易跳出站立吸引子

**算法设计**（`FlowDenoiserA2ATransformer` + `A2ARectifiedFlow`）：

**结果**：
- 站立死局现象消失（stand → walk → punch 组合不再卡死）
- 推理 NFE 从 10 → **1-2** 步，onboard 实时推理跑通
- 相邻帧 latent 拉近，肉眼可见 smoothness 提升

**遗留问题与后续**：第一版 A2A "幻觉" 偏重，并且动作偏软（挥拳绵软、跳变成轻跃）。

---

### 2.3 训练工程

- **多卡 DDP 并行**：MVAE 与 Flow 都做了 DDP，单步吞吐 ≈ 单卡的 0.85N。
- **VAE + Flow 联合训练**：在第三代生成器上做 joint training（`train_mvae_then_flow.py`），让 latent 空间在生成阶段进一步整形。
- **阶段式训练管理器**：`manager_stage3_stable` / `manager_flow_highroll` / `manager_flow_constrained`，分阶段切换数据采样策略与 rollout 长度。
- **梯度健壮性**：`train_flow.py` 在每步显式检查 grad 的 NaN/Inf 并跳过 `optimizer.step()`（避免污染 EMA / Adam 状态，至少救过 3 次训练）。
- **采样权重**：`cal_action_statistics.py` 按动作类别频率倒数生成采样权重，减轻 BABEL 分布长尾。

---

### 2.4 部署侧（Sim2Sim / Sim2Real）

**Sim2Sim（Unitree Mujoco）**
- 链路：`textop_onnx_controller`（ROS2 launch，C++）↔ `unitree_mujoco`（C++ 仿真）↔ `rmdar.py`（Python 推理节点）。
- ONNX Runtime 1.22 + ROS2 Foxy。
- 自定义 joystick（含键盘代理），由 `physics_joystick.h` 接管，模拟真机 L2/R2/A/B 按键流程。
- DAR/Flow 的开关由 `/dar/toggle` 话题控制，由 ONNX controller 主动管理。

**Sim2Real（G1 Edu）**
- 部署拓扑：Tracker 跑在 G1 onboard 板（aarch64 编译），RobotMDAR 跑在带 GPU 的 PC，PC 通过 ROS2 网络下发参考动作。
- 执行流程：进入 Debug Mode → onboard zero-torque → Start 平滑过渡到默认姿态 → A 启动 Tracker + RobotMDAR → 文本指令实时下发。
- **安全机制**：onboard controller 对所有关节角度 / 关节速度做硬阈值监控，越界立即切断；joystick `B` 键紧急停 Tracker；网络断流时 Tracker 持续跟踪最后一帧参考。
- **Sim/Real 一致性 ~95%**：通过对齐仿真与真机的 PD 增益、传感器频率、滤波器、控制周期实现。

