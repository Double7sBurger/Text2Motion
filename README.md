# Text2Motion：从自然语言文本到人形机器人全身运动的端到端生成与部署系统

> 一句话概括：文本进去，机器人在仿真/真机中按指令走、跑、打拳、站立。
> 技术栈核心：**MVAE + 条件扩散 / Flow Matching + 跟踪策略**，三段式打通"语言 → 运动隐空间 → 关节轨迹 → 真机控制"。

---

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

### 2.1 数据侧（工作量最大、代码可见的最少）

**(a) 数据集蒸馏流水线（代码层）**
完整搭建 `export_babel_feat_list.py` → `filter_babel_release_by_feat_list.py` → `pack_dataset.py` → `cal_action_statistics.py` 的可复现流水线：从已打包数据导出 feat-id 列表，回到 BABEL release 过滤，重新打包并算归一化统计量。便于做消融与快速迭代。

**(b) Retarget 后数据筛选（人工，但关键）**
SMPL（24 关节软体）→ G1/X2（23/31-DOF 刚体）retarget 后存在大量"非拟人"样本：
- 突变帧（IK 解算失败，手瞬移）
- 自穿模（胳膊穿胸腔、膝盖反关节）
- 整体姿态扭曲

我把 retarget 结果**全部渲染成视频逐条审核**，人工剔除不合格样本。这一步直接决定了模型动作质量上限。

> 改进方向（已规划）：用关节角速度阈值 / 末端执行器速度异常 / link-pair 距离 / 重心跳变 几个指标做自动化清洗。

**(c) BABEL 脏标签清洗——`stand` 的定向修复（最关键的一段）**
BABEL 是人工标注的，`stand` 标签里混入了大量"挥手 / 扭身 / 踱步"。`stand` 在系统里语义等价于"待机"——这种污染直接导致模型在收到 `stand` 时表现不稳、用户体验崩坏。我针对 `stand` 类做了定向重清洗：用关键关节速度统计 + 重心位移阈值过滤，把假阳性 `stand` 样本从训练集中剔除。修复后模型 `stand` 行为显著稳定。

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
- 引入 `rollout_cap=0.8` 截断历史窗口里离当前最远的 20% 信号，避免长序列分布漂移导致的爆炸。

#### 第三代：Flow A2A（1-2 步 NFE，自研）
**问题**：
1. **站立死局**——数据集 `stand` 占比偏高，模型形成"站立吸引子"，发完 `stand` 再发新指令时有概率卡死不动。
2. 10 步 Euler 在 onboard 仍偏慢。

**思路来源**：A2A (Action-to-Action) 论文把 flow 起点从噪声换成"上一个 action latent"，流场学的是 latent → latent 而不是 noise → latent。我判断这能直接迁移到 text2motion：
- 起点和终点语义接近 → 流场近似直线 → **1-2 步 NFE 即可**
- 起点已含历史动作信息 → smoothness 自然变好
- 不再"从纯噪声出发" → 更容易跳出站立吸引子

**算法设计**（`FlowDenoiserA2ATransformer` + `A2ARectifiedFlow`）：

起点构造：
```
z_0 = Pool_f(history_latent) + γ · Proj_in(CLIP(text))
```
- `Pool_f`：1D Conv (k=1) + adaptive_avg_pool 把 history latent 时间维对齐到 noise 的 T_f
- `Proj_in`：CLIP(512) → 2 层 MLP → noise latent 形状
- `γ`：可学习标量，初值 0.3（让模型早期更依赖 history，后期自己学最优注入强度）
- Transformer backbone 完全保持原 `FlowDenoiser` 不变 → `RectifiedFlow` / `loop_flow` 链路可零修改切换

**两个防坍缩 trick**：
1. **Hybrid-noise fallback**：每个 batch 以 `hybrid_noise_prob=0.3` 随机把 z_0 换回纯高斯噪声，强制模型保留"无 history 也要听 text"的能力，CFG 不失效。
2. **Anchor loss**：`MSE(Proj_out(z_1_hat), CLIP(text))`，把预测出来的 latent 反投回 CLIP 空间，给 text 语义加一条直接监督锚（权重 0.3）。

**训练-推理 gap 缓解**：训练时 z_0 由 GT history 编码得到，推理时由上一帧生成结果得到。给 z_0 加 `z0_noise_scale=0.05` 的高斯扰动模拟暴露偏差。

**结果**：
- 站立死局现象消失（stand → walk → punch 组合不再卡死）
- 推理 NFE 从 10 → **1-2** 步，onboard 实时推理跑通
- 相邻帧 latent 拉近，肉眼可见 smoothness 提升

**遗留问题与后续**：第一版 A2A "幻觉" 偏轻——动作偏软（挥拳绵软、跳变成轻跃）。分析认为 z_0 太接近 history 让 velocity 输出被拉住。计划方向：γ / z0_noise_scale / hybrid_noise_prob 调参；reflow 第二阶段蒸馏；训练 loss 中加一个动作能量项鼓励大幅度动作。

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

---

## 3. 关键工程坑（按重要度）

| 坑 | 现象 | 解决 |
|---|---|---|
| **站立吸引子** | 发 stand 后下一条指令卡死 | Flow A2A：起点引入 history + text 双路径 |
| **Rollout 漂移** | 闭环 10s+ 后动作发散 | HighRoll Manager + rollout_cap=0.8 |
| **DDPM 5 步幻觉** | 同输入采样结果差异大 | 替换为 Flow Matching，10 步 Euler |
| **Retarget 不拟人** | IK 失败、自穿模 | 渲染逐条审核 + 人工剔除 |
| **BABEL `stand` 脏标签** | stand 表现不稳定 | 关节速度 + 重心位移阈值定向重清洗 |
| **训练 NaN** | 偶发 grad 爆炸污染 EMA | 每步 grad NaN 检测 + 跳过 step |
| **CLIP 维度选择** | ViT-L/14 (768) 在小数据集过拟合 | 退回 ViT-B/32 (512) |
| **history_len 选择** | history=4 时 VAE 重建反而变差 | 定在 history_len=2 |
| **Flow 训练不稳** | Adam 1e-4 直接训不稳 | 加 warmup + `anneal_lr=True` |

---

## 4. Flow Matching 演进对照表

| 版本 | Manager | 起点 z_0 | NFE | 主要解决 | 代价 |
|---|---|---|---|---|---|
| **DAR (DDPM)** | manager.py | 高斯噪声 | 5 | onboard 实时基线 | 幻觉重，长 rollout 漂移 |
| **Flow v1** | manager.py | 高斯噪声 | 10 | 替代 DDPM，低 NFE 质量更稳 | rollout 后期会漂 |
| **Flow HighRoll** | FlowHighRollManager | 高斯噪声 | 10 | 闭环稳定 30s+ | 训练时间略增 |
| **Flow A2A**（自研） | 同 highroll | **Pool(history) + γ·Proj_in(text)** | **1-2** | 消除站立死局 + onboard 真正实时 | 第一版动作偏软 |

---

## 5. 复盘 & 后续方向

1. **Reflow 蒸馏**：A2A 已经把 NFE 压到 1-2，再做 Rectified Flow 第二阶段 reflow 有机会蒸到单步。
2. **物理约束进训练 loss**：目前 physics cost 仅作 sampling-time guidance，纳入训练 loss 可让模型自身朝物理可行收敛。
3. **多模态条件**：CLIP text 之外加目标位姿 / 朝向 / 示范片段做 few-shot。
4. **真机闭环自适应**：当前 Tracker 离线训练，下一步想做生成器在 loop 里根据 Tracker residual 在线自适应。
5. **数据自动清洗**：用关节速度 / 末端执行器异常 / 自穿模检测 替代人工逐条审核，规模化扩到新数据源。

---

## 6. 一句话 takeaway

这个项目把 **生成模型（VAE / DDPM / Flow Matching / A2A 变体）** 与 **机器人控制（retarget / 自回归 rollout / Sim2Real 部署）** 两条线扎实打通。
**5 步 DDPM 幻觉重 → 10 步 Flow Matching 稳定 → HighRoll 闭环 30s+ → 自研 A2A 把 NFE 压到 1-2 步并消除站立死局**——每一步都是亲自踩坑、分析、找论文、改代码做出来的。

---

## 附：Git push 配置说明

我已经在你机器上验证过：
- `git config --global user.name` → `yukang hu`
- `git config --global user.email` → `yukanghu7@163.com`
- `ssh -T git@github.com` → 成功识别为 **Double7sBurger**

所以**不需要再额外配置**，直接可以 push 到 `git@github.com:Double7sBurger/Text2Motion.git`：

```bash
cd /home/agiuser/yukang/workspace/motion2text
git init
git add README.md
git commit -m "Add Text2Motion technical README"
git branch -M main
git remote add origin git@github.com:Double7sBurger/Text2Motion.git
git push -u origin main
```

如果远端仓库已经有 README/初始 commit 导致 push 被拒，可以先 `git pull --rebase origin main` 再 push；如果仓库是空的就直接 push 即可。
