# TCO-DRL 方法说明 README

> Trust-Aware and Cost-Optimized Blockchain Oracle Selection with Deep Reinforcement Learning

本仓库面向区块链预言机（Blockchain Oracle）选择问题：在一组具有不同服务类型、成本、处理速度、质押代币、信誉、验证成功率和潜在恶意行为风险的预言机中，为连续到达的请求选择合适的预言机或预言机组合，使系统在满足服务类型与截止时间约束的同时，尽可能提高任务成功率、降低响应时间、降低成本并减少恶意预言机风险。

本 README 重点说明仓库中各类方法的含义、适用场景和运行方式，便于读者理解不同 baseline 与改进方法之间的区别。

---

## 1. 仓库结构

```text
TCO-DRL/
├── TCO-DRL_with baseline/          # 仿真实验主代码：环境、模型、参数、结果输出
│   ├── main.py                     # 训练与评估入口
│   ├── env.py                      # 预言机选择仿真环境
│   ├── model.py                    # Random/RR/Earliest/SemiGreedy/BLOR/DQN/PPO/RA-DDQN/HCRL 等模型
│   ├── param_parser.py             # 所有实验参数与方法开关
│   ├── utils.py                    # 参数解析与辅助函数
│   └── output/                     # 每次实验自动生成的日志、结果表和模型参数
├── TCO-DRL_on blockchain/          # Ethereum/Ganache/Truffle 区块链部署版本
│   ├── contracts/                  # 智能合约
│   ├── build/contracts/            # 编译后的合约 ABI/JSON
│   └── python project/             # 与链上合约交互的 Python 代码
├── TCO-DRL_update smart contracts/ # 更新版智能合约相关代码
└── scripts/                        # PowerShell 一键运行脚本：quick test、COBRA、HCRL、消融、多种子实验
```

---

## 2. 问题定义

给定一个连续到达的请求序列，每个请求包含服务类型、到达时间、任务长度和截止时间。系统中存在多个预言机，每个预言机具有：

- **Service type**：可以处理的请求类型；
- **Cost**：调用成本；
- **Accuracy / processing speed**：处理能力或执行速度；
- **Token / stake**：质押代币，可作为可信先验；
- **Validation probability**：返回结果被验证为正确的概率；
- **Behavior probability**：正常、延迟、异常或恶意行为概率；
- **Reputation**：根据历史表现动态更新的信誉；
- **Load / idle time**：当前负载与等待时间。

目标是在每个请求到达时选择一个预言机，或选择“主预言机 + 备份预言机/委员会模式”，以优化以下指标：

- 总奖励（total reward）；
- 平均响应时间（average response time）；
- 成功率（success rate）；
- 截止时间内成功率（success-in-time rate）；
- 总完成时间（finish time）；
- 总成本（cost）；
- 服务类型匹配率（match rate）；
- 恶意/正常/可信预言机分配数量；
- 对 primary-backup 方法，还统计 backup 使用率、backup 恢复率、跳过 backup 比例等诊断指标；
- 对 HCRL，还统计 single/serial/parallel 模式比例，以及 cost/latency/risk 约束违反情况。

---

## 3. 方法总览

| 方法 | 类型 | 是否学习 | 是否考虑信誉 | 是否考虑成本 | 是否支持 backup | 核心作用 |
|---|---|---:|---:|---:|---:|---|
| Random | 基础 baseline | 否 | 否 | 否 | 否 | 随机选择预言机，作为最低基准 |
| Round-Robin | 基础 baseline | 否 | 否 | 否 | 否 | 按顺序轮询预言机，测试公平分配效果 |
| Earliest | 启发式 baseline | 否 | 间接 | 否 | 否 | 选择最早可用/等待时间最短的预言机 |
| BLOR | Bayesian bandit baseline | 部分 | 是 | 是 | 否 | 基于成功/失败历史估计可靠性，并加入成本惩罚 |
| SemiGreedy | 贪心 baseline | 否 | 取决于 reward | 是 | 否 | 基于当前即时 reward 和 cost 的一步贪心选择 |
| DQN / TCO-DRL | 深度强化学习 | 是 | 是 | 是 | 否 | 原始 TCO-DRL 主方法，学习长期 oracle selection 策略 |
| PPO | 策略梯度 RL baseline | 是 | 是 | 是 | 否 | 使用随机策略进行选择，作为 DQN 的策略梯度对照 |
| RA-DDQN | 风险感知 DDQN | 是 | 是 | 是 | 否 | 使用 Dueling Double DQN 降低 Q 值高估并增强风险感知 |
| PB-SafeDQN | Primary-backup 安全选择 | 是 | 是 | 是 | 是 | Dueling DDQN 选主预言机，规则型安全模块选备份 |
| COBRA-Oracle | 成本约束恢复感知 RL | 是 | 是 | 是 | 是 | teacher-guided primary selector + adaptive constrained backup gate |
| HCRL-Oracle | 层次化约束 RL | 是 | 是 | 是 | 是 | 学习 single/serial/parallel 模式、主预言机和备份预言机 |

---

## 4. 各个方法说明

### 4.1 Random

**Random** 是最简单的随机基线方法。对于每个请求，它在所有预言机中均匀随机选择一个预言机执行任务。

该方法不使用请求类型、成本、信誉、历史成功率或负载信息，因此主要用于判断其他方法是否显著优于随机选择。

**特点：**

- 不需要训练；
- 不使用状态信息；
- 不考虑服务类型匹配；
- 不考虑信誉、成本和风险；
- 适合作为最低性能参考。

---

### 4.2 Round-Robin

**Round-Robin** 按照固定顺序循环选择预言机。第 1 个请求选择第 1 个预言机，第 2 个请求选择第 2 个预言机，以此类推，超过预言机数量后重新从第 1 个开始。

该方法可以避免单个预言机被过度使用，但它不根据任务类型、信誉、成本或实时负载做自适应调整。

**特点：**

- 不需要训练；
- 分配较均匀；
- 不考虑当前请求类型；
- 不考虑预言机可靠性；
- 适合作为简单公平调度 baseline。

---

### 4.3 Earliest

**Earliest** 选择当前最早可用或等待时间最短的预言机。它关注系统排队与响应时间，试图减少请求等待。

该方法对降低短期等待时间有帮助，但可能选择低信誉、高成本或服务类型不匹配的预言机，因此在 validation-aware 或 risk-aware 场景中可能不稳定。

**特点：**

- 不需要训练；
- 主要优化短期等待时间；
- 不显式优化信誉和成本；
- 适合与 RL 方法比较“短视时间优化”和“长期收益优化”的差异。

---

### 4.4 BLOR

**BLOR** 是一个 Bayesian bandit 风格的 oracle selection baseline。它根据每个预言机历史成功次数和失败次数构建 Beta 后验分布，采样得到可靠性估计，并加入成本惩罚项：

```text
score = sampled_reliability - cost_weight * oracle_cost
```

得分最高的预言机会被选中。该方法比 Random/Round-Robin 更关注历史表现，也比纯贪心更具有探索性。

**特点：**

- 使用历史成功/失败反馈；
- 用 Beta 后验近似预言机可靠性；
- 通过 cost penalty 控制高成本预言机；
- 不使用深度神经网络；
- 适合作为非深度学习的自适应 baseline。

---

### 4.5 SemiGreedy

**SemiGreedy** 是一步贪心式启发方法。它先根据当前请求和候选预言机计算即时 reward 与 cost，然后在高 reward 候选中选择成本较低的预言机。

该方法可以在静态或简单环境中取得较好表现，但它只关注当前请求，不建模未来信誉变化、负载积累或验证概率衰减，因此在 rl_hard / rl_harder 等非平稳场景中容易陷入“低成本但高风险”的选择。

**特点：**

- 不需要训练；
- 计算当前即时 reward；
- 在近似最优 reward 候选中偏向低成本；
- 属于 myopic 策略；
- 可作为 RL 是否学到长期策略的对照。

---

### 4.6 DQN / TCO-DRL

**DQN** 是原始 TCO-DRL 的主要深度强化学习方法。系统将 oracle selection 建模为序列决策问题：

- **State**：请求类型、请求长度、截止时间、各预言机等待时间、信誉、成本、验证历史等信息；
- **Action**：选择一个预言机；
- **Reward**：由响应时间、成本、信誉、服务类型匹配和成功情况构成；
- **Policy**：通过 Q-learning 学习在不同状态下选择哪个预言机。

在默认设置中，DQN 对应原始 TCO-DRL 单预言机选择策略。在增强场景中，它可以结合 enhanced state、risk-aware reward 和 action mask 进行更稳健的训练。

**特点：**

- 学习长期回报，而非只看当前请求；
- 支持 replay buffer 和 target network；
- 可使用 action mask 限制只选择服务类型匹配的预言机；
- 是后续 RA-DDQN、PB-SafeDQN、COBRA 和 HCRL 的基础对照方法。

---

### 4.7 PPO

**PPO** 是策略梯度类强化学习 baseline。与 DQN 学习 Q 值不同，PPO 直接学习一个随机策略，即在给定状态下输出每个预言机被选择的概率。

仓库中的 PPO 是轻量级、稳定版实现，主要用于作为 DQN 类方法的策略梯度对照。它支持 action mask、reward clipping、return normalization 和概率裁剪，以减少数值不稳定。

**特点：**

- 学习随机策略；
- 适合作为 DQN 的对照；
- 可使用 action mask；
- 对连续训练和复杂 reward 更稳定；
- 通常不是本仓库最强主方法，而是重要 RL baseline。

---

### 4.8 RA-DDQN

**RA-DDQN** 是 Risk-Aware Dueling Double DQN。它在 DQN 的基础上加入两个重要改进：

1. **Double DQN**：缓解标准 DQN 对动作价值的过高估计；
2. **Dueling network**：将状态价值和动作优势分开建模，使模型更容易判断“当前状态本身好不好”以及“哪个动作更好”。

在 validation_stress、rl_hard 和 rl_harder 场景中，RA-DDQN 更适合作为风险感知的单预言机选择方法。

**特点：**

- 比普通 DQN 更稳定；
- 更适合风险感知 reward；
- 仍然是单预言机选择；
- 不直接引入 backup，但可作为 COBRA/HCRL 的 teacher。

---

### 4.9 PB-SafeDQN

**PB-SafeDQN** 是 primary-backup 风格的安全选择方法。它使用 Dueling Double DQN 选择主预言机（primary oracle），当主预言机失败或风险较高时，使用规则型安全模块选择同类型备份预言机（backup oracle）。

备份预言机的选择综合考虑：

- 最近成功率；
- 信誉；
- 当前负载；
- 成本；
- 质押代币；
- 行为风险；
- 延迟估计。

PB-SafeDQN 可以设置 backup 模式：

- **serial**：主预言机失败后再调用 backup；
- **parallel**：backup 作为 warm-standby 并行准备，降低恢复延迟；
- **always trigger**：主预言机失败就使用 backup；
- **cost-aware trigger**：只有当 backup 的预期收益超过成本和风险时才启用。

**特点：**

- 主预言机由 RL 学习；
- 备份预言机由可解释规则选择；
- 可以提高 failure recovery；
- 适合高风险、高验证失败率场景；
- 是 COBRA-Oracle 的重要前置版本。

---

### 4.10 COBRA-Oracle

**COBRA-Oracle** 可以理解为 PB-SafeDQN 的进一步强化版本，全称可写作 **Cost-Bounded Recovery-Aware Oracle Selection**。它保留 primary-backup 架构，但重点改进了三个方面：

1. **Teacher-guided primary selection**  
   早期训练阶段可以从 DQN 或 RA-DDQN 复制参数，并以逐渐衰减的概率跟随 teacher action，避免 primary selector 初期过弱。

2. **Adaptive backup gate**  
   backup 是否启用不再只依赖固定阈值，而是基于最近 backup utility 的均值和标准差动态调整阈值：

   ```text
   adaptive_threshold = max(min_backup_score, recent_mean + alpha * recent_std)
   ```

3. **Constrained reward**  
   同时考虑成本、延迟和恶意风险预算，通过 lambda penalty 控制违反约束的行为。

**特点：**

- 使用 Dueling Double DQN 作为 primary selector；
- 支持 DQN 或 RA-DDQN teacher warm-start；
- 使用 adaptive gate 控制 backup 使用频率；
- 将 cost、latency、risk 作为软约束；
- 目标是在高成功率和低冗余成本之间取得平衡；
- 是本仓库中 primary-backup 方向的核心改进方法。

---

### 4.11 HCRL-Oracle

**HCRL-Oracle** 是 Hierarchical Constrained Reinforcement Learning for Oracle Selection。它不再只判断“是否需要 backup”，而是将 oracle selection 拆成层次化决策：

1. **High-level mode policy**：选择执行模式；
   - `single`：只使用一个主预言机；
   - `serial`：主预言机失败后串行调用 backup；
   - `parallel`：主预言机和 backup 并行或 warm-standby 执行。

2. **Primary selector**：选择主预言机；
3. **Backup selector**：在需要 backup 时选择备份预言机。

HCRL 使用 Option Actor-Critic 风格的轻量级 actor-critic 策略分别学习 mode、primary 和 backup 三个子策略。它还支持：

- teacher warm-start；
- backup teacher guidance；
- action mask；
- GNN-style oracle state encoder；
- cost/latency/risk 的 primal-dual Lagrange multiplier 动态更新。

**特点：**

- 比 COBRA 更灵活，因为它学习 single/serial/parallel 模式；
- 主预言机和 backup 都由学习策略选择；
- 可以根据场景动态决定是否使用冗余；
- 适合 rl_harder 这类高风险、非平稳、隐藏真实验证概率的场景；
- 是本仓库中最完整、最复杂的改进方法。

---

## 5. 状态、奖励与场景设置

### 5.1 State Mode

```text
--State_Mode original
```

原始状态，主要包含请求类型、各预言机等待时间和信誉。

```text
--State_Mode enhanced
```

增强状态，在 original 基础上加入请求长度、截止时间、成本、处理能力、类型匹配、验证概率或历史成功率、最近负载等信息。

---

### 5.2 GNN Encoder

```text
--Use_GNN_Encoder
```

启用动态图消息传递式 oracle encoder。它将预言机视作图节点，根据服务类型、可靠性、负载和成本等关系进行 message passing，再将编码后的 oracle 表征输入 RL 模型。

HCRL-Oracle 默认适合搭配 GNN encoder，因为层次化策略需要更丰富的 oracle context。

---

### 5.3 Reward Mode

```text
--Reward_Mode original
```

保留原始 TCO-DRL 风格 reward，主要考虑成本、执行时间、信誉和服务类型匹配。

```text
--Reward_Mode risk_aware
```

引入 validation-aware success、行为风险、超时惩罚、响应时间惩罚和成本惩罚，更适合模拟恶意或不稳定预言机环境。

---

### 5.4 Success Mode

```text
--Success_Mode original
```

成功主要由 deadline 和服务类型匹配决定。

```text
--Success_Mode validation_aware
```

成功还要求预言机返回结果通过验证，因此更适合区块链 oracle 场景。

---

### 5.5 Scenario

```text
--Scenario static
```

原始静态场景，适合复现实验和基础方法比较。

```text
--Scenario validation_stress
```

低成本预言机不再总是可靠，用于测试方法是否能识别低成本高风险节点。

```text
--Scenario rl_hard
```

加入 bursty requests 和 fatigue traps。过度使用某些低成本节点会导致验证概率下降。

```text
--Scenario rl_harder
```

更困难的非平稳场景：隐藏真实验证概率、增强疲劳效应、收紧 deadline，并设置更强的 bait oracle。该场景更能体现长期 RL 策略、backup 和层次化控制的价值。

---

## 6. 快速运行

进入仿真实验目录：

```bash
cd "TCO-DRL_with baseline"
```

运行默认 baseline：

```bash
python main.py
```

运行包含 RA-DDQN、PB-SafeDQN 和 COBRA 的主实验：

```bash
python main.py \
  --Scenario rl_harder \
  --Use_RA_DDQN \
  --Use_PB_SafeDQN \
  --Use_COBRA \
  --Oracles_Per_Type 10 \
  --Epoch 30 \
  --Request_Num 6000 \
  --Reward_Scale 2.0 \
  --Reward_Clip 2.0 \
  --Dqn_lr 0.0015 \
  --RA_lr 0.0012 \
  --COBRA_lr 0.0014 \
  --Dqn_batch_size 128 \
  --Dqn_memory_size 10000 \
  --Dqn_epsilon_increment 0.0008
```

运行包含 HCRL-Oracle 的主实验：

```bash
python main.py \
  --Scenario rl_harder \
  --Use_RA_DDQN \
  --Use_PB_SafeDQN \
  --Use_COBRA \
  --Use_HCRL \
  --Oracles_Per_Type 10 \
  --Epoch 30 \
  --Request_Num 6000 \
  --Reward_Scale 2.0 \
  --Reward_Clip 2.0 \
  --Dqn_lr 0.0015 \
  --RA_lr 0.0012 \
  --COBRA_lr 0.0014 \
  --HCRL_lr 0.0014 \
  --HCRL_Mode_lr 0.0012 \
  --Dqn_batch_size 128 \
  --Dqn_memory_size 10000 \
  --Dqn_epsilon_increment 0.0008 \
  --Run_Tag hcrl_main
```

Windows PowerShell 用户也可以直接运行 `scripts/` 下的一键脚本，例如：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_01_main_cobra.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_09_hcrl_main.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_10_hcrl_ablation.ps1
```

---

## 7. 消融实验建议

为了验证每个模块的贡献，建议依次进行以下消融：

### 7.1 COBRA 消融

```bash
# 禁用 teacher warm-start / teacher guidance
python main.py --Scenario rl_harder --Use_RA_DDQN --Use_PB_SafeDQN --Use_COBRA --COBRA_No_Teacher

# 使用随机 backup，检验 safety-ranked backup selector 是否有效
python main.py --Scenario rl_harder --Use_RA_DDQN --Use_PB_SafeDQN --Use_COBRA --COBRA_Random_Backup

# 禁用 decoupled reward，检验 primary-only reward 是否有帮助
python main.py --Scenario rl_harder --Use_RA_DDQN --Use_PB_SafeDQN --Use_COBRA --COBRA_No_Decoupled_Reward
```

### 7.2 HCRL 消融

```bash
# 禁用 teacher
python main.py --Scenario rl_harder --Use_HCRL --HCRL_No_Teacher

# 禁用约束项
python main.py --Scenario rl_harder --Use_HCRL --HCRL_No_Constrained

# 使用随机 backup
python main.py --Scenario rl_harder --Use_HCRL --HCRL_Random_Backup

# 强制 single mode，检验 backup/committee 模式是否有效
python main.py --Scenario rl_harder --Use_HCRL --HCRL_Fixed_Single_Mode

# 强制 parallel mode，检验 learned mode policy 是否优于固定策略
python main.py --Scenario rl_harder --Use_HCRL --HCRL_Fixed_Parallel_Mode
```

---

## 8. 输出结果

每次运行会在 `output/` 下自动创建一个独立文件夹，文件夹名包含运行时间、Epoch、Request_Num、Scenario、Seed 和 Run_Tag。

典型输出包括：

```text
output/
└── YY_M_D_HH_MM_Epoch{N}_Req{M}_{Scenario}_Seed{S}_{Run_Tag}/
    ├── *.txt                         # 完整控制台日志
    ├── *_final_results.csv           # 最终结果表
    ├── *_final_results.json          # 最终结果 JSON
    ├── *.npz                         # DQN/PPO/RA-DDQN/COBRA/HCRL 模型参数
    └── *.pdf                         # reward 或性能曲线图
```

建议论文或报告中优先展示：

- success rate；
- success-in-time rate；
- average response time；
- cost；
- cost per success；
- malicious oracle assignment；
- trusted oracle assignment；
- backup recovery rate；
- conditional backup recovery rate；
- constraint violation rate；
- HCRL single/serial/parallel mode usage。

---

## 9. 区块链部署版本

`TCO-DRL_on blockchain/` 提供了链上测试版本，主要用于将 oracle selection 合约部署到 Ethereum/Ganache/Truffle 环境中，并通过 Python 与合约交互。

基本流程：

```bash
# 1. 启动 Ganache 并生成账户
端口和账户数量根据实验设置调整

# 2. 编译智能合约
truffle compile

# 3. 部署智能合约
truffle migrate

# 4. 修改 Python 代码中的 Web3 Provider、合约地址和 ABI 路径
# 5. 运行链上 oracle selection 实验
python main.py
```

注意：链上版本需要正确配置 Geth/Truffle/Ganache/Node/Web3 等依赖，并记录部署后的 contract address。

---

## 10. 方法选择建议

如果只是复现实验或快速检查环境，建议先运行：

```bash
python main.py --Epoch 1 --Request_Num 500
```

如果要比较原始 TCO-DRL 与传统 baseline，使用默认方法即可：

```bash
python main.py
```

如果要测试风险感知单预言机选择，建议加入：

```bash
--Use_RA_DDQN --Scenario rl_harder
```

如果要测试 primary-backup 安全恢复机制，建议运行：

```bash
--Use_RA_DDQN --Use_PB_SafeDQN --Use_COBRA --Scenario rl_harder
```

如果要测试最完整的层次化约束强化学习方法，建议运行：

```bash
--Use_RA_DDQN --Use_PB_SafeDQN --Use_COBRA --Use_HCRL --Scenario rl_harder
```

---

## 11. 一句话总结各方法

- **Random**：完全随机，最低基准。
- **Round-Robin**：均匀轮询，测试公平分配。
- **Earliest**：优先选等待时间最短的预言机。
- **BLOR**：用 Bayesian bandit 根据历史成功率和成本选预言机。
- **SemiGreedy**：根据当前 reward/cost 做一步贪心选择。
- **DQN / TCO-DRL**：原始深度 Q-learning oracle selection 方法。
- **PPO**：策略梯度 RL baseline。
- **RA-DDQN**：更稳定、更风险感知的 Dueling Double DQN。
- **PB-SafeDQN**：Dueling DDQN 主预言机 + 规则型安全 backup。
- **COBRA-Oracle**：teacher-guided primary selector + adaptive constrained backup gate。
- **HCRL-Oracle**：层次化学习 single/serial/parallel 模式、主预言机和 backup 的完整约束 RL 方法。
