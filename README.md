# TCO-DRL with HCRL-Oracle：面向区块链预言机选择的层次化约束强化学习框架

> **Trust-aware and Cost-optimized Blockchain Oracle Selection with Hierarchical Constrained Reinforcement Learning**

本仓库围绕区块链预言机（Blockchain Oracle）选择问题展开：在连续到达的请求流中，系统需要从一组异构预言机中选择合适的执行节点，或选择“主预言机 + 备份预言机”的组合，使任务在满足服务类型与截止时间约束的同时，尽可能提高验证成功率、降低响应时间、降低调用成本，并减少恶意或不可靠预言机被分配的概率。

当前版本的核心方法是 **HCRL-Oracle**，即 **Hierarchical Constrained Reinforcement Learning for Oracle Selection**。该方法将传统的“选择一个预言机”问题扩展为“选择执行模式、选择主预言机、选择备份预言机”的层次化决策问题，并进一步引入审计感知声誉修正、图结构预言机编码和成本—延迟—风险约束优化。

---

## 目录

- [1. 仓库结构](#1-仓库结构)
- [2. 问题定义](#2-问题定义)
- [3. 方法总览](#3-方法总览)
- [4. HCRL-Oracle 算法设计](#4-hcrl-oracle-算法设计)
- [5. HCRL 的状态建模](#5-hcrl-的状态建模)
- [6. 审计感知声誉机制](#6-审计感知声誉机制)
- [7. 图结构预言机编码器](#7-图结构预言机编码器)
- [8. 层次化策略与执行模式](#8-层次化策略与执行模式)
- [9. 成本—延迟—风险约束奖励](#9-成本延迟风险约束奖励)
- [10. HCRL 与其他方法的对比](#10-hcrl-与其他方法的对比)
- [11. 安装与运行](#11-安装与运行)
- [12. 复现实验命令](#12-复现实验命令)
- [13. 消融实验](#13-消融实验)
- [14. 输出文件与指标解释](#14-输出文件与指标解释)
- [15. 推荐实验设置](#15-推荐实验设置)

---

## 1. 仓库结构

```text
TCO-DRL/
├── README.md
├── TCO-DRL_with baseline/
│   ├── main.py                     # 训练与评估主入口
│   ├── env.py                      # 预言机选择仿真环境与 HCRL 反馈逻辑
│   ├── model.py                    # DQN、PPO、RA-DDQN、Option Actor-Critic 等模型
│   ├── param_parser.py             # 方法选择、场景配置与超参数定义
│   ├── utils.py                    # 参数解析辅助函数
│   ├── scripts/                    # 一键运行实验脚本
│   ├── tools/                      # 实验结果汇总工具
│   └── output/                     # 自动保存日志、结果表和模型权重
├── TCO-DRL_on blockchain/          # 区块链部署相关代码
└── TCO-DRL_update smart contracts/ # 智能合约更新相关代码
```

主要仿真实验代码位于：

```text
TCO-DRL_with baseline/
```

---

## 2. 问题定义

### 2.1 区块链预言机选择任务

考虑一个连续到达的请求序列：

$$
\mathcal{R}=\{r_1,r_2,\ldots,r_T\}.
$$

每个请求 $r_t$ 由以下属性描述：

$$
r_t = (\tau_t, a_t, \ell_t, d_t),
$$

其中：

- $\tau_t$ 表示请求类型；
- $a_t$ 表示请求到达时间；
- $\ell_t$ 表示任务长度；
- $d_t$ 表示截止时间约束。

系统中存在 $N$ 个候选预言机：

$$
\mathcal{O}=\{o_1,o_2,\ldots,o_N\}.
$$

每个预言机 $o_i$ 具有服务类型、成本、处理能力、质押代币、验证概率、行为风险、历史信誉和当前负载等属性：

$$
o_i = (\tau_i, c_i, v_i, q_i, p_i^{val}, p_i^{beh}, rep_i, load_i).
$$

调度器需要在每个请求到达时选择一个执行动作。对于单预言机方法，动作是：

$$
a_t \in \{1,2,\ldots,N\}.
$$

对于 HCRL-Oracle，动作被拆分为层次化三元组：

$$
a_t^{HCRL} = (m_t, p_t, b_t),
$$

其中：

- $m_t$ 表示高层执行模式；
- $p_t$ 表示主预言机；
- $b_t$ 表示备份预言机，若当前模式不需要备份，则 $b_t=-1$。

### 2.2 优化目标

系统的总体目标是在长期请求序列上最大化累计收益：

$$
\max_\pi \; \mathbb{E}_{\pi}\left[\sum_{t=1}^{T}\gamma^{t-1} r_t\right],
$$

其中 $\pi$ 为调度策略，$\gamma$ 为折扣因子。奖励函数不仅关注任务是否成功，还综合考虑：

- 验证成功率；
- 截止时间内成功率；
- 响应时间；
- 调用成本；
- 恶意预言机分配率；
- 备份恢复能力；
- 审计后的可信程度；
- 成本、延迟和风险预算约束。

因此，该问题并不是简单的最短队列调度或最低成本选择，而是一个带有隐藏风险、非平稳可靠性和约束优化目标的序列决策问题。

---

## 3. 方法总览

当前代码支持以下方法。

| 方法 | 类型 | 是否学习 | 是否支持备份 | 是否显式考虑风险 | 核心思想 |
|---|---|---:|---:|---:|---|
| `Random` | 随机基线 | 否 | 否 | 否 | 在候选预言机中随机选择。 |
| `Round-Robin` | 轮询基线 | 否 | 否 | 否 | 按固定顺序循环选择预言机。 |
| `Earliest` | 启发式基线 | 否 | 否 | 间接 | 选择最早空闲或等待时间最短的预言机。 |
| `BLOR` | Bayesian bandit | 部分 | 否 | 间接 | 基于历史成功/失败记录估计可靠性并加入成本惩罚。 |
| `SemiGreedy` | 贪心基线 | 否 | 否 | 间接 | 根据当前即时收益和成本进行一步贪心选择。 |
| `DQN` | 深度强化学习 | 是 | 否 | 间接 | 学习单预言机选择的长期 Q 值。 |
| `PPO` | 策略梯度 RL | 是 | 否 | 间接 | 学习随机策略形式的单预言机选择器。 |
| `RA-DDQN` | 风险感知 DDQN | 是 | 否 | 是 | 使用 Dueling Double DQN 提高稳定性并适配风险奖励。 |
| `PB-SafeDQN` | Primary-backup RL | 是 | 是 | 是 | RL 选择主预言机，规则模块选择备份预言机。 |
| `COBRA-Oracle` | 约束恢复感知 RL | 是 | 是 | 是 | Teacher-guided primary selector + adaptive backup gate。 |
| `HCRL-Oracle` | 层次化约束 RL | 是 | 是 | 是 | 同时学习执行模式、主预言机和备份预言机。 |

其中，**HCRL-Oracle 是当前版本的主方法**，其设计目标是在困难场景下实现更稳健的安全调度。

---

## 4. HCRL-Oracle 算法设计

### 4.1 设计动机

传统预言机选择方法通常只回答一个问题：

> 当前请求应该交给哪个预言机？

然而，在更真实的区块链预言机环境中，这一问题往往过于简化。原因包括：

1. **低成本预言机可能是风险诱饵**：某些节点成本较低，但验证概率低或恶意行为概率高；
2. **历史信誉并不完全可靠**：节点可能通过短期正常行为积累信誉，然后在高负载或关键请求中作恶；
3. **可靠性具有非平稳性**：在 `rl_hard` 和 `rl_harder` 场景中，节点过度使用会导致疲劳效应，验证概率下降；
4. **备份机制存在成本—安全权衡**：无条件使用备份会提高成本，完全不用备份会提高失败风险；
5. **串行备份与并行备份适用场景不同**：串行备份节省成本但可能增加延迟，并行备份提高安全性但成本更高。

因此，HCRL-Oracle 不再将问题建模为单层动作选择，而是引入层次化控制结构：

$$
\pi_{HCRL}(a_t|s_t)=
\pi_m(m_t|s_t^m)\cdot
\pi_p(p_t|s_t^p)\cdot
\pi_b(b_t|s_t^b,m_t,p_t).
$$

其中：

- $\pi_m$ 是高层模式策略；
- $\pi_p$ 是主预言机策略；
- $\pi_b$ 是备份预言机策略；
- $s_t^m$、$s_t^p$、$s_t^b$ 分别表示模式、主预言机和备份预言机使用的状态表示。

这种设计使模型能够自适应地决定：

- 什么时候只使用单个预言机；
- 什么时候需要串行备份；
- 什么时候需要并行安全冗余；
- 哪个节点适合作为主节点；
- 哪个节点适合作为备份节点。

### 4.2 HCRL 的三个策略模块

在当前代码实现中，HCRL-Oracle 由三个可学习模块组成：

```text
HCRL_Mode
HCRL_Primary
HCRL_Backup
```

对应关系如下：

| 模块 | 策略符号 | 动作空间 | 功能 |
|---|---|---:|---|
| `HCRL_Mode` | $\pi_m$ | 执行模式数量 | 选择 single / serial / parallel 等高层模式。 |
| `HCRL_Primary` | $\pi_p$ | 预言机数量 $N$ | 选择主预言机。 |
| `HCRL_Backup` | $\pi_b$ | 预言机数量 $N$ | 在需要备份时选择备份预言机。 |

三个模块均采用轻量级 **Option Actor-Critic** 风格实现。对于任意策略 $\pi_\theta$，actor 输出动作分布，critic 估计状态价值：

$$
\pi_\theta(a|s)=\mathrm{Softmax}(f_\theta(s)),
$$

$$
V_\phi(s)=g_\phi(s).
$$

策略更新可以抽象表示为：

$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) A_t\right]
+ \beta \nabla_\theta \mathcal{H}(\pi_\theta),
$$

其中：

- $A_t$ 为优势函数；
- $\mathcal{H}(\pi_\theta)$ 为策略熵；
- $\beta$ 为熵正则系数，对应代码中的 `HCRL_AC_Entropy`。

critic 的目标是最小化价值估计误差：

$$
\mathcal{L}_V = \left(V_\phi(s_t)-\hat{R}_t\right)^2.
$$

代码中通过 `HCRL_AC_Value_Coef` 控制价值损失在更新中的权重。

---

## 5. HCRL 的状态建模

### 5.1 请求级状态

对于请求 $r_t$，模型首先使用请求类型、任务长度和截止时间构成请求级上下文：

$$
x_t^{req} =
\left[
\frac{\tau_t}{\tau_{max}},
\frac{\ell_t}{\bar{\ell}},
\frac{d_t}{d_{hard}}
\right].
$$

其中 $d_{hard}$ 对应困难场景中的 deadline 标准化常数。

### 5.2 预言机级状态

对于每个预言机 $o_i$，代码构造一组 oracle feature：

$$
x_{t,i}^{oracle}=\left[
wait_i,
rep_i^{eff},
cost_i,
acc_i,
match_i,
val_i,
load_i,
riskdelay_i,
token_i,
truth_i,
fail_i^{audit},
cooldown_i
\right].
$$

各项含义如下：

| 特征 | 含义 |
|---|---|
| $wait_i$ | 当前请求到达时，预言机 $i$ 的归一化等待时间。 |
| $rep_i^{eff}$ | 审计修正后的有效信誉。 |
| $cost_i$ | 归一化调用成本。 |
| $acc_i$ | 归一化处理能力。 |
| $match_i$ | 预言机服务类型是否与请求类型匹配。 |
| $val_i$ | 验证成功概率或历史验证成功估计。 |
| $load_i$ | 近期负载水平。 |
| $riskdelay_i$ | 行为风险与延迟风险的混合估计。 |
| $token_i$ | 归一化质押代币。 |
| $truth_i$ | 审计后验可信度。 |
| $fail_i^{audit}$ | 审计失败率估计。 |
| $cooldown_i$ | 审计冷却惩罚状态。 |

因此，增强状态可以写为：

$$
s_t^p = \left[x_t^{req}, x_{t,1}^{oracle},x_{t,2}^{oracle},\ldots,x_{t,N}^{oracle}\right].
$$

该状态用于主预言机策略 $\pi_p$ 和备份预言机策略 $\pi_b$。

### 5.3 模式策略状态

HCRL 的模式策略不仅使用基础状态，还额外加入全局风险摘要特征：

$$
s_t^m = [s_t^p, z_t],
$$

其中 $z_t$ 包含：

$$
z_t = [
slack_t,
risk_t^p,
ontime_t^p,
score_t^b,
gain_t^b,
pressure_t^c,
succ_t^{recent},
risk_t^{recent},
fail_t^{audit},
truth_t^{best\_backup}
].
$$

这些特征分别表示：

- primary deadline slack；
- primary estimated risk；
- primary on-time probability；
- best backup score；
- backup gain；
- backup cost pressure；
- recent success rate；
- recent risk level；
- recent audit failure rate；
- best backup audit truth score。

这使模式策略能够判断当前请求是否需要安全冗余，而不是固定使用某一种 backup 策略。

---

## 6. 审计感知声誉机制

### 6.1 基本思想

传统 reputation update 依赖历史成功率，容易受到伪装节点影响。恶意节点可以在早期保持正常行为积累信誉，然后在高价值或高负载场景中作恶。

为缓解该问题，代码中引入 **audit-aware reputation correction**。每个预言机维护一个审计后验：

$$
\alpha_i, \beta_i,
$$

其中：

- $\alpha_i$ 表示审计通过的可信证据；
- $\beta_i$ 表示审计失败或风险证据。

审计后验可信度定义为：

$$
truth_i = \frac{\alpha_i}{\alpha_i+\beta_i}.
$$

### 6.2 有效信誉

调度策略实际使用的不是原始信誉 $rep_i$，而是融合审计后验后的有效信誉：

$$
rep_i^{eff}
=
\mathrm{clip}\left(
(1-w_a)rep_i + w_a truth_i - \eta \cdot cooldown_i,
0,1
\right),
$$

其中：

- $w_a$ 对应 `Audit_Weight_In_Reputation`；
- $cooldown_i$ 表示审计失败后的冷却状态；
- $\eta$ 对应 `Audit_Cooldown_Penalty`；
- `clip` 将信誉限制在 $[0,1]$。

### 6.3 审计触发

审计由基础触发率与风险触发率共同决定：

$$
P(audit_i) = p_{base} + p_{risk}\cdot \mathbb{I}(risk_i > \delta_r),
$$

其中：

- $p_{base}$ 对应 `Audit_Base_Rate`；
- $p_{risk}$ 对应 `Audit_Risk_Rate`；
- $\delta_r$ 对应高风险阈值。

当节点审计失败时，其信誉会被快速惩罚；当节点持续通过审计时，其信誉会缓慢恢复。这种非对称更新符合安全场景中的常见假设：**失信应快速惩罚，可信应谨慎恢复**。

---

## 7. 图结构预言机编码器

### 7.1 预言机图建模

HCRL-Oracle 支持 GNN-style oracle encoder。预言机被视为图中的节点：

$$
G=(\mathcal{O}, E),
$$

其中边权由服务类型、可靠性、负载和成本相似性共同决定。

对于两个预言机 $o_i$ 和 $o_j$，邻接权重可以表示为：

$$
A_{ij}
\propto
w_s \mathbb{I}(\tau_i=\tau_j)
+ w_r(1-|rel_i-rel_j|)
+ w_l(1-|load_i-load_j|)
+ w_c(1-|cost_i-cost_j|),
$$

其中：

- $w_s$ 对应 `GNN_Service_Weight`；
- $w_r$ 对应 `GNN_Reliability_Weight`；
- $w_l$ 对应 `GNN_Load_Weight`；
- $w_c$ 对应 `GNN_Cost_Weight`。

邻接矩阵经过归一化后用于消息传递。

### 7.2 消息传递

设第 $k$ 层中预言机 $i$ 的表示为 $h_i^{(k)}$，则一次消息传递可写为：

$$
h_i^{(k+1)} =
\tanh\left(
\lambda_{self}h_i^{(k)}
+
\lambda_{neigh}\sum_{j\ne i}A_{ij}h_j^{(k)}
+
\lambda_{req}g_i
\right),
$$

其中：

- $\lambda_{self}$ 对应 `GNN_Self_Weight`；
- $\lambda_{neigh}$ 对应 `GNN_Neighbor_Weight`；
- $g_i$ 表示当前请求类型与预言机类型的匹配门控；
- 消息传递步数由 `GNN_Message_Steps` 控制。

该编码器使模型不仅能看到单个预言机的属性，还能建模 oracle pool 内部的结构关系。例如，同服务类型节点、相似可靠性节点和相似负载节点之间可以相互传递上下文信息。

---

## 8. 层次化策略与执行模式

### 8.1 执行模式集合

HCRL 默认使用以下五种模式：

```text
single_cost
single_safe
serial_safe
parallel_fast
parallel_safe
```

可以记为：

$$
\mathcal{M}=\{m_1,m_2,m_3,m_4,m_5\}.
$$

| 模式 | 含义 | 适用场景 |
|---|---|---|
| `single_cost` | 只调用一个主预言机，偏向成本效率。 | 风险低、deadline 不紧、成本敏感。 |
| `single_safe` | 只调用一个主预言机，但偏向可信节点。 | 风险中等、无需备份但不能过度追求低成本。 |
| `serial_safe` | 先调用主预言机，失败或风险触发后再调用备份。 | 成本敏感但需要恢复能力。 |
| `parallel_fast` | 主预言机和备份以并行或 warm-standby 方式执行。 | deadline 紧张，需要降低响应延迟。 |
| `parallel_safe` | 并行冗余并更强调安全性。 | 高风险请求或主预言机可信度不足。 |

### 8.2 模式掩码

代码中不是所有模式在每个状态下都可用。HCRL 使用 mode mask 排除明显不合理的模式。

设模式可用性为：

$$
mask_m(s_t)\in\{0,1\}^{|\mathcal{M}|}.
$$

例如：

1. 如果没有可用 backup，禁用 serial 和 parallel 模式；
2. 如果 backup score 低于阈值，禁用 backup 相关模式；
3. 如果 deadline slack 不足，禁用 `serial_safe`；
4. 如果 cost pressure 过高，禁用 parallel 模式；
5. 如果 primary risk 过高或 audit truth 过低，禁用 `single_cost`；
6. 如果 primary truth 过低但存在 backup，禁用 `single_safe`。

该机制可以写为：

$$
\pi_m(m|s)=0,\quad \forall m \; \text{s.t.}\; mask_m(s)=0.
$$

最终策略在可行动作集合上重新归一化：

$$
\pi_m(m|s)=
\frac{\exp(f_m(s))\cdot mask_m(s)}
{\sum_{m'}\exp(f_{m'}(s))\cdot mask_m(s)}.
$$

### 8.3 主预言机评分先验

代码中存在一个可解释的主预言机评分函数，用于模式状态、启发式选择和安全判断。其形式可概括为：

$$
Score_i^{primary}
=
0.35rep_i
+0.25obs_i
+0.20ontime_i
+0.15match_i
-0.15cost_i
-0.25risk_i.
$$

其中：

- $rep_i$ 为有效信誉；
- $obs_i$ 为观测到的成功率估计；
- $ontime_i$ 为按时完成概率估计；
- $match_i$ 为服务类型匹配项；
- $cost_i$ 为成本项；
- $risk_i$ 为综合风险项。

该评分不是最终的 HCRL policy，而是为高层模式判断和安全约束提供结构化先验。

### 8.4 备份预言机评分先验

备份预言机评分函数综合最近成功率、信誉、质押、按时概率、成本和风险：

$$
Score_i^{backup}
=
w_{succ}obs_i
+w_{rep}rep_i
+w_{token}token_i
+0.18ontime_i
-w_{cost}cost_i
-w_{risk}risk_i
-0.08\mathbb{I}(cost_i>C_{cap}).
$$

对应代码参数包括：

```text
PB_W_RECENT_SUCCESS
PB_W_REPUTATION
PB_W_TOKEN
PB_W_COST
PB_W_BEHAVIOR_RISK
PB_Backup_Cost_Limit
```

同时，备份候选集合会排除主预言机：

$$
b_t \ne p_t.
$$

这种设计避免 primary 和 backup 选择同一个节点导致的“伪冗余”。

---

## 9. 成本—延迟—风险约束奖励

### 9.1 基础奖励

对于一次请求执行，基础奖励可以抽象为：

$$
r_t^{base}
=
\omega_s S_t
+
\omega_m M_t
+
\omega_r Rep_t
-
\omega_c C_t
-
\omega_l L_t
-
\omega_o O_t,
$$

其中：

- $S_t$ 表示是否验证成功；
- $M_t$ 表示服务类型是否匹配；
- $Rep_t$ 表示所选预言机信誉；
- $C_t$ 表示调用成本；
- $L_t$ 表示响应时间或延迟；
- $O_t$ 表示超时或失败惩罚。

### 9.2 HCRL 约束项

HCRL 进一步引入成本、延迟和风险预算：

$$
C_t \le B_C,
$$

$$
L_t \le B_L,
$$

$$
\rho_t \le B_R,
$$

其中：

- $B_C$ 对应 `HCRL_Cost_Budget`；
- $B_L$ 对应 `HCRL_Latency_Budget`；
- $B_R$ 对应 `HCRL_Risk_Budget`；
- $\rho_t$ 表示当前 primary-backup 组合的风险估计。

约束惩罚可以写为：

$$
\mathcal{P}_t
=
\lambda_C [C_t-B_C]_+
+
\lambda_L [L_t-B_L]_+
+
\lambda_R [\rho_t-B_R]_+,
$$

其中 $[x]_+=\max(x,0)$。

### 9.3 HCRL 总奖励

HCRL 的总奖励可以概括为：

$$
r_t^{HCRL}
=
r_t^{base}
+eta_1\mathbb{I}(success)
+eta_2\mathbb{I}(backup\_recovery)
+eta_3\mathbb{I}(trusted\_selection)
-eta_4\mathbb{I}(malicious\_primary)
-eta_5\mathbb{I}(malicious\_backup)
-eta_6\mathbb{I}(unnecessary\_backup)
-eta_7\mathbb{I}(skip\_needed\_backup)
-
\mathcal{P}_t.
$$

其中与 HCRL 相关的重要参数包括：

```text
HCRL_Primary_Success_Bonus
HCRL_Backup_Recovery_Bonus
HCRL_Backup_Used_Penalty
HCRL_Unnecessary_Backup_Penalty
HCRL_Skip_Recovery_Penalty
HCRL_Primary_Malicious_Penalty
HCRL_Backup_Malicious_Penalty
HCRL_Final_Success_Bonus
HCRL_Success_Gain_Bonus
HCRL_Safety_Override_Bonus
HCRL_Estimated_Risk_Penalty
HCRL_Total_Cost_Penalty
HCRL_Trusted_Selection_Bonus
HCRL_Backup_Trust_Bonus
```

### 9.4 Primal-dual 约束更新

代码支持 primal-dual 风格的动态约束权重。约束乘子可以抽象更新为：

$$
\lambda_k \leftarrow
\mathrm{clip}\left(
\lambda_k + \eta_\lambda (g_k(s_t,a_t)-B_k),
\lambda_{min},
\lambda_{max}
\right),
$$

其中 $k\in\{C,L,R\}$ 分别表示成本、延迟和风险约束。

对应参数为：

```text
HCRL_Primal_Dual
HCRL_Lambda_LR
HCRL_Lambda_Min
HCRL_Lambda_Max
```

该机制使 HCRL 不需要固定地惩罚所有约束，而是可以根据当前训练过程中约束违反情况动态调整惩罚强度。

---

## 10. HCRL 与其他方法的对比

### 10.1 总体对比

| 维度 | DQN | RA-DDQN | PB-SafeDQN | COBRA-Oracle | HCRL-Oracle |
|---|---:|---:|---:|---:|---:|
| 单预言机选择 | 是 | 是 | 是 | 是 | 是 |
| 主备机制 | 否 | 否 | 是 | 是 | 是 |
| 备份选择可学习 | 否 | 否 | 否 | 部分 | 是 |
| 执行模式可学习 | 否 | 否 | 否 | 否 | 是 |
| 支持 single/serial/parallel | 否 | 否 | 部分 | 部分 | 是 |
| 支持 teacher guidance | 否 | 否 | 否 | 是 | 是 |
| 支持 GNN oracle encoder | 可选 | 可选 | 可选 | 可选 | 默认支持 |
| 支持审计声誉 | 是 | 是 | 是 | 是 | 是 |
| 支持成本约束 | 间接 | 间接 | 部分 | 是 | 是 |
| 支持延迟约束 | 间接 | 间接 | 部分 | 是 | 是 |
| 支持风险约束 | 间接 | 是 | 是 | 是 | 是 |
| 主要优势 | 基础 RL | 稳定 Q 学习 | 安全恢复 | 约束 backup gate | 层次化安全控制 |

### 10.2 与 DQN 的区别

DQN 学习的是：

$$
Q(s,o_i),
$$

即在状态 $s$ 下选择预言机 $o_i$ 的长期价值。

HCRL 学习的是：

$$
\pi_m(m|s^m),\quad \pi_p(p|s^p),\quad \pi_b(b|s^b,m,p).
$$

因此 HCRL 不只是回答“选哪个节点”，还回答“采用何种安全执行结构”。

### 10.3 与 RA-DDQN 的区别

RA-DDQN 通过 Dueling Double DQN 提高单预言机选择的稳定性，但其动作仍然是单节点：

$$
a_t=o_i.
$$

HCRL 的动作是结构化组合：

$$
a_t=(m_t,p_t,b_t).
$$

因此，HCRL 更适合处理需要冗余、恢复和安全模式切换的场景。

### 10.4 与 PB-SafeDQN 的区别

PB-SafeDQN 已经引入 primary-backup 思想，但其备份更多依赖规则评分，而不是独立学习策略。HCRL 则显式学习 backup policy：

$$
b_t \sim \pi_b(\cdot|s_t^b,m_t,p_t).
$$

同时，PB-SafeDQN 的执行模式较固定，而 HCRL 可以根据上下文在 single、serial 和 parallel 之间动态切换。

### 10.5 与 COBRA-Oracle 的区别

COBRA-Oracle 强调 constrained backup gate，即是否启用 backup 由 adaptive gate 控制。其核心仍然接近：

$$
primary\;selection + backup\;gate.
$$

HCRL 进一步将 gate 扩展为 mode policy：

$$
backup\;gate \Rightarrow mode\;policy.
$$

这意味着 HCRL 不只是决定“用不用 backup”，还决定：

- 使用 cost-oriented single；
- 使用 safety-oriented single；
- 使用 serial recovery；
- 使用 parallel fast；
- 使用 parallel safe。

因此，HCRL 的动作表达能力更强，也更适合复杂非平稳环境。

---

## 11. 安装与运行

### 11.1 克隆仓库

```bash
git clone -b hcrl https://github.com/flankerLym/TCO-DRL.git
cd "TCO-DRL/TCO-DRL_with baseline"
```

### 11.2 安装依赖

```bash
python -m pip install numpy pandas scipy matplotlib
```

当前主要学习模型为 NumPy 实现，不依赖 GPU。

---

## 12. 复现实验命令

### 12.1 查看可用方法

```bash
python main.py --List_Methods
```

### 12.2 快速测试

```bash
python main.py \
  --Seed 6 \
  --Method_Preset all \
  --Scenario rl_harder \
  --Oracles_Per_Type 10 \
  --Epoch 3 \
  --Request_Num 1000 \
  --Reward_Scale 2.0 \
  --Reward_Clip 2.0 \
  --Run_Tag quick_all_methods
```

Windows PowerShell：

```powershell
python main.py `
  --Seed 6 `
  --Method_Preset all `
  --Scenario rl_harder `
  --Oracles_Per_Type 10 `
  --Epoch 3 `
  --Request_Num 1000 `
  --Reward_Scale 2.0 `
  --Reward_Clip 2.0 `
  --Run_Tag quick_all_methods
```

### 12.3 训练全部方法

```bash
python main.py \
  --Seed 6 \
  --Method_Preset all \
  --Scenario rl_harder \
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
  --Run_Tag all_methods_seed6
```

Windows PowerShell：

```powershell
python main.py `
  --Seed 6 `
  --Method_Preset all `
  --Scenario rl_harder `
  --Oracles_Per_Type 10 `
  --Epoch 30 `
  --Request_Num 6000 `
  --Reward_Scale 2.0 `
  --Reward_Clip 2.0 `
  --Dqn_lr 0.0015 `
  --RA_lr 0.0012 `
  --COBRA_lr 0.0014 `
  --HCRL_lr 0.0014 `
  --HCRL_Mode_lr 0.0012 `
  --Dqn_batch_size 128 `
  --Dqn_memory_size 10000 `
  --Dqn_epsilon_increment 0.0008 `
  --Run_Tag all_methods_seed6
```

### 12.4 只训练 HCRL-Oracle

```bash
python main.py \
  --Seed 6 \
  --Method_Preset hcrl_only \
  --Scenario rl_harder \
  --Oracles_Per_Type 10 \
  --Epoch 30 \
  --Request_Num 6000 \
  --Reward_Scale 2.0 \
  --Reward_Clip 2.0 \
  --HCRL_lr 0.0014 \
  --HCRL_Mode_lr 0.0012 \
  --Dqn_batch_size 128 \
  --Dqn_memory_size 10000 \
  --Run_Tag hcrl_only_seed6
```

---

## 13. 消融实验

### 13.1 HCRL 完整版本

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_20_full_hcrl_gnn.ps1"
```

### 13.2 去除 GNN 编码器

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_21_hcrl_no_gnn.ps1"
```

该实验用于验证图结构 oracle encoder 是否提升了状态表征能力。

### 13.3 去除 teacher guidance

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_24_hcrl_no_teacher.ps1"
```

该实验用于验证 DQN / RA-DDQN / COBRA 等 teacher 是否有助于 HCRL 初期稳定训练。

### 13.4 去除约束惩罚

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_25_hcrl_no_constrained.ps1"
```

该实验用于验证成本—延迟—风险约束是否真正降低约束违反和恶意分配。

### 13.5 随机备份选择

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_26_hcrl_random_backup.ps1"
```

该实验用于验证 learned backup selector 是否优于随机 backup。

### 13.6 固定 single 模式

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_27_hcrl_fixed_single.ps1"
```

该实验用于验证 serial / parallel 模式是否带来额外安全收益。

### 13.7 固定 parallel 模式

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_28_hcrl_fixed_parallel.ps1"
```

该实验用于验证 learned mode policy 是否优于固定并行冗余策略。

### 13.8 一键运行论文实验集合

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_29_all_paper_experiments.ps1"
```

该脚本会依次运行 HCRL 完整版本、HCRL 消融、COBRA 公平对照和 RA-DDQN 公平对照，并最终汇总结果。

---

## 14. 输出文件与指标解释

每次运行会在 `output/` 下生成独立文件夹：

```text
output/
└── YY_M_D_HH_MM_Epoch{N}_Req{M}_{Scenario}_Seed{S}_{Run_Tag}/
    ├── *.txt                         # 完整日志
    ├── *_final_results.csv           # 最终结果表
    ├── *_final_results.json          # 最终结果 JSON
    ├── DQN.npz                       # DQN 权重
    ├── PPO.npz                       # PPO 权重
    ├── RA-DDQN.npz                   # RA-DDQN 权重
    ├── PB-SafeDQN.npz                # PB-SafeDQN 权重
    ├── COBRA-Oracle.npz              # COBRA 权重
    ├── HCRL_Mode.npz                 # HCRL 模式策略权重
    ├── HCRL_Primary.npz              # HCRL 主预言机策略权重
    └── HCRL_Backup.npz               # HCRL 备份策略权重
```

主要指标如下。

| 指标 | 含义 |
|---|---|
| `reward` | 累计奖励。 |
| `avg_responseT` | 平均响应时间。 |
| `success_rate` | 总体成功率。 |
| `success_time_rate` | 截止时间内成功率。 |
| `finishT` | 总完成时间。 |
| `Cost` | 总成本。 |
| `cost_per_success` | 每次成功请求的平均成本。 |
| `malicious_rate` | 恶意预言机分配比例。 |
| `normal_rate` | 普通预言机分配比例。 |
| `trusted_rate` | 可信预言机分配比例。 |
| `primary_success_rate` | 主预言机成功率。 |
| `backup_used_rate` | 备份使用比例。 |
| `backup_recovery_rate` | 总体备份恢复比例。 |
| `conditional_backup_recovery_rate` | 在使用备份条件下的恢复成功率。 |
| `backup_skipped_rate` | 备份跳过比例。 |
| `backup_score_mean` | 平均备份评分。 |
| `single_mode_rate` | HCRL 使用 single 模式比例。 |
| `serial_mode_rate` | HCRL 使用 serial 模式比例。 |
| `parallel_mode_rate` | HCRL 使用 parallel 模式比例。 |
| `audit_rate` | 审计触发比例。 |
| `audit_pass_rate` | 审计通过比例。 |
| `audit_fail_rate` | 审计失败比例。 |
| `audit_truth_mean` | 平均审计后验可信度。 |

对于 HCRL-Oracle，建议重点报告：

```text
success_rate
success_time_rate
avg_responseT
Cost
cost_per_success
malicious_rate
trusted_rate
backup_used_rate
conditional_backup_recovery_rate
single_mode_rate
serial_mode_rate
parallel_mode_rate
audit_fail_rate
audit_truth_mean
```

---

## 15. 推荐实验设置

### 15.1 单次主实验

推荐使用：

```text
Scenario = rl_harder
Epoch = 30
Request_Num = 6000
Oracles_Per_Type = 10
Seed = 6
```

### 15.2 多随机种子实验

为了进行更稳定的学术报告，建议使用多个随机种子并报告均值与标准差：

```text
Seed = 6, 42, 43, 2024, 2025
```

PowerShell 示例：

```powershell
$seeds = @(6, 42, 43, 2024, 2025)

foreach ($s in $seeds) {
  python main.py `
    --Seed $s `
    --Method_Preset all `
    --Scenario rl_harder `
    --Oracles_Per_Type 10 `
    --Epoch 30 `
    --Request_Num 6000 `
    --Reward_Scale 2.0 `
    --Reward_Clip 2.0 `
    --Dqn_lr 0.0015 `
    --RA_lr 0.0012 `
    --COBRA_lr 0.0014 `
    --HCRL_lr 0.0014 `
    --HCRL_Mode_lr 0.0012 `
    --Dqn_batch_size 128 `
    --Dqn_memory_size 10000 `
    --Dqn_epsilon_increment 0.0008 `
    --Run_Tag "all_methods_seed$s"
}
```

最终报告格式建议为：

$$
\mathrm{Metric}=\mathrm{mean}\pm\mathrm{std}.
$$

---

## 16. 方法总结

HCRL-Oracle 的核心贡献可以概括为：

1. **层次化决策结构**：将 oracle selection 拆分为模式选择、主预言机选择和备份预言机选择；
2. **自适应执行模式**：在 `single_cost`、`single_safe`、`serial_safe`、`parallel_fast` 和 `parallel_safe` 之间动态切换；
3. **审计感知声誉修正**：通过隐藏审计后验修正被恶意节点操纵的历史信誉；
4. **图结构 oracle encoder**：基于服务类型、可靠性、负载和成本相似性建模 oracle pool 的结构关系；
5. **成本—延迟—风险约束优化**：在提高成功率的同时控制冗余成本、响应延迟和恶意分配风险；
6. **teacher-guided 稳定训练**：使用 DQN、RA-DDQN 或 COBRA 作为早期 teacher，提高 HCRL 初期训练稳定性；
7. **可解释消融设计**：通过 no-GNN、no-teacher、no-constrained、random-backup、fixed-single 和 fixed-parallel 验证各模块贡献。

相比于 DQN、RA-DDQN、PB-SafeDQN 和 COBRA-Oracle，HCRL-Oracle 的主要优势在于它不再将 backup 视为一个固定附加模块，而是将安全冗余本身纳入可学习的层次化决策结构中。因此，在高风险、非平稳、验证感知和成本受限的区块链预言机环境中，HCRL-Oracle 提供了一种更灵活、更稳健且更具解释性的调度框架。
