# 一种面向区块链预言机选择的零知识审计感知层次化约束强化学习方法

> **中文论文初稿（Markdown 版）**  
> 说明：本文档按照你指定的章节骨架组织。第 **1、2、3、8、9** 章为占位性内容，后续可在投稿 FGCS 前统一重写为英文；第 **4、5、6、7** 章为当前重点撰写内容，围绕代码仓库中的 HCRL-Oracle、审计声誉、主备调度和拟扩展的 ZK-VOS 验证机制展开。实验数值位置暂以“待填入”标注，避免在未运行完整实验前编造结果。

---

## Abstract

区块链预言机承担着将链下数据与计算结果安全传递至链上智能合约的重要职责。然而，在开放且异构的预言机网络中，不同预言机在服务类型、响应延迟、调用成本、历史声誉、验证成功率以及潜在恶意行为方面存在显著差异。传统预言机选择方法通常将该问题简化为单步节点选择，难以同时处理成本优化、低延迟响应、恶意节点规避、审计追踪和结果可验证性等目标。为此，本文提出一种面向区块链预言机选择的零知识审计感知层次化约束强化学习模型，称为 **ZK-HCRL Oracle**。该模型将预言机选择过程建模为由执行模式选择、主预言机选择和备份预言机选择组成的层次化决策问题，并结合审计触发、贝叶斯后验声誉修正、风险感知奖励和成本—延迟—风险约束优化，实现对动态请求流的鲁棒调度。进一步地，本文设计了 **ZK-VOS（Zero-Knowledge Verifiable Oracle Scheduling）** 验证机制，在不泄露预言机内部风险评分、审计证据和策略细节的情况下，向链上合约证明调度过程满足预定义的有效性、预算和安全约束。仿真实验部分计划在静态、困难和恶意攻击场景下评估所提方法，并与 Random、Round-Robin、Earliest、DQN、PPO、RA-DDQN、PB-SafeDQN 和 COBRA-Oracle 等基线进行比较。预期结果表明，ZK-HCRL Oracle 能够在维持较高任务成功率和截止时间内成功率的同时，降低恶意预言机分配率、约束违反率和单位成功成本，从而为可信、可审计和可验证的区块链预言机服务提供一种新的智能调度框架。

**Keywords:** Blockchain oracle; hierarchical reinforcement learning; constrained reinforcement learning; audit-aware reputation; zero-knowledge proof; oracle scheduling; trust management.

---

# 1. Introduction

> 本章按你的要求暂时“不动”，此处先放占位内容，后续可在英文投稿前统一重写。

区块链智能合约在去中心化金融、供应链、物联网、保险和跨链服务等场景中得到广泛应用。然而，智能合约自身无法直接访问链下世界的数据和计算结果，因此需要依赖区块链预言机将外部信息输入链上环境。预言机的可靠性、响应速度和成本直接影响智能合约执行的正确性与经济效率。在开放预言机网络中，候选节点通常具有不同的服务能力、费用结构、历史表现和潜在风险，这使得预言机选择成为一个复杂的动态优化问题。

现有预言机选择方法主要关注信誉排序、成本最小化或简单的强化学习调度。然而，在真实区块链环境中，调度器不仅需要选择一个看似最优的预言机，还需要考虑该节点是否可能伪装信誉、是否在高负载下出现性能衰退、是否需要备份预言机进行安全恢复，以及调度决策本身是否能够被链上合约和外部审计者验证。特别是在涉及高价值交易或跨链状态同步时，单一预言机失败可能导致错误结算、资产损失或智能合约状态污染。

为了解决上述问题，本文提出 ZK-HCRL Oracle 框架。该框架将预言机选择扩展为“执行模式—主预言机—备份预言机”的层次化决策过程，并引入审计感知声誉修正与零知识可验证调度机制。与传统单层选择方法相比，该框架能够根据请求风险、截止时间、成本压力和候选备份质量自适应地选择单预言机、串行备份或并行备份模式，从而在性能与安全之间取得更灵活的平衡。

本文的主要贡献可概括如下：

1. 提出一种面向区块链预言机选择的层次化约束强化学习模型，将调度动作分解为执行模式、主预言机和备份预言机三个层次；
2. 设计审计感知信任管理机制，通过审计后验、冷却惩罚和非对称声誉更新抑制伪装型恶意预言机；
3. 引入成本—延迟—风险联合约束奖励，使模型能够在长期收益最大化的同时降低预算和安全约束违反；
4. 构建 ZK-VOS 零知识可验证调度机制，在不泄露内部策略分数和审计证据的前提下证明调度过程满足链上验证条件；
5. 规划系统化仿真实验，在不同请求强度、恶意节点比例和攻击模式下验证所提方法的鲁棒性。

---

# 2. Preliminaries

> 本章按你的要求暂时“不动”，此处先放占位内容，后续可在英文投稿前统一重写。

## 2.1 Blockchain Oracle

区块链预言机是连接链上智能合约与链下数据源的中间组件。设预言机集合为：

$$
\mathcal{O}=\{o_1,o_2,\ldots,o_N\},
$$

其中每个预言机 $o_i$ 可以由服务类型、调用成本、处理能力、验证概率、行为风险、质押代币和历史信誉等属性描述：

$$
o_i=(\tau_i,c_i,a_i,p_i^{val},p_i^{beh},q_i,rep_i).
$$

调度器需要在每个请求到达时从候选预言机集合中选择合适节点执行任务。

## 2.2 Markov Decision Process

预言机选择问题可以表示为马尔可夫决策过程：

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,R,\gamma),
$$

其中 $\mathcal{S}$ 表示状态空间，$\mathcal{A}$ 表示动作空间，$P$ 表示状态转移概率，$R$ 表示奖励函数，$\gamma$ 为折扣因子。与传统单动作 MDP 不同，本文方法中的动作具有层次结构。

## 2.3 Zero-Knowledge Verification

零知识证明允许证明者向验证者证明某一命题为真，同时不泄露除命题真实性以外的额外信息。在本文场景中，调度器需要证明其选择的预言机满足类型匹配、预算限制、风险限制和审计约束，但不应公开内部策略网络参数、候选节点的完整风险评分或审计证据。因此，零知识证明适合用于构建隐私保护的链上调度验证机制。

---

# 3. Related Work

> 本章按你的要求暂时“不动”，此处先放占位内容，后续可在英文投稿前统一重写。

现有研究主要从三个方向讨论区块链预言机选择问题。第一类方法基于静态信誉或历史成功率进行排序，优点是实现简单，但难以适应动态负载和伪装型恶意节点。第二类方法基于博弈论、拍卖机制或成本优化模型，在一定程度上考虑了费用与激励兼容性，但通常缺少对长期序列决策和任务时延约束的建模。第三类方法将强化学习引入区块链资源调度，通过学习长期收益改进节点选择策略，但多数工作仍以单节点选择为主，缺少对主备恢复、审计惩罚和可验证执行的统一建模。

此外，零知识证明已被广泛用于区块链隐私交易、身份认证、链下计算验证和可验证机器学习等场景。然而，将零知识证明与强化学习驱动的预言机调度结合仍是一个值得探索的问题。本文工作尝试在审计感知 HCRL 调度与 ZK-VOS 验证之间建立桥梁，使链下智能调度策略能够以隐私保护方式接受链上验证。

---

# 4. The Proposed ZK-HCRL Oracle Model

## 4.1 System Overview

本文提出的 **ZK-HCRL Oracle** 由四个核心模块组成：请求建模模块、审计感知状态编码模块、层次化约束强化学习调度模块和零知识可验证调度模块。整体流程如下：

1. 链上或链下应用产生请求 $r_t$，该请求包含服务类型、到达时间、任务长度和截止时间；
2. 调度环境收集候选预言机的等待时间、成本、处理能力、服务类型、验证历史、声誉、质押代币和审计后验等信息；
3. 审计感知状态编码模块将请求级特征与预言机级特征拼接为强化学习状态；
4. HCRL 调度器依次选择执行模式 $m_t$、主预言机 $p_t$ 和备份预言机 $b_t$；
5. 调度结果执行后，根据响应时间、成本、验证结果、行为记录和审计结果更新奖励与信誉；
6. ZK-VOS 模块为关键调度约束生成零知识证明，链上合约只验证证明与公开承诺，而不直接访问内部风险分数或策略参数。

与传统方法只选择一个预言机不同，ZK-HCRL Oracle 的核心思想是将调度决策扩展为：

$$
a_t^{ZK\text{-}HCRL}=(m_t,p_t,b_t,\pi_t^{zk}),
$$

其中 $m_t$ 表示执行模式，$p_t$ 表示主预言机，$b_t$ 表示备份预言机，$\pi_t^{zk}$ 表示当前调度决策对应的零知识验证证明。

## 4.2 Problem Formulation

考虑连续到达的请求序列：

$$
\mathcal{R}=\{r_1,r_2,\ldots,r_T\}.
$$

每个请求可表示为：

$$
r_t=(\tau_t,A_t,L_t,D_t),
$$

其中 $\tau_t$ 为请求类型，$A_t$ 为到达时间，$L_t$ 为任务长度，$D_t$ 为截止时间。预言机集合为：

$$
\mathcal{O}=\{o_1,o_2,\ldots,o_N\}.
$$

每个预言机 $o_i$ 具有如下属性：

$$
o_i=(\tau_i,c_i,acc_i,q_i,p_i^{val},p_i^{beh},rep_i,load_i),
$$

其中 $\tau_i$ 表示服务类型，$c_i$ 表示调用成本，$acc_i$ 表示处理能力，$q_i$ 表示质押代币，$p_i^{val}$ 表示验证成功概率，$p_i^{beh}$ 表示行为风险分布，$rep_i$ 表示历史信誉，$load_i$ 表示当前或近期负载。

系统优化目标是在长期请求序列中最大化累计折扣收益：

$$
\max_{\pi}\; \mathbb{E}_{\pi}\left[\sum_{t=1}^{T}\gamma^{t-1}R_t\right],
$$

同时满足成本、延迟和风险约束：

$$
\mathbb{E}[C_t]\leq B_c,
$$

$$
\mathbb{E}[T_t]\leq B_l,
$$

$$
\mathbb{E}[\rho_t]\leq B_r,
$$

其中 $C_t$ 表示当前调度成本，$T_t$ 表示响应时间，$\rho_t$ 表示调度风险，$B_c$、$B_l$ 和 $B_r$ 分别表示成本预算、延迟预算和风险预算。

## 4.3 Audit-aware State Representation

ZK-HCRL Oracle 的状态表示由请求级特征、预言机级特征和模式级摘要特征组成。

### 4.3.1 Request-level Features

请求级特征用于描述当前任务本身：

$$
x_t^{req}=\left[\frac{\tau_t}{\tau_{max}},\frac{L_t}{\bar{L}},\frac{D_t}{D_{hard}}\right],
$$

其中 $\tau_{max}$ 为最大服务类型编号，$\bar{L}$ 为平均任务长度，$D_{hard}$ 为困难场景中的截止时间归一化常数。

### 4.3.2 Oracle-level Features

对于每个候选预言机 $o_i$，构造如下特征：

$$
x_{t,i}^{oracle}=
[wait_i, rep_i^{eff}, cost_i, acc_i, match_i, val_i, load_i, riskdelay_i, token_i, truth_i, fail_i^{audit}, cooldown_i].
$$

各特征含义如下表所示。

| 特征 | 含义 |
|---|---|
| $wait_i$ | 当前请求到达时预言机 $i$ 的归一化等待时间 |
| $rep_i^{eff}$ | 审计修正后的有效信誉 |
| $cost_i$ | 归一化调用成本 |
| $acc_i$ | 归一化处理能力 |
| $match_i$ | 预言机服务类型是否与请求类型匹配 |
| $val_i$ | 验证成功概率或历史验证成功估计 |
| $load_i$ | 近期负载水平 |
| $riskdelay_i$ | 行为风险与延迟风险的混合估计 |
| $token_i$ | 归一化质押代币 |
| $truth_i$ | 审计后验可信度 |
| $fail_i^{audit}$ | 审计失败率估计 |
| $cooldown_i$ | 审计失败后的冷却惩罚状态 |

因此，主预言机和备份预言机策略使用的基础状态为：

$$
s_t^p=[x_t^{req},x_{t,1}^{oracle},x_{t,2}^{oracle},\ldots,x_{t,N}^{oracle}].
$$

### 4.3.3 Mode-level Summary Features

高层模式策略需要判断当前请求是否需要备份、采用串行还是并行模式。因此，在基础状态 $s_t^p$ 之外，加入模式级摘要特征：

$$
s_t^m=[s_t^p,z_t],
$$

其中：

$$
z_t=[slack_t,risk_t^p,ontime_t^p,score_t^b,gain_t^b,pressure_t^c,succ_t^{recent},risk_t^{recent},fail_t^{audit},truth_t^{best\_backup}].
$$

这些特征分别表示主预言机截止时间余量、主预言机风险、主预言机准时概率、最佳备份评分、备份收益、备份成本压力、近期成功率、近期风险水平、近期审计失败率和最佳备份审计可信度。

## 4.4 Hierarchical Action Space

传统预言机选择方法的动作空间为：

$$
a_t\in\{1,2,\ldots,N\}.
$$

而 ZK-HCRL Oracle 将动作分解为三个层次：

$$
a_t=(m_t,p_t,b_t).
$$

其中：

- $m_t\in\mathcal{M}$ 表示执行模式；
- $p_t\in\mathcal{O}$ 表示主预言机；
- $b_t\in\mathcal{O}\cup\{-1\}$ 表示备份预言机，若当前模式不需要备份，则 $b_t=-1$。

默认执行模式集合为：

$$
\mathcal{M}=\{single\_cost,single\_safe,serial\_safe,parallel\_fast,parallel\_safe\}.
$$

| 模式 | 调度含义 | 适用场景 |
|---|---|---|
| `single_cost` | 仅调用主预言机，强调低成本 | 低风险、成本敏感请求 |
| `single_safe` | 仅调用主预言机，但强调高信誉和低风险 | 中低风险、无需备份请求 |
| `serial_safe` | 先调用主预言机，失败后再调用备份 | 成本敏感但需要恢复能力 |
| `parallel_fast` | 主备并行执行，选择更快或成功结果 | 截止时间紧张请求 |
| `parallel_safe` | 主备并行执行，并更强调安全冗余 | 高风险或高价值请求 |

这种分解使模型能够根据场景自动决定是否需要冗余验证，而不是固定地使用单预言机或固定地使用备份。

## 4.5 Graph-style Oracle Encoder

为了建模预言机池内部结构关系，ZK-HCRL Oracle 可以将预言机集合视为图：

$$
G=(\mathcal{O},E),
$$

其中节点表示预言机，边表示预言机之间的服务类型、可靠性、负载和成本相似性。对于预言机 $o_i$ 和 $o_j$，边权可表示为：

$$
A_{ij}\propto
w_s\mathbb{I}(\tau_i=\tau_j)
+w_r(1-|rel_i-rel_j|)
+w_l(1-|load_i-load_j|)
+w_c(1-|cost_i-cost_j|).
$$

其中 $w_s$、$w_r$、$w_l$、$w_c$ 分别控制服务类型、可靠性、负载和成本相似性的权重。归一化后的邻接矩阵用于消息传递：

$$
h_i^{(k+1)}=\tanh\left(\lambda_{self}h_i^{(k)}+\lambda_{neigh}\sum_{j\neq i}A_{ij}h_j^{(k)}+\lambda_{req}g_i\right),
$$

其中 $g_i$ 表示当前请求类型与预言机类型是否匹配的门控项。该编码器使模型不仅能观察单个预言机的局部属性，还能学习预言机池中相似节点之间的上下文关系。

## 4.6 Constrained Reward Design

ZK-HCRL Oracle 的奖励函数同时考虑任务成功、验证成功、类型匹配、信誉、成本、响应时间、行为风险、超时和审计风险。可抽象表示为：

$$
R_t=R_t^{pos}-R_t^{neg},
$$

其中正向项为：

$$
R_t^{pos}=w_s\cdot success_t+w_v\cdot validation_t+w_m\cdot match_t+w_r\cdot rep_t,
$$

负向项为：

$$
R_t^{neg}=w_c\cdot cost_t+w_l\cdot latency_t+w_b\cdot behavior_t+w_o\cdot timeout_t+w_a\cdot auditRisk_t.
$$

对于层次化模式，还加入模式相关奖励修正。例如：

- `single_cost` 给予成本节省奖励，但惩罚高审计风险；
- `single_safe` 奖励高信誉主预言机，但更强惩罚低可信度；
- `serial_safe` 奖励主预言机失败后的备份恢复；
- `parallel_fast` 奖励截止时间内完成，但惩罚过高成本；
- `parallel_safe` 奖励最终成功和安全冗余，但惩罚高风险备份。

为了处理成本、延迟和风险约束，引入拉格朗日形式：

$$
\tilde{R}_t=R_t-\lambda_c[C_t-B_c]_+-\lambda_l[T_t-B_l]_+-\lambda_r[\rho_t-B_r]_+,
$$

其中 $[x]_+=\max(0,x)$，$\lambda_c$、$\lambda_l$、$\lambda_r$ 是成本、延迟和风险约束的惩罚系数。

## 4.7 Learning Procedure

ZK-HCRL Oracle 包含三个可学习策略：

$$
\pi_m(m_t|s_t^m),\quad \pi_p(p_t|s_t^p),\quad \pi_b(b_t|s_t^p,p_t,m_t).
$$

三个策略均可采用轻量级 Actor-Critic 实现。Actor 输出动作概率分布，Critic 估计状态价值：

$$
\pi_\theta(a|s)=Softmax(f_\theta(s)),
$$

$$
V_\phi(s)=g_\phi(s).
$$

优势函数定义为：

$$
A_t=\tilde{R}_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

策略梯度可表示为：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t)A_t+\beta\nabla_\theta H(\pi_\theta(\cdot|s_t))\right],
$$

其中 $H(\cdot)$ 为策略熵，$\beta$ 为熵正则系数，用于鼓励探索。

**Algorithm 1: ZK-HCRL Oracle Scheduling**

```text
Input: request r_t, oracle pool O, audit posterior, policy networks π_m, π_p, π_b
Output: scheduling action (m_t, p_t, b_t) and ZK proof π_t^zk

1. Construct request-level feature x_t^req.
2. For each oracle o_i, compute wait_i, rep_i^eff, cost_i, acc_i, match_i, val_i, load_i, risk_i, truth_i and cooldown_i.
3. Build base state s_t^p and mode state s_t^m.
4. Generate primary action mask and mode action mask.
5. Select primary oracle p_t using π_p(s_t^p).
6. Select execution mode m_t using π_m(s_t^m).
7. If m_t requires backup:
       Select backup oracle b_t using π_b(s_t^p, p_t, m_t).
   Else:
       Set b_t = -1.
8. Execute scheduling according to selected mode.
9. Observe validation result, response time, cost and behavior record.
10. Trigger audit according to risk-aware audit probability.
11. Update reputation and audit posterior.
12. Compute constrained reward and update π_m, π_p, π_b.
13. Generate ZK-VOS proof π_t^zk for the validity of scheduling constraints.
14. Return (m_t, p_t, b_t, π_t^zk).
```

---

# 5. Audit-aware Trust Management in ZK-HCRL Oracle

## 5.1 Motivation and Threat Model

在开放预言机网络中，预言机节点可能存在以下风险：

1. **低质量节点**：验证成功率低、响应慢或成本异常；
2. **伪装型恶意节点**：在早期表现正常以积累信誉，在关键请求或高负载阶段作恶；
3. **疲劳型节点**：在连续高负载下性能下降，导致验证失败率上升；
4. **类型不匹配节点**：不能正确处理当前请求类型；
5. **高成本诱导节点**：虽然可靠性较高，但调用成本过高，长期使用会降低系统经济性。

传统信誉机制通常依赖历史成功率更新，因此对伪装型恶意节点不够敏感。为此，本文引入审计感知信任管理机制，将随机审计、风险触发审计和贝叶斯后验可信度纳入调度状态、奖励函数和模式掩码。

## 5.2 Audit Posterior

对于每个预言机 $o_i$，维护一组审计后验参数：

$$
\alpha_i,\beta_i,
$$

其中 $\alpha_i$ 表示审计通过证据，$\beta_i$ 表示审计失败或风险证据。审计后验可信度定义为：

$$
truth_i=\frac{\alpha_i}{\alpha_i+\beta_i}.
$$

初始时，所有预言机采用相同先验：

$$
\alpha_i=\alpha_0,\quad \beta_i=\beta_0.
$$

在当前实现中，可设置 $\alpha_0=2.0$，$\beta_0=2.0$，表示调度器在开始时对预言机保持中性信任。

## 5.3 Effective Reputation

调度器实际使用的信誉不是原始历史信誉 $rep_i$，而是融合审计后验和冷却惩罚后的有效信誉：

$$
rep_i^{eff}=clip((1-w_a)rep_i+w_a truth_i-\eta\cdot cooldown_i,0,1),
$$

其中 $w_a$ 为审计后验权重，$\eta$ 为冷却惩罚系数，$cooldown_i$ 表示审计失败后的冷却状态。该设计使得审计失败会直接降低预言机在后续调度中的可选性，而审计长期通过则可以逐步恢复其信誉。

## 5.4 Risk-triggered Audit

审计触发概率由基础审计率和风险触发项组成：

$$
P(audit_i)=clip(p_{base}+p_{risk}\cdot risk_i,0,p_{max}),
$$

其中 $p_{base}$ 是基础审计率，$p_{risk}$ 控制风险对审计概率的影响，$p_{max}$ 为最大审计概率上限。风险估计可由有效信誉、审计后验、近期失败率、冷却状态和距离上次审计的时间共同决定：

$$
risk_i=w_1(1-rep_i^{eff})+w_2(1-truth_i)+w_3fail_i^{recent}+w_4cooldown_i+w_5stale_i.
$$

其中 $stale_i$ 表示距离上次审计的时间间隔归一化值。该机制能够对长期未审计但近期表现不稳定的预言机提高审计概率。

## 5.5 Asymmetric Reputation Update

审计结果分为通过和失败两类。当审计通过时：

$$
\alpha_i\leftarrow\alpha_i+1,
$$

同时增加连续清洁记录 $clean_i$。若连续通过次数超过阈值，则信誉缓慢恢复：

$$
rep_i\leftarrow rep_i+\delta_{pass}(1-rep_i).
$$

当审计失败时，根据信任违规严重程度 $sev_i$ 更新：

$$
\beta_i\leftarrow\beta_i+sev_i,
$$

$$
rep_i\leftarrow rep_i-\delta_{fail}\cdot sev_i.
$$

若失败严重程度超过阈值，则进入冷却状态：

$$
cooldown_i\leftarrow C_{max}.
$$

这种非对称更新遵循安全系统中的基本原则：**失信应快速惩罚，可信应谨慎恢复**。这有助于防止恶意预言机通过短期正常行为快速恢复声誉。

## 5.6 Audit Severity

审计失败严重程度由多种因素组成。例如：

$$
sev_i=\omega_1\mathbb{I}(T_i>D_t)+\omega_2\mathbb{I}(match_i=0)+\omega_3\mathbb{I}(val_i=0)+\omega_4behavior_i+\omega_5malicious_i.
$$

其中超时、类型不匹配、验证失败、异常行为记录和恶意节点标记都会提高审计严重度。最终严重度被限制在合理范围内：

$$
sev_i\in[sev_{min},sev_{max}].
$$

## 5.7 Integration with HCRL Decision Making

审计感知信任管理机制通过三条路径影响 HCRL 调度：

1. **状态路径**：$truth_i$、$fail_i^{audit}$ 和 $cooldown_i$ 被加入预言机级状态；
2. **动作路径**：当主预言机审计可信度过低或风险过高时，模式掩码会禁用低成本单节点模式；
3. **奖励路径**：审计风险作为负向奖励项，惩罚选择高风险预言机的策略。

因此，审计模块不是一个独立的后处理组件，而是与强化学习调度闭环耦合的信任管理机制。

---

# 6. HCRL-based Scheduling and ZK-VOS Verification

## 6.1 HCRL-based Scheduling Pipeline

ZK-HCRL Oracle 的调度过程由三个策略模块协同完成：

| 策略模块 | 输入状态 | 输出动作 | 功能 |
|---|---|---|---|
| $\pi_p$ | $s_t^p$ | 主预言机 $p_t$ | 选择最适合执行当前请求的主节点 |
| $\pi_m$ | $s_t^m$ | 执行模式 $m_t$ | 判断是否使用备份以及采用何种备份形式 |
| $\pi_b$ | $s_t^p$ 与主预言机信息 | 备份预言机 $b_t$ | 在需要冗余时选择恢复节点 |

执行顺序可写为：

$$
p_t\sim\pi_p(\cdot|s_t^p),
$$

$$
m_t\sim\pi_m(\cdot|s_t^m),
$$

$$
b_t\sim\pi_b(\cdot|s_t^p,p_t,m_t).
$$

## 6.2 Mode Masking Mechanism

并非所有执行模式在每个状态下都合理。为了减少无效探索和违反约束的动作，HCRL 引入模式掩码：

$$
mask_m(s_t)\in\{0,1\}^{|\mathcal{M}|}.
$$

模式掩码遵循以下规则：

1. 若不存在可用备份，则禁用 `serial_safe`、`parallel_fast` 和 `parallel_safe`；
2. 若最佳备份评分低于阈值，则禁用所有备份模式；
3. 若主预言机截止时间余量不足，则禁用 `serial_safe`；
4. 若成本压力过高，则禁用并行模式；
5. 若主预言机风险过高或审计可信度过低，则禁用 `single_cost`；
6. 若主预言机可信度过低且存在备份，则禁用 `single_safe`。

该机制可抽象为：

$$
\pi_m(m|s_t^m)=0,\quad \forall m\notin\mathcal{M}_{valid}(s_t).
$$

通过模式掩码，模型能够在安全边界内进行策略学习。

## 6.3 Primary-backup Execution Semantics

当执行模式为单预言机模式时，系统只调用主预言机：

$$
result_t=result(p_t).
$$

当执行模式为 `serial_safe` 时，系统先调用主预言机。若主预言机成功，则不调用备份；若主预言机失败，则调用备份：

$$
result_t=\begin{cases}
result(p_t), & success(p_t)=1,\\
result(b_t), & success(p_t)=0.
\end{cases}
$$

该模式成本较低，但可能增加响应延迟。

当执行模式为 `parallel_fast` 或 `parallel_safe` 时，主预言机和备份预言机并行执行：

$$
result_t=select(result(p_t),result(b_t)).
$$

若至少一个预言机成功，则最终任务成功：

$$
success_t=success(p_t)\lor success(b_t).
$$

并行模式可提高截止时间内成功率和恢复能力，但会增加调用成本。

## 6.4 ZK-VOS: Zero-Knowledge Verifiable Oracle Scheduling

虽然 HCRL 可以在链下完成复杂调度，但区块链系统要求关键决策能够被验证。如果直接公开调度器内部状态，可能泄露以下敏感信息：

- 预言机的内部风险评分；
- 审计失败记录和行为证据；
- 策略网络输出概率或 Q 值；
- 备份节点选择逻辑；
- 链下服务质量数据。

因此，本文提出 **ZK-VOS** 验证机制，使调度器可以证明其决策满足约束，而无需公开完整内部状态。

对于每个请求 $r_t$，调度器生成公开承诺：

$$
C_t=Com(r_t,m_t,p_t,b_t,B_c,B_l,B_r,h_t),
$$

其中 $h_t$ 是链下状态、审计记录和策略输出的哈希承诺。调度器随后生成零知识证明：

$$
\pi_t^{zk}=Prove(w_t,C_t),
$$

其中 $w_t$ 为私有 witness，包括内部风险评分、有效信誉、候选备份评分、策略输出和审计证据等。

链上验证者只需执行：

$$
Verify(C_t,\pi_t^{zk})\rightarrow\{0,1\}.
$$

若验证通过，则说明调度决策满足预定义约束。

## 6.5 Verification Statements

ZK-VOS 证明的核心命题可以设计为以下几类。

### 6.5.1 Valid Oracle Selection

证明被选择的主预言机和备份预言机属于候选集合：

$$
p_t\in\mathcal{O},\quad b_t\in\mathcal{O}\cup\{-1\}.
$$

若使用备份，则还需证明：

$$
b_t\neq p_t.
$$

### 6.5.2 Service Type Compatibility

证明被选择的主预言机或最终成功预言机满足请求服务类型：

$$
\tau_{selected}=\tau_t.
$$

在并行或串行备份模式下，只需证明至少一个最终采用的成功结果满足类型约束。

### 6.5.3 Budget Feasibility

证明调度成本不超过预算或相应违反项已被记录：

$$
C_t\leq B_c.
$$

对于并行模式，成本可写为：

$$
C_t=c_{p_t}+\kappa c_{b_t},
$$

其中 $\kappa$ 为并行成本折扣因子。

### 6.5.4 Latency Feasibility

证明最终响应时间满足截止时间约束：

$$
T_t\leq D_t.
$$

对于串行模式，若主预言机失败后再调用备份，则最终延迟由主预言机失败时间和备份执行时间共同决定。对于并行模式，最终延迟可取成功结果中的较小完成时间。

### 6.5.5 Risk Feasibility

证明所选预言机的审计风险或组合风险不超过风险预算：

$$
\rho_t\leq B_r.
$$

其中风险 $\rho_t$ 可以由有效信誉、审计后验、近期失败率和行为风险共同计算，但具体分数作为 witness 保持私有。

### 6.5.6 Audit Update Correctness

若当前请求触发审计，则证明审计后验更新遵循预定义规则：

$$
(\alpha_i',\beta_i')=Update(\alpha_i,\beta_i,audit_i,sev_i).
$$

该命题保证调度器不能在链下任意篡改预言机信誉。

## 6.6 Privacy-preserving Trust Update

ZK-VOS 的关键优势在于既保证链上可验证性，又保护链下敏感信任信息。公开信息仅包括：

- 请求承诺；
- 选择结果承诺；
- 预算阈值；
- 状态哈希；
- 零知识证明。

私有信息包括：

- 预言机内部风险评分；
- 策略网络输出；
- 审计失败严重度；
- 未公开的验证概率；
- 候选备份评分。

因此，恶意预言机无法通过观察链上记录反推出调度器的完整安全策略，从而降低策略被规避的风险。

## 6.7 ZK-VOS Protocol

**Algorithm 2: ZK-VOS Verification Protocol**

```text
Input: scheduling decision (m_t, p_t, b_t), request r_t, private witness w_t
Output: public commitment C_t and zero-knowledge proof π_t^zk

1. Compute commitment C_r = Com(r_t).
2. Compute commitment C_a = Com(m_t, p_t, b_t).
3. Compute commitment C_s = Com(hidden state summary and audit posterior).
4. Construct public input C_t = (C_r, C_a, C_s, B_c, B_l, B_r).
5. Encode verification circuit:
       a) selected oracle belongs to candidate set;
       b) selected oracle satisfies service type mask;
       c) selected mode is allowed by mode mask;
       d) total cost is within budget or violation flag is recorded;
       e) final latency is within deadline or violation flag is recorded;
       f) audit-aware risk is within budget or violation flag is recorded;
       g) reputation and audit posterior update follows predefined rule.
6. Generate π_t^zk = Prove(w_t, C_t).
7. Submit (C_t, π_t^zk) to smart contract.
8. Smart contract verifies Verify(C_t, π_t^zk).
9. If verification succeeds, accept scheduling result and update public ledger state.
```

## 6.8 Complexity Analysis

ZK-HCRL Oracle 的计算复杂度主要由三个部分组成。

首先，状态编码需要遍历所有预言机，因此复杂度为：

$$
O(Nd),
$$

其中 $N$ 为预言机数量，$d$ 为单个预言机特征维度。

其次，若使用图结构编码器，则需要计算预言机间相似度，朴素实现复杂度为：

$$
O(N^2d).
$$

当预言机数量较大时，可通过稀疏邻接、同服务类型局部图或近邻采样降低复杂度。

最后，ZK-VOS 证明复杂度取决于电路规模。若只验证模式掩码、预算约束和审计更新，则电路规模与预言机数量近似线性或准线性相关；若完整验证策略网络前向传播，则证明成本将显著增加。因此，本文建议优先验证调度约束和审计更新，而不是直接验证整个强化学习网络。

---

# 7. Simulation Results and Evaluations

## 7.1 Experimental Setup

本文计划在模拟区块链预言机服务环境中评估 ZK-HCRL Oracle。请求按照泊松过程到达，任务长度服从正态分布并截断为正值。每个请求包含服务类型、到达时间、任务长度和截止时间。预言机池由多个服务类型的异构节点组成，不同节点具有不同成本、处理能力、验证成功概率、质押代币和行为风险。

实验场景包括：

| 场景 | 描述 | 目的 |
|---|---|---|
| Static | 预言机验证概率和行为分布相对稳定 | 评估基础调度性能 |
| RL-hard | 存在请求突发和节点疲劳效应 | 评估动态负载鲁棒性 |
| RL-harder | 更强疲劳、更紧截止时间和更高风险 | 评估困难场景下的安全恢复能力 |
| Attack-mild | 轻度恶意行为 | 测试低强度攻击下的稳定性 |
| Attack-stealth | 伪装型恶意节点 | 测试审计机制对隐藏风险的识别能力 |
| Attack-severe | 高强度恶意行为 | 测试安全边界和恢复能力 |
| Attack-burst | 突发式攻击 | 测试非平稳攻击下的鲁棒性 |

## 7.2 Baselines

计划比较以下方法：

| 方法 | 类型 | 是否学习 | 是否备份 | 是否审计感知 |
|---|---|---:|---:|---:|
| Random | 随机选择 | 否 | 否 | 否 |
| Round-Robin | 轮询选择 | 否 | 否 | 否 |
| Earliest | 最早空闲选择 | 否 | 否 | 否 |
| BLOR | 贝叶斯 bandit | 部分 | 否 | 否 |
| SemiGreedy | 半贪心选择 | 否 | 否 | 间接 |
| DQN | 单层强化学习 | 是 | 否 | 否 |
| PPO | 策略梯度强化学习 | 是 | 否 | 否 |
| RA-DDQN | 风险感知 Dueling Double DQN | 是 | 否 | 部分 |
| PB-SafeDQN | 主备安全 DQN | 是 | 是 | 部分 |
| COBRA-Oracle | 约束恢复感知预言机选择 | 是 | 是 | 部分 |
| ZK-HCRL Oracle | 本文方法 | 是 | 是 | 是 |

## 7.3 Evaluation Metrics

实验指标包括：

| 指标 | 含义 |
|---|---|
| Total reward | 累计奖励，反映总体长期收益 |
| Success rate | 任务成功率 |
| Success-in-time rate | 截止时间内成功率 |
| Average response time | 平均响应时间 |
| Average cost | 平均调用成本 |
| Cost per success | 单位成功成本 |
| Malicious assignment rate | 恶意预言机分配率 |
| Trusted assignment rate | 可信预言机分配率 |
| Primary success rate | 主预言机成功率 |
| Backup used rate | 备份使用率 |
| Backup recovery rate | 备份恢复率 |
| Conditional backup recovery rate | 在使用备份条件下的恢复成功率 |
| Constraint violation rate | 成本、延迟或风险约束违反率 |
| Cost violation rate | 成本预算违反率 |
| Latency violation rate | 延迟预算违反率 |
| Risk violation rate | 风险预算违反率 |
| Audit rate | 审计触发率 |
| Audit fail rate | 审计失败率 |
| Audit truth mean | 平均审计后验可信度 |
| Single/serial/parallel mode rate | HCRL 不同执行模式占比 |

## 7.4 Main Results

> 下面表格为论文写作模板。真实数值需要在完成实验运行后填入。

| Method | Reward ↑ | Success Rate ↑ | Success-in-time ↑ | Avg. Response ↓ | Cost ↓ | Cost/Success ↓ | Malicious Rate ↓ | Trusted Rate ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| Round-Robin | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| Earliest | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| DQN | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| PPO | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| RA-DDQN | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| PB-SafeDQN | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| COBRA-Oracle | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |

从预期趋势来看，随机与轮询方法由于不感知预言机风险和服务类型匹配，通常会产生较高的恶意分配率和较低的任务成功率。Earliest 方法能够改善响应时间，但不一定能保证验证成功。DQN 和 PPO 可以学习长期收益，但由于动作空间仍是单层预言机选择，在高风险场景中缺乏恢复能力。PB-SafeDQN 和 COBRA-Oracle 通过主备机制提高鲁棒性，但其模式选择能力有限。相比之下，ZK-HCRL Oracle 通过执行模式、主预言机和备份预言机的联合学习，有望在成功率、单位成功成本和风险控制之间取得更好的平衡。

## 7.5 Primary-backup Diagnostics

| Method | Primary Success ↑ | Backup Used | Backup Recovery ↑ | Conditional Recovery ↑ | Backup Skipped | Backup Score ↑ |
|---|---:|---:|---:|---:|---:|---:|
| PB-SafeDQN | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| COBRA-Oracle | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |

该表用于分析主备机制的实际贡献。若 ZK-HCRL Oracle 的备份使用率低于 PB-SafeDQN，但条件备份恢复率更高，说明其能够更精准地判断何时需要备份，而不是盲目调用备份节点。

## 7.6 Constraint Violation Analysis

| Method | Overall Violation ↓ | Cost Violation ↓ | Latency Violation ↓ | Risk Violation ↓ |
|---|---:|---:|---:|---:|
| DQN | 待填入 | 待填入 | 待填入 | 待填入 |
| RA-DDQN | 待填入 | 待填入 | 待填入 | 待填入 |
| PB-SafeDQN | 待填入 | 待填入 | 待填入 | 待填入 |
| COBRA-Oracle | 待填入 | 待填入 | 待填入 | 待填入 |
| ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 | 待填入 |

该实验用于验证约束强化学习设计是否真正降低成本、延迟和风险预算违反。如果 ZK-HCRL Oracle 在风险违反率上明显低于其他方法，则说明审计感知状态与风险预算约束有效。

## 7.7 Audit-aware Trust Evaluation

| Method | Audit Rate | Audit Pass Rate ↑ | Audit Fail Rate ↓ | Audit Truth Mean ↑ | Malicious Assignment ↓ |
|---|---:|---:|---:|---:|---:|
| HCRL without audit | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| HCRL with audit | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |
| ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 | 待填入 | 待填入 |

该消融实验用于验证审计机制的作用。若启用审计后恶意预言机分配率下降，同时 audit truth mean 上升，说明审计后验能够帮助策略更准确地区分可靠节点和伪装节点。

## 7.8 Mode Distribution

| Method | Single Mode Rate | Serial Mode Rate | Parallel Mode Rate |
|---|---:|---:|---:|
| ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 |

模式分布可以反映 HCRL 是否真正学到了自适应调度策略。在低风险场景中，模型应更多使用 `single_cost` 或 `single_safe`，以降低调用成本；在高风险或高攻击强度场景中，模型应增加 `serial_safe` 或 `parallel_safe` 的使用比例，以提高恢复能力和安全性。

## 7.9 Malicious Ratio Stress Test

为了进一步评估鲁棒性，可以设置不同恶意预言机比例：

$$
\eta_m\in\{0\%,10\%,20\%,30\%,40\%,50\%\}.
$$

| Malicious Ratio | Method | Success Rate ↑ | Malicious Assignment ↓ | Risk Violation ↓ | Cost/Success ↓ |
|---:|---|---:|---:|---:|---:|
| 10% | DQN | 待填入 | 待填入 | 待填入 | 待填入 |
| 10% | ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 | 待填入 |
| 30% | DQN | 待填入 | 待填入 | 待填入 | 待填入 |
| 30% | ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 | 待填入 |
| 50% | DQN | 待填入 | 待填入 | 待填入 | 待填入 |
| 50% | ZK-HCRL Oracle | 待填入 | 待填入 | 待填入 | 待填入 |

预期趋势是，随着恶意节点比例升高，所有方法性能都会下降，但 ZK-HCRL Oracle 应通过审计修正和备份恢复机制保持更低的恶意分配率和更稳定的成功率。

## 7.10 ZK-VOS Overhead Evaluation

ZK-VOS 需要额外的证明生成和验证开销。建议评估以下指标：

| Circuit Setting | Proving Time ↓ | Verification Time ↓ | Proof Size ↓ | On-chain Gas ↓ |
|---|---:|---:|---:|---:|
| Constraint-only proof | 待填入 | 待填入 | 待填入 | 待填入 |
| Audit-update proof | 待填入 | 待填入 | 待填入 | 待填入 |
| Full scheduling proof | 待填入 | 待填入 | 待填入 | 待填入 |

在初版系统中，建议优先实现 constraint-only proof 和 audit-update proof，因为它们可以验证关键安全约束，同时避免完整神经网络证明带来的过高开销。

---

# 8. Discussion

> 本章按你的要求暂时“不动”，此处先放占位内容，后续可在英文投稿前统一重写。

本文提出的 ZK-HCRL Oracle 将强化学习调度、审计感知信任管理和零知识可验证机制结合在一起，为区块链预言机选择提供了一种新的系统化思路。与单层预言机选择相比，层次化策略能够更灵活地处理低成本、低延迟和高安全之间的冲突。与固定主备机制相比，HCRL 可以根据请求风险和备份质量决定是否使用备份，从而避免不必要的成本开销。

然而，当前方法仍存在一些需要进一步完善的方面。首先，ZK-VOS 的证明电路设计需要在安全性和效率之间平衡。若验证整个强化学习策略网络，证明成本可能较高；若只验证约束和审计更新，则需要额外假设链下调度器不会偏离策略。其次，仿真环境虽然可以控制恶意节点比例、攻击强度和负载条件，但真实预言机网络中的行为更加复杂。未来工作可以将该框架部署到真实智能合约测试网络中，并引入实际 oracle service traces 进行验证。

---

# 9. Conclusion

> 本章按你的要求暂时“不动”，此处先放占位内容，后续可在英文投稿前统一重写。

本文提出了一种面向区块链预言机选择的 ZK-HCRL Oracle 框架。该框架通过层次化约束强化学习实现执行模式、主预言机和备份预言机的联合决策，通过审计感知声誉机制提高对伪装型恶意节点的识别能力，并通过 ZK-VOS 机制支持隐私保护的链上可验证调度。与传统单预言机选择和固定主备策略相比，ZK-HCRL Oracle 具有更强的动态适应性、安全恢复能力和可审计性。

未来工作将进一步完善零知识证明电路实现，补充真实链上验证开销实验，并在更多攻击场景和真实预言机数据上评估模型泛化能力。

---

# Appendix A. Suggested Commands for Experiments

> 以下命令需要根据服务器路径和实际代码配置进一步调整。

```bash
python main.py --Method_Preset hcrl --Epoch 10 --Request_Num 6000 --Scenario static --State_Mode enhanced --Reward_Mode risk_aware --Success_Mode validation_aware --Use_GNN_Encoder --Run_Tag static_hcrl
```

```bash
python main.py --Method_Preset hcrl --Epoch 10 --Request_Num 6000 --Scenario rl_hard --State_Mode enhanced --Reward_Mode risk_aware --Success_Mode validation_aware --Use_GNN_Encoder --Run_Tag hard_hcrl
```

```bash
python main.py --Method_Preset hcrl --Epoch 10 --Request_Num 6000 --Scenario rl_harder --Attack_Profile stealth --Malicious_Ratio 0.3 --State_Mode enhanced --Reward_Mode risk_aware --Success_Mode validation_aware --Use_GNN_Encoder --Run_Tag stealth_30
```

---

# Appendix B. Paper Writing Notes for FGCS Conversion

1. 第 4 章可作为英文稿的 **Methodology** 主体；
2. 第 5 章可作为 **Audit-aware Trust Management** 或并入 Methodology；
3. 第 6 章中的 ZK-VOS 需要在英文稿中明确区分“已实现的 HCRL 调度代码”和“拟扩展的零知识验证协议”；
4. 第 7 章必须使用真实实验结果替换“待填入”；
5. 若投稿 FGCS，建议补充系统架构图、HCRL 决策流程图、审计更新流程图和 ZK-VOS 验证流程图；
6. 英文稿中需增加 Related Work 的真实引用，包括 blockchain oracle、trust management、constrained RL、hierarchical RL 和 zero-knowledge proof。
