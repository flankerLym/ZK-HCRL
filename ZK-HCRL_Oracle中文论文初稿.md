# ZK-HCRL Oracle：融合审计反馈与零知识合规证明的 DeFi 可验证预言机服务调度框架

> **中文论文初稿（Markdown 版）**  
> 说明：本文档按照你指定的章节骨架组织。第 **1、2、3、8、9** 章为占位性内容，后续可在投稿 FGCS 前统一重写为英文；第 **4、5、6、7** 章为当前重点撰写内容，围绕 DeFi 场景下的 HCRL-Oracle 调度、审计反馈驱动的信任更新、主备预言机服务编排和 ZK-VOS 零知识合规证明机制展开。实验数值位置暂以“待填入”标注，避免在未运行完整实验前编造结果。

---

## Abstract

去中心化金融（Decentralized Finance, DeFi）协议高度依赖区块链预言机获取链下价格、清算状态、借贷参数和跨链数据。然而，在开放且异构的预言机网络中，候选节点在响应延迟、调用成本、服务类型、历史可信度、验证成功率以及潜在恶意行为方面存在显著差异，单一预言机失效或被操纵可能直接导致错误清算、资产定价偏差和智能合约状态污染。传统预言机选择方法通常将调度过程简化为单步节点选择，难以同时满足 DeFi 服务对低延迟、高可靠、可审计和链上可验证合规性的要求。为此，本文提出 **ZK-HCRL Oracle：融合审计反馈与零知识合规证明的 DeFi 可验证预言机服务调度框架**。该框架将 DeFi 预言机服务调度建模为由执行模式选择、主预言机选择和备份预言机选择组成的层次化约束强化学习问题，并通过审计反馈、贝叶斯后验声誉修正、风险感知奖励和成本—延迟—风险约束优化，实现对动态请求流的鲁棒调度。进一步地，本文设计 **ZK-VOS（Zero-Knowledge Verifiable Oracle Scheduling）** 机制，将节点选择合法性、服务类型匹配、成本预算、延迟约束、风险约束和审计更新正确性编码为零知识合规证明，使链上合约能够在不暴露内部风险评分、审计证据和策略输出的情况下验证调度结果的合规性。仿真实验部分计划在静态负载、困难负载和恶意攻击场景下评估所提框架，并与 Random、Round-Robin、Earliest、DQN、PPO、RA-DDQN、PB-SafeDQN 和 COBRA-Oracle 等基线进行比较。预期结果表明，ZK-HCRL Oracle 能够在维持较高服务成功率和截止时间内成功率的同时，降低恶意预言机分配率、约束违反率和单位成功成本，从而为 DeFi 场景中的可验证预言机服务提供一种兼具智能调度、审计反馈和隐私保护合规证明能力的系统框架。

**Keywords:** DeFi; blockchain oracle; verifiable oracle service; hierarchical reinforcement learning; audit feedback; constrained reinforcement learning; zero-knowledge compliance proof; oracle scheduling.

---

# 1. Introduction

> 本章按你的要求暂时“不动”，此处先放占位内容，后续可在英文投稿前统一重写。

区块链智能合约在去中心化金融、供应链、物联网、保险和跨链服务等场景中得到广泛应用。然而，智能合约自身无法直接访问链下世界的数据和计算结果，因此需要依赖区块链预言机将外部信息输入链上环境。预言机的可靠性、响应速度和成本直接影响智能合约执行的正确性与经济效率。在开放预言机网络中，候选节点通常具有不同的服务能力、费用结构、历史表现和潜在风险，这使得预言机选择成为一个复杂的动态优化问题。

现有预言机选择方法主要关注信誉排序、成本最小化或简单的强化学习调度。然而，在真实区块链环境中，调度器不仅需要选择一个看似最优的预言机，还需要考虑该节点是否可能伪装信誉、是否在高负载下出现性能衰退、是否需要备份预言机进行安全恢复，以及调度决策本身是否能够被链上合约和外部审计者验证。特别是在涉及高价值交易或跨链状态同步时，单一预言机失败可能导致错误结算、资产损失或智能合约状态污染。

为了解决上述问题，本文提出 **ZK-HCRL Oracle：融合审计反馈与零知识合规证明的 DeFi 可验证预言机服务调度框架**。该框架面向 DeFi 请求中的价格查询、清算触发、借贷结算和跨链状态同步等预言机服务，将传统“选择一个预言机”的问题扩展为“执行模式—主预言机—备份预言机—合规证明”的复合调度过程，并引入审计反馈驱动的信任更新与零知识合规证明机制。与传统单层选择方法相比，该框架能够根据请求风险、截止时间、成本压力和候选备份质量自适应地选择单预言机、串行备份或并行备份模式，从而在 DeFi 服务的执行效率、安全冗余和链上可验证性之间取得更灵活的平衡。

本文的主要贡献可概括如下：

1. 面向 DeFi 可验证预言机服务调度场景，提出 ZK-HCRL Oracle 框架，将调度动作分解为执行模式、主预言机、备份预言机和零知识合规证明四个相互关联的组成部分；
2. 设计融合审计反馈的信任管理机制，通过审计后验、冷却惩罚和非对称声誉更新抑制伪装型恶意预言机，并将更新后的信任信号反馈至后续调度策略；
3. 引入成本—延迟—风险联合约束奖励，使模型能够在长期收益最大化的同时降低预算违反、延迟违反和安全风险违反；
4. 构建 ZK-VOS 零知识合规证明机制，在不泄露内部策略分数、风险评分和审计证据的前提下，证明调度结果满足链上可验证的服务合规条件；
5. 规划系统化仿真实验，在不同请求强度、恶意节点比例和攻击模式下验证所提框架在 DeFi 预言机服务调度中的鲁棒性。

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

# 4. The Proposed ZK-HCRL Oracle Framework

## 4.1 System Overview

图 1 展示了本文提出的 **ZK-HCRL Oracle** 总体框架。面向 DeFi 场景中价格查询、清算触发、借贷结算和跨链状态同步等高实时性、高经济敏感性的预言机服务需求，本文将传统的预言机节点选择问题扩展为一个可验证的服务调度问题。与仅输出单一节点索引的方法不同，ZK-HCRL Oracle 将调度决策建模为由执行模式、主预言机、备份预言机和链上合规证明共同构成的复合动作，从而在服务成功率、响应延迟、调用成本、节点风险和链上可验证性之间实现联合优化。

<p align="center">
  <img src="./paperMuter/技术架构图.png" alt="Figure 1. Overall framework of ZK-HCRL Oracle" width="550" height="550"/>
</p>

<p align="center"><strong>Figure 1.</strong> Overall framework of ZK-HCRL Oracle.</p>

从系统组成上看，ZK-HCRL Oracle 包含 DeFi 应用层、链上验证与调度层、预言机服务层和外部数据源层。DeFi 应用层产生价格查询、抵押率更新、清算判定和结算触发等业务请求；外部数据源层提供市场数据、交易所接口、Web API、跨链数据源或链下计算结果；预言机服务层由异构预言机节点构成，不同节点在服务类型、处理能力、调用费用、历史验证表现和审计记录方面存在差异。链上验证与调度层是本文框架的核心，其内部包括 HCRL/Audit Scheduler、OracleSchedule Registry、ZK-VOS Proof Generation 和 Solidity Groth16 Verifier，分别用于生成调度决策、登记调度结果、构造零知识合规证明并完成链上验证。

当 DeFi 预言机服务请求到达时，HCRL/Audit Scheduler 首先构造审计感知状态表示。该状态融合请求类型、任务长度、到达时间和截止时间等请求级特征，以及候选预言机的等待时间、服务类型匹配关系、调用成本、处理能力、历史验证成功率、有效信誉、审计后验可信度、冷却状态和风险监测指标。随后，层次化约束强化学习策略依次完成执行模式选择、主预言机选择和备份预言机选择。执行模式用于决定当前请求采用成本优先单节点执行、可信优先单节点执行、串行备份恢复、并行快速响应或并行安全冗余；主预言机负责主要服务响应，备份预言机则在高风险或高时延敏感场景中提供恢复或冗余验证能力。该结构使系统能够在低风险请求中降低冗余成本，在高风险请求中增强服务可靠性。

审计反馈机制为调度过程提供持续的信任校正能力。每次服务执行后，系统会收集响应延迟、验证结果、服务类型匹配情况、异常行为记录、审计结果和失败严重度等反馈信息。Reputation/Audit Module 基于上述反馈更新预言机的审计后验和有效信誉，Monitoring/Risk Module 则持续估计节点负载、异常行为和风险趋势。更新后的审计信息进一步参与后续状态编码、动作掩码和奖励计算，使调度策略能够逐步降低伪装型恶意节点、疲劳节点和高风险节点的被选概率，从而形成“调度—执行—审计反馈—再调度”的闭环优化机制。

零知识合规证明机制用于解决智能调度结果的链上可验证问题。由于 HCRL 调度器的内部状态包含节点风险评分、审计证据、历史行为记录和策略输出等敏感信息，直接公开这些信息可能泄露防御策略并被恶意节点利用。因此，ZK-VOS 不要求链上合约复现完整的强化学习推理过程，而是围绕调度结果的合规性构造可验证命题，包括候选节点合法性、服务类型兼容性、成本预算满足性、延迟约束满足性、风险预算满足性和审计更新正确性。调度器以内部风险评分和审计证据作为私有 witness 生成零知识证明，链上合约仅验证证明是否成立，即可确认调度结果满足 DeFi 服务约束，而无需获知内部策略细节。

形式化地，对于时刻 $t$ 到达的 DeFi 预言机服务请求，ZK-HCRL Oracle 输出的调度动作定义为

$$
a_t^{ZK\text{-}HCRL}=(m_t,p_t,b_t,\pi_t^{zk}),
$$

其中 $m_t$ 表示执行模式，$p_t$ 表示主预言机，$b_t$ 表示备份预言机，$\pi_t^{zk}$ 表示与当前调度结果绑定的零知识合规证明。该定义表明，本文方法的输出并非单一预言机节点，而是同时包含服务编排、风险冗余和可验证合规性的复合调度结果。基于上述设计，ZK-HCRL Oracle 将层次化约束强化学习、审计反馈驱动的信任管理和零知识合规证明统一到同一框架中，为 DeFi 场景下的可验证预言机服务调度提供了系统化建模基础。
## 4.2 Problem Formulation

本文研究 DeFi 场景下的可验证预言机服务调度问题。设系统在调度周期内接收连续到达的服务请求序列：

$$
\mathcal{R}=\{r_1,r_2,\ldots,r_T\},
$$

其中 $T$ 表示请求总数。每个请求 $r_t$ 定义为：

$$
r_t=(\tau_t,A_t,L_t,D_t,\omega_t),
$$

其中 $\tau_t$ 表示请求服务类型，$A_t$ 表示到达时间，$L_t$ 表示任务规模或数据处理长度，$D_t$ 表示截止时间，$\omega_t$ 表示请求重要性或风险权重。在 DeFi 应用中，不同请求具有不同的经济敏感性。例如，普通价格查询、抵押率更新、清算触发和大额结算对时延、可信度和风险控制的要求并不相同。因此，预言机调度不仅需要返回可用结果，还需要兼顾服务质量、节点可信性、调用成本和链上可验证性。

设候选预言机集合为：

$$
\mathcal{O}=\{o_1,o_2,\ldots,o_N\},
$$

其中 $N$ 为候选节点数量。每个预言机 $o_i$ 由静态服务属性和动态信任状态共同描述：

$$
o_i=(\tau_i,c_i,acc_i,q_i,p_i^{val},p_i^{beh},rep_i,load_i,\alpha_i,\beta_i),
$$

其中 $\tau_i$ 表示节点支持的服务类型，$c_i$ 表示调用成本，$acc_i$ 表示处理能力，$q_i$ 表示质押代币或经济担保水平，$p_i^{val}$ 表示验证成功概率，$p_i^{beh}$ 表示行为风险分布，$rep_i$ 表示历史信誉，$load_i$ 表示当前或近期负载，$\alpha_i$ 和 $\beta_i$ 分别表示审计通过与审计失败所累积的后验证据。这些变量共同决定节点在当前请求下的可用性、可靠性和风险水平。

对于请求 $r_t$，ZK-HCRL Oracle 输出的不是单一预言机索引，而是一个复合调度动作：

$$
a_t=(m_t,p_t,b_t,\pi_t^{zk}),
$$

其中 $m_t$ 表示执行模式，$p_t$ 表示主预言机，$b_t$ 表示备份预言机，$\pi_t^{zk}$ 表示与该次调度绑定的零知识合规证明。执行模式从集合 $\mathcal{M}$ 中选择：

$$
m_t\in\mathcal{M}=\{single\_cost,single\_safe,serial\_safe,parallel\_fast,parallel\_safe\}.
$$

主备节点满足：

$$
p_t\in\mathcal{O},\quad b_t\in\mathcal{O}\cup\{-1\}.
$$

其中 $b_t=-1$ 表示当前模式不启用备份；若执行模式需要备份，则要求 $b_t\neq p_t$，以保证主备冗余的有效性。不同模式对应不同的服务编排方式：单节点模式主要降低调用成本，串行备份模式强调失败恢复，并行模式则通过冗余执行提高实时性和安全性。

给定状态 $s_t$ 和动作 $a_t$ 后，系统产生服务结果和反馈信息。记 $Y_t\in\{0,1\}$ 表示请求是否成功完成，$C_t$ 表示总调用成本，$T_t$ 表示最终响应时间，$\rho_t$ 表示调度风险，$R_t$ 表示即时奖励。其中 $C_t$ 与所选预言机数量、节点调用费用和执行模式有关；$T_t$ 由节点等待时间、执行时间和主备执行语义共同决定；$\rho_t$ 则由节点信誉、审计后验、近期失败率、行为风险和冷却状态等因素估计。由此，预言机调度被建模为一个同时受服务质量、经济成本和安全风险约束的序列决策问题。

本文将该问题形式化为带约束的层次化马尔可夫决策过程。系统状态 $s_t$ 包括请求级特征、候选预言机状态、审计后验和近期风险统计；动作 $a_t$ 由执行模式策略、主预言机策略和备份预言机策略联合生成；奖励函数 $R_t$ 综合刻画服务成功、验证结果、类型匹配、信誉水平、响应延迟、调用成本、行为风险和审计风险。调度策略 $\pi$ 的目标是在长期请求序列中最大化期望累计折扣收益：

$$
\max_{\pi}\;J(\pi)=
\mathbb{E}_{\pi}\left[
\sum_{t=1}^{T}\gamma^{t-1}R_t
\right],
$$

其中 $\gamma\in(0,1]$ 为折扣因子，$\pi$ 表示由 $\pi_m$、$\pi_p$ 和 $\pi_b$ 组成的层次化调度策略。

同时，DeFi 预言机服务需要满足成本、延迟和风险约束：

$$
\mathbb{E}_{\pi}[C_t]\leq B_c,
$$

$$
\mathbb{E}_{\pi}[T_t]\leq B_l,
$$

$$
\mathbb{E}_{\pi}[\rho_t]\leq B_r,
$$

其中 $B_c$、$B_l$ 和 $B_r$ 分别表示成本预算、延迟预算和风险预算。成本约束用于避免过度调用高费用节点或不必要的并行冗余；延迟约束保证服务响应满足 DeFi 业务的实时性要求；风险约束限制低信誉、高异常概率或审计失败节点的使用频率。对于经济敏感性更高的请求，可通过提高 $\omega_t$ 或收紧风险预算 $B_r$ 强化安全要求。

除上述优化约束外，本文进一步引入链上合规验证条件。设 $\mathcal{C}_{zk}$ 表示 ZK-VOS 需要证明的调度合规命题集合，包括节点选择合法性、服务类型兼容性、成本预算满足性、延迟约束满足性、风险预算满足性和审计更新正确性。调度器根据私有 witness 生成零知识证明 $\pi_t^{zk}$，链上合约验证：

$$
Verify(C_t^{pub},\pi_t^{zk})=1,
$$

其中 $C_t^{pub}$ 表示公开输入或承诺。该条件保证链上合约能够确认调度结果满足预定义规则，而无需获取内部风险评分、审计证据、历史行为记录或策略网络输出。

综上，本文的目标是在连续到达的 DeFi 预言机服务请求下，学习一个层次化调度策略 $\pi$，使其在满足成本、延迟、风险和零知识合规验证约束的同时，最大化长期服务收益。该问题的核心挑战在于：预言机节点具有异构能力和动态信任状态，请求具有不同的实时性与经济风险，而调度结果还必须在保护内部策略隐私的前提下接受链上验证。

## 4.3 Audit-aware State and Oracle Representation

在上述问题定义基础上，ZK-HCRL Oracle 将调度状态构造为请求特征、预言机特征和候选池结构信息的联合表示。设当前请求的归一化特征为

$$
x_t^{req}=\left[\frac{\tau_t}{\tau_{\max}},\frac{L_t}{\bar{L}},\frac{D_t-A_t}{D_{\max}},\omega_t\right],
$$

其中 $D_t-A_t$ 表示请求可用时间窗口，$\omega_t$ 表示请求经济敏感性或风险权重。该表示用于刻画 DeFi 请求对服务类型、响应时限和安全等级的需求。

对于候选预言机 $o_i$，构造审计感知节点特征

$$
x_{t,i}^{oracle}=
[wait_i,c_i,acc_i,match_i,val_i,load_i,rep_i^{eff},truth_i,fail_i^{audit},cooldown_i,risk_i],
$$

其中 $wait_i$ 表示当前排队等待时间，$match_i$ 表示服务类型匹配关系，$rep_i^{eff}$ 表示审计修正后的有效信誉，$truth_i$ 表示审计后验可信度，$fail_i^{audit}$ 表示近期审计失败水平，$cooldown_i$ 表示审计失败后的冷却状态，$risk_i$ 表示综合风险估计。与仅依赖历史成功率的状态表示不同，该特征同时编码了性能、成本、服务匹配和审计反馈，使策略能够区分短期高表现但存在潜在风险的节点。

为保留预言机池中的结构关系，本文进一步将候选节点视为图 $G_t=(\mathcal{O},E_t)$。边权 $A_{ij}$ 根据服务类型相似性、信誉差异、负载差异和成本差异构造：

$$
A_{ij}\propto
w_s\mathbb{I}(\tau_i=\tau_j)
+w_r(1-|rep_i^{eff}-rep_j^{eff}|)
+w_l(1-|load_i-load_j|)
+w_c(1-|c_i-c_j|).
$$

图编码器通过邻域聚合获得节点上下文表示：

$$
h_i^{(k+1)}=\sigma\left(
W_s h_i^{(k)}+W_n\sum_{j\neq i}A_{ij}h_j^{(k)}+W_q x_t^{req}
\right),
$$

其中 $\sigma(\cdot)$ 为非线性激活函数。该结构使模型能够在单节点属性之外感知候选池整体状态，例如同类服务节点是否拥塞、低成本节点是否集中伴随高风险，以及可信备份是否充足。

最终，主预言机与备份预言机策略使用基础状态

$$
s_t^p=[x_t^{req},h_1,\ldots,h_N],
$$

而执行模式策略额外引入全局摘要向量

$$
z_t=[slack_t,risk_t^p,score_t^b,gain_t^b,pressure_t^c,truth_t^{best}],
$$

形成

$$
s_t^m=[s_t^p,z_t].
$$

其中 $slack_t$ 表示截止时间余量，$risk_t^p$ 表示当前主节点风险估计，$score_t^b$ 和 $gain_t^b$ 分别表示最优备份评分及其边际收益，$pressure_t^c$ 表示成本压力，$truth_t^{best}$ 表示候选备份的最高审计可信度。上述状态设计为后续层次化调度提供统一输入。

## 4.4 Hierarchical Constrained Scheduling Policy

ZK-HCRL Oracle 采用层次化策略将调度动作分解为执行模式选择、主预言机选择和备份预言机选择。三个策略分别表示为

$$
\pi_m(m_t|s_t^m),\quad
\pi_p(p_t|s_t^p),\quad
\pi_b(b_t|s_t^p,p_t,m_t).
$$

其中 $\pi_m$ 控制服务编排方式，$\pi_p$ 选择主要响应节点，$\pi_b$ 在需要冗余时选择恢复或并行验证节点。该分解降低了复合动作空间的搜索难度，并使策略能够根据请求风险和候选池状态自适应调整冗余强度。

执行模式集合 $\mathcal{M}$ 包含成本优先单节点、可信优先单节点、串行安全备份、并行快速响应和并行安全冗余五类模式。单节点模式适用于低风险或成本敏感请求；串行备份模式在主节点失败后调用备份，适合成本受限但需要恢复能力的请求；并行模式同时调用主备节点，适合清算、大额结算等高时效或高风险场景。为避免无效探索，策略训练中引入模式掩码：

$$
\pi_m(m|s_t^m)=0,\quad \forall m\notin \mathcal{M}_{valid}(s_t).
$$

当不存在合格备份、成本压力过高、截止时间不足或主节点风险超过阈值时，相应模式会被禁用。该机制将业务约束显式注入策略空间，有助于降低成本、延迟和风险违反。

调度执行后，系统根据服务结果计算约束奖励。即时奖励由正向收益和惩罚项组成：

$$
R_t=w_sY_t+w_vV_t+w_mM_t+w_qQ_t
-\left(w_cC_t+w_lT_t+w_r\rho_t+w_aA_t^{risk}\right),
$$

其中 $Y_t$ 表示服务成功，$V_t$ 表示验证结果，$M_t$ 表示服务类型匹配，$Q_t$ 表示有效信誉收益，$C_t$、$T_t$、$\rho_t$ 和 $A_t^{risk}$ 分别表示成本、延迟、综合风险和审计风险。为处理长期预算限制，引入约束惩罚：

$$
\tilde{R}_t=R_t-\lambda_c[C_t-B_c]_+-\lambda_l[T_t-B_l]_+-\lambda_r[\rho_t-B_r]_+.
$$

其中 $[x]_+=\max(0,x)$。策略更新以 $\tilde{R}_t$ 为学习信号，从而在最大化长期收益的同时抑制预算、延迟和风险越界。

<p align="center">
  <img src="./paperMuter/zk-hcrl oracle schedule.png" alt="Algorithm 1: ZK-HCRL Oracle Scheduling" width="850"/>
</p>

<p align="center"><strong>Algorithm 1.</strong> ZK-HCRL Oracle Scheduling.</p>

## 4.5 Scheduling Performance Evaluation

本节评估 HCRL 调度策略在 DeFi 预言机服务中的综合性能，重点考察其在服务成功率、时效性、单位成功成本、恶意节点规避和可信节点覆盖方面的表现。与模型设计相对应，该实验不作为独立实验章节展开，而是用于直接验证层次化约束调度机制的有效性。比较方法包括基于规则的 Reputation-Greedy、Cost-Aware-Greedy 和 Risk-Aware-Greedy，以及基于强化学习的 PPO、DQN、RA-DDQN、PB-SafeDQN 和 COBRA-Oracle。各方法在相同请求流、候选预言机集合和攻击扰动条件下重复运行，并报告均值及标准差。

**Table 1. Scheduling performance comparison under DeFi oracle service requests.**

| Method | Success Rate | On-time Rate | Avg. Response | Cost/Success | Malicious Rate | Trusted Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Reputation-Greedy | 36.05 ± 1.40% | 90.41 ± 1.53% | 5.62 ± 0.08 | 1.86 ± 0.06 | 8.94 ± 1.79% | 80.89 ± 3.47% |
| Cost-Aware-Greedy | 19.71 ± 0.56% | 100.00 ± 0.00% | 4.82 ± 0.00 | 0.95 ± 0.02 | 45.81 ± 3.15% | 54.17 ± 3.14% |
| Risk-Aware-Greedy | 37.35 ± 0.71% | 85.45 ± 0.97% | 5.85 ± 0.06 | 1.81 ± 0.08 | 8.08 ± 1.25% | 82.58 ± 3.17% |
| PPO | 70.20 ± 1.20% | 97.00 ± 0.17% | 5.32 ± 0.01 | 1.13 ± 0.01 | 3.64 ± 0.78% | 92.69 ± 1.27% |
| DQN | 60.73 ± 2.46% | 91.91 ± 1.04% | 5.51 ± 0.05 | 1.35 ± 0.04 | 3.22 ± 0.36% | 92.68 ± 1.05% |
| RA-DDQN | 59.06 ± 2.83% | 91.64 ± 1.48% | 5.51 ± 0.07 | 1.39 ± 0.05 | 3.07 ± 0.18% | 93.01 ± 0.94% |
| PB-SafeDQN | 52.23 ± 3.18% | 89.39 ± 1.07% | 5.53 ± 0.06 | 1.92 ± 0.06 | 7.32 ± 2.06% | 95.79 ± 1.59% |
| COBRA-Oracle | 56.20 ± 2.64% | 88.19 ± 1.11% | 5.65 ± 0.05 | 2.04 ± 0.06 | 10.37 ± 4.85% | 96.87 ± 0.87% |
| **HCRL-Oracle** | **73.41 ± 0.85%** | **96.77 ± 0.23%** | **5.30 ± 0.02** | **1.55 ± 0.04** | **1.12 ± 0.74%** | **99.27 ± 0.32%** |

如 Table 1 所示，HCRL-Oracle 在服务成功率和可信节点覆盖率上取得最优结果，分别达到 73.41% 和 99.27%，同时将恶意节点分配率降至 1.12%。与 PPO、DQN 和 RA-DDQN 等单层强化学习方法相比，HCRL-Oracle 通过执行模式、主预言机和备份预言机的分层决策提升了调度成功率；与 PB-SafeDQN 和 COBRA-Oracle 等主备调度方法相比，其恶意节点规避能力更强，说明层次化动作空间与审计感知状态能够更有效地识别高风险节点。虽然 Cost-Aware-Greedy 在平均响应时间和单位成功成本方面较低，但其成功率与可信覆盖率明显不足，表明单纯成本优化难以满足 DeFi 预言机服务的可靠性要求。

为说明实验数据的基本特征，本文对所使用的 DeFi 预言机价格数据集进行统计，结果如 Table 2 所示。该数据集包含 BTC/USD、ETH/USD 和 LINK/USD 三类资产对，每类资产对均包含 4307 条价格样本，并记录价格偏差、数据陈旧度和验证成功率等信息。这些统计量用于刻画不同资产请求在价格波动、数据时效性和可验证性方面的差异，为后续调度实验提供数据基础。

**Table 2. Dataset statistics of DeFi oracle price samples.**

| Asset | Samples | Mean Deviation | P95 Deviation | Mean Staleness | Validation Success Rate |
|---|---:|---:|---:|---:|---:|
| BTC/USD | 4307 | 0.1400% | 0.3746% | 1641.76 s | 20.66% |
| ETH/USD | 4307 | 0.1565% | 0.4064% | 1571.22 s | 23.13% |
| LINK/USD | 4307 | 0.1672% | 0.4260% | 1531.62 s | 23.96% |

Table 2 显示，三类资产对的样本规模一致，平均价格偏差均低于 0.17%，P95 偏差均低于 0.43%，说明数据集中价格偏离整体处于较低水平。然而，各资产对的平均数据陈旧度均超过 1500 s，验证成功率约为 20.66%–23.96%，表明该数据集不仅包含价格偏差信息，还反映了 DeFi 预言机服务中常见的数据时效性不足和验证失败问题。因此，该数据集适合用于评估调度策略在异构资产请求、数据新鲜度约束和验证不确定性条件下的综合表现。
# 5. Audit-feedback Trust Management

## 5.1 Threat Model and Audit Signals

DeFi 预言机网络中的节点并非始终保持稳定可信。除能够持续提供类型匹配、及时且可验证数据的可信节点外，系统还可能面临低质量节点、疲劳节点和恶意节点。低质量节点通常表现为响应延迟较高、验证失败率较高或调用成本异常；疲劳节点在连续高负载下出现性能衰减；恶意节点则可能通过信誉投毒、休眠攻击、合谋转移、突发攻击、间歇规避和渐进漂移等方式隐藏真实风险。由于 DeFi 请求通常关联资产价格、清算条件和资金结算，预言机异常行为可能直接导致错误定价、错误清算或资金损失。

为刻画上述动态风险，本文将审计信号统一建模为服务结果信号、行为异常信号和历史趋势信号三类。其中，服务结果信号描述单次请求的执行质量，包括是否按时完成、是否通过验证、服务类型是否匹配以及最终响应延迟；行为异常信号用于捕捉潜在攻击行为，包括异常报价、拒绝服务、连续失败和与历史表现不一致的突发变化；历史趋势信号反映节点的持续风险状态，包括近期失败率、审计失败率、负载变化和冷却状态。与传统信誉机制仅依赖历史成功率不同，本文将上述信号作为审计反馈输入，使节点的近期异常行为能够及时影响后验可信度、有效信誉和后续调度约束。

因此，审计反馈在本文框架中并非简单的事后记录，而是与 HCRL 调度策略耦合的动态信任信号。一方面，它为调度器提供对低质量节点、疲劳节点和伪装型恶意节点的细粒度风险刻画；另一方面，它通过状态编码、动作掩码和奖励惩罚影响后续请求的节点选择，从而形成“服务执行—审计反馈—信任修正—再调度”的闭环信任管理机制。

## 5.2 Audit Posterior and Risk-aware Trust Update

对每个预言机 $o_i$，系统维护审计后验参数 $(\alpha_i,\beta_i)$，其中 $\alpha_i$ 表示审计通过证据，$\beta_i$ 表示审计失败或风险证据。基于该后验，节点的审计可信度定义为

$$
truth_i=\frac{\alpha_i}{\alpha_i+\beta_i}.
$$

与仅依赖历史成功率的信誉值不同，$truth_i$ 能够更直接地反映近期审计证据对节点可信度的影响。为将历史信誉、审计后验和惩罚状态统一到调度状态中，本文定义有效信誉：

$$
rep_i^{eff}=clip((1-w_a)rep_i+w_a truth_i-\eta cooldown_i,0,1),
$$

其中 $rep_i$ 为历史信誉，$w_a$ 为审计后验权重，$\eta$ 为冷却惩罚系数，$cooldown_i$ 表示节点在审计失败后的惩罚状态。有效信誉直接参与状态编码、动作掩码和奖励计算，从而使近期异常节点在后续调度中被自动降权。

审计触发采用风险感知机制。对于节点 $o_i$，审计概率定义为

$$
P(audit_i)=clip(p_{base}+p_{risk}risk_i,0,p_{max}),
$$

其中 $p_{base}$ 保证最低审计覆盖，$p_{risk}$ 控制风险敏感性，$p_{max}$ 限制最大审计频率。综合风险 $risk_i$ 由有效信誉、审计可信度、近期失败、冷却状态和审计陈旧度共同决定：

$$
risk_i=w_1(1-rep_i^{eff})+w_2(1-truth_i)+w_3fail_i^{recent}+w_4cooldown_i+w_5stale_i,
$$

其中 $stale_i$ 表示距离上次审计的时间间隔。该机制避免对所有节点进行高频审计，同时提高长期未审计且近期表现不稳定节点的审计概率。

审计更新遵循非对称原则。若审计通过，则系统缓慢累积可信证据：

$$
\alpha_i\leftarrow \alpha_i+1.
$$

若审计失败，则系统根据失败严重度 $sev_i$ 快速增加风险证据并降低历史信誉：

$$
\beta_i\leftarrow \beta_i+sev_i,\quad
rep_i\leftarrow rep_i-\delta_{fail}sev_i.
$$

当 $sev_i$ 超过阈值时，节点进入冷却状态。该设计体现了安全关键系统中的保守信任原则，即信任建立应缓慢，而失信惩罚应迅速。由此，恶意节点难以通过短期正常行为快速恢复信誉，同时偶发网络波动对可信节点的长期影响也被限制在可控范围内。
## 5.3 Audit-feedback Evaluation

为验证审计反馈机制的有效性，本文在多类动态攻击场景下分析可信节点与恶意节点的有效信誉变化。实验覆盖 reputation poisoning、sleeper attack、collusion shift、burst attack、intermittent evasion 和 gradual drift 六类攻击模式。评价重点不是单次请求是否成功，而是审计反馈能否在攻击阶段快速降低恶意节点信誉，并在恢复阶段保持较慢的信誉回升速度，从而抑制伪装型和机会主义攻击。

<p align="center">
  <img src="./paperMuter/audit_reputation_recovery_all_attacks_figure.png" alt="Audit reputation dynamics under dynamic attacks" width="850"/>
</p>

<p align="center"><strong>Figure 5.</strong> Asymmetric audit reputation dynamics under six dynamic attacks.</p>

如 Figure 5 所示，在六类动态攻击下，可信节点信誉保持稳定上升，而恶意节点在攻击阶段均出现明显下降。其中 reputation poisoning 和 sleeper attack 的信誉下降幅度分别达到 43.3% 和 45.4%，collusion shift、burst attack、intermittent evasion 和 gradual drift 也分别产生 39.8%、37.1%、32.8% 和 33.1% 的信誉下降。攻击结束后，恶意节点信誉并未立即恢复至可信节点水平，而是在非对称更新机制作用下缓慢回升。该现象说明审计后验与冷却惩罚能够延长恶意行为的负面影响，降低节点通过短期正常服务快速恢复信誉的可能性。


# 6. ZK-VOS Compliance Proof for Verifiable Scheduling

## 6.1 Design Rationale and Verification Scope

HCRL 调度器在链下执行复杂的状态编码、模式选择和主备预言机决策，能够处理动态负载、节点风险和审计反馈等高维信息。然而，DeFi 合约无法直接信任链下调度结果，尤其在价格查询、清算触发和资金结算等高价值场景中，调度结果必须满足可验证的合规约束。若将调度器的完整内部状态公开到链上，又会暴露节点风险评分、审计证据、策略输出和备份选择逻辑，从而为恶意预言机规避调度策略提供信息。

因此，本文提出 ZK-VOS（Zero-Knowledge Verifiable Oracle Scheduling）机制。其目标不是在链上复现完整的强化学习推理过程，而是在隐藏私有信任信息的前提下，证明给定调度结果满足预定义的服务合规条件。具体而言，ZK-VOS 将链下调度结果映射为一组可验证命题，包括节点选择合法性、服务类型兼容性、冷却状态约束、信誉阈值约束、成本预算、延迟约束和风险预算等。该设计将“智能调度”和“链上验证”解耦，使合约只需验证证明成立，即可接受调度结果。

ZK-VOS 的验证范围聚焦于调度合规性，而非完整神经网络计算。这样做有两个原因：一方面，HCRL 策略网络的前向传播包含大量私有状态与模型输出，直接证明完整推理会显著增加电路规模；另一方面，DeFi 合约真正需要确认的是调度结果是否满足业务安全边界，而不是调度策略的每个中间计算。因此，本文优先验证调度约束和审计相关规则，在保证合规性的同时降低证明成本。

## 6.2 Public Inputs, Private Witnesses, and Compliance Statements

对于时刻 $t$ 的调度动作 $(m_t,p_t,b_t)$，调度器首先生成公开承诺：

$$
C_t^{pub}=Com(r_t,m_t,p_t,b_t,B_c,B_l,B_r,h_t),
$$

其中 $r_t$ 表示 DeFi 请求，$B_c$、$B_l$ 和 $B_r$ 分别表示成本、延迟和风险阈值，$h_t$ 为隐藏状态、审计记录和策略输出的哈希承诺。公开输入仅包含请求承诺、调度承诺、预算阈值、状态哈希和证明本身；私有 witness 则包含节点风险评分、有效信誉、审计证据、冷却状态、候选备份评分、策略输出和审计严重度等敏感信息。

调度器基于私有 witness 生成零知识证明：

$$
\pi_t^{zk}=Prove(w_t,C_t^{pub}),
$$

链上合约执行验证：

$$
Verify(C_t^{pub},\pi_t^{zk})=1.
$$

若验证通过，则说明调度结果满足预定义合规规则，但链上合约无法获知内部风险评分、审计证据或策略输出。本文将 ZK-VOS 的核心合规命题表示为集合 $\mathcal{C}_{zk}$，其主要内容如下。

| Compliance statement | Verification target |
|---|---|
| Valid schedule | selected primary and backup oracles belong to the candidate set |
| Cooldown constraint | selected oracle is not in a forbidden cooldown state |
| Membership validity | selected oracle index is valid under the current oracle pool |
| Reputation threshold | selected oracle satisfies the minimum effective reputation requirement |
| Cost feasibility | scheduling cost does not exceed the predefined budget |
| Latency feasibility | expected or observed latency satisfies the deadline constraint |
| Risk feasibility | audit-aware risk score remains within the risk budget |
| Service compatibility | selected oracle supports the required service type |

上述命题共同保证调度结果不是任意链下输出，而是满足 DeFi 预言机服务合规边界的可验证服务编排结果。对于需要更高安全保证的应用场景，可进一步扩展电路以覆盖更复杂的审计更新或模式选择逻辑；但在默认设置中，本文优先验证与链上安全直接相关的约束，以避免过高的证明开销。

## 6.3 ZK-VOS Protocol

ZK-VOS 协议由承诺生成、证明生成、链上验证和结果接受四个阶段组成。首先，调度器对请求、调度动作和隐藏状态摘要生成公开承诺；随后根据私有 witness 构造合规证明；链上合约调用 Groth16 验证器检查证明是否成立；若验证通过，OracleScheduleRegistry 接受该调度结果并记录公开状态，否则拒绝该调度。

<p align="center">
  <img src="./paperMuter/zk-hcrl oracle schedule.png" alt="Algorithm 2: ZK-VOS Compliance Verification" width="850"/>
</p>

<p align="center"><strong>Algorithm 2.</strong> ZK-VOS Compliance Verification.</p>

## 6.4 Verification Results and On-chain Cost Analysis

为验证 ZK-VOS 对调度合规性的判别能力，本文首先构造合法调度与多类违规调度样本，包括冷却状态违规、成员索引非法、低信誉节点、成本超限、延迟超限、风险超限和服务类型不匹配。每类场景包含 1000 个样本。实验结果表明，ZK-VOS 能够正确接受全部合法调度，并拒绝全部违规调度，所有测试类型的判别准确率均达到 100%。在 witness 检查开销方面，Figure 6(a) 展示了不同合规场景下的时间分布。多数违规样例的检查时间集中在 151--157 ms 区间，而合法调度和成员合法性检查的平均时间约为 210 ms，说明该验证过程能够在较低开销下覆盖多类关键调度约束。

<p align="center">
  <img src="./paperMuter/fig_zk_vos_witness_time_distribution.png" alt="Distribution of witness-check time by compliance case" width="425"/>
</p>

<p align="center"><strong>Figure 6(a).</strong> Distribution of witness-check time across ZK-VOS compliance cases.</p>

进一步地，本文统计完整零知识证明流程的链下时间构成。如 Figure 6(b) 所示，单次证明流程的平均总耗时约为 6920 ms，其中 Groth16 proving 占 50.4%，snarkJS verification 占 45.8%，witness generation 仅占 3.9%。该结果表明，ZK-VOS 的主要链下开销来自证明生成与离线验证，而 witness 构造本身相对轻量。由于这些计算均在链下完成，其开销不会直接增加链上合约执行负担。

<p align="center">
  <img src="./paperMuter/fig_zk_vos_proof_time_donut.png" alt="Off-chain proof time composition" width="310"/>
</p>

<p align="center"><strong>Figure 6(b).</strong> Off-chain proof time composition of ZK-VOS.</p>

链上 gas 消耗如 Figure 6(c) 所示。Groth16 Verifier 部署消耗 525,594 gas，OracleScheduleRegistry 部署消耗 433,213 gas，真实 Groth16 验证下的 submitSchedule 调用消耗 272,132 gas。其中前两项属于一次性部署成本，而 submitSchedule 表示每次调度合规验证所需的链上开销。结果说明，ZK-VOS 将复杂状态编码、策略推理和证明生成保留在链下执行，链上仅负责证明验证和结果登记，从而在保持可验证合规性的同时将单次链上验证成本控制在可接受范围内。

<p align="center">
  <img src="./paperMuter/gas_scheme5_segmented_bar.png" alt="Relative on-chain gas usage of ZK-VOS" width="410"/>
</p>

<p align="center"><strong>Figure 6(c).</strong> Relative on-chain gas usage of ZK-VOS.</p>

# 7. Integrated Experimental Analysis

## 7.1 Experimental Settings and Stability Analysis

本节进一步给出 ZK-HCRL Oracle 的实现设置与稳定性分析。与第 4.5 节的主调度性能比较、第 5.3 节的审计反馈验证以及第 6.4 节的 ZK-VOS 合规验证不同，本节关注模型在随机初始化、学习率变化、恶意节点比例变化和系统规模扩展下的整体表现。实验中的请求流按照泊松过程生成，候选预言机具有异构服务类型、调用成本、处理能力、验证成功率和行为风险。比较方法包括 Random、Round-Robin、Earliest、DQN、BLOR、SemiGreedy、PPO、RA-DCQN、PB-SafeDQN、COBRA-Oracle 和 HCRL-Oracle。

Table 7 汇总了不同方法在多随机种子下的平均表现。HCRL-Oracle 在成功率、准时成功率和恶意节点规避方面取得较优结果，其成功率达到 77.19%，准时成功率达到 99.13%，恶意节点选择率仅为 0.84%。与 PPO 相比，HCRL-Oracle 的成功率更高，且恶意节点选择率更低；与 DQN、RA-DCQN 等单层强化学习方法相比，HCRL-Oracle 在保持较高可信节点选择比例的同时进一步降低了恶意节点分配风险。

**Table 7. Stability analysis across random seeds.**

| Method | n | Reward ↑ | Success Rate ↑ | Success Time Rate ↑ | Malicious Rate ↓ | Trusted Rate ↑ | CPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 2 | 1507.26 ± 84.63 | 54.83 ± 1.79% | 97.03 ± 0.40% | 20.44 ± 0.51% | 59.93 ± 0.11% | 0.681 ± 0.004 |
| Round-Robin | 2 | 1839.00 ± 66.71 | 59.52 ± 1.25% | 99.93 ± 0.04% | 20.00 ± 0.00% | 60.00 ± 0.00% | 0.794 ± 0.017 |
| Earliest | 2 | 1827.01 ± 44.69 | 59.26 ± 0.72% | 99.88 ± 0.05% | 20.04 ± 0.06% | 59.98 ± 0.00% | 0.797 ± 0.011 |
| DQN | 3 | 2028.80 ± 67.56 | 71.90 ± 1.05% | 94.78 ± 0.21% | 1.74 ± 0.27% | 95.02 ± 0.75% | 1.184 ± 0.008 |
| BLOR | 2 | 1088.90 ± 21.86 | 48.69 ± 0.55% | 94.95 ± 0.68% | 11.20 ± 0.11% | 74.18 ± 1.10% | 1.155 ± 0.021 |
| SemiGreedy | 2 | 316.46 ± 253.41 | 31.60 ± 5.90% | 92.40 ± 0.07% | 7.03 ± 1.49% | 84.32 ± 1.96% | 1.165 ± 0.035 |
| PPO | 2 | **2173.57 ± 23.79** | 73.15 ± 0.04% | 93.73 ± 0.43% | 1.85 ± 0.35% | 93.43 ± 0.53% | **1.039 ± 0.002** |
| RA-DCQN | 2 | 2094.38 ± 99.43 | 72.06 ± 1.50% | 94.84 ± 0.59% | 1.78 ± 0.04% | **95.41 ± 1.17%** | 1.181 ± 0.016 |
| PB-SafeDQN | 2 | 1385.08 ± 11.60 | 69.82 ± 0.04% | 92.69 ± 0.51% | 3.46 ± 0.93% | 89.98 ± 0.32% | 1.881 ± 0.040 |
| COBRA-Oracle | 2 | 1317.62 ± 26.28 | 72.33 ± 0.69% | 91.39 ± 0.62% | 3.04 ± 0.90% | 89.85 ± 5.33% | 2.028 ± 0.012 |
| **HCRL-Oracle** | 3 | 1823.41 ± 160.93 | **77.19 ± 0.29%** | **99.13 ± 0.89%** | **0.84 ± 0.16%** | 91.45 ± 4.36% | 1.558 ± 0.099 |

此外，本文评估了不同学习率下 HCRL-Oracle 的训练收敛特性。Figure 7 给出了总奖励、服务成功率和单位成功成本随训练轮次的变化情况。总体来看，各学习率设置均在训练初期快速收敛，并在约 15–25 个 episode 后趋于稳定。较小学习率通常表现出更平滑的收敛轨迹，但早期提升相对较慢；较大学习率虽能加快前期学习，却更易在后期产生波动。综合三项指标可见，HCRL-Oracle 对学习率具有较好的适应性，但仍需在收敛速度、稳定性与成本效率之间进行权衡。

<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="./paperMuter/hcrl_convergence_reward_wideaxis.png" alt="Reward convergence" width="100%"/><br/>
      <strong>(a)</strong> Reward
    </td>
    <td align="center" width="33%">
      <img src="./paperMuter/hcrl_convergence_success_rate_wideaxis.png" alt="Success-rate convergence" width="100%"/><br/>
      <strong>(b)</strong> Success rate
    </td>
    <td align="center" width="33%">
      <img src="./paperMuter/hcrl_convergence_cost_per_success_wideaxis.png" alt="Cost-per-success convergence" width="100%"/><br/>
      <strong>(c)</strong> Cost per success
    </td>
  </tr>
</table>

<p align="center"><strong>Figure 7.</strong> Learning-rate convergence analysis of HCRL-Oracle.</p>

## 7.2 Robustness and Scalability Analysis

为进一步评估 ZK-HCRL Oracle 在复杂环境中的适应能力，本文从恶意节点比例和预言机池规模两个角度开展补充实验。前者用于检验模型在高风险场景下的鲁棒性，后者用于分析模型在更大候选空间中的可扩展性。与标准性能比较不同，本节重点关注模型在风险增强和系统规模变化时的稳定性、风险控制能力和泛化表现。

### (1) Robustness under increasing malicious-node ratios

本文将恶意节点比例从 10% 逐步提高至 50%，观察 HCRL-Oracle 的性能变化。结果显示，随着恶意节点比例升高，服务成功率由 85.99% 下降至 75.18%，说明高风险环境会对任务完成率造成一定影响。然而，在 50% 恶意节点场景下，模型的准时成功率仍保持在 98.88% 以上，恶意节点选择率最高仅为 1.73%，可信节点选择率稳定在约 94%–96%。这表明 HCRL-Oracle 在风险增强时虽存在一定性能退化，但仍能维持较强的恶意节点规避和时效性保障能力。

从效率指标看，平均响应时间始终维持在约 5.16–5.18，说明恶意比例升高并未导致明显调度延迟。单位成功成本由 1.507 上升至 1.683，审计失败率由 4.90% 上升至 6.40%，表明系统需要付出更多审计和备份开销来维持可靠性。总体而言，模型呈现出“成功率有限下降、成本适度增加、恶意选择率持续受控”的特征，说明审计感知状态、风险约束和主备调度机制能够共同增强高风险场景下的鲁棒性。

<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="./paperMuter/fig1_malicious_ratio_reliability.png" alt="Reliability under different malicious-node ratios" width="100%"/><br/>
      <strong>(a)</strong> Reliability
    </td>
    <td align="center" width="33%">
      <img src="./paperMuter/fig2_malicious_ratio_efficiency.png" alt="Efficiency under different malicious-node ratios" width="100%"/><br/>
      <strong>(b)</strong> Efficiency
    </td>
    <td align="center" width="33%">
      <img src="./paperMuter/fig3_malicious_ratio_risk_metrics.png" alt="Risk-related metrics under different malicious-node ratios" width="100%"/><br/>
      <strong>(c)</strong> Risk-related metrics
    </td>
  </tr>
</table>

<p align="center"><strong>Figure 8.</strong> Robustness analysis of HCRL-Oracle under different malicious-node ratios.</p>

如 Figure 8 所示，恶意节点比例升高后，成功率逐步下降，但准时成功率始终保持高位；单位成功成本上升，而平均响应时间基本稳定；恶意节点选择率和审计失败率随风险增强而增加，但整体仍处于较低水平。上述结果表明，HCRL-Oracle 能够在恶意节点比例显著提高时保持较强的风险抑制能力。

### (2) Scalability under different oracle-pool sizes

为验证模型在不同系统规模下的适应能力，本文扩大预言机池规模并保持相应请求压力。实验结果表明，当预言机规模由 15 扩展至 150 时，模型成功率由 67.46% 提升至 84.30%，平均响应时间由 5.594 降低至 5.145，说明更大的候选空间为调度器提供了更多高质量节点选择，从而提升任务完成率和响应效率。

随着系统规模增长，模型总奖励由 1308.30 上升至 2239.37，准时成功率由 89.48% 提升至约 96.96%–98.99%，恶意节点选择率始终低于 1%。可信节点选择率在中大规模场景中上升至约 92%–94%，说明层次化策略、审计反馈和图表示能够帮助模型在更复杂的候选集合中保持较好的节点区分能力。

<table align="center">
  <tr>
    <td align="center" width="33%">
      <img src="./paperMuter/fig4_oracle_scale_reward.png" alt="Reward across different oracle-pool scales" width="100%"/><br/>
      <strong>(a)</strong> Reward
    </td>
    <td align="center" width="33%">
      <img src="./paperMuter/fig5_oracle_scale_quality.png" alt="Scheduling quality across different oracle-pool scales" width="100%"/><br/>
      <strong>(b)</strong> Scheduling quality
    </td>
    <td align="center" width="33%">
      <img src="./paperMuter/fig6_oracle_scale_cost_recovery.png" alt="Cost and recovery across different oracle-pool scales" width="100%"/><br/>
      <strong>(c)</strong> Cost and recovery
    </td>
  </tr>
</table>

<p align="center"><strong>Figure 9.</strong> Scalability analysis of HCRL-Oracle under different oracle-pool scales.</p>

Figure 9 显示，随着预言机池规模扩大，总奖励整体上升，成功率、准时成功率和可信节点选择率在中大规模场景下明显改善。同时，单位成功成本、备份恢复率和审计失败率保持在合理范围内，未随规模扩大而失控。综合 Figure 8 和 Figure 9 可见，HCRL-Oracle 在高恶意比例和大规模预言机池中均表现出较好的稳定性，说明本文提出的审计感知层次化调度机制具有良好的环境适应性和系统规模泛化能力。

## 7.3 Ablation and Deployment Discussion

为分析各模块对整体性能的贡献，本文进一步进行消融实验。实验比较完整模型 ZK-HCRL-Full 与去除 GNN 编码器、审计反馈、约束引导、随机备份、固定单节点执行和固定并行执行等变体。完整模型并非在所有单项指标上均最优，而是在成功率、恶意节点规避、可信节点覆盖、恢复能力和单位成功成本之间取得更优综合权衡。完整结果如 Figure 10 和 Table 10 所示。

**Table 10. Ablation results of ZK-HCRL Oracle.**

| Variant | Reward ↑ | Success ↑ | On-time ↑ | Resp. ↓ | Cost ↓ | CPS ↓ | Malicious ↓ | Trusted ↑ | Recovery ↑ | Single | Serial | Parallel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ZK-HCRL-Full | **2131.76** | 79.38% | 94.38% | 5.376 | 1.092 | 1.376 | 0.59% | **96.08%** | 29.31% | 2.43% | 18.55% | 79.03% |
| w/o GNN | 1992.94 | 69.88% | 96.15% | 5.291 | 1.145 | 1.639 | 1.85% | 87.40% | 23.17% | 2.15% | 34.88% | 62.98% |
| w/o Audit | 1657.39 | 72.00% | 96.47% | 5.257 | 1.272 | 1.767 | 4.05% | 82.70% | **32.93%** | 1.77% | 24.30% | 73.92% |
| w/o Constraint | 1657.39 | 76.88% | 94.38% | 5.376 | 1.282 | 1.668 | 0.83% | 94.08% | 29.31% | 2.43% | 18.55% | 79.03% |
| Random Backup | 1519.82 | 67.83% | **97.10%** | **5.207** | 1.136 | 1.675 | **0.50%** | 93.42% | 18.03% | 2.55% | 30.18% | 67.27% |
| Fixed Single | 1076.54 | 65.58% | 88.28% | 5.631 | **0.837** | **1.276** | 0.78% | 96.33% | 0.00% | 100.00% | 0.00% | 0.00% |
| Fixed Parallel | 1842.93 | 77.55% | 95.85% | 5.293 | 1.279 | 1.649 | 0.65% | 89.48% | 32.10% | 0.00% | 0.00% | 100.00% |

如 Table 10 所示，ZK-HCRL-Full 在总奖励和可信节点选择率上取得最优结果，分别达到 2131.76 和 96.08%，同时保持较低的恶意节点选择率 0.59%。去除 GNN 后，成功率从 79.38% 降至 69.88%，可信节点选择率下降至 87.40%，恶意节点选择率上升至 1.85%，说明图结构表征有助于建模预言机之间的服务关系和风险关联。去除审计反馈后，恶意节点选择率进一步升至 4.05%，可信节点选择率降至 82.70%，表明审计后验对动态风险修正和恶意节点抑制具有关键作用。

约束引导消融中，w/o Constraint 的总奖励下降至 1657.39，单位成功成本升至 1.668，说明成本、风险和可行性约束能够提升长期调度收益和成本效率。由于该变体的执行模式分布与完整模型接近，本文将其解释为去除约束惩罚或约束引导后的消融，而非完全移除动作掩码。

备份相关消融进一步验证了主备机制的必要性。Random Backup 的恢复率仅为 18.03%，明显低于完整模型的 29.31%，说明学习式备份选择器能够提升故障恢复能力。Fixed Single 虽然成本最低，但成功率仅为 65.58%，且恢复率为 0%，难以满足高风险 DeFi 请求的可靠性需求。Fixed Parallel 的恢复率较高，但成本和单位成功成本均高于完整模型，说明无差别并行会带来冗余开销，而自适应模式策略能够在可靠性与成本之间取得更优平衡。

部署方面，ZK-HCRL Oracle 采用链下调度与链上验证的混合架构。高维状态编码、策略推理、审计更新和证明生成在链下完成，链上仅保存承诺、证明和调度接受结果。第 6.4 节结果表明，ZK-VOS 可通过 222 个 template instances、2700 个非线性约束和 3112 个线性约束完成调度合规验证，单次 submitSchedule 的真实 Groth16 验证成本为 272,132 gas。该设计在降低链上计算负担的同时，保留了 DeFi 预言机服务所需的可验证合规性。

总体而言，消融实验表明：GNN 编码器和审计反馈提升了可信节点识别与风险抑制能力；学习式备份和主备执行增强了失败恢复能力；自适应模式策略避免了固定单节点可靠性不足和固定并行冗余成本过高的问题。结合 ZK-VOS，本文框架实现了“智能调度—审计反馈—零知识验证”的闭环式可验证预言机服务调度。

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
