# Audit Reputation Recovery Experiments

这是一个整理好的、可直接放到 `TCO-DRL` 根目录运行的审计信誉实验文件夹。

实验目标：展示 HCRL 审计感知信誉机制的**非对称动态**：

```text
正常/伪装阶段 → 作恶阶段 → 恢复良好行为阶段
信誉保持/积累 → 快速下降 → 缓慢回升
```

该实验不做复杂消融，只验证：

1. 作恶行为会触发审计失败并快速降低恶意预言机信誉；
2. 攻击结束后，如果节点持续保持良好行为，信誉会谨慎、缓慢恢复；
3. trusted 与 malicious 节点的信誉曲线存在清晰分离。

## 文件结构

```text
audit_reputation_recovery_experiments/
├── README.md
├── requirements.txt
├── run_recovery_experiment.ps1
├── run_audit_reputation_recovery.py
├── plot_reputation_recovery_curves.py
└── plot_all_attack_recovery_curves.py
```

## 运行方式 PowerShell

在 `TCO-DRL` 根目录执行：

```powershell
python .\audit_reputation_recovery_experiments\run_audit_reputation_recovery.py `
  --trace .\experiments_real_trace\data\real_oracle_trace.csv `
  --out .\audit_reputation_recovery_experiments\output `
  --seeds 3,4,5,6,7 `
  --requests 12000 `
  --oracles 120 `
  --malicious-ratio 0.30 `
  --attack-onset-ratio 0.25 `
  --attack-end-ratio 0.65
```

然后绘制代表性三场景图：

```powershell
python .\audit_reputation_recovery_experiments\plot_reputation_recovery_curves.py `
  --curve-csv .\audit_reputation_recovery_experiments\output\audit_reputation_recovery_curve.csv `
  --summary-csv .\audit_reputation_recovery_experiments\output\audit_reputation_recovery_summary_mean_std.csv `
  --out .\audit_reputation_recovery_experiments\output\audit_reputation_recovery_figure.png
```

如果正文或附录需要六类攻击全部曲线，运行：

```powershell
python .\audit_reputation_recovery_experiments\plot_all_attack_recovery_curves.py `
  --curve-csv .\audit_reputation_recovery_experiments\output\audit_reputation_recovery_curve.csv `
  --summary-csv .\audit_reputation_recovery_experiments\output\audit_reputation_recovery_summary_mean_std.csv `
  --out .\audit_reputation_recovery_experiments\output\audit_reputation_recovery_all_attacks_figure.png
```

当前整理版默认使用更长攻击窗口：`requests=12000, attack_onset_ratio=0.25, attack_end_ratio=0.65`，即约 3000 步正常/伪装、4800 步攻击、4200 步恢复。

也可以直接运行：

```powershell
.\audit_reputation_recovery_experiments\run_recovery_experiment.ps1
```

## 输出文件

```text
output/
├── audit_reputation_recovery_curve.csv
├── audit_reputation_recovery_event_timeline.csv
├── audit_reputation_recovery_summary_by_seed.csv
├── audit_reputation_recovery_summary_mean_std.csv
├── paper_table_reputation_recovery.csv
├── paper_table_reputation_recovery.tex
├── audit_reputation_recovery_figure.png
├── audit_reputation_recovery_figure.pdf
├── audit_reputation_recovery_all_attacks_figure.png
└── audit_reputation_recovery_all_attacks_figure.pdf
```

## 建议论文标题

```text
Asymmetric Reputation Dynamics under Dynamic Oracle Attacks
```

## 可直接写进论文的描述

> The audit-aware reputation mechanism exhibits asymmetric dynamics: malicious behavior triggers rapid reputation degradation, whereas sustained benign behavior after the attack leads to conservative and gradual reputation recovery.

中文解释：

> 审计感知信誉机制表现出非对称动态特性：作恶行为会触发信誉快速下降，而攻击结束后持续良好行为只会带来谨慎、缓慢的信誉恢复。



## 六类动态攻击含义

| Scenario | 中文含义 | 攻击逻辑 |
|---|---|---|
| `reputation_poisoning` | 声誉投毒 | 前期表现良好积累信誉，攻击阶段开始作恶。 |
| `sleeper_attack` | 潜伏攻击 | 长时间正常/潜伏，在指定窗口突然激活攻击。 |
| `collusion_shift` | 合谋偏移 | 多个恶意 oracle 在攻击阶段协同偏移或共同异常。 |
| `burst_attack` | 突发攻击 | 短时间窗口内恶意行为集中爆发。 |
| `intermittent_evasion` | 间歇规避 | 间歇性作恶与正常行为交替，试图规避审计。 |
| `gradual_drift` | 渐进漂移 | 恶意偏差逐步增大，模拟缓慢异常漂移。 |

正文如果版面有限，可以使用 `audit_reputation_recovery_figure.png`；如果要求六类攻击全部展示，使用 `audit_reputation_recovery_all_attacks_figure.png`。


## 拉长攻击阶段

本版脚本已经支持直接调节攻击阶段比例：

```powershell
python .\audit_reputation_recovery_experiments\run_audit_reputation_recovery.py `
  --trace .\experiments_real_trace\data\real_oracle_trace.csv `
  --out .\audit_reputation_recovery_experiments\output `
  --seeds 3,4,5,6,7 `
  --requests 12000 `
  --oracles 120 `
  --malicious-ratio 0.30 `
  --attack-onset-ratio 0.25 `
  --attack-end-ratio 0.65
```

含义：

```text
0%–25%   Benign / camouflage
25%–65%  Attack-active
65%–100% Recovery
```

如果需要更极端攻击，可以使用：

```powershell
--requests 15000 --attack-onset-ratio 0.25 --attack-end-ratio 0.70
```

但恢复窗口不能太短，否则无法展示“缓慢回升”。
