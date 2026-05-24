# Audit Reputation Drop Experiments for HCRL-Oracle

本目录专门用于一个更窄、更清晰的论文问题：

> 在不同动态攻击场景下，HCRL 的审计算法是否能识别异常行为，并成功降低恶意预言机的信誉值？

本实验**不做 Full vs w/o Audit 消融**，也不重点比较调度成功率。它只保留完整 HCRL 风格审计机制，并输出恶意节点信誉曲线、可信节点信誉曲线、信誉差距和审计失败率。

## 场景

- `static_malicious`: 恶意节点全程攻击。
- `reputation_poisoning`: 恶意节点前期正常服务刷高信誉，后期攻击。
- `burst_attack`: 中间窗口突发攻击。
- `sleeper_attack`: 前半程休眠，后半程激活攻击。
- `collusion_shift`: 部分恶意节点协同偏移。
- `gradual_drift`: 恶意行为逐步增强。
- `intermittent_evasion`: 间歇攻击，模拟规避审计。

## 运行

Linux/macOS:

```bash
bash audit_reputation_drop_experiments/scripts/run_reputation_drop.sh
```

Windows PowerShell:

```powershell
.\audit_reputation_drop_experiments\scripts\run_reputation_drop.ps1
```

也可以手动运行：

```bash
python audit_reputation_drop_experiments/run_audit_reputation_drop.py \
  --trace experiments_real_trace/data/real_oracle_trace.csv \
  --out audit_reputation_drop_experiments/output \
  --seeds 3,4,5,6,7 \
  --requests 6000 \
  --oracles 120 \
  --malicious-ratio 0.30
```

## 输出

```text
audit_reputation_event_timeline.csv      # 每个请求级事件
audit_reputation_curve.csv               # 每个 interval 的信誉曲线与审计统计
audit_reputation_summary_by_seed.csv     # 每个场景/seed 的信誉下降统计
audit_reputation_summary_mean_std.csv    # mean±std 汇总
paper_table_reputation_drop.csv          # 论文表格
paper_table_reputation_drop.tex          # LaTeX 表格
plots/*_reputation_curve.png             # 恶意/可信信誉曲线
plots/*_reputation_gap.png               # 信誉差距曲线
```

## 论文中建议报告的核心指标

- `pre_attack_malicious_rep`: 攻击前恶意节点平均信誉。
- `post_attack_malicious_rep`: 攻击后恶意节点平均信誉。
- `malicious_rep_drop_abs`: 恶意节点信誉绝对下降。
- `malicious_rep_drop_pct`: 恶意节点信誉相对下降百分比。
- `malicious_truth_drop_abs`: 审计 Beta 后验 truth score 下降。
- `reputation_gap_increase`: 可信节点与恶意节点信誉差距扩大程度。
- `drop_lag_intervals`: 攻击开始后多少个 interval 出现至少 0.05 的恶意信誉下降。
- `attack_audit_fail_rate`: 攻击阶段审计失败率。

## 推荐论文表述

> Under all dynamic attack scenarios, the audit-aware reputation mechanism consistently reduces the effective reputation of malicious oracles after attack activation, while preserving a higher reputation for trusted oracles. This demonstrates that the audit module provides an interpretable and responsive reputation degradation mechanism against dynamic oracle misbehavior.

