# -*- coding: utf-8 -*-
"""
Plot multi-seed mean ± std figures for paper.

Usage:
    python scripts/plot_seed_results_cn.py \
      --summary_csv "TCO-DRL_with baseline/output/seed_xxx/seed_mean_std.csv" \
      --all_csv "TCO-DRL_with baseline/output/seed_xxx/seed_all_runs.csv" \
      --output_dir "TCO-DRL_with baseline/output/seed_xxx/seed_figs"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--summary_csv', required=True)
parser.add_argument('--all_csv', required=False, default=None)
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(args.summary_csv)
all_df = pd.read_csv(args.all_csv) if args.all_csv and Path(args.all_csv).exists() else None

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

order = ['Random','Round-Robin','Earliest','DQN','BLOR','SemiGreedy','PPO','RA-DDQN','PB-SafeDQN','COBRA-Oracle','HCRL-Oracle']
summary['__order'] = summary['method'].apply(lambda m: order.index(m) if m in order else 999)
summary = summary.sort_values('__order').drop(columns='__order')

def bar_mean_std(metric, title, ylabel, filename, higher_better=True):
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    if mean_col not in summary.columns:
        return
    means = summary[mean_col].astype(float).values
    stds = summary[std_col].astype(float).values if std_col in summary.columns else np.zeros_like(means)
    labels = summary['method'].tolist()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_title(title)
    ax.set_xlabel('方法')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    y_min = np.nanmin(means - stds)
    y_max = np.nanmax(means + stds)
    pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.1
    ax.set_ylim(max(0, y_min - pad) if higher_better else y_min - pad, y_max + pad)
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+pad*0.15, f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / filename, bbox_inches='tight')
    plt.close(fig)

bar_mean_std('success_rate', '多随机种子最终成功率对比（均值±标准差）', '成功率', 'seed_成功率_mean_std.png')
bar_mean_std('total_rewards', '多随机种子最终总奖励对比（均值±标准差）', '总奖励', 'seed_总奖励_mean_std.png')
bar_mean_std('cost', '多随机种子平均成本对比（均值±标准差）', '平均成本', 'seed_平均成本_mean_std.png', higher_better=False)
bar_mean_std('avg_responseT', '多随机种子平均响应时间对比（均值±标准差）', '平均响应时间', 'seed_平均响应时间_mean_std.png', higher_better=False)
bar_mean_std('assigned_malicious', '多随机种子恶意节点分配次数对比（均值±标准差）', '恶意节点分配次数', 'seed_恶意节点分配_mean_std.png', higher_better=False)
bar_mean_std('assigned_trusted', '多随机种子可信节点分配次数对比（均值±标准差）', '可信节点分配次数', 'seed_可信节点分配_mean_std.png')
bar_mean_std('cost_per_success', '多随机种子单位成功成本对比（均值±标准差）', '单位成功成本', 'seed_单位成功成本_mean_std.png', higher_better=False)

# Success-cost tradeoff scatter.
if 'success_rate_mean' in summary.columns and 'cost_mean' in summary.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in summary.iterrows():
        ax.scatter(r['cost_mean'], r['success_rate_mean'], s=80)
        ax.text(r['cost_mean'], r['success_rate_mean'], ' ' + r['method'], fontsize=10, va='center')
    ax.set_title('成功率-成本权衡图（多 seed 均值）')
    ax.set_xlabel('平均成本')
    ax.set_ylabel('成功率')
    ax.grid(True, linestyle='--', alpha=0.35)
    fig.tight_layout()
    fig.savefig(out / 'seed_成功率_成本权衡.png', bbox_inches='tight')
    plt.close(fig)

# Per-seed line/box style for key methods.
if all_df is not None and 'seed' in all_df.columns and 'success_rate' in all_df.columns:
    key_methods = [m for m in ['DQN','PPO','RA-DDQN','PB-SafeDQN','COBRA-Oracle','HCRL-Oracle'] if m in set(all_df['method'])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in key_methods:
        g = all_df[all_df['method'] == method].sort_values('seed')
        ax.plot(g['seed'], g['success_rate'], marker='o', label=method)
    ax.set_title('关键方法在不同随机种子下的成功率')
    ax.set_xlabel('随机种子')
    ax.set_ylabel('成功率')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'seed_关键方法按种子成功率.png', bbox_inches='tight')
    plt.close(fig)

print('Saved seed figures to:', out)
