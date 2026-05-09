# -*- coding: utf-8 -*-
"""
Aggregate multi-seed COBRA/TCO-DRL final result CSVs.

Usage:
    python scripts/aggregate_seed_results.py --output_dir "TCO-DRL_with baseline/output/seed_formal_xxx"

Outputs under output_dir:
    seed_all_runs.csv
    seed_mean_std.csv
    seed_paper_table.csv
    seed_summary.md
"""
import argparse
from pathlib import Path
import math
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True, help='Folder containing seed run subfolders or final_results.csv files')
parser.add_argument('--pattern', type=str, default='**/*_final_results.csv')
args = parser.parse_args()

base = Path(args.output_dir)
files = sorted(base.glob(args.pattern))
if not files:
    raise SystemExit(f'No *_final_results.csv found under {base.resolve()}')

frames = []
for f in files:
    df = pd.read_csv(f)
    if 'seed' not in df.columns:
        # Best-effort fallback for older output files: parse SeedXX from filename/folder.
        import re
        m = re.search(r'Seed(\d+)', str(f))
        df['seed'] = int(m.group(1)) if m else -1
    if 'run_id' not in df.columns:
        df['run_id'] = f.parent.name
    df['run_folder'] = str(f.parent)
    df['run_file'] = f.name
    # Derived metrics useful for paper review.
    if 'cost' in df.columns and 'success_rate' in df.columns:
        df['cost_per_success'] = df['cost'] / df['success_rate'].replace(0, float('nan'))
    if 'assigned_malicious' in df.columns and 'assigned_trusted' in df.columns:
        total_assigned = df.get('assigned_malicious', 0) + df.get('assigned_normal', 0) + df.get('assigned_trusted', 0)
        df['malicious_assignment_rate'] = df['assigned_malicious'] / total_assigned.replace(0, float('nan'))
        df['trusted_assignment_rate'] = df['assigned_trusted'] / total_assigned.replace(0, float('nan'))
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv(base / 'seed_all_runs.csv', index=False, encoding='utf-8-sig')

id_cols = {'method', 'run_id', 'run_tag', 'run_folder', 'run_file'}
num_cols = [c for c in all_df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(all_df[c])]
# Do not average seed itself as a metric.
num_cols = [c for c in num_cols if c != 'seed']

rows = []
for method, g in all_df.groupby('method'):
    row = {'method': method, 'n_seeds': int(g['seed'].nunique()) if 'seed' in g.columns else int(len(g))}
    for col in num_cols:
        vals = pd.to_numeric(g[col], errors='coerce').dropna()
        n = len(vals)
        if n == 0:
            continue
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * sem if n > 1 else 0.0
        row[f'{col}_mean'] = mean
        row[f'{col}_std'] = std
        row[f'{col}_ci95'] = ci95
    rows.append(row)
summary = pd.DataFrame(rows)
# Put common baselines in a stable order.
order = ['Random','Round-Robin','Earliest','DQN','BLOR','SemiGreedy','PPO','RA-DDQN','PB-SafeDQN','COBRA-Oracle','HCRL-Oracle']
summary['__order'] = summary['method'].apply(lambda m: order.index(m) if m in order else 999)
summary = summary.sort_values('__order').drop(columns='__order')
summary.to_csv(base / 'seed_mean_std.csv', index=False, encoding='utf-8-sig')

# Compact paper table with mean ± std for key metrics.
key_metrics = [
    'success_rate', 'total_rewards', 'cost', 'avg_responseT',
    'cost_per_success', 'assigned_malicious', 'assigned_trusted',
    'malicious_assignment_rate', 'trusted_assignment_rate',
    'backup_used_rate', 'backup_recovery_rate', 'conditional_backup_recovery_rate', 'backup_skipped_rate', 'constraint_violation_rate', 'hcrl_single_mode_rate', 'hcrl_serial_mode_rate', 'hcrl_parallel_mode_rate'
]
paper = pd.DataFrame({'method': summary['method'], 'n_seeds': summary['n_seeds']})
for metric in key_metrics:
    mean_col, std_col = f'{metric}_mean', f'{metric}_std'
    if mean_col in summary.columns and std_col in summary.columns:
        paper[metric] = summary.apply(lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}", axis=1)
paper.to_csv(base / 'seed_paper_table.csv', index=False, encoding='utf-8-sig')

# Markdown summary emphasizing COBRA vs key baselines.
md = []
md.append('# Multi-seed Summary')
md.append('')
md.append(f'- Number of final result files: {len(files)}')
md.append(f'- Output directory: `{base.resolve()}`')
md.append('')
if 'success_rate_mean' in summary.columns:
    best = summary.sort_values('success_rate_mean', ascending=False).iloc[0]
    md.append(f"- Best mean success rate: **{best['method']}** = {best['success_rate_mean']:.4f} ± {best.get('success_rate_std', 0):.4f}")
if 'assigned_malicious_mean' in summary.columns:
    best_safe = summary.sort_values('assigned_malicious_mean', ascending=True).iloc[0]
    md.append(f"- Lowest malicious assignment: **{best_safe['method']}** = {best_safe['assigned_malicious_mean']:.2f} ± {best_safe.get('assigned_malicious_std', 0):.2f}")
if 'cost_per_success_mean' in summary.columns:
    cps = summary.sort_values('cost_per_success_mean', ascending=True).iloc[0]
    md.append(f"- Best cost per success: **{cps['method']}** = {cps['cost_per_success_mean']:.4f} ± {cps.get('cost_per_success_std', 0):.4f}")
md.append('')
md.append('## Files')
md.append('- `seed_all_runs.csv`: all method-level results for every seed')
md.append('- `seed_mean_std.csv`: mean/std/95% CI grouped by method')
md.append('- `seed_paper_table.csv`: compact mean ± std table')
(base / 'seed_summary.md').write_text('\n'.join(md), encoding='utf-8')

print('Saved:', base / 'seed_all_runs.csv')
print('Saved:', base / 'seed_mean_std.csv')
print('Saved:', base / 'seed_paper_table.csv')
print('Saved:', base / 'seed_summary.md')
print(paper)
