# -*- coding: utf-8 -*-
"""Aggregate *_final_results.csv files under output/ into mean/std tables.
Usage:
    python scripts/aggregate_final_results.py --output_dir "TCO-DRL_with baseline/output"
"""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='TCO-DRL_with baseline/output')
args = parser.parse_args()
base = Path(args.output_dir)
files = sorted(base.glob('**/*_final_results.csv'))
if not files:
    raise SystemExit(f'No *_final_results.csv found under {base}')
frames = []
for f in files:
    df = pd.read_csv(f)
    df['run_folder'] = str(f.parent)
    df['run_file'] = f.name
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)
out_all = base / 'aggregated_all_runs.csv'
all_df.to_csv(out_all, index=False, encoding='utf-8-sig')
num_cols = [c for c in all_df.columns if c not in ['method', 'run_folder', 'run_file']]
summary = all_df.groupby('method')[num_cols].agg(['mean','std','count'])
summary.columns = ['_'.join([str(x) for x in col]).strip('_') for col in summary.columns.values]
summary = summary.reset_index()
out_summary = base / 'aggregated_mean_std.csv'
summary.to_csv(out_summary, index=False, encoding='utf-8-sig')
print('Saved:', out_all)
print('Saved:', out_summary)
print(summary)
