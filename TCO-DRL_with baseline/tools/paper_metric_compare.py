#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
paper_metric_compare.py

Purpose
-------
Parse TCO-DRL / HCRL experiment logs and generate paper-style metric comparisons
aligned with the original TCO-DRL paper:
  1) convergence reward curves
  2) main performance table
  3) malicious / benign / trusted assignment distribution
  4) mean ± std robustness summary
  5) response time and cost comparison
  6) noise robustness plots, if run tags contain noise levels
  7) increasing-malicious-oracle plots, if run tags contain malicious counts
  8) backup and HCRL mode diagnostics

Recommended location
--------------------
Copy this file to:
    TCO-DRL_with baseline/tools/paper_metric_compare.py

Basic usage
-----------
From "TCO-DRL_with baseline":

    python tools/paper_metric_compare.py --input_dir .\output --out_dir .\paper_metric_outputs

For a single log:

    python tools/paper_metric_compare.py --logs ".\output\xxx\xxx.txt" --out_dir .\paper_metric_outputs

If your log says "after 2000 requests" but you want paper-style counts normalized
to 6000 requests, use:

    python tools/paper_metric_compare.py --input_dir .\output --out_dir .\paper_metric_outputs --request_num_override 6000

Notes
-----
- Percent metrics in logs such as "success_rate: 75.20%" are stored as 75.20.
- If "match_rate" is not in your log, the script will skip the match-rate figure.
- For original-paper-style robustness, use multiple independent seeds. With one log,
  mean/std across runs is not scientifically equivalent to multi-seed robustness.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHOD_ORDER = [
    "Random",
    "Round-Robin",
    "Earliest",
    "BLOR",
    "SemiGreedy",
    "DQN",
    "PPO",
    "RA-DDQN",
    "PB-SafeDQN",
    "COBRA-Oracle",
    "HCRL-Oracle",
]

PAPER_CORE_METHODS = [
    "Round-Robin",
    "BLOR",
    "SemiGreedy",
    "DQN",
    "HCRL-Oracle",
]

# Metrics in the uploaded logs are already in percentage units when they have "%"
PERCENT_METRICS = {
    "success_rate",
    "success_time_rate",
    "malicious_rate",
    "trusted_rate",
    "benign_rate",
    "audit_rate",
    "audit_fail_rate",
    "primary_success_rate",
    "backup_used_rate",
    "backup_recovery_rate",
    "conditional_backup_recovery_rate",
    "backup_skipped_rate",
    "single_mode_rate",
    "serial_mode_rate",
    "parallel_mode_rate",
    "match_rate",
}

MAIN_METRICS = [
    "reward",
    "success_rate",
    "success_time_rate",
    "avg_responseT",
    "Cost",
    "cost_per_success",
    "malicious_rate",
    "benign_rate",
    "trusted_rate",
    "audit_rate",
    "audit_fail_rate",
    "audit_truth_mean",
]

BACKUP_METRICS = [
    "primary_success_rate",
    "backup_used_rate",
    "backup_recovery_rate",
    "conditional_backup_recovery_rate",
    "backup_skipped_rate",
    "backup_score_mean",
    "single_mode_rate",
    "serial_mode_rate",
    "parallel_mode_rate",
]

LOWER_BETTER = {
    "avg_responseT",
    "Cost",
    "cost_per_success",
    "malicious_rate",
    "audit_fail_rate",
}


def ordered_methods(methods: Iterable[str]) -> List[str]:
    methods = list(dict.fromkeys(methods))
    ordered = [m for m in METHOD_ORDER if m in methods]
    rest = sorted([m for m in methods if m not in ordered])
    return ordered + rest


def safe_float(x: str) -> float:
    return float(x.strip())


def discover_logs(input_dir: Optional[str], logs: Optional[List[str]]) -> List[Path]:
    result: List[Path] = []

    if logs:
        for item in logs:
            p = Path(item)
            # Support glob-like input
            if any(ch in item for ch in ["*", "?", "["]):
                result.extend(Path().glob(item))
            elif p.exists():
                result.append(p)

    if input_dir:
        root = Path(input_dir)
        if root.exists():
            result.extend(root.rglob("*.txt"))

    # Remove duplicates and skip empty/non-log files
    unique = []
    seen = set()
    for p in result:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file() and p.stat().st_size > 0:
            unique.append(p)
    return unique


def parse_run_context(path: Path, text: str) -> Dict[str, object]:
    run_id = path.parent.name
    full = f"{path} {run_id} {text[:2000]}"

    def find_first(patterns: List[str], cast=None):
        for pat in patterns:
            m = re.search(pat, full, flags=re.IGNORECASE)
            if m:
                val = m.group(1)
                return cast(val) if cast else val
        return None

    seed = find_first([r"Seed(\d+)", r"--Seed\s+(\d+)", r"seed[_-]?(\d+)"], int)
    epoch = find_first([r"Epoch(\d+)", r"--Epoch\s+(\d+)"], int)
    req = find_first([r"Req(\d+)", r"Request[_-]?Num[_-]?(\d+)", r"--Request_Num\s+(\d+)"], int)

    scenario = find_first([
        r"(rl_harder)",
        r"(rl_hard)",
        r"(validation_stress)",
        r"(static)",
    ])

    noise = find_first([
        r"noise[_-]?(\d+(?:\.\d+)?)",
        r"noisy[_-]?(\d+(?:\.\d+)?)",
        r"Noise[_-]?(\d+(?:\.\d+)?)",
    ], float)
    if noise is not None and noise > 1.0:
        # If tag is Noise20, interpret it as 20 percent.
        noise = noise / 100.0

    malicious_num = find_first([
        r"malicious[_-]?num[_-]?(\d+)",
        r"malnum[_-]?(\d+)",
        r"mal[_-]?(\d+)",
    ], int)

    oracle_num = find_first([
        r"Oracle[_-]?Num[_-]?(\d+)",
        r"oracles[_-]?(\d+)",
        r"Oracle(\d+)",
    ], int)

    return {
        "run_id": run_id,
        "log_file": str(path),
        "seed": seed,
        "epoch_total": epoch,
        "request_num_from_name": req,
        "scenario": scenario,
        "noise": noise,
        "malicious_num": malicious_num,
        "oracle_num": oracle_num,
    }


def parse_key_values(body: str) -> Dict[str, float]:
    kv_re = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<pct>%)?")
    out: Dict[str, float] = {}
    for kv in kv_re.finditer(body):
        out[kv.group("key")] = safe_float(kv.group("value"))
    return out


def parse_log_file(path: Path, request_num_override: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    ctx = parse_run_context(path, text)

    episode_re = re.compile(r"Episode\s+(\d+)")
    req_re = re.compile(r"after\s+(\d+)\s+requests", flags=re.IGNORECASE)
    main_line_re = re.compile(r"^\[(?P<method>[^\]]+)\]\s+(?P<body>.+)$")
    diag_line_re = re.compile(r"^\[(?P<method>[^\]]+)\s+diagnostics\]\s+(?P<body>.+)$")

    episode = None
    requests_after = None
    main_rows = []
    diag_rows = []

    for line in text.splitlines():
        line = line.strip()
        m_ep = episode_re.search(line)
        if m_ep:
            episode = int(m_ep.group(1))

        m_req = req_re.search(line)
        if m_req:
            requests_after = int(m_req.group(1))

        m_diag = diag_line_re.match(line)
        if m_diag:
            row = dict(ctx)
            row.update({
                "episode": episode,
                "requests_after": request_num_override or requests_after,
                "method": m_diag.group("method").strip(),
            })
            row.update(parse_key_values(m_diag.group("body")))
            diag_rows.append(row)
            continue

        m_main = main_line_re.match(line)
        if m_main and "diagnostics" not in m_main.group("method"):
            body = m_main.group("body")
            if "reward:" not in body:
                continue
            row = dict(ctx)
            row.update({
                "episode": episode,
                "requests_after": request_num_override or requests_after,
                "method": m_main.group("method").strip(),
            })
            row.update(parse_key_values(body))
            main_rows.append(row)

    main_df = pd.DataFrame(main_rows)
    diag_df = pd.DataFrame(diag_rows)

    return main_df, diag_df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "malicious_rate" in df.columns and "trusted_rate" in df.columns:
        df["benign_rate"] = 100.0 - df["malicious_rate"] - df["trusted_rate"]
        df["benign_rate"] = df["benign_rate"].clip(lower=0.0)

    if "requests_after" in df.columns:
        for rate_col, count_col in [
            ("malicious_rate", "malicious_count"),
            ("benign_rate", "benign_count"),
            ("trusted_rate", "trusted_count"),
        ]:
            if rate_col in df.columns:
                df[count_col] = df[rate_col] / 100.0 * df["requests_after"].astype(float)

    return df


def merge_main_and_diagnostics(main_df: pd.DataFrame, diag_df: pd.DataFrame) -> pd.DataFrame:
    if main_df.empty:
        return main_df
    if diag_df.empty:
        return add_derived_metrics(main_df)

    keys = ["run_id", "log_file", "episode", "method"]
    diag_cols = [c for c in diag_df.columns if c not in main_df.columns or c in keys]
    merged = main_df.merge(diag_df[diag_cols], on=keys, how="left")
    return add_derived_metrics(merged)


def select_analysis_rows(df: pd.DataFrame, replicate: str = "final", final_window: int = 1) -> pd.DataFrame:
    if df.empty:
        return df

    if replicate == "episode":
        return df.copy()

    # One result per run/method. Use the last episode or average over last N episodes.
    grouped_rows = []
    for (run_id, method), g in df.groupby(["run_id", "method"], dropna=False):
        g = g.sort_values("episode")
        if final_window <= 1:
            selected = g.tail(1).copy()
            grouped_rows.append(selected)
        else:
            selected = g.tail(final_window).copy()
            numeric_cols = selected.select_dtypes(include=[np.number]).columns.tolist()
            base = selected.tail(1).iloc[0].to_dict()
            for c in numeric_cols:
                base[c] = selected[c].mean()
            base["episode"] = int(selected["episode"].max())
            base["final_window"] = final_window
            grouped_rows.append(pd.DataFrame([base]))

    return pd.concat(grouped_rows, ignore_index=True)


def mean_std_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    existing = [m for m in metrics if m in df.columns]
    rows = []
    for method, g in df.groupby("method", dropna=False):
        row = {"method": method, "n": len(g)}
        for m in existing:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        cats = ordered_methods(out["method"].tolist())
        out["method"] = pd.Categorical(out["method"], categories=cats, ordered=True)
        out = out.sort_values("method").reset_index(drop=True)
    return out


def formatted_mean_std_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    summary = mean_std_table(df, metrics)
    if summary.empty:
        return summary
    out = summary[["method", "n"]].copy()
    for m in [x for x in metrics if f"{x}_mean" in summary.columns]:
        vals = []
        for _, r in summary.iterrows():
            mean = r[f"{m}_mean"]
            std = r[f"{m}_std"]
            if pd.isna(std):
                vals.append(f"{mean:.4f}")
            else:
                vals.append(f"{mean:.4f} ± {std:.4f}")
        out[m] = vals
    return out


def relative_improvement_table(summary: pd.DataFrame, target_method: str) -> pd.DataFrame:
    if summary.empty or target_method not in set(summary["method"].astype(str)):
        return pd.DataFrame()

    s = summary.set_index(summary["method"].astype(str))
    target = s.loc[target_method]
    rows = []

    def get(row, col):
        name = f"{col}_mean"
        return float(row[name]) if name in row.index and pd.notna(row[name]) else np.nan

    for method, row in s.iterrows():
        if method == target_method:
            continue

        base_mal = get(row, "malicious_rate")
        tar_mal = get(target, "malicious_rate")
        base_trusted = get(row, "trusted_rate")
        tar_trusted = get(target, "trusted_rate")
        base_cost = get(row, "Cost")
        tar_cost = get(target, "Cost")
        base_cps = get(row, "cost_per_success")
        tar_cps = get(target, "cost_per_success")
        base_rt = get(row, "avg_responseT")
        tar_rt = get(target, "avg_responseT")
        base_success = get(row, "success_rate")
        tar_success = get(target, "success_rate")
        base_sit = get(row, "success_time_rate")
        tar_sit = get(target, "success_time_rate")

        def pct_reduction(base, tar):
            if np.isnan(base) or np.isnan(tar) or abs(base) < 1e-12:
                return np.nan
            return (base - tar) / base * 100.0

        def pct_increase(tar, base):
            if np.isnan(base) or np.isnan(tar) or abs(base) < 1e-12:
                return np.nan
            return (tar - base) / base * 100.0

        rows.append({
            "baseline": method,
            "target": target_method,
            "malicious_reduction_pct": pct_reduction(base_mal, tar_mal),
            "trusted_improvement_pct": pct_increase(tar_trusted, base_trusted),
            "cost_saving_pct": pct_reduction(base_cost, tar_cost),
            "cost_per_success_saving_pct": pct_reduction(base_cps, tar_cps),
            "response_time_reduction_pct": pct_reduction(base_rt, tar_rt),
            "success_rate_gain_points": tar_success - base_success if not np.isnan(tar_success + base_success) else np.nan,
            "success_time_rate_gain_points": tar_sit - base_sit if not np.isnan(tar_sit + base_sit) else np.nan,
        })

    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_bar(df: pd.DataFrame, metric: str, out_dir: Path, title: Optional[str] = None, ylabel: Optional[str] = None):
    if df.empty or metric not in df.columns:
        return

    data = df[["method", metric]].dropna().copy()
    if data.empty:
        return

    methods = ordered_methods(data["method"].astype(str).tolist())
    grouped = data.groupby("method", observed=False)[metric].agg(["mean", "std"]).reindex(methods).dropna(how="all")

    plt.figure(figsize=(10, 5))
    plt.bar(grouped.index.astype(str), grouped["mean"], yerr=grouped["std"].fillna(0.0), capsize=4)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel or metric)
    plt.title(title or metric)
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_bar_{metric}.png", dpi=300)
    plt.savefig(out_dir / f"fig_bar_{metric}.pdf")
    plt.close()


def plot_convergence(all_df: pd.DataFrame, out_dir: Path):
    if all_df.empty or "reward" not in all_df.columns or "episode" not in all_df.columns:
        return

    data = all_df.dropna(subset=["episode", "reward"]).copy()
    if data.empty:
        return

    methods = ordered_methods(data["method"].astype(str).unique())
    plt.figure(figsize=(10, 5))
    for method in methods:
        g = data[data["method"] == method]
        if g.empty:
            continue
        curve = g.groupby("episode", observed=False)["reward"].mean().sort_index()
        plt.plot(curve.index, curve.values, marker="o", linewidth=1.5, label=method)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Convergence performance")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_convergence_reward.png", dpi=300)
    plt.savefig(out_dir / "fig_convergence_reward.pdf")
    plt.close()


def plot_assignment_distribution(final_df: pd.DataFrame, out_dir: Path):
    needed = {"malicious_rate", "benign_rate", "trusted_rate"}
    if final_df.empty or not needed.issubset(final_df.columns):
        return

    data = final_df.groupby("method", observed=False)[["malicious_rate", "benign_rate", "trusted_rate"]].mean()
    methods = ordered_methods(data.index.astype(str).tolist())
    data = data.reindex(methods).dropna(how="all")
    if data.empty:
        return

    x = np.arange(len(data))
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(data))
    for col in ["malicious_rate", "benign_rate", "trusted_rate"]:
        vals = data[col].fillna(0.0).values
        plt.bar(x, vals, bottom=bottom, label=col.replace("_rate", ""))
        bottom += vals

    plt.xticks(x, data.index.astype(str), rotation=35, ha="right")
    plt.ylabel("Assignment distribution (%)")
    plt.title("Distribution of assigned requests")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_assignment_distribution_stacked.png", dpi=300)
    plt.savefig(out_dir / "fig_assignment_distribution_stacked.pdf")
    plt.close()


def plot_noise_curves(final_df: pd.DataFrame, out_dir: Path):
    if final_df.empty or "noise" not in final_df.columns:
        return
    data = final_df.dropna(subset=["noise"])
    if data["noise"].nunique() < 2:
        return

    for metric in ["avg_responseT", "Cost", "malicious_rate", "success_time_rate"]:
        if metric not in data.columns:
            continue
        plt.figure(figsize=(8, 5))
        for method in ordered_methods(data["method"].astype(str).unique()):
            g = data[data["method"] == method]
            curve = g.groupby("noise", observed=False)[metric].mean().sort_index()
            plt.plot(curve.index, curve.values, marker="o", label=method)
        plt.xlabel("Noise percentage")
        plt.ylabel(metric)
        plt.title(f"{metric} against noise observations")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_noise_{metric}.png", dpi=300)
        plt.savefig(out_dir / f"fig_noise_{metric}.pdf")
        plt.close()


def plot_malicious_num_curves(final_df: pd.DataFrame, out_dir: Path):
    if final_df.empty or "malicious_num" not in final_df.columns:
        return
    data = final_df.dropna(subset=["malicious_num"])
    if data["malicious_num"].nunique() < 2:
        return

    for metric in ["Cost", "malicious_rate", "trusted_rate", "success_time_rate", "parallel_mode_rate"]:
        if metric not in data.columns:
            continue
        plt.figure(figsize=(8, 5))
        for method in ordered_methods(data["method"].astype(str).unique()):
            g = data[data["method"] == method]
            curve = g.groupby("malicious_num", observed=False)[metric].mean().sort_index()
            if curve.empty:
                continue
            plt.plot(curve.index, curve.values, marker="o", label=method)
        plt.xlabel("Number of malicious oracles")
        plt.ylabel(metric)
        plt.title(f"{metric} with increasing malicious oracles")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"fig_malicious_num_{metric}.png", dpi=300)
        plt.savefig(out_dir / f"fig_malicious_num_{metric}.pdf")
        plt.close()


def write_report(
    out_dir: Path,
    all_df: pd.DataFrame,
    final_df: pd.DataFrame,
    summary_fmt: pd.DataFrame,
    rel: pd.DataFrame,
    target_method: str,
):
    lines = []
    lines.append("# Paper-style Metric Comparison Report\n")
    lines.append("This report is generated by `paper_metric_compare.py`.\n")
    lines.append("## Inputs\n")
    lines.append(f"- Parsed log files: {all_df['log_file'].nunique() if 'log_file' in all_df.columns else 0}\n")
    lines.append(f"- Parsed rows: {len(all_df)}\n")
    lines.append(f"- Analysis rows: {len(final_df)}\n")
    if "episode" in all_df.columns:
        lines.append(f"- Episode range: {int(all_df['episode'].min())}–{int(all_df['episode'].max())}\n")
    lines.append("\n## Original-paper-aligned metrics generated\n")
    lines.append("- Convergence reward curves\n")
    lines.append("- Assignment distribution among malicious / benign / trusted oracles\n")
    lines.append("- Average response time and cost comparison\n")
    lines.append("- Mean ± standard deviation tables\n")
    lines.append("- Relative improvement against baselines\n")
    lines.append("- Backup recovery and HCRL mode diagnostics\n")
    lines.append("- Noise and increasing-malicious-oracle curves if corresponding run tags exist\n")

    if not summary_fmt.empty:
        lines.append("\n## Main mean ± std table\n")
        lines.append(summary_fmt.to_markdown(index=False))
        lines.append("\n")

    if not rel.empty:
        lines.append(f"\n## Relative comparison versus {target_method}\n")
        lines.append(rel.to_markdown(index=False, floatfmt=".2f"))
        lines.append("\n")

    lines.append("\n## Files\n")
    for p in sorted(out_dir.glob("*")):
        if p.name != "paper_metric_report.md":
            lines.append(f"- `{p.name}`\n")

    (out_dir / "paper_metric_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output", help="Directory containing run folders and .txt logs.")
    parser.add_argument("--logs", nargs="*", default=None, help="Specific log files or glob patterns.")
    parser.add_argument("--out_dir", default="paper_metric_outputs", help="Output directory.")
    parser.add_argument("--target_method", default="HCRL-Oracle", help="Method used as the proposed method.")
    parser.add_argument("--replicate", choices=["final", "episode"], default="final",
                        help="Use one final row per run/method, or treat all episodes as replicates.")
    parser.add_argument("--final_window", type=int, default=1,
                        help="Average the last N episodes when replicate=final.")
    parser.add_argument("--request_num_override", type=int, default=None,
                        help="Override request count when computing assignment counts.")
    parser.add_argument("--paper_core_only", action="store_true",
                        help="Only keep Round-Robin, BLOR, SemiGreedy, DQN, and target method.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = discover_logs(args.input_dir, args.logs)
    if not logs:
        raise FileNotFoundError("No .txt logs found. Check --input_dir or --logs.")

    main_frames = []
    diag_frames = []
    for log in logs:
        main_df, diag_df = parse_log_file(log, request_num_override=args.request_num_override)
        if not main_df.empty:
            main_frames.append(main_df)
        if not diag_df.empty:
            diag_frames.append(diag_df)

    if not main_frames:
        raise ValueError("No metric rows were parsed from the provided logs.")

    main_df = pd.concat(main_frames, ignore_index=True)
    diag_df = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    all_df = merge_main_and_diagnostics(main_df, diag_df)

    if args.paper_core_only:
        keep = set(PAPER_CORE_METHODS + [args.target_method])
        all_df = all_df[all_df["method"].isin(keep)].copy()

    # Save raw parsed episode-level data
    save_csv(all_df, out_dir / "parsed_all_episode_metrics.csv")

    final_df = select_analysis_rows(all_df, replicate=args.replicate, final_window=args.final_window)
    save_csv(final_df, out_dir / "paper_analysis_rows.csv")

    # Main summary tables
    summary = mean_std_table(final_df, MAIN_METRICS + BACKUP_METRICS + ["match_rate"])
    save_csv(summary, out_dir / "paper_main_metrics_mean_std_numeric.csv")

    summary_fmt = formatted_mean_std_table(final_df, MAIN_METRICS + BACKUP_METRICS + ["match_rate"])
    save_csv(summary_fmt, out_dir / "paper_main_metrics_mean_std_formatted.csv")

    # Assignment distribution table
    assignment_cols = [
        "method",
        "requests_after",
        "malicious_rate",
        "benign_rate",
        "trusted_rate",
        "malicious_count",
        "benign_count",
        "trusted_count",
    ]
    assignment_cols = [c for c in assignment_cols if c in final_df.columns]
    if assignment_cols:
        assign_summary = mean_std_table(final_df, [c for c in assignment_cols if c not in ["method", "requests_after"]])
        save_csv(final_df[assignment_cols], out_dir / "paper_assignment_distribution_rows.csv")
        save_csv(assign_summary, out_dir / "paper_assignment_distribution_mean_std.csv")

    # Relative improvement table
    rel = relative_improvement_table(summary, args.target_method)
    save_csv(rel, out_dir / f"paper_relative_improvement_vs_{args.target_method.replace('-', '_')}.csv")

    # Backup / HCRL-specific rows
    existing_backup = [m for m in BACKUP_METRICS if m in final_df.columns]
    if existing_backup:
        backup_df = final_df[["method"] + existing_backup].dropna(how="all", subset=existing_backup)
        save_csv(backup_df, out_dir / "paper_backup_and_mode_rows.csv")
        backup_summary = formatted_mean_std_table(backup_df, existing_backup)
        save_csv(backup_summary, out_dir / "paper_backup_and_mode_mean_std.csv")

    # Plots
    plot_convergence(all_df, out_dir)
    plot_assignment_distribution(final_df, out_dir)

    for metric in [
        "success_rate",
        "success_time_rate",
        "avg_responseT",
        "Cost",
        "cost_per_success",
        "malicious_rate",
        "trusted_rate",
        "audit_truth_mean",
        "backup_used_rate",
        "conditional_backup_recovery_rate",
        "single_mode_rate",
        "serial_mode_rate",
        "parallel_mode_rate",
        "match_rate",
    ]:
        plot_bar(final_df, metric, out_dir)

    plot_noise_curves(final_df, out_dir)
    plot_malicious_num_curves(final_df, out_dir)

    write_report(out_dir, all_df, final_df, summary_fmt, rel, args.target_method)

    print(f"[OK] Parsed logs: {len(logs)}")
    print(f"[OK] Episode-level rows: {len(all_df)}")
    print(f"[OK] Analysis rows: {len(final_df)}")
    print(f"[OK] Output directory: {out_dir.resolve()}")
    print("\nKey outputs:")
    print(f"  - {out_dir / 'paper_metric_report.md'}")
    print(f"  - {out_dir / 'paper_main_metrics_mean_std_formatted.csv'}")
    print(f"  - {out_dir / 'paper_assignment_distribution_mean_std.csv'}")
    rel_file = "paper_relative_improvement_vs_" + args.target_method.replace("-", "_") + ".csv"
    print(f"  - {out_dir / rel_file}")
    print(f"  - {out_dir / 'fig_convergence_reward.png'}")
    print(f"  - {out_dir / 'fig_assignment_distribution_stacked.png'}")


if __name__ == "__main__":
    main()
