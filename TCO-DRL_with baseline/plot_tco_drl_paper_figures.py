#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper-quality plotting script for TCO-DRL / Audit-aware HCRL-Oracle logs.

It parses training logs like:
[HCRL-Oracle] reward: ... success_rate: 79.77% ...
[HCRL-Oracle diagnostics] primary_success_rate: ... parallel_mode_rate: ...

Outputs:
  - parsed_episode_metrics.csv
  - parsed_diagnostics.csv
  - summary_lastN.csv
  - Fig1_main_performance.(png/pdf)
  - Fig2_learning_curves.(png/pdf)
  - Fig3_security_cost_tradeoff.(png/pdf)
  - Fig4_recovery_and_audit.(png/pdf)
  - Fig5_metric_heatmap.(png/pdf)

Example:
  python plot_tco_drl_paper_figures.py `
    --log "E:\keyan\code\TCO\useful\ablation\26_5_15_18_40_Epoch30_Req6000_rl_harder_Seed3_ablation_hcrl_full\26_5_15_18_40_Epoch30_Req6000_rl_harder_Seed3_ablation_hcrl_full_final_results.csv" `
    --out_dir "paper_figures" `
    --last_n 5
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RATE_KEYS = {
    "success_rate", "success_time_rate", "malicious_rate", "trusted_rate",
    "audit_rate", "audit_fail_rate", "audit_pass_rate", "audit_detect",
    "primary_success_rate", "backup_used_rate", "backup_recovery_rate",
    "conditional_backup_recovery_rate", "backup_skipped_rate",
    "single_mode_rate", "serial_mode_rate", "parallel_mode_rate",
}

METHOD_ORDER = [
    "Random", "Round-Robin", "Earliest", "BLOR", "SemiGreedy",
    "DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle",
]

DISPLAY_NAMES = {
    "Round-Robin": "Round\nRobin",
    "SemiGreedy": "Semi\nGreedy",
    "PB-SafeDQN": "PB-\nSafeDQN",
    "COBRA-Oracle": "COBRA",
    "HCRL-Oracle": "HCRL",
}

# User explicitly requested beautiful paper plots, so a fixed color palette is used.
PALETTE = {
    "Random": "#9aa0a6",
    "Round-Robin": "#7f8c8d",
    "Earliest": "#95a5a6",
    "BLOR": "#8e6bbd",
    "SemiGreedy": "#b0a160",
    "DQN": "#4c78a8",
    "PPO": "#72b7b2",
    "RA-DDQN": "#54a24b",
    "PB-SafeDQN": "#f58518",
    "COBRA-Oracle": "#e45756",
    "HCRL-Oracle": "#6f4eeb",
}

LOWER_BETTER = {"avg_responseT", "Cost", "cost_per_success", "malicious_rate", "audit_fail_rate"}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "font.family": "DejaVu Sans",
    })


def parse_value(raw: str, key: str) -> float:
    raw = raw.strip()
    is_percent = raw.endswith("%")
    raw = raw.rstrip("%")
    try:
        val = float(raw)
    except ValueError:
        return np.nan
    if key in RATE_KEYS:
        if is_percent:
            return val
        # Logs without % often print rates as fractions.
        if 0.0 <= val <= 1.0:
            return val * 100.0
    return val


def kv_get(line: str, key: str, rate: bool = False) -> float:
    # Accept either "key: 12.3" or "key: 12.3%" with arbitrary spaces.
    m = re.search(rf"\b{re.escape(key)}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?)", line)
    if not m:
        return np.nan
    return parse_value(m.group(1), key)


def parse_log(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    diag_rows: List[Dict[str, float]] = []
    current_ep: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            m_ep = re.match(r"-+\s*Episode\s+(\d+)\s*-+", line)
            if m_ep:
                current_ep = int(m_ep.group(1))
                continue

            if current_ep is None or not line.startswith("["):
                continue

            m_name = re.match(r"\[([^\]]+)\]", line)
            if not m_name:
                continue
            name = m_name.group(1)

            if name.endswith(" diagnostics"):
                method = name.replace(" diagnostics", "")
                row = {"episode": current_ep, "method": method}
                for k in [
                    "primary_success_rate", "backup_used_rate", "backup_recovery_rate",
                    "conditional_backup_recovery_rate", "backup_skipped_rate",
                    "backup_score_mean", "single_mode_rate", "serial_mode_rate", "parallel_mode_rate",
                ]:
                    row[k] = kv_get(line, k)
                diag_rows.append(row)
                continue

            if "reward:" not in line:
                continue

            row = {"episode": current_ep, "method": name}
            for k in [
                "reward", "avg_responseT", "success_rate", "success_time_rate",
                "finishT", "Cost", "cost_per_success", "malicious_rate", "trusted_rate",
                "audit_rate", "audit_fail_rate", "audit_truth_mean", "audit_detect",
            ]:
                row[k] = kv_get(line, k)
            rows.append(row)

    metrics = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diag_rows)
    if metrics.empty:
        raise ValueError(f"No method metric lines were parsed from {path}. Please check log format.")
    return metrics, diagnostics


def ordered_methods(methods: Iterable[str]) -> List[str]:
    methods = list(dict.fromkeys(methods))
    order = [m for m in METHOD_ORDER if m in methods]
    order += [m for m in methods if m not in order]
    return order


def summarize(metrics: pd.DataFrame, diagnostics: pd.DataFrame, last_n: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    episodes = sorted(metrics["episode"].dropna().unique().tolist())
    selected_eps = episodes[-last_n:] if last_n > 0 and len(episodes) > last_n else episodes
    metric_part = metrics[metrics["episode"].isin(selected_eps)].copy()
    diag_part = diagnostics[diagnostics["episode"].isin(selected_eps)].copy() if not diagnostics.empty else diagnostics

    metric_cols = [c for c in metric_part.columns if c not in ["episode", "method"]]
    metric_summary = metric_part.groupby("method")[metric_cols].agg(["mean", "std"]).reset_index()
    metric_summary.columns = ["_".join([x for x in col if x]) for col in metric_summary.columns.to_flat_index()]

    if diag_part.empty:
        diag_summary = pd.DataFrame(columns=["method"])
    else:
        diag_cols = [c for c in diag_part.columns if c not in ["episode", "method"]]
        diag_summary = diag_part.groupby("method")[diag_cols].agg(["mean", "std"]).reset_index()
        diag_summary.columns = ["_".join([x for x in col if x]) for col in diag_summary.columns.to_flat_index()]

    summary = metric_summary.merge(diag_summary, on="method", how="left")
    summary["method"] = pd.Categorical(summary["method"], categories=ordered_methods(summary["method"]), ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)
    return summary, metric_part, selected_eps


def label_for(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def colors_for(methods: List[str]) -> List[str]:
    return [PALETTE.get(m, "#444444") for m in methods]


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def bar_with_error(ax, summary: pd.DataFrame, metric: str, title: str, ylabel: str, methods: List[str], highlight: str = "HCRL-Oracle"):
    sub = summary.set_index("method").loc[methods]
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    values = sub[mean_col].astype(float).values
    errors = sub[std_col].fillna(0).astype(float).values if std_col in sub.columns else np.zeros_like(values)
    x = np.arange(len(methods))
    bars = ax.bar(x, values, yerr=errors, capsize=2.5, width=0.72,
                  color=colors_for(methods), edgecolor="#222222", linewidth=0.55)
    for bar, method in zip(bars, methods):
        if method == highlight:
            bar.set_edgecolor("#000000")
            bar.set_linewidth(1.5)
            bar.set_hatch("///")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(m) for m in methods], rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if metric in RATE_KEYS:
        ax.set_ylim(bottom=0)
    if metric in LOWER_BETTER:
        best_idx = np.nanargmin(values)
    else:
        best_idx = np.nanargmax(values)
    ax.scatter([best_idx], [values[best_idx]], s=28, marker="*", color="black", zorder=5)


def plot_main_performance(summary: pd.DataFrame, out_dir: Path, methods: List[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    specs = [
        ("success_rate", "Task success rate", "Success rate (%)"),
        ("success_time_rate", "On-time completion rate", "On-time rate (%)"),
        ("cost_per_success", "Cost per successful request", "Cost / success"),
        ("malicious_rate", "Malicious oracle assignment", "Malicious rate (%)"),
    ]
    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        if f"{metric}_mean" not in summary.columns:
            ax.axis("off")
            continue
        bar_with_error(ax, summary, metric, title, ylabel, methods)
    savefig(fig, out_dir, "Fig1_main_performance")


def plot_learning_curves(metrics: pd.DataFrame, out_dir: Path, methods: List[str]) -> None:
    curve_metrics = [
        ("reward", "Reward"),
        ("success_rate", "Success rate (%)"),
        ("success_time_rate", "On-time rate (%)"),
        ("malicious_rate", "Malicious rate (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), curve_metrics):
        for method in methods:
            sub = metrics[metrics["method"] == method].sort_values("episode")
            if sub.empty or metric not in sub.columns:
                continue
            ax.plot(sub["episode"], sub[metric], label=label_for(method), color=PALETTE.get(method, None), linewidth=2.0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode")
        ax.set_title(ylabel.replace(" (%)", ""))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 1.03))
    savefig(fig, out_dir, "Fig2_learning_curves")


def plot_tradeoff(summary: pd.DataFrame, out_dir: Path, methods: List[str]) -> None:
    sub = summary.set_index("method").loc[methods].reset_index()
    x = sub["cost_per_success_mean"].astype(float).values
    y = sub["success_rate_mean"].astype(float).values
    size_metric = sub["malicious_rate_mean"].astype(float).fillna(0).values
    sizes = 80 + 15 * size_metric
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i, method in enumerate(sub["method"]):
        ax.scatter(x[i], y[i], s=sizes[i], color=PALETTE.get(method, "#333333"),
                   edgecolor="#111111", linewidth=0.8, alpha=0.92)
        ax.text(x[i] + 0.012, y[i] + 0.15, label_for(method).replace("\n", " "), fontsize=8.5)
    ax.set_xlabel("Cost per successful request ↓")
    ax.set_ylabel("Task success rate (%) ↑")
    ax.set_title("Reliability-cost-security trade-off")
    ax.grid(True, alpha=0.22)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Bubble size = malicious assignment rate',
                   markerfacecolor='#bbbbbb', markeredgecolor='#111111', markersize=9)
    ]
    ax.legend(handles=legend_elements, frameon=False, loc="lower right")
    savefig(fig, out_dir, "Fig3_security_cost_tradeoff")


def plot_recovery_audit(summary: pd.DataFrame, out_dir: Path) -> None:
    methods = [m for m in ["PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"] if m in summary["method"].astype(str).tolist()]
    if not methods:
        return
    sub = summary.set_index("method").loc[methods]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # Recovery diagnostics
    x = np.arange(len(methods))
    width = 0.26
    rec_metrics = ["primary_success_rate", "backup_recovery_rate", "conditional_backup_recovery_rate"]
    rec_labels = ["Primary success", "Backup recovery", "Conditional recovery"]
    rec_colors = ["#4c78a8", "#f58518", "#54a24b"]
    for j, (metric, label, color) in enumerate(zip(rec_metrics, rec_labels, rec_colors)):
        col = f"{metric}_mean"
        if col in sub.columns:
            axes[0].bar(x + (j - 1) * width, sub[col].astype(float).values, width=width,
                        label=label, color=color, edgecolor="#222222", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label_for(m) for m in methods])
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Recovery diagnostics")
    axes[0].legend(frameon=False)

    # HCRL mode stacked bar or audit rates if diagnostics absent.
    mode_cols = ["single_mode_rate_mean", "serial_mode_rate_mean", "parallel_mode_rate_mean"]
    if all(c in sub.columns for c in mode_cols):
        bottom = np.zeros(len(methods))
        for col, label, color in zip(mode_cols, ["Single", "Serial", "Parallel"], ["#9aa0a6", "#72b7b2", "#6f4eeb"]):
            vals = sub[col].fillna(0).astype(float).values
            axes[1].bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="#222222", linewidth=0.5)
            bottom += vals
        axes[1].set_ylabel("Mode usage (%)")
        axes[1].set_title("Execution-mode distribution")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([label_for(m) for m in methods])
        axes[1].legend(frameon=False)
    savefig(fig, out_dir, "Fig4_recovery_and_audit")


def plot_heatmap(summary: pd.DataFrame, out_dir: Path, methods: List[str]) -> None:
    metrics = [
        "reward", "success_rate", "success_time_rate", "trusted_rate",
        "malicious_rate", "avg_responseT", "Cost", "cost_per_success",
    ]
    available = [m for m in metrics if f"{m}_mean" in summary.columns]
    sub = summary.set_index("method").loc[methods]
    mat = []
    labels = []
    for metric in available:
        vals = sub[f"{metric}_mean"].astype(float).values
        # normalize to [0,1], reversing lower-better metrics
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if math.isclose(vmax, vmin):
            norm = np.ones_like(vals) * 0.5
        else:
            norm = (vals - vmin) / (vmax - vmin)
        if metric in LOWER_BETTER:
            norm = 1.0 - norm
        mat.append(norm)
        labels.append(metric.replace("avg_responseT", "responseT").replace("success_time_rate", "on_time"))
    data = np.array(mat).T
    fig, ax = plt.subplots(figsize=(8.8, max(4.2, 0.42 * len(methods))))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([label_for(m).replace("\n", " ") for m in methods])
    ax.set_title("Normalized overall comparison (higher is better)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized score")
    savefig(fig, out_dir, "Fig5_metric_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to run log .txt")
    parser.add_argument("--out_dir", type=str, default="paper_figures", help="Output directory")
    parser.add_argument("--last_n", type=int, default=10, help="Use last N episodes for bar summaries")
    parser.add_argument("--curve_methods", nargs="*", default=["DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"],
                        help="Methods shown in learning curves")
    parser.add_argument("--bar_methods", nargs="*", default=["DQN", "PPO", "RA-DDQN", "PB-SafeDQN", "COBRA-Oracle", "HCRL-Oracle"],
                        help="Methods shown in main bar figures")
    args = parser.parse_args()

    setup_style()
    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, diagnostics = parse_log(log_path)
    metrics.to_csv(out_dir / "parsed_episode_metrics.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(out_dir / "parsed_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary, metric_last, eps = summarize(metrics, diagnostics, args.last_n)
    summary.to_csv(out_dir / "summary_lastN.csv", index=False, encoding="utf-8-sig")

    available_methods = set(summary["method"].astype(str))
    bar_methods = [m for m in args.bar_methods if m in available_methods]
    if not bar_methods:
        bar_methods = ordered_methods(available_methods)
    curve_methods = [m for m in args.curve_methods if m in set(metrics["method"].astype(str))]
    if not curve_methods:
        curve_methods = bar_methods

    plot_main_performance(summary, out_dir, bar_methods)
    plot_learning_curves(metrics, out_dir, curve_methods)
    plot_tradeoff(summary, out_dir, bar_methods)
    plot_recovery_audit(summary, out_dir)
    plot_heatmap(summary, out_dir, bar_methods)

    print("Parsed episodes:", min(eps), "to", max(eps), f"(summary uses {len(eps)} episodes)")
    print("Saved figures and CSV files to:", out_dir.resolve())
    print("Summary table:")
    keep = ["method", "reward_mean", "success_rate_mean", "success_time_rate_mean", "cost_per_success_mean", "malicious_rate_mean"]
    print(summary[[c for c in keep if c in summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
