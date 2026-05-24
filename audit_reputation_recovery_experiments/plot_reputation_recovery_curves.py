"""Plot paper-ready reputation degradation-and-recovery figure."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

LABELS = {
    "reputation_poisoning": "Reputation poisoning",
    "sleeper_attack": "Sleeper attack",
    "collusion_shift": "Collusion shift",
    "burst_attack": "Burst attack",
    "gradual_drift": "Gradual drift",
    "intermittent_evasion": "Intermittent evasion",
}
DEFAULT_REPRESENTATIVE = ["reputation_poisoning", "sleeper_attack", "collusion_shift"]


def parse_args():
    p = argparse.ArgumentParser(description="Plot audit reputation degradation and recovery curves.")
    p.add_argument("--curve-csv", required=True)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pdf-out", default=None)
    p.add_argument("--scenarios", nargs="*", default=DEFAULT_REPRESENTATIVE)
    return p.parse_args()


def label(s: str) -> str:
    return LABELS.get(s, s.replace("_", " ").title())


def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def aggregate_curve(curve: pd.DataFrame) -> pd.DataFrame:
    cols = ["malicious_rep_mean", "trusted_rep_mean", "normal_rep_mean", "reputation_gap", "attack_onset_step", "attack_end_step"]
    agg = curve.groupby(["scenario", "step"], as_index=False)[cols].mean()
    return agg


def add_phase_background(ax, onset: float, end: float, xmax: float):
    ax.axvspan(0, onset, alpha=0.04)
    ax.axvspan(onset, end, alpha=0.13)
    ax.axvspan(end, xmax, alpha=0.06)
    ax.axvline(onset, linestyle="--", linewidth=1.2)
    ax.axvline(end, linestyle=":", linewidth=1.2)
    y = 0.98
    ax.text(onset / 2, y, "Benign / camouflage", ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text((onset + end) / 2, y, "Attack", ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text((end + xmax) / 2, y, "Recovery", ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())


def add_timeseries(ax, curve: pd.DataFrame, summary_row: pd.Series, scenario: str, legend=False):
    sdf = curve[curve["scenario"] == scenario].sort_values("step")
    if sdf.empty:
        ax.text(0.5, 0.5, f"No data: {scenario}", ha="center", va="center", transform=ax.transAxes)
        return
    x = sdf["step"].to_numpy()
    onset = float(sdf["attack_onset_step"].iloc[0])
    end = float(sdf["attack_end_step"].iloc[0])
    xmax = float(x.max())
    add_phase_background(ax, onset, end, xmax)
    ax.plot(x, sdf["trusted_rep_mean"], linewidth=2.2, label="Trusted reputation")
    ax.plot(x, sdf["malicious_rep_mean"], linewidth=2.2, label="Malicious reputation")

    drop = float(summary_row.get("malicious_rep_drop_pct_mean", np.nan))
    recovery = float(summary_row.get("malicious_rep_recovery_ratio_mean", np.nan))
    asym = float(summary_row.get("asymmetry_score_mean", np.nan))
    pre = float(summary_row.get("pre_attack_malicious_rep_mean", np.nan))
    low = float(summary_row.get("attack_end_malicious_rep_mean", np.nan))
    endrep = float(summary_row.get("recovery_end_malicious_rep_mean", np.nan))
    txt = f"Drop: {drop:.1f}%\nRecovery ratio: {recovery:.2f}\nRep: {pre:.3f}→{low:.3f}→{endrep:.3f}\nAsymmetry: {asym:.1f}×"
    ax.text(0.98, 0.05, txt, ha="right", va="bottom", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86, linewidth=0.8))
    ax.set_title(label(scenario))
    ax.set_xlabel("Request step")
    ax.set_ylabel("Effective reputation")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(loc="upper right", frameon=False)


def add_summary_bar(ax, summary: pd.DataFrame):
    df = summary.copy()
    df["scenario_label"] = df["scenario"].map(label)
    df = df.sort_values("malicious_rep_drop_pct_mean", ascending=False)
    x = np.arange(len(df))
    width = 0.36
    drop = df["malicious_rep_drop_pct_mean"].to_numpy()
    rec = 100.0 * df["malicious_rep_recovery_ratio_mean"].to_numpy()
    ax.bar(x - width / 2, drop, width=width, label="Reputation drop")
    ax.bar(x + width / 2, rec, width=width, label="Recovered fraction")
    ax.set_title("Drop and conservative recovery across attacks")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario_label"], rotation=35, ha="right")
    ax.set_ylim(0, max(105, np.nanmax([np.nanmax(drop), np.nanmax(rec)]) + 8))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for xi, val in zip(x - width / 2, drop):
        ax.text(xi, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, val in zip(x + width / 2, rec):
        ax.text(xi, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8)


def main():
    args = parse_args()
    setup_style()
    curve = pd.read_csv(args.curve_csv)
    summary = pd.read_csv(args.summary_csv)
    curve_agg = aggregate_curve(curve)

    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    for i, (ax, scenario) in enumerate(zip(axes, args.scenarios[:3])):
        row = summary[summary["scenario"] == scenario]
        if row.empty:
            ax.text(0.5, 0.5, f"Missing summary: {scenario}", ha="center", va="center", transform=ax.transAxes)
        else:
            add_timeseries(ax, curve_agg, row.iloc[0], scenario, legend=(i == 0))
        ax.text(-0.12, 1.05, f"({chr(ord('a') + i)})", transform=ax.transAxes, fontsize=13, fontweight="bold")

    axb = fig.add_subplot(gs[1, 1])
    add_summary_bar(axb, summary)
    axb.text(-0.12, 1.05, "(d)", transform=axb.transAxes, fontsize=13, fontweight="bold")

    fig.suptitle("Asymmetric audit reputation dynamics under dynamic attacks", fontsize=15, y=1.01)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    pdf_out = Path(args.pdf_out) if args.pdf_out else out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"[Done] wrote {out}")
    print(f"[Done] wrote {pdf_out}")


if __name__ == "__main__":
    main()
