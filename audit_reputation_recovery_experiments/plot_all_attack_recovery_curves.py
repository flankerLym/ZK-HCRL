"""Plot all six dynamic attack reputation degradation-and-recovery curves.

This figure is intended for paper/appendix use when every attack scenario needs
an explicit curve instead of only three representative panels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    "reputation_poisoning": "Reputation poisoning",
    "sleeper_attack": "Sleeper attack",
    "collusion_shift": "Collusion shift",
    "burst_attack": "Burst attack",
    "intermittent_evasion": "Intermittent evasion",
    "gradual_drift": "Gradual drift",
}

DEFAULT_ALL_SCENARIOS = [
    "reputation_poisoning",
    "sleeper_attack",
    "collusion_shift",
    "burst_attack",
    "intermittent_evasion",
    "gradual_drift",
]


def parse_args():
    p = argparse.ArgumentParser(description="Plot all dynamic attack reputation recovery curves.")
    p.add_argument("--curve-csv", required=True, help="Path to audit_reputation_recovery_curve.csv")
    p.add_argument("--summary-csv", required=True, help="Path to audit_reputation_recovery_summary_mean_std.csv")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--pdf-out", default=None, help="Optional output PDF path")
    p.add_argument("--scenarios", nargs="*", default=DEFAULT_ALL_SCENARIOS,
                   help="Scenario order to plot. Default: all six attack scenarios.")
    p.add_argument("--ncols", type=int, default=3, help="Number of columns. Default: 3")
    p.add_argument("--title", default="Asymmetric audit reputation dynamics across six dynamic attacks")
    return p.parse_args()


def label(s: str) -> str:
    return LABELS.get(s, s.replace("_", " ").title())


def setup_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def aggregate_curve(curve: pd.DataFrame) -> pd.DataFrame:
    required = [
        "scenario", "step", "malicious_rep_mean", "trusted_rep_mean",
        "attack_onset_step", "attack_end_step",
    ]
    require_cols(curve, required, "curve csv")
    optional = [c for c in ["normal_rep_mean", "reputation_gap"] if c in curve.columns]
    cols = ["malicious_rep_mean", "trusted_rep_mean", "attack_onset_step", "attack_end_step"] + optional
    return curve.groupby(["scenario", "step"], as_index=False)[cols].mean()


def add_phase_background(ax, onset: float, end: float, xmax: float):
    ax.axvspan(0, onset, alpha=0.04)
    ax.axvspan(onset, end, alpha=0.13)
    ax.axvspan(end, xmax, alpha=0.06)
    ax.axvline(onset, linestyle="--", linewidth=1.1)
    ax.axvline(end, linestyle=":", linewidth=1.1)
    y = 0.985
    ax.text(onset / 2, y, "Benign / camouflage", ha="center", va="top", fontsize=7.5,
            transform=ax.get_xaxis_transform())
    ax.text((onset + end) / 2, y, "Attack", ha="center", va="top", fontsize=7.5,
            transform=ax.get_xaxis_transform())
    ax.text((end + xmax) / 2, y, "Recovery", ha="center", va="top", fontsize=7.5,
            transform=ax.get_xaxis_transform())


def get_row(summary: pd.DataFrame, scenario: str) -> pd.Series | None:
    row = summary[summary["scenario"] == scenario]
    if row.empty:
        return None
    return row.iloc[0]


def add_panel(ax, curve: pd.DataFrame, summary: pd.DataFrame, scenario: str, panel_id: str, legend: bool = False):
    sdf = curve[curve["scenario"] == scenario].sort_values("step")
    if sdf.empty:
        ax.text(0.5, 0.5, f"No curve data: {scenario}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = sdf["step"].to_numpy()
    onset = float(sdf["attack_onset_step"].iloc[0])
    end = float(sdf["attack_end_step"].iloc[0])
    xmax = float(x.max())
    add_phase_background(ax, onset, end, xmax)

    ax.plot(x, sdf["trusted_rep_mean"], linewidth=2.1, label="Trusted reputation")
    ax.plot(x, sdf["malicious_rep_mean"], linewidth=2.1, label="Malicious reputation")

    row = get_row(summary, scenario)
    if row is not None:
        drop = float(row.get("malicious_rep_drop_pct_mean", np.nan))
        asym = float(row.get("asymmetry_score_mean", np.nan))
        pre = float(row.get("pre_attack_malicious_rep_mean", np.nan))
        low = float(row.get("attack_end_malicious_rep_mean", np.nan))
        rec_end = float(row.get("recovery_end_malicious_rep_mean", np.nan))
        rec_ratio = float(row.get("malicious_rep_recovery_ratio_mean", np.nan))
        rec_text = "full" if np.isfinite(rec_ratio) and rec_ratio >= 1.0 else (f"{rec_ratio:.2f}" if np.isfinite(rec_ratio) else "NA")
        txt = (
            f"Drop: {drop:.1f}%\n"
            f"Rep: {pre:.3f}→{low:.3f}→{rec_end:.3f}\n"
            f"Recovery: {rec_text}\n"
            f"Asym.: {asym:.1f}×"
        )
        ax.text(0.98, 0.05, txt, ha="right", va="bottom", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86, linewidth=0.8),
                fontsize=9.2)

    ax.set_title(label(scenario))
    ax.set_xlabel("Request step")
    ax.set_ylabel("Effective reputation")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.text(-0.12, 1.05, f"({panel_id})", transform=ax.transAxes, fontsize=13, fontweight="bold")
    if legend:
        ax.legend(loc="upper right", frameon=False)


def main():
    args = parse_args()
    setup_style()
    curve_raw = pd.read_csv(args.curve_csv)
    summary = pd.read_csv(args.summary_csv)
    require_cols(summary, ["scenario"], "summary csv")
    curve = aggregate_curve(curve_raw)

    scenarios = [s for s in args.scenarios if s in set(curve["scenario"])]
    if not scenarios:
        raise ValueError("No requested scenarios were found in curve csv.")

    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig_w = 5.2 * ncols
    fig_h = 4.25 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for i, scenario in enumerate(scenarios):
        r, c = divmod(i, ncols)
        add_panel(axes_arr[r, c], curve, summary, scenario, chr(ord("a") + i), legend=(i == 0))

    for j in range(len(scenarios), nrows * ncols):
        r, c = divmod(j, ncols)
        axes_arr[r, c].set_axis_off()

    fig.suptitle(args.title, fontsize=15, y=1.02)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    pdf_out = Path(args.pdf_out) if args.pdf_out else out.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"[Done] wrote {out}")
    print(f"[Done] wrote {pdf_out}")


if __name__ == "__main__":
    main()
