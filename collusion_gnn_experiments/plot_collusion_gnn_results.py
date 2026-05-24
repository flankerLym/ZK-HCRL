from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from collusion_simulator import SCENARIO_LABELS, SCENARIOS

METHOD_ORDER = ["Heuristic-Risk", "Feature-MLP", "Collusion-GNN"]


def parse_args():
    p = argparse.ArgumentParser(description="Plot collusion-aware GNN experiment results.")
    p.add_argument("--input", default="collusion_gnn_experiments/output")
    p.add_argument("--out", default="collusion_gnn_experiments/output/figures")
    return p.parse_args()


def setup():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_performance(summary: pd.DataFrame, out: Path):
    scenarios = [s for s in SCENARIOS if s in set(summary["scenario"])]
    x = np.arange(len(scenarios))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for j, method in enumerate(METHOD_ORDER):
        vals, errs = [], []
        for s in scenarios:
            row = summary[(summary["scenario"] == s) & (summary["method"] == method)]
            vals.append(float(row["auc_mean"].iloc[0]) * 100 if len(row) else np.nan)
            errs.append(float(row["auc_std"].iloc[0]) * 100 if len(row) and not pd.isna(row["auc_std"].iloc[0]) else 0.0)
        ax.bar(x + (j - 1) * width, vals, width=width, yerr=errs, capsize=3, label=method)
    ax.set_ylabel("AUROC (%)")
    ax.set_title("Collusion detection performance across dynamic attack scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], rotation=25, ha="right")
    ax.set_ylim(45, 105)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "collusion_gnn_performance.png")
    fig.savefig(out / "collusion_gnn_performance.pdf")


def plot_risk_curves(risk: pd.DataFrame, out: Path):
    # Use Collusion-GNN curves averaged over seeds.
    df = risk[risk["method"] == "Collusion-GNN"].copy()
    agg = df.groupby(["scenario", "window_id", "group"], as_index=False)["risk"].mean()
    scenarios = [s for s in SCENARIOS if s in set(agg["scenario"])]
    n = len(scenarios)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.ravel()
    for idx, s in enumerate(scenarios):
        ax = axes[idx]
        sdf = agg[agg["scenario"] == s]
        for group in ["benign", "colluder"]:
            g = sdf[sdf["group"] == group].sort_values("window_id")
            ax.plot(g["window_id"], g["risk"], linewidth=2, label=group.capitalize())
        ax.set_title(SCENARIO_LABELS.get(s, s))
        ax.set_xlabel("Window")
        ax.set_ylabel("Predicted collusion risk")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(frameon=False)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("GNN-predicted collusion risk over time", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "collusion_risk_curves_all_scenarios.png")
    fig.savefig(out / "collusion_risk_curves_all_scenarios.pdf")


def plot_graph_diagnostics(graphs: pd.DataFrame, out: Path):
    agg = graphs.groupby(["scenario", "phase"], as_index=False)[["colluder_colluder_weight", "colluder_benign_weight"]].mean()
    scenarios = [s for s in SCENARIOS if s in set(agg["scenario"])]
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    x = np.arange(len(scenarios))
    attack = agg[agg["phase"] == "attack"].set_index("scenario")
    cc = [attack.loc[s, "colluder_colluder_weight"] if s in attack.index else np.nan for s in scenarios]
    cb = [attack.loc[s, "colluder_benign_weight"] if s in attack.index else np.nan for s in scenarios]
    ax.bar(x - 0.18, cc, width=0.36, label="Colluder–colluder edge")
    ax.bar(x + 0.18, cb, width=0.36, label="Colluder–benign edge")
    ax.set_ylabel("Mean graph edge weight during attack")
    ax.set_title("Behavior graph captures group-level collusion structure")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "collusion_graph_edge_gap.png")
    fig.savefig(out / "collusion_graph_edge_gap.pdf")


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    summary = pd.read_csv(inp / "collusion_gnn_summary_mean_std.csv")
    risk = pd.read_csv(inp / "collusion_gnn_window_risk.csv")
    graphs = pd.read_csv(inp / "collusion_graph_diagnostics.csv")
    plot_performance(summary, out)
    plot_risk_curves(risk, out)
    plot_graph_diagnostics(graphs, out)
    print(f"[Done] figures written to {out}")


if __name__ == "__main__":
    main()
