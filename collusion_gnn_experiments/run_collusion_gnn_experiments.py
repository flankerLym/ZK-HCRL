from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils import set_seed, ensure_dir
from data_loader import load_real_trace
from collusion_simulator import SCENARIOS, SCENARIO_LABELS, simulate_oracle_panel
from graph_builder import build_graph_dataset, graph_diagnostics
from models import FeatureMLP, CollusionGCN, train_model, heuristic_risk_scores
from metrics import roc_auc_score_np, average_precision_np, binary_metrics, precision_recall_at_k


def parse_args():
    p = argparse.ArgumentParser(description="Standalone collusion-aware GNN experiment for HCRL oracle risk modeling.")
    p.add_argument("--trace", default="experiments_real_trace/data/real_oracle_trace.csv")
    p.add_argument("--out", default="collusion_gnn_experiments/output")
    p.add_argument("--seeds", default="3,4,5,6,7")
    p.add_argument("--requests", type=int, default=12000)
    p.add_argument("--oracles", type=int, default=120)
    p.add_argument("--malicious-ratio", type=float, default=0.30)
    p.add_argument("--window-size", type=int, default=120)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--attack-onset-ratio", type=float, default=0.25)
    p.add_argument("--attack-end-ratio", type=float, default=0.70)
    return p.parse_args()


def split_windows(windows):
    # Train on benign + early attack windows, test on later attack + recovery.
    n = len(windows)
    split = max(2, int(n * 0.55))
    train = windows[:split]
    test = windows[split:]
    if len(test) == 0:
        test = windows[-1:]
        train = windows[:-1]
    return train, test


def eval_scores(scenario, seed, method, y, scores, wids, n_colluders):
    out = {"scenario": scenario, "seed": seed, "method": method}
    out["auc"] = roc_auc_score_np(y, scores)
    out["auprc"] = average_precision_np(y, scores)
    out.update(binary_metrics(y, scores, threshold=0.5))
    p_at_k, r_at_k = precision_recall_at_k(y, scores, k=n_colluders)
    out["precision_at_num_colluders"] = p_at_k
    out["recall_at_num_colluders"] = r_at_k
    out["risk_colluder_mean"] = float(np.mean(scores[y == 1])) if np.any(y == 1) else np.nan
    out["risk_benign_mean"] = float(np.mean(scores[y == 0])) if np.any(y == 0) else np.nan
    out["risk_gap"] = out["risk_colluder_mean"] - out["risk_benign_mean"]
    return out


def window_risk_df(scenario, seed, method, y, scores, wids):
    df = pd.DataFrame({"scenario": scenario, "seed": seed, "method": method, "window_id": wids, "label": y, "risk": scores})
    agg = df.groupby(["scenario", "seed", "method", "window_id", "label"], as_index=False)["risk"].mean()
    agg["group"] = np.where(agg["label"] == 1, "colluder", "benign")
    return agg


def main():
    args = parse_args()
    out_dir = Path(args.out)
    ensure_dir(str(out_dir))
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    trace = load_real_trace(args.trace, args.requests, seeds[0] if seeds else 0)

    metric_rows = []
    risk_rows = []
    graph_rows = []
    node_diag_frames = []

    for seed in seeds:
        set_seed(seed)
        for scenario in SCENARIOS:
            print(f"[run] scenario={scenario} seed={seed}")
            panel = simulate_oracle_panel(
                trace=trace,
                scenario=scenario,
                seed=seed,
                requests=args.requests,
                n_oracles=args.oracles,
                malicious_ratio=args.malicious_ratio,
                attack_onset_ratio=args.attack_onset_ratio,
                attack_end_ratio=args.attack_end_ratio,
            )
            panel_out = out_dir / "panels"
            ensure_dir(str(panel_out))
            if seed == seeds[0]:
                panel.head(2000).to_csv(panel_out / f"{scenario}_panel_head.csv", index=False)

            windows, node_df = build_graph_dataset(panel, args.oracles, args.window_size, top_k=args.top_k)
            train_windows, test_windows = split_windows(windows)
            n_colluders = int(round(args.oracles * args.malicious_ratio))

            # Diagnostics averaged over attack windows.
            for w in windows:
                d = graph_diagnostics(w["A"], w["y"])
                d.update({"scenario": scenario, "seed": seed, "window_id": w["window_id"], "phase": w["phase"]})
                graph_rows.append(d)
            node_diag_frames.append(node_df.assign(scenario=scenario, seed=seed))

            # Heuristic risk.
            y_h, s_h, wid_h = heuristic_risk_scores(test_windows)
            metric_rows.append(eval_scores(scenario, seed, "Heuristic-Risk", y_h, s_h, wid_h, n_colluders))
            risk_rows.append(window_risk_df(scenario, seed, "Heuristic-Risk", y_h, s_h, wid_h))

            in_dim = train_windows[0]["X"].shape[1]
            # Feature-only MLP.
            y_m, s_m, wid_m = train_model(FeatureMLP(in_dim), train_windows, test_windows, epochs=args.epochs, seed=seed)
            metric_rows.append(eval_scores(scenario, seed, "Feature-MLP", y_m, s_m, wid_m, n_colluders))
            risk_rows.append(window_risk_df(scenario, seed, "Feature-MLP", y_m, s_m, wid_m))

            # Graph-aware GNN.
            y_g, s_g, wid_g = train_model(CollusionGCN(in_dim), train_windows, test_windows, epochs=args.epochs, seed=seed)
            metric_rows.append(eval_scores(scenario, seed, "Collusion-GNN", y_g, s_g, wid_g, n_colluders))
            risk_rows.append(window_risk_df(scenario, seed, "Collusion-GNN", y_g, s_g, wid_g))

    metrics = pd.DataFrame(metric_rows)
    risks = pd.concat(risk_rows, ignore_index=True)
    graphs = pd.DataFrame(graph_rows)
    nodes = pd.concat(node_diag_frames, ignore_index=True)

    metrics.to_csv(out_dir / "collusion_gnn_metrics_by_seed.csv", index=False)
    risks.to_csv(out_dir / "collusion_gnn_window_risk.csv", index=False)
    graphs.to_csv(out_dir / "collusion_graph_diagnostics.csv", index=False)
    nodes.to_csv(out_dir / "collusion_node_window_features.csv", index=False)

    metric_cols = ["auc", "auprc", "f1", "precision", "recall", "precision_at_num_colluders", "recall_at_num_colluders", "risk_gap"]
    summary_parts = []
    for col in metric_cols:
        tmp = metrics.groupby(["scenario", "method"])[col].agg(["mean", "std"]).reset_index()
        tmp = tmp.rename(columns={"mean": f"{col}_mean", "std": f"{col}_std"})
        summary_parts.append(tmp)
    summary = summary_parts[0]
    for tmp in summary_parts[1:]:
        summary = summary.merge(tmp, on=["scenario", "method"], how="outer")
    summary["scenario_label"] = summary["scenario"].map(SCENARIO_LABELS).fillna(summary["scenario"])
    summary = summary.sort_values(["scenario", "method"])
    summary.to_csv(out_dir / "collusion_gnn_summary_mean_std.csv", index=False)

    # Compact paper table.
    paper = summary.copy()
    def fmt_pct(m, s):
        return f"{100*m:.2f} ± {100*(0 if pd.isna(s) else s):.2f}%"
    paper_table = pd.DataFrame({
        "Scenario": paper["scenario_label"],
        "Method": paper["method"],
        "AUROC": [fmt_pct(m, s) for m, s in zip(paper["auc_mean"], paper["auc_std"])],
        "AUPRC": [fmt_pct(m, s) for m, s in zip(paper["auprc_mean"], paper["auprc_std"])],
        "F1": [fmt_pct(m, s) for m, s in zip(paper["f1_mean"], paper["f1_std"])],
        "Recall@K": [fmt_pct(m, s) for m, s in zip(paper["recall_at_num_colluders_mean"], paper["recall_at_num_colluders_std"])],
        "Risk Gap": [f"{m:.3f} ± {(0 if pd.isna(s) else s):.3f}" for m, s in zip(paper["risk_gap_mean"], paper["risk_gap_std"])],
    })
    paper_table.to_csv(out_dir / "paper_table_collusion_gnn.csv", index=False)
    try:
        paper_table.to_latex(out_dir / "paper_table_collusion_gnn.tex", index=False)
    except Exception:
        pass

    print("\n[Done] outputs written to", out_dir)
    print(paper_table.head(18).to_string(index=False))


if __name__ == "__main__":
    main()
