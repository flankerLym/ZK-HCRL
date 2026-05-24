"""Run audit reputation-drop validation under multiple dynamic attacks.

Example:
    python audit_reputation_drop_experiments/run_audit_reputation_drop.py \
      --trace experiments_real_trace/data/real_oracle_trace.csv \
      --out audit_reputation_drop_experiments/output \
      --seeds 3,4,5,6,7 --requests 6000 --oracles 120 --malicious-ratio 0.30
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

try:
    from .audit_core import AuditConfig
    from .attack_scenarios import scenario_by_names
    from .simulator import ReputationDropSimulator, TraceProvider
except ImportError:
    from audit_core import AuditConfig
    from attack_scenarios import scenario_by_names
    from simulator import ReputationDropSimulator, TraceProvider


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="experiments_real_trace/data/real_oracle_trace.csv")
    ap.add_argument("--out", default="audit_reputation_drop_experiments/output")
    ap.add_argument("--seeds", default="3,4,5,6,7")
    ap.add_argument("--requests", type=int, default=6000)
    ap.add_argument("--oracles", type=int, default=120)
    ap.add_argument("--services", type=int, default=3)
    ap.add_argument("--malicious-ratio", type=float, default=0.30)
    ap.add_argument("--trusted-ratio", type=float, default=0.50)
    ap.add_argument("--interval", type=int, default=100)
    ap.add_argument("--trace-split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--train-days", type=int, default=20)
    ap.add_argument("--scenarios", default="", help="Comma-separated subset. Empty means all scenarios.")
    ap.add_argument("--audit-base-rate", type=float, default=0.08)
    ap.add_argument("--audit-risk-rate", type=float, default=0.45)
    ap.add_argument("--audit-fail-penalty", type=float, default=0.18)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = TraceProvider(args.trace, split=args.trace_split, train_days=args.train_days)
    scenarios = scenario_by_names(parse_csv_strs(args.scenarios))
    seeds = parse_csv_ints(args.seeds)
    cfg = AuditConfig(base_rate=args.audit_base_rate, risk_rate=args.audit_risk_rate, fail_penalty=args.audit_fail_penalty)

    all_timeline = []
    all_rep = []
    all_summary = []
    for scenario in scenarios:
        for seed in seeds:
            print("[Run] scenario=%s seed=%s" % (scenario.name, seed))
            sim = ReputationDropSimulator(
                trace=trace,
                scenario=scenario,
                seed=seed,
                n_oracles=args.oracles,
                n_services=args.services,
                malicious_ratio=args.malicious_ratio,
                trusted_ratio=args.trusted_ratio,
                request_num=args.requests,
                interval=args.interval,
                audit_cfg=cfg,
            )
            timeline, rep, summary = sim.run()
            all_timeline.append(timeline)
            all_rep.append(rep)
            all_summary.append(summary)

    timeline_df = pd.concat(all_timeline, ignore_index=True)
    rep_df = pd.concat(all_rep, ignore_index=True)
    summary_df = pd.DataFrame(all_summary)
    timeline_df.to_csv(out / "audit_reputation_event_timeline.csv", index=False)
    rep_df.to_csv(out / "audit_reputation_curve.csv", index=False)
    summary_df.to_csv(out / "audit_reputation_summary_by_seed.csv", index=False)

    numeric_cols = [c for c in summary_df.columns if c not in {"scenario"}]
    agg = summary_df.groupby("scenario")[numeric_cols].agg(["mean", "std"])
    agg.columns = ["%s_%s" % (a, b) for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "audit_reputation_summary_mean_std.csv", index=False)

    paper_cols = [
        "scenario",
        "pre_attack_malicious_rep_mean", "post_attack_malicious_rep_mean",
        "malicious_rep_drop_abs_mean", "malicious_rep_drop_pct_mean",
        "pre_attack_malicious_truth_mean", "post_attack_malicious_truth_mean",
        "malicious_truth_drop_abs_mean", "reputation_gap_increase_mean",
        "drop_lag_intervals_mean", "attack_audit_rate_mean", "attack_audit_fail_rate_mean",
        "audit_degradation_success_mean",
    ]
    available = [c for c in paper_cols if c in agg.columns]
    paper = agg[available].copy()
    paper.to_csv(out / "paper_table_reputation_drop.csv", index=False)
    try:
        paper.to_latex(out / "paper_table_reputation_drop.tex", index=False, float_format="%.4f")
    except Exception as exc:
        print("[Warn] failed to export latex: %s" % exc)

    print("[Done] outputs written to %s" % out)
    print(paper.to_string(index=False))


if __name__ == "__main__":
    main()
