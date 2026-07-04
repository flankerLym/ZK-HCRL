#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize real_oracle_trace.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/real_oracle_trace.csv")
    ap.add_argument("--out", default="data/processed/real_oracle_trace_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    group_cols = ["source", "asset"]
    summary = (
        df.groupby(group_cols)
        .agg(
            n=("timestamp", "count"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            mean_price=("oracle_price", "mean"),
            mean_reference=("reference_price", "mean"),
            mean_deviation=("deviation", "mean"),
            p95_deviation=("deviation", lambda x: x.quantile(0.95)),
            mean_staleness=("staleness", "mean"),
            p95_staleness=("staleness", lambda x: x.quantile(0.95)),
            validation_success_rate=("validation_success", "mean"),
            mean_gas_cost=("gas_cost", "mean"),
            mean_latency=("latency", "mean"),
        )
        .reset_index()
    )
    label = df.pivot_table(index=group_cols, columns="anomaly_label", values="timestamp", aggfunc="count", fill_value=0).reset_index()
    out = summary.merge(label, on=group_cols, how="left")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"[summary] wrote {out_path}")


if __name__ == "__main__":
    main()
