#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create simple numpy arrays from real_oracle_trace.csv for quick HCRL integration.

This does not modify your environment. It gives you a compact NPZ file that can
be loaded by a custom real-trace environment wrapper.

Output keys:
  timestamps, service_types, oracle_ids, prices, reference_prices, deviations,
  staleness, gas_cost, latency, validation_success, anomaly_is_risky
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/real_oracle_trace.csv")
    ap.add_argument("--out", default="data/processed/real_oracle_trace_hcrl.npz")
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    oracle_codes, oracle_uniques = pd.factorize(df["oracle_id"])
    np.savez_compressed(
        args.out,
        timestamps=df["timestamp"].astype("int64").to_numpy() // 10**9,
        service_types=df["service_type"].astype(int).to_numpy(),
        oracle_codes=oracle_codes.astype(np.int64),
        oracle_ids=oracle_uniques.astype(str).to_numpy(),
        prices=df["oracle_price"].astype(float).to_numpy(),
        reference_prices=df["reference_price"].astype(float).to_numpy(),
        deviations=df["deviation"].astype(float).to_numpy(),
        staleness=df["staleness"].astype(float).to_numpy(),
        gas_cost=df["gas_cost"].astype(float).to_numpy(),
        latency=df["latency"].astype(float).to_numpy(),
        validation_success=df["validation_success"].astype(int).to_numpy(),
        anomaly_is_risky=(df["anomaly_label"].isin(["suspicious", "anomalous"]).astype(int).to_numpy()),
    )
    print(f"[adapter] wrote {args.out} rows={len(df)} oracles={len(oracle_uniques)}")


if __name__ == "__main__":
    main()
