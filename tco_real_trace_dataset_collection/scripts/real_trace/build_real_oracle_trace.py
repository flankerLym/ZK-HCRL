#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build HCRL-compatible real_oracle_trace.csv from collected data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_binance(binance_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(binance_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if "timestamp" not in df.columns or "close" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["reference_price"] = pd.to_numeric(df["close"], errors="coerce")
        frames.append(df[["timestamp", "symbol", "reference_price"]])
    if not frames:
        raise RuntimeError(f"No Binance csv files found in {binance_dir}")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])


def infer_binance_symbol(asset: str) -> str:
    base = asset.split("/")[0].upper()
    return f"{base}USDT"


def load_source_dir(source_dir: Path, source_name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(source_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "timestamp" not in df.columns or "oracle_price" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["source"] = df.get("source", source_name)
        if "asset" not in df.columns:
            asset = path.stem.replace("_", "/")
            df["asset"] = asset
        if "oracle_id" not in df.columns:
            df["oracle_id"] = source_name + "_" + df["asset"].astype(str).str.replace("/", "_", regex=False)
        if "gas_cost" not in df.columns:
            df["gas_cost"] = np.nan
        if "confidence_ratio" not in df.columns:
            df["confidence_ratio"] = np.nan
        if "service_type" not in df.columns:
            df["service_type"] = np.nan
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["asset", "timestamp"])


def merge_reference(oracle_df: pd.DataFrame, binance_df: pd.DataFrame, tolerance: str = "10min") -> pd.DataFrame:
    oracle_df = oracle_df.copy()
    oracle_df["symbol"] = oracle_df["asset"].map(infer_binance_symbol)
    out_frames = []

    for symbol, odf in oracle_df.groupby("symbol"):
        bdf = binance_df[binance_df["symbol"] == symbol].copy()
        if bdf.empty:
            print(f"[warn] missing Binance reference for {symbol}; rows will be dropped")
            continue
        odf = odf.sort_values("timestamp")
        bdf = bdf.sort_values("timestamp")
        merged = pd.merge_asof(
            odf,
            bdf[["timestamp", "reference_price"]],
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )
        out_frames.append(merged)

    if not out_frames:
        raise RuntimeError("No oracle rows could be aligned with Binance references.")
    return pd.concat(out_frames, ignore_index=True)


def add_features(
    df: pd.DataFrame,
    deviation_threshold: float,
    staleness_threshold_sec: int,
    suspicious_deviation: float,
    anomalous_deviation: float,
    suspicious_staleness: int,
    anomalous_staleness: int,
) -> pd.DataFrame:
    df = df.copy()
    df["oracle_price"] = pd.to_numeric(df["oracle_price"], errors="coerce")
    df["reference_price"] = pd.to_numeric(df["reference_price"], errors="coerce")
    df = df.dropna(subset=["oracle_price", "reference_price", "timestamp"])

    df["deviation"] = (df["oracle_price"] - df["reference_price"]).abs() / df["reference_price"].replace(0, np.nan)

    df = df.sort_values(["oracle_id", "timestamp"])
    # Event-log rows usually have timestamp == updated_at. Historical-call rows use
    # timestamp as the sampled request/block time and feed_updated_at/updated_at as
    # the true Chainlink feed update time. Prefer this true staleness when present.
    if "feed_updated_at" in df.columns or "updated_at" in df.columns:
        src_col = "feed_updated_at" if "feed_updated_at" in df.columns else "updated_at"
        upd = pd.to_numeric(df[src_col], errors="coerce")
        ts_epoch = df["timestamp"].astype("int64") / 1e9
        true_staleness = ts_epoch - upd
        df["staleness"] = true_staleness.where(upd.notna(), np.nan)
        # Fallback for rows without update timestamps.
        fallback = df.groupby("oracle_id")["timestamp"].diff().dt.total_seconds()
        df["staleness"] = df["staleness"].fillna(fallback)
    else:
        df["staleness"] = df.groupby("oracle_id")["timestamp"].diff().dt.total_seconds()
    # For the first update per oracle, set staleness to zero. This avoids falsely marking all first records as stale.
    df["staleness"] = df["staleness"].fillna(0).clip(lower=0)
    df["latency"] = df["staleness"]

    if "gas_cost" not in df.columns:
        df["gas_cost"] = np.nan
    df["gas_cost"] = pd.to_numeric(df["gas_cost"], errors="coerce")
    # Fill missing gas cost with per-source median, then global median, then zero.
    df["gas_cost"] = df.groupby("source")["gas_cost"].transform(lambda s: s.fillna(s.median()))
    df["gas_cost"] = df["gas_cost"].fillna(df["gas_cost"].median()).fillna(0.0)

    df["validation_success"] = (
        (df["deviation"] <= deviation_threshold) &
        (df["staleness"] <= staleness_threshold_sec)
    ).astype(int)

    conditions_anom = (df["deviation"] > anomalous_deviation) | (df["staleness"] > anomalous_staleness)
    conditions_susp = (df["deviation"] > suspicious_deviation) | (df["staleness"] > suspicious_staleness)
    df["anomaly_label"] = np.where(conditions_anom, "anomalous", np.where(conditions_susp, "suspicious", "normal"))

    if df["service_type"].isna().all():
        assets = {a: i for i, a in enumerate(sorted(df["asset"].dropna().unique()))}
        df["service_type"] = df["asset"].map(assets)
    df["service_type"] = pd.to_numeric(df["service_type"], errors="coerce").fillna(0).astype(int)

    keep = [
        "timestamp", "asset", "source", "oracle_id", "oracle_price", "reference_price",
        "deviation", "staleness", "confidence_ratio", "gas_cost", "latency",
        "validation_success", "anomaly_label", "service_type",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep].sort_values("timestamp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chainlink-dir", default="data/raw/chainlink")
    ap.add_argument("--binance-dir", default="data/raw/binance")
    ap.add_argument("--pyth-dir", default="data/raw/pyth")
    ap.add_argument("--out", default="data/processed/real_oracle_trace.csv")
    ap.add_argument("--include-pyth", action="store_true")
    ap.add_argument("--reference-tolerance", default="10min")
    ap.add_argument("--deviation-threshold", type=float, default=0.01)
    ap.add_argument("--staleness-threshold-sec", type=int, default=600)
    ap.add_argument("--suspicious-deviation", type=float, default=0.005)
    ap.add_argument("--anomalous-deviation", type=float, default=0.015)
    ap.add_argument("--suspicious-staleness-sec", type=int, default=300)
    ap.add_argument("--anomalous-staleness-sec", type=int, default=900)
    args = ap.parse_args()

    binance = load_binance(Path(args.binance_dir))
    chainlink = load_source_dir(Path(args.chainlink_dir), "chainlink")
    frames = []
    if not chainlink.empty:
        frames.append(chainlink)
    if args.include_pyth:
        pyth = load_source_dir(Path(args.pyth_dir), "pyth")
        if not pyth.empty:
            frames.append(pyth)
    if not frames:
        raise RuntimeError("No Chainlink/Pyth oracle data found.")

    oracle_df = pd.concat(frames, ignore_index=True)
    merged = merge_reference(oracle_df, binance, tolerance=args.reference_tolerance)
    trace = add_features(
        merged,
        deviation_threshold=args.deviation_threshold,
        staleness_threshold_sec=args.staleness_threshold_sec,
        suspicious_deviation=args.suspicious_deviation,
        anomalous_deviation=args.anomalous_deviation,
        suspicious_staleness=args.suspicious_staleness_sec,
        anomalous_staleness=args.anomalous_staleness_sec,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(out_path, index=False)
    print(f"[trace] wrote {out_path} rows={len(trace)}")
    print(trace.groupby(["source", "asset", "anomaly_label"]).size().rename("n").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
