#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline smoke test for build_real_oracle_trace.py using tiny fake data."""

from pathlib import Path
import pandas as pd
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
binance = root / "data/raw/binance"
chainlink = root / "data/raw/chainlink"
binance.mkdir(parents=True, exist_ok=True)
chainlink.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
    "symbol": ["ETHUSDT"] * 5,
    "interval": ["1m"] * 5,
    "open": [100, 101, 102, 103, 104],
    "high": [100, 101, 102, 103, 104],
    "low": [100, 101, 102, 103, 104],
    "close": [100, 101, 102, 103, 104],
    "volume": [1] * 5,
    "open_time": [0] * 5,
    "close_time": [0] * 5,
}).to_csv(binance / "ETHUSDT_1m.csv", index=False)

pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC"),
    "asset": ["ETH/USD"] * 5,
    "source": ["chainlink"] * 5,
    "oracle_id": ["chainlink_ETH_USD"] * 5,
    "oracle_price": [100, 101.1, 105, 103, 104],
    "gas_cost": [0.001] * 5,
    "service_type": [0] * 5,
}).to_csv(chainlink / "ETH_USD.csv", index=False)

out = root / "data/processed/smoke_real_oracle_trace.csv"
cmd = [
    sys.executable,
    str(root / "scripts/real_trace/build_real_oracle_trace.py"),
    "--chainlink-dir", str(chainlink),
    "--binance-dir", str(binance),
    "--out", str(out),
]
subprocess.check_call(cmd)
print(pd.read_csv(out).head().to_string(index=False))
