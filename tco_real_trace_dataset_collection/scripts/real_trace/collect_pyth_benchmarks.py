#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect Pyth benchmark historical prices using a TradingView-compatible endpoint.

The benchmark endpoint returns OHLC-style historical prices. This script stores
close prices as `oracle_price`. Historical confidence intervals are not always
available from this endpoint, so `confidence_ratio` is left empty here.

Output:
  data/raw/pyth/<SYMBOL>.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

try:
    from net_utils import make_requests_session
except ImportError:  # pragma: no cover
    from scripts.real_trace.net_utils import make_requests_session


BASE = "https://benchmarks.pyth.network/v1/shims/tradingview/history"


def parse_ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def collect_symbol(symbol: str, resolution: str, start_ts: int, end_ts: int, session: requests.Session | None = None) -> pd.DataFrame:
    url = (
        f"{BASE}?symbol={quote(symbol, safe='')}"
        f"&resolution={resolution}&from={start_ts}&to={end_ts}"
    )
    sess = session or make_requests_session(trust_env=False)
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("s") not in ("ok", "no_data"):
        raise RuntimeError(f"Pyth benchmark error for {symbol}: {data}")
    if data.get("s") == "no_data" or not data.get("t"):
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(data["t"], unit="s", utc=True),
            "oracle_price": data.get("c"),
            "open": data.get("o"),
            "high": data.get("h"),
            "low": data.get("l"),
        }
    )
    df["asset"] = symbol.replace("Crypto.", "")
    df["source"] = "pyth"
    df["oracle_id"] = "pyth_" + df["asset"].str.replace("/", "_", regex=False)
    df["confidence_ratio"] = pd.NA
    df["gas_cost"] = pd.NA
    df["service_type"] = pd.NA
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True, help="Example: Crypto.ETH/USD Crypto.BTC/USD")
    ap.add_argument("--resolution", default="1", help="TradingView resolution. 1 means one minute.")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD exclusive")
    ap.add_argument("--out", default="data/raw/pyth")
    ap.add_argument("--trust-env", action="store_true", help="Allow requests to use system/env proxies. Default: disabled for TUN/global VPN mode.")
    args = ap.parse_args()

    start_ts, end_ts = parse_ts(args.start), parse_ts(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_requests_session(trust_env=args.trust_env)
    print(f"[network] trust_env={args.trust_env} for Pyth requests")

    for symbol in args.symbols:
        print(f"[Pyth] downloading {symbol}")
        df = collect_symbol(symbol, args.resolution, start_ts, end_ts, session=session)
        safe = symbol.replace("Crypto.", "").replace("/", "_")
        out_path = out_dir / f"{safe}.csv"
        df.to_csv(out_path, index=False)
        print(f"[Pyth] wrote {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
