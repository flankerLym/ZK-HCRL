#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download Binance public kline data from Binance Data Vision.

The script downloads daily zip files first. If a daily file is unavailable,
it falls back to monthly zip files when possible.

Output:
  data/raw/binance/<SYMBOL>_<INTERVAL>.csv
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

try:
    from net_utils import make_requests_session
except ImportError:  # pragma: no cover
    from scripts.real_trace.net_utils import make_requests_session


BASE = "https://data.binance.vision/data/spot"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def month_range(start: date, end: date) -> Iterable[date]:
    cur = date(start.year, start.month, 1)
    while cur < end:
        yield cur
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def download_zip_csv(url: str, timeout: int = 30, session: requests.Session | None = None) -> pd.DataFrame | None:
    sess = session or make_requests_session(trust_env=False)
    r = sess.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return None
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, header=None)
    return df


def _safe_to_datetime_epoch(values: pd.Series, field_name: str) -> tuple[pd.Series, str]:
    """Convert Binance epoch timestamps with automatic unit detection.

    Binance historical archives were traditionally millisecond based, but some
    newer spot kline archives can contain microsecond timestamps. Interpreting
    microseconds as milliseconds produces years such as 58217 and triggers
    pandas OutOfBoundsDatetime. We infer the unit by magnitude and validate that
    the converted timestamps fall into a reasonable calendar range.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    sample = numeric.dropna()
    if sample.empty:
        return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce"), "ms"

    max_abs = float(sample.abs().max())
    candidate_units: list[str]
    if max_abs >= 1e17:
        candidate_units = ["ns", "us", "ms", "s"]
    elif max_abs >= 1e14:
        candidate_units = ["us", "ms", "ns", "s"]
    elif max_abs >= 1e11:
        candidate_units = ["ms", "us", "s", "ns"]
    else:
        candidate_units = ["s", "ms", "us", "ns"]

    lower = pd.Timestamp("2000-01-01", tz="UTC")
    upper = pd.Timestamp("2100-01-01", tz="UTC")
    last_result = None
    last_unit = candidate_units[0]
    for unit in candidate_units:
        converted = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        last_result, last_unit = converted, unit
        valid = converted.dropna()
        if valid.empty:
            continue
        if (valid >= lower).all() and (valid < upper).all():
            return converted, unit

    print(f"[warn] could not confidently infer Binance {field_name} timestamp unit; using {last_unit}")
    return last_result, last_unit


def standardize_klines(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "num_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    df = df.iloc[:, : len(cols)].copy()
    df.columns = cols[: df.shape[1]]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["timestamp"], open_unit = _safe_to_datetime_epoch(df["open_time"], "open_time")
    if "close_time" in df.columns:
        _, close_unit = _safe_to_datetime_epoch(df["close_time"], "close_time")
    else:
        close_unit = open_unit
    if open_unit != "ms" or close_unit != "ms":
        print(f"[Binance] detected timestamp units for {symbol}: open_time={open_unit}, close_time={close_unit}")

    df = df.dropna(subset=["timestamp", "close"]).copy()
    df["symbol"] = symbol
    df["interval"] = interval
    return df[["timestamp", "symbol", "interval", "open", "high", "low", "close", "volume", "open_time", "close_time"]]


def collect_symbol(symbol: str, interval: str, start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    # Daily archives.
    for d in daterange(start, end):
        url = f"{BASE}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d.isoformat()}.zip"
        df = download_zip_csv(url, session=session)
        if df is not None and len(df):
            frames.append(standardize_klines(df, symbol, interval))

    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time")
        out = out.sort_values("timestamp")
        return out

    # Fallback to monthly archives.
    for m in month_range(start, end):
        ym = f"{m.year}-{m.month:02d}"
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        df = download_zip_csv(url, session=session)
        if df is not None and len(df):
            frames.append(standardize_klines(df, symbol, interval))

    if not frames:
        raise RuntimeError(f"No Binance kline data downloaded for {symbol} {interval} between {start} and {end}")

    out = pd.concat(frames, ignore_index=True).drop_duplicates("open_time")
    out = out[(out["timestamp"] >= pd.Timestamp(start.isoformat(), tz="UTC")) &
              (out["timestamp"] < pd.Timestamp(end.isoformat(), tz="UTC"))]
    return out.sort_values("timestamp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True, help="Example: ETHUSDT BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD exclusive")
    ap.add_argument("--out", default="data/raw/binance")
    ap.add_argument("--trust-env", action="store_true", help="Allow requests to use system/env proxies. Default: disabled for TUN/global VPN mode.")
    args = ap.parse_args()

    start, end = parse_date(args.start), parse_date(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_requests_session(trust_env=args.trust_env)
    print(f"[network] trust_env={args.trust_env} for Binance requests")

    for symbol in args.symbols:
        print(f"[Binance] downloading {symbol} {args.interval} {start} -> {end}")
        df = collect_symbol(symbol.upper(), args.interval, start, end, session=session)
        out_path = out_dir / f"{symbol.upper()}_{args.interval}.csv"
        df.to_csv(out_path, index=False)
        print(f"[Binance] wrote {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
