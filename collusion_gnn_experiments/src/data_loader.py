from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_real_trace(trace_path: str, requests: int, seed: int) -> pd.DataFrame:
    """Load real oracle trace and normalize the column names used by this experiment.

    If the real trace file is missing, a deterministic synthetic fallback is generated so that
    users can smoke-test the code before collecting real trace data.
    """
    p = Path(trace_path)
    rng = np.random.default_rng(seed)
    if not p.exists():
        n = max(requests, 1000)
        t = np.arange(n)
        assets = np.array(["ETH/USD", "BTC/USD", "LINK/USD"])
        df = pd.DataFrame({
            "timestamp": t,
            "asset": rng.choice(assets, size=n),
            "deviation": np.clip(rng.lognormal(mean=-6.2, sigma=0.45, size=n), 0.0001, 0.03),
            "staleness": np.clip(rng.gamma(shape=3.0, scale=60.0, size=n), 10, 1200),
            "validation_success": rng.binomial(1, 0.78, size=n),
            "service_type": rng.integers(0, 3, size=n),
        })
        return df

    df = pd.read_csv(p)
    out = pd.DataFrame()

    asset_col = _find_col(df, ["asset", "symbol", "feed", "pair"])
    out["asset"] = df[asset_col].astype(str) if asset_col else "UNKNOWN"

    ts_col = _find_col(df, ["timestamp", "time", "datetime", "block_time", "updated_at"])
    if ts_col:
        out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        # If parse failed, fall back to row order.
        if out["timestamp"].isna().all():
            out["timestamp"] = np.arange(len(df))
    else:
        out["timestamp"] = np.arange(len(df))

    dev_col = _find_col(df, ["deviation", "abs_deviation", "relative_deviation", "mean_deviation"])
    if dev_col:
        out["deviation"] = pd.to_numeric(df[dev_col], errors="coerce").fillna(0.0).abs()
    else:
        price_col = _find_col(df, ["price", "oracle_price", "answer"])
        ref_col = _find_col(df, ["reference", "reference_price", "mean_reference", "binance_price"])
        if price_col and ref_col:
            price = pd.to_numeric(df[price_col], errors="coerce")
            ref = pd.to_numeric(df[ref_col], errors="coerce").replace(0, np.nan)
            out["deviation"] = ((price - ref).abs() / ref.abs()).fillna(0.0)
        else:
            out["deviation"] = 0.001

    stale_col = _find_col(df, ["staleness", "latency", "mean_staleness", "mean_latency"])
    out["staleness"] = pd.to_numeric(df[stale_col], errors="coerce").fillna(120.0) if stale_col else 120.0

    succ_col = _find_col(df, ["validation_success", "success", "is_valid", "valid"])
    if succ_col:
        out["validation_success"] = pd.to_numeric(df[succ_col], errors="coerce").fillna(1.0).clip(0, 1)
    else:
        # Infer from anomaly label when available.
        lab_col = _find_col(df, ["anomaly_label", "label"])
        if lab_col:
            labels = df[lab_col].astype(str).str.lower()
            out["validation_success"] = (~labels.isin(["anomalous", "attack", "bad", "malicious"])).astype(float)
        else:
            out["validation_success"] = 1.0

    svc_col = _find_col(df, ["service_type", "service", "type"])
    if svc_col:
        out["service_type"] = pd.to_numeric(df[svc_col], errors="coerce").fillna(0).astype(int)
    else:
        # Map assets to service types.
        cats = {a: i for i, a in enumerate(sorted(out["asset"].astype(str).unique()))}
        out["service_type"] = out["asset"].map(cats).astype(int) % 3

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["deviation", "staleness", "validation_success"])
    if out.empty:
        raise ValueError(f"Trace file {trace_path} could not be normalized into usable rows.")
    return out.reset_index(drop=True)
