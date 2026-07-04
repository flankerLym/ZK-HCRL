#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect Chainlink price-feed traces via an Ethereum RPC.

Output:
  data/raw/chainlink/<ASSET>.csv

Collection modes:
  auto   : try AnswerUpdated event logs first; if Alchemy/RPC rejects logs or
           a proxy emits no logs, fall back to historical latestRoundData calls.
  events : only collect AnswerUpdated logs.
  calls  : sample historical latestRoundData at fixed block intervals.

Robustness features for long historical-call jobs:
  * per-RPC-call retry with exponential backoff
  * skip failed sample points instead of killing the whole run
  * partial CSV checkpointing and resume from existing CSV
  * failed point log for reproducibility/debugging
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
import yaml
from tqdm import tqdm
from web3 import Web3

try:
    from net_utils import make_web3_http_provider
except ImportError:  # pragma: no cover
    from scripts.real_trace.net_utils import make_web3_http_provider


ANSWER_UPDATED_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "internalType": "int256", "name": "current", "type": "int256"},
        {"indexed": True, "internalType": "uint256", "name": "roundId", "type": "uint256"},
        {"indexed": False, "internalType": "uint256", "name": "updatedAt", "type": "uint256"},
    ],
    "name": "AnswerUpdated",
    "type": "event",
}

DECIMALS_ABI = {
    "inputs": [],
    "name": "decimals",
    "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
    "stateMutability": "view",
    "type": "function",
}

LATEST_ROUND_DATA_ABI = {
    "inputs": [],
    "name": "latestRoundData",
    "outputs": [
        {"internalType": "uint80", "name": "roundId", "type": "uint80"},
        {"internalType": "int256", "name": "answer", "type": "int256"},
        {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
        {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
        {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"},
    ],
    "name": "latestRoundData",
    "stateMutability": "view",
    "type": "function",
}

T = TypeVar("T")


def parse_dt(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def iso_from_ts(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def call_with_retry(
    fn: Callable[[], T],
    *,
    desc: str,
    max_retries: int = 5,
    backoff_sec: float = 2.0,
    max_backoff_sec: float = 60.0,
    jitter: float = 0.25,
) -> T:
    """Run an RPC function with retry/backoff.

    Alchemy/free RPC endpoints may occasionally close the connection for long
    historical eth_call jobs. Retrying is usually enough. If all retries fail,
    the original exception is raised so the caller can either skip or stop.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max(1, max_retries) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - RPC providers raise many exception types
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = min(max_backoff_sec, backoff_sec * (2 ** attempt))
            if jitter > 0:
                delay *= random.uniform(1.0 - jitter, 1.0 + jitter)
            print(f"[retry] {desc} failed ({attempt + 1}/{max_retries}); sleep {delay:.1f}s; error={exc}")
            time.sleep(max(0.0, delay))
    assert last_exc is not None
    raise last_exc


def get_block_ts(w3: Web3, block_number: int, max_retries: int = 5, backoff_sec: float = 2.0) -> int:
    block = call_with_retry(
        lambda: w3.eth.get_block(block_number),
        desc=f"get_block({block_number})",
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    return int(block["timestamp"])


def find_block_by_timestamp(w3: Web3, target_ts: int, max_retries: int = 5, backoff_sec: float = 2.0) -> int:
    latest_block = call_with_retry(
        lambda: int(w3.eth.block_number),
        desc="eth.block_number",
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    lo, hi = 0, latest_block
    while lo < hi:
        mid = (lo + hi) // 2
        mid_ts = get_block_ts(w3, mid, max_retries=max_retries, backoff_sec=backoff_sec)
        if mid_ts < target_ts:
            lo = mid + 1
        else:
            hi = mid
    return lo


def safe_get_receipt_cost_eth(w3: Web3, tx_hash) -> tuple[float | None, int | None, int | None]:
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        tx = w3.eth.get_transaction(tx_hash)
        gas_used = int(receipt.get("gasUsed", 0))
        gas_price = int(tx.get("gasPrice", 0))
        cost_eth = gas_used * gas_price / 1e18
        return cost_eth, gas_used, gas_price
    except Exception:
        return None, None, None


def load_feeds(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj["feeds"]


def make_contract(w3: Web3, feed: Dict):
    address = Web3.to_checksum_address(feed["address"])
    return address, w3.eth.contract(address=address, abi=[ANSWER_UPDATED_ABI, DECIMALS_ABI, LATEST_ROUND_DATA_ABI])


def get_decimals(contract, feed: Dict, max_retries: int = 5, backoff_sec: float = 2.0) -> int:
    decimals = int(feed.get("decimals", 8))
    try:
        decimals = int(call_with_retry(
            lambda: contract.functions.decimals().call(),
            desc=f"decimals({feed['asset']})",
            max_retries=max_retries,
            backoff_sec=backoff_sec,
        ))
    except Exception:
        pass
    return decimals


def get_answer_updated_logs(contract, from_block: int, to_block: int):
    """Return AnswerUpdated logs across common web3.py versions."""
    event_obj = contract.events.AnswerUpdated
    attempts = [
        lambda: event_obj.get_logs(from_block=from_block, to_block=to_block),
        lambda: event_obj().get_logs(from_block=from_block, to_block=to_block),
        lambda: event_obj.get_logs(fromBlock=from_block, toBlock=to_block),
        lambda: event_obj().get_logs(fromBlock=from_block, toBlock=to_block),
    ]
    last_type_error = None
    last_other_error = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as exc:
            last_type_error = exc
            continue
        except Exception as exc:
            last_other_error = exc
            break
    if last_other_error is not None:
        raise last_other_error
    if last_type_error is not None:
        raise last_type_error
    return []


def event_row(w3: Web3, ev, feed: Dict, address: str, decimals: int, fetch_receipts: bool) -> Dict:
    args = ev["args"]
    updated_at = int(args["updatedAt"])
    raw_answer = int(args["current"])
    price = raw_answer / (10 ** decimals)
    block_number = int(ev["blockNumber"])
    tx_hash = ev["transactionHash"].hex()
    gas_cost, gas_used, gas_price = (None, None, None)
    if fetch_receipts:
        gas_cost, gas_used, gas_price = safe_get_receipt_cost_eth(w3, ev["transactionHash"])

    return {
        "timestamp": iso_from_ts(updated_at),
        "sample_timestamp": iso_from_ts(updated_at),
        "updated_at": updated_at,
        "feed_updated_at": updated_at,
        "block_number": block_number,
        "tx_hash": tx_hash,
        "asset": feed["asset"],
        "symbol": feed.get("symbol"),
        "source": "chainlink",
        "feed_address": address,
        "round_id": int(args["roundId"]),
        "raw_answer": raw_answer,
        "decimals": decimals,
        "oracle_price": price,
        "gas_cost": gas_cost,
        "gas_used": gas_used,
        "gas_price": gas_price,
        "service_type": feed.get("service_type"),
        "oracle_id": f"chainlink_{feed['asset'].replace('/', '_')}",
        "collection_mode": "event_logs",
    }


def collect_feed_events(
    w3: Web3,
    feed: Dict,
    start_block: int,
    end_block: int,
    block_chunk: int,
    fetch_receipts: bool = False,
    sleep_sec: float = 0.0,
    min_chunk: int = 500,
    max_retries: int = 5,
    retry_backoff_sec: float = 2.0,
) -> pd.DataFrame:
    address, contract = make_contract(w3, feed)
    decimals = get_decimals(contract, feed, max_retries=max_retries, backoff_sec=retry_backoff_sec)

    rows = []
    current = start_block
    pbar = tqdm(total=max(0, end_block - start_block + 1), desc=f"Chainlink {feed['asset']} events", leave=False)
    while current <= end_block:
        to_block = min(current + block_chunk - 1, end_block)
        try:
            logs = call_with_retry(
                lambda: get_answer_updated_logs(contract, current, to_block),
                desc=f"get_logs({feed['asset']} {current}-{to_block})",
                max_retries=max_retries,
                backoff_sec=retry_backoff_sec,
            )
        except Exception as e:
            if block_chunk > min_chunk:
                smaller = max(min_chunk, block_chunk // 2)
                pbar.close()
                print(f"[warn] get_logs failed for {feed['asset']} {current}-{to_block}: {e}; retry with chunk={smaller}")
                return collect_feed_events(w3, feed, start_block, end_block, smaller, fetch_receipts, sleep_sec, min_chunk, max_retries, retry_backoff_sec)
            pbar.close()
            raise

        for ev in logs:
            rows.append(event_row(w3, ev, feed, address, decimals, fetch_receipts))
        pbar.update(to_block - current + 1)
        current = to_block + 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    pbar.close()
    return pd.DataFrame(rows).sort_values("updated_at") if rows else pd.DataFrame()


def call_latest_round_data(contract, block_number: int):
    """Call latestRoundData at a historical block, across web3.py versions."""
    try:
        return contract.functions.latestRoundData().call(block_identifier=block_number)
    except TypeError:
        return contract.functions.latestRoundData().call(block_number)


def load_existing_partial(out_path: Optional[Path]) -> pd.DataFrame:
    if out_path is None or not out_path.exists() or out_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(out_path)
        if "block_number" in df.columns:
            df = df.drop_duplicates(subset=["block_number", "round_id"], keep="last")
        return df
    except Exception as exc:
        print(f"[warn] cannot read existing partial file {out_path}: {exc}; start from scratch")
        return pd.DataFrame()


def save_partial(df: pd.DataFrame, out_path: Optional[Path]) -> None:
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.sort_values(["block_number", "updated_at"]).to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)


def append_failure(failure_log_path: Optional[Path], row: Dict) -> None:
    if failure_log_path is None:
        return
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = failure_log_path.exists()
    with failure_log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def collect_feed_historical_calls(
    w3: Web3,
    feed: Dict,
    start_block: int,
    end_block: int,
    sample_sec: int = 60,
    sleep_sec: float = 0.0,
    out_path: Optional[Path] = None,
    resume: bool = True,
    save_every: int = 25,
    max_retries: int = 5,
    retry_backoff_sec: float = 2.0,
    retry_max_backoff_sec: float = 60.0,
    skip_failed_points: bool = True,
    failure_log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fallback collector that samples latestRoundData at historical blocks.

    This avoids eth_getLogs. It requires the RPC provider to support historical
    eth_call/archive state for the requested date. For long jobs, every sampled
    block is retried, optionally skipped on persistent failure, and partial rows
    are flushed to out_path for resuming.
    """
    address, contract = make_contract(w3, feed)
    decimals = get_decimals(contract, feed, max_retries=max_retries, backoff_sec=retry_backoff_sec)

    # Ethereum mainnet block time is roughly 12s. Use a block stride that gives
    # about one sample per sample_sec.
    block_step = max(1, int(round(sample_sec / 12.0)))

    existing = load_existing_partial(out_path) if resume else pd.DataFrame()
    rows: List[Dict] = existing.to_dict("records") if not existing.empty else []
    done_blocks = set(int(x) for x in existing.get("block_number", pd.Series(dtype="int64")).dropna().astype(int).tolist())
    seen = set()
    for r in rows:
        try:
            seen.add((int(r.get("round_id")), int(r.get("block_number"))))
        except Exception:
            continue

    block_numbers = list(range(start_block, end_block + 1, block_step))
    pending_blocks = [b for b in block_numbers if b not in done_blocks]
    if done_blocks:
        print(f"[resume] {feed['asset']}: loaded {len(rows)} rows from {out_path}; skip {len(done_blocks)} sampled blocks")

    pbar = tqdm(total=len(block_numbers), initial=len(block_numbers) - len(pending_blocks), desc=f"Chainlink {feed['asset']} historical calls", leave=False)
    new_since_save = 0
    skipped = 0

    for block_number in pending_blocks:
        try:
            block_ts = call_with_retry(
                lambda: get_block_ts(w3, block_number, max_retries=max_retries, backoff_sec=retry_backoff_sec),
                desc=f"{feed['asset']} get_block_ts({block_number})",
                max_retries=max_retries,
                backoff_sec=retry_backoff_sec,
                max_backoff_sec=retry_max_backoff_sec,
            )
            data = call_with_retry(
                lambda: call_latest_round_data(contract, block_number),
                desc=f"{feed['asset']} latestRoundData(block={block_number})",
                max_retries=max_retries,
                backoff_sec=retry_backoff_sec,
                max_backoff_sec=retry_max_backoff_sec,
            )
        except Exception as exc:  # noqa: BLE001
            if skip_failed_points:
                skipped += 1
                print(f"[skip] {feed['asset']} block={block_number} failed after retries; skip. error={exc}")
                append_failure(
                    failure_log_path,
                    {
                        "asset": feed["asset"],
                        "symbol": feed.get("symbol"),
                        "block_number": block_number,
                        "error": repr(exc),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                pbar.update(1)
                continue
            pbar.close()
            raise RuntimeError(
                f"Historical latestRoundData call failed for {feed['asset']} at block {block_number}. "
                f"This usually means the RPC does not provide archive/historical state, the provider disconnected, "
                f"or the feed address is invalid. Original error: {exc}"
            ) from exc

        round_id, answer, started_at, updated_at, answered_in_round = data
        round_id = int(round_id)
        raw_answer = int(answer)
        updated_at = int(updated_at)
        if updated_at <= 0 or raw_answer <= 0:
            pbar.update(1)
            continue

        # Avoid duplicate rows when several sampled blocks have the same round.
        key = (round_id, block_number)
        if key in seen:
            pbar.update(1)
            continue
        seen.add(key)

        price = raw_answer / (10 ** decimals)
        rows.append(
            {
                "timestamp": iso_from_ts(block_ts),
                "sample_timestamp": iso_from_ts(block_ts),
                "updated_at": updated_at,
                "feed_updated_at": updated_at,
                "feed_update_time": iso_from_ts(updated_at),
                "block_number": block_number,
                "tx_hash": None,
                "asset": feed["asset"],
                "symbol": feed.get("symbol"),
                "source": "chainlink",
                "feed_address": address,
                "round_id": round_id,
                "answered_in_round": int(answered_in_round),
                "raw_answer": raw_answer,
                "decimals": decimals,
                "oracle_price": price,
                "gas_cost": None,
                "gas_used": None,
                "gas_price": None,
                "service_type": feed.get("service_type"),
                "oracle_id": f"chainlink_{feed['asset'].replace('/', '_')}",
                "collection_mode": "historical_call",
            }
        )
        new_since_save += 1
        if out_path is not None and new_since_save >= max(1, save_every):
            save_partial(pd.DataFrame(rows), out_path)
            new_since_save = 0
        pbar.update(1)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    pbar.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["block_number", "round_id"], keep="last").sort_values(["block_number", "updated_at"])
    save_partial(df, out_path)
    if skipped:
        print(f"[warn] {feed['asset']}: skipped {skipped} failed historical sample points. See {failure_log_path}")
    return df


def collect_feed(
    w3: Web3,
    feed: Dict,
    start_block: int,
    end_block: int,
    block_chunk: int,
    fetch_receipts: bool = False,
    sleep_sec: float = 0.0,
    mode: str = "auto",
    historical_sample_sec: int = 60,
    out_path: Optional[Path] = None,
    resume: bool = True,
    save_every: int = 25,
    max_retries: int = 5,
    retry_backoff_sec: float = 2.0,
    retry_max_backoff_sec: float = 60.0,
    skip_failed_points: bool = True,
    failure_log_path: Optional[Path] = None,
) -> pd.DataFrame:
    mode = mode.lower()
    if mode not in {"auto", "events", "calls"}:
        raise ValueError(f"Unknown chainlink mode: {mode}")

    if mode == "calls":
        return collect_feed_historical_calls(
            w3, feed, start_block, end_block, historical_sample_sec, sleep_sec,
            out_path=out_path, resume=resume, save_every=save_every,
            max_retries=max_retries, retry_backoff_sec=retry_backoff_sec,
            retry_max_backoff_sec=retry_max_backoff_sec,
            skip_failed_points=skip_failed_points, failure_log_path=failure_log_path,
        )

    if mode in {"auto", "events"}:
        try:
            df = collect_feed_events(
                w3, feed, start_block, end_block, block_chunk, fetch_receipts, sleep_sec,
                max_retries=max_retries, retry_backoff_sec=retry_backoff_sec,
            )
            if not df.empty:
                save_partial(df, out_path)
                return df
            if mode == "events":
                save_partial(df, out_path)
                return df
            print(f"[warn] no AnswerUpdated logs found for {feed['asset']} on the proxy; falling back to historical calls")
        except Exception as exc:
            if mode == "events":
                raise
            print(f"[warn] event-log collection failed for {feed['asset']}: {exc}")
            print(f"[warn] falling back to historical latestRoundData calls for {feed['asset']}")
        return collect_feed_historical_calls(
            w3, feed, start_block, end_block, historical_sample_sec, sleep_sec,
            out_path=out_path, resume=resume, save_every=save_every,
            max_retries=max_retries, retry_backoff_sec=retry_backoff_sec,
            retry_max_backoff_sec=retry_max_backoff_sec,
            skip_failed_points=skip_failed_points, failure_log_path=failure_log_path,
        )

    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc-url", required=True)
    ap.add_argument("--feeds", default="config/chainlink_feeds.yaml")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD exclusive")
    ap.add_argument("--out", default="data/raw/chainlink")
    ap.add_argument("--block-chunk", type=int, default=5000)
    ap.add_argument("--fetch-receipts", action="store_true")
    ap.add_argument("--sleep-sec", type=float, default=0.0)
    ap.add_argument("--trust-env", action="store_true", help="Allow Web3 requests to use system/env proxies. Default: disabled for TUN/global VPN mode.")
    ap.add_argument("--mode", choices=["auto", "events", "calls"], default="auto", help="Chainlink collection mode. auto tries events, then historical calls.")
    ap.add_argument("--historical-sample-sec", type=int, default=60, help="Sampling interval for historical latestRoundData fallback.")
    ap.add_argument("--max-retries", type=int, default=5, help="Retry count for every RPC request in historical-call mode.")
    ap.add_argument("--retry-backoff-sec", type=float, default=2.0, help="Initial exponential backoff seconds for RPC retries.")
    ap.add_argument("--retry-max-backoff-sec", type=float, default=60.0, help="Maximum backoff seconds for RPC retries.")
    ap.add_argument("--save-every", type=int, default=25, help="Flush partial Chainlink CSV every N newly collected rows.")
    ap.add_argument("--no-resume", action="store_false", dest="resume", default=True, help="Disable resume from existing Chainlink CSV files.")
    ap.add_argument("--strict-failures", action="store_false", dest="skip_failed_points", default=True, help="Stop the job when a historical sample point fails after retries. Default skips failed points.")
    args = ap.parse_args()

    provider = make_web3_http_provider(args.rpc_url, timeout=60, trust_env=args.trust_env)
    w3 = Web3(provider)
    print(f"[network] trust_env={args.trust_env} for Chainlink RPC")
    print(f"[Chainlink] collection mode={args.mode}, historical_sample_sec={args.historical_sample_sec}")
    print(f"[Chainlink] retry max_retries={args.max_retries}, resume={args.resume}, skip_failed_points={args.skip_failed_points}, save_every={args.save_every}")
    if not w3.is_connected():
        raise RuntimeError("Cannot connect to Ethereum RPC. Check --rpc-url.")

    start_ts, end_ts = parse_dt(args.start), parse_dt(args.end)
    print("[Chainlink] locating block range by timestamp...")
    start_block = find_block_by_timestamp(w3, start_ts, max_retries=args.max_retries, backoff_sec=args.retry_backoff_sec)
    end_block = find_block_by_timestamp(w3, end_ts, max_retries=args.max_retries, backoff_sec=args.retry_backoff_sec)
    print(f"[Chainlink] blocks {start_block} -> {end_block}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = out_dir / "failed_points"
    failed_dir.mkdir(parents=True, exist_ok=True)

    any_rows = False
    for feed in load_feeds(args.feeds):
        safe_asset = feed["asset"].replace("/", "_")
        out_path = out_dir / f"{safe_asset}.csv"
        failure_log_path = failed_dir / f"{safe_asset}_failed_points.csv"
        df = collect_feed(
            w3=w3,
            feed=feed,
            start_block=start_block,
            end_block=end_block,
            block_chunk=args.block_chunk,
            fetch_receipts=args.fetch_receipts,
            sleep_sec=args.sleep_sec,
            mode=args.mode,
            historical_sample_sec=args.historical_sample_sec,
            out_path=out_path,
            resume=args.resume,
            save_every=args.save_every,
            max_retries=args.max_retries,
            retry_backoff_sec=args.retry_backoff_sec,
            retry_max_backoff_sec=args.retry_max_backoff_sec,
            skip_failed_points=args.skip_failed_points,
            failure_log_path=failure_log_path,
        )
        # collect_feed already saves partial/final CSV, but write once more for non-historical paths.
        df.to_csv(out_path, index=False)
        any_rows = any_rows or (not df.empty)
        print(f"[Chainlink] wrote {out_path} rows={len(df)}")

    if not any_rows:
        raise RuntimeError("Chainlink collection produced zero rows for all feeds. Check feed addresses, RPC archive support, and date range.")


if __name__ == "__main__":
    main()
