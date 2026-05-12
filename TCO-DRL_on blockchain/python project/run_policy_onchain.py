"""
Run an offline-trained oracle-selection policy and submit inference results on-chain.

Example:
    python run_policy_onchain.py --Chain_Config config_chain.json \
        --Method_Preset hcrl_only --Scenario rl_harder --Seed 6 --Request_Num 6000

The --Chain_Config option is consumed by this script. All remaining CLI options
are passed to the simulation argument parser from TCO-DRL_with baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from web3 import Web3


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
SIM_DIR = REPO_ROOT / "TCO-DRL_with baseline"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from env import SchedulingEnv  # noqa: E402
from utils import get_args  # noqa: E402
from policy_adapter import OnChainPolicyAdapter  # noqa: E402


def parse_onchain_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--Chain_Config", default="config_chain.json")
    parser.add_argument("--Max_Requests", type=int, default=None)
    parser.add_argument("--Dry_Run", action="store_true")
    onchain_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return onchain_args


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_value: str, base_dir: Path) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def is_connected(w3: Web3) -> bool:
    if hasattr(w3, "is_connected"):
        return bool(w3.is_connected())
    return bool(w3.isConnected())


def wait_receipt(w3: Web3, tx_hash):
    if hasattr(w3.eth, "wait_for_transaction_receipt"):
        return w3.eth.wait_for_transaction_receipt(tx_hash)
    return w3.eth.waitForTransactionReceipt(tx_hash)


def to_checksum(w3: Web3, address: str) -> str:
    if hasattr(w3, "to_checksum_address"):
        return w3.to_checksum_address(address)
    return w3.toChecksumAddress(address)


def load_contract(w3: Web3, artifact_path: Path, explicit_address: Optional[str] = None):
    artifact = load_json(artifact_path)
    abi = artifact.get("abi", artifact)

    address = explicit_address
    if not address:
        networks = artifact.get("networks", {})
        if not networks:
            raise ValueError(
                "contract_address is empty and no deployed network address was found "
                f"in artifact: {artifact_path}"
            )
        # Truffle network IDs are usually numeric strings. Use the largest one.
        network_id = sorted(networks.keys(), key=lambda x: int(x) if str(x).isdigit() else -1)[-1]
        address = networks[network_id]["address"]

    return w3.eth.contract(address=to_checksum(w3, address), abi=abi)


def submit_decision(w3: Web3, contract, account: str, request_id: int, decision):
    try:
        tx_hash = contract.functions.submitSelection(
            *decision.as_contract_args(request_id)
        ).transact({"from": account})
        receipt = wait_receipt(w3, tx_hash)
        return receipt, "submitSelection"
    except Exception as e:
        # Backward-compatible fallback for the original contract.
        print(f"[WARN] submitSelection failed, falling back to ReAction(uint): {e}")
        tx_hash = contract.functions.ReAction(int(decision.primary_oracle)).transact({"from": account})
        receipt = wait_receipt(w3, tx_hash)
        return receipt, "ReAction"


def main():
    onchain_args = parse_onchain_args()
    args = get_args()

    config_path = resolve_path(onchain_args.Chain_Config, Path.cwd())
    config_dir = config_path.parent
    cfg = load_json(config_path)

    method = cfg.get("method")
    if not method:
        if len(args.Baselines) != 1:
            raise ValueError(
                "config_chain.json must set 'method', or CLI must select exactly one method."
            )
        method = args.Baselines[0]

    weights = cfg.get("weights", {})
    if "weight_path" in cfg and "primary" not in weights:
        weights["primary"] = cfg["weight_path"]

    dry_run = bool(onchain_args.Dry_Run or cfg.get("dry_run", False))

    w3 = Web3(Web3.HTTPProvider(cfg.get("rpc_http", "http://127.0.0.1:8545")))
    if not dry_run and not is_connected(w3):
        raise RuntimeError("Web3 connection failed. Check Ganache RPC and config_chain.json.")

    contract = None
    account = None
    if not dry_run:
        accounts = w3.eth.accounts
        if not accounts:
            raise RuntimeError("No unlocked account found in Web3 provider.")
        account_index = int(cfg.get("from_account_index", 0))
        account = cfg.get("from_account") or accounts[account_index]
        artifact_path = resolve_path(cfg.get("contract_artifact", "../build/contracts/Selection.json"), config_dir)
        contract = load_contract(w3, artifact_path, cfg.get("contract_address") or None)

    env = SchedulingEnv(args)
    adapter = OnChainPolicyAdapter(args, env, method=method, weights=weights, config_dir=config_dir)

    env.reset(args)
    env.reset_reputation_factors()
    env.initial_reputation()

    max_requests = onchain_args.Max_Requests or cfg.get("max_requests") or args.Request_Num
    max_requests = int(max_requests)

    out_dir = resolve_path(cfg.get("output_dir", "onchain_output"), config_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"onchain_{method.replace('/', '_').replace(' ', '_')}_{int(time.time())}.csv"

    print(f"[on-chain deploy] method={method}")
    print(f"[on-chain deploy] policy_hash={adapter.policy_hash}")
    print(f"[on-chain deploy] dry_run={dry_run}")
    print(f"[on-chain deploy] output={out_csv}")

    rows = []
    request_c = 1
    time_period = 1

    while request_c <= max_requests:
        if request_c % args.Time_Period_Size == 0:
            time_period += 1
            env.update_reputation(env.get_reputation_factors(method), time_period, method)
            env.reset_reputation_factors()

        finish, request_attrs = env.workload(request_c)
        decision = adapter.infer(request_attrs)

        if dry_run:
            gas_used = 0
            latency = 0.0
            submit_fn = "dry_run"
            tx_hash = ""
        else:
            start = time.time()
            receipt, submit_fn = submit_decision(w3, contract, account, request_c, decision)
            latency = time.time() - start
            gas_used = int(receipt["gasUsed"])
            tx_hash = receipt["transactionHash"].hex() if hasattr(receipt["transactionHash"], "hex") else str(receipt["transactionHash"])

        adapter.update_environment_after_submit(request_attrs, decision)

        row = {
            "request_id": request_c,
            "method": method,
            "method_id": decision.method_id,
            "mode_code": decision.mode_code,
            "mode_index": decision.mode_index,
            "mode_name": decision.mode_name,
            "primary_oracle": decision.primary_oracle,
            "backup_oracle": decision.backup_oracle,
            "policy_hash": decision.policy_hash,
            "submit_function": submit_fn,
            "gas_used": gas_used,
            "latency_sec": round(latency, 6),
            "tx_hash": tx_hash,
        }
        rows.append(row)

        print(
            f"[request={request_c}] mode={decision.mode_name} "
            f"primary={decision.primary_oracle} backup={decision.backup_oracle} "
            f"gas={gas_used} latency={latency:.4f}s"
        )

        request_c += 1
        if finish:
            break

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"[done] saved on-chain decisions: {out_csv}")


if __name__ == "__main__":
    main()
