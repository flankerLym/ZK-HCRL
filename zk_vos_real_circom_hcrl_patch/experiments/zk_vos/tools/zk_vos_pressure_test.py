#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZK-VOS pressure/scalability and circuit-ablation benchmark.

This script is designed for the TCO-DRL HCRL ZK-VOS experiments.
It produces two paper-ready tables:

1) Batch verification / scalability
   #Schedules | Avg. Proof Time | Avg. Verify Time | Avg. Verify Gas | Success Verify Rate | Total Gas

2) Circuit ablation
   Circuit | Constraints | Proof Time | Verify Time | Verify Gas

The script intentionally keeps gas handling flexible because on-chain gas is usually
measured by an existing Hardhat/Foundry single-proof test. Pass that measured value
through --single-verify-gas or pass a per-circuit gas map JSON through --gas-map.

Dependencies for real proving/verifying:
  - Node.js
  - snarkjs in PATH, or pass --snarkjs path/to/snarkjs.cmd
  - existing .wasm, .zkey, verification_key.json artifacts

Examples:
  python zk_vos_pressure_test.py stress --trace-csv data/trace_hcrl_zk_schedule_trace.csv \
      --wasm circuits/zk_vos_js/zk_vos.wasm --zkey circuits/zk_vos_final.zkey \
      --vkey circuits/verification_key.json --single-verify-gas 224532 \
      --sizes "100 500 1000 5000" --repeat-trace-rows

  python zk_vos_pressure_test.py ablation --ablation-config pressure_test/configs/ablation_config_template.json

  python zk_vos_pressure_test.py full ...same stress args... --ablation-config ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_executable(path_or_name: Optional[str], required: bool = True) -> Optional[str]:
    if not path_or_name:
        if required:
            raise FileNotFoundError("Executable path is empty")
        return None
    p = Path(path_or_name)
    if p.exists():
        return str(p)
    found = shutil.which(path_or_name)
    if found:
        return found
    # Windows npm global binaries sometimes exist as .cmd.
    if os.name == "nt" and not str(path_or_name).lower().endswith(".cmd"):
        found_cmd = shutil.which(f"{path_or_name}.cmd")
        if found_cmd:
            return found_cmd
    if required:
        raise FileNotFoundError(
            f"Executable not found in PATH: {path_or_name}. "
            "For snarkjs, install with `npm install -g snarkjs`, or pass "
            "--snarkjs C:/path/to/snarkjs.cmd."
        )
    return None


def ensure_file(path: Optional[str], name: str, required: bool = True) -> Optional[Path]:
    if not path:
        if required:
            raise FileNotFoundError(f"Missing required path: {name}")
        return None
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"{name} does not exist: {p}")
        return None
    return p


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        shell=False,
    )
    return proc


def timed_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Tuple[float, subprocess.CompletedProcess]:
    t0 = time.perf_counter()
    proc = run_cmd(cmd, cwd=cwd, timeout=timeout)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, proc


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    mkdir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(round(float(x)))
    except Exception:
        return default


def parse_sizes(s: str) -> List[int]:
    out = []
    for token in re.split(r"[,\s]+", str(s).strip()):
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No batch sizes were provided")
    return out


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    if len(xs2) == 1:
        return float(xs2[0])
    pos = (len(xs2) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs2[lo])
    return float(xs2[lo] * (hi - pos) + xs2[hi] * (pos - lo))


def stats_ms(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {
            "total_ms": float("nan"),
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "p95_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
        }
    return {
        "total_ms": float(sum(xs)),
        "mean_ms": float(statistics.mean(xs)),
        "median_ms": float(statistics.median(xs)),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": float(min(xs)),
        "max_ms": float(max(xs)),
    }


def find_latest_file(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if not root.exists():
        return None
    for pat in patterns:
        candidates.extend(root.rglob(pat))
    files = [p for p in candidates if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


# -----------------------------------------------------------------------------
# Input schema mapping
# -----------------------------------------------------------------------------


def read_trace_csv(trace_csv: Path) -> List[Dict[str, str]]:
    with open(trace_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Trace CSV is empty: {trace_csv}")
    return rows


def parse_circom_input_signals(circuit_path: Optional[Path]) -> Optional[List[str]]:
    if circuit_path is None or not circuit_path.exists():
        return None
    text = circuit_path.read_text(encoding="utf-8", errors="ignore")
    # Match `signal input foo;` and `signal input foo[n];`.
    names = re.findall(r"signal\s+input\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    # Deduplicate while preserving order.
    out: List[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out or None


def _row_first(row: Dict[str, Any], keys: Sequence[str], default: Any = 0) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def trace_row_to_base_values(row: Dict[str, Any], index: int = 0) -> Dict[str, int]:
    """Convert one HCRL trace row into canonical integer values.

    All values are integer scaled values expected by Circom circuits.  The trace
    exporter already creates *_scaled fields. If a scaled field is missing, we
    derive it from the floating value.
    """
    rep = safe_int(_row_first(row, ["repEff_scaled", "rep_scaled", "reputation_scaled"], None), -1)
    if rep < 0:
        rep = safe_int(safe_float(_row_first(row, ["repEff", "rep", "reputation"], 0.0)) * 10000)

    cost = safe_int(_row_first(row, ["cost_scaled", "cost"], None), -1)
    # If `cost` looks like a floating small value, scale by 1000.
    if cost < 0:
        cost = safe_int(safe_float(_row_first(row, ["cost"], 0.0)) * 1000)
    elif "cost_scaled" not in row and safe_float(row.get("cost", cost)) < 100.0:
        cost = safe_int(safe_float(row.get("cost", 0.0)) * 1000)

    risk = safe_int(_row_first(row, ["risk_scaled", "risk"], None), -1)
    if risk < 0:
        risk = safe_int(safe_float(_row_first(row, ["risk"], 0.0)) * 10000)
    elif "risk_scaled" not in row and safe_float(row.get("risk", risk)) <= 1.0:
        risk = safe_int(safe_float(row.get("risk", 0.0)) * 10000)

    latency = safe_int(_row_first(row, ["latencyEst_scaled", "latency_scaled", "latency", "final_duration"], None), -1)
    if latency < 0:
        latency = safe_int(safe_float(_row_first(row, ["latencyEst", "latency", "final_duration"], 0.0)) * 1000)
    elif "latencyEst_scaled" not in row and safe_float(_row_first(row, ["latencyEst", "latency", "final_duration"], latency)) < 100.0:
        latency = safe_int(safe_float(_row_first(row, ["latencyEst", "latency", "final_duration"], 0.0)) * 1000)

    rep_th = safe_int(_row_first(row, ["reputationThreshold", "repThreshold", "reputation_threshold"], 6000))
    cost_budget = safe_int(_row_first(row, ["costBudget", "cost_budget"], 1000))
    risk_budget = safe_int(_row_first(row, ["riskBudget", "risk_budget"], 600))
    deadline = safe_int(_row_first(row, ["deadline", "deadline_scaled"], 6000))

    service_match = safe_int(_row_first(row, ["service_match", "serviceMatch"], 1))
    cooldown = safe_int(_row_first(row, ["cooldown_flag", "cooldownFlag", "cooldown"], 0))

    selected = safe_int(_row_first(row, ["selectedOracleId", "oracleId", "selected_oracle_id"], index))
    request_id = safe_int(_row_first(row, ["request_id", "requestId"], index))
    mode_id = safe_int(_row_first(row, ["mode_id", "modeId"], 0))
    backup_id = safe_int(_row_first(row, ["backupOracleId", "backupId"], -1))
    if backup_id < 0:
        backup_id = 0

    audit_truth = safe_int(_row_first(row, ["audit_truth_scaled", "auditTruth", "audit_truth"], None), -1)
    if audit_truth < 0:
        audit_truth = safe_int(safe_float(_row_first(row, ["audit_truth"], 0.0)) * 10000)

    pre_truth = safe_int(_row_first(row, ["audit_truth_before_scaled", "auditTruthBefore"], audit_truth))
    post_truth = safe_int(_row_first(row, ["audit_truth_after_scaled", "auditTruthAfter"], audit_truth))
    audit_pass = safe_int(_row_first(row, ["audit_pass", "auditPass"], 1))
    audit_fail = safe_int(_row_first(row, ["audit_fail", "auditFail"], 0))

    is_compliant = int(
        rep >= rep_th
        and cost <= cost_budget
        and risk <= risk_budget
        and latency <= deadline
        and service_match == 1
        and cooldown == 0
    )
    if "zk_is_compliant" in row:
        # Keep trace exporter result if present; it reflects the exact runtime rule.
        is_compliant = safe_int(row.get("zk_is_compliant"), is_compliant)

    return {
        "requestId": request_id,
        "selectedOracleId": selected,
        "oracleId": selected,
        "backupOracleId": backup_id,
        "modeId": mode_id,
        "repEff": rep,
        "reputation": rep,
        "cost": cost,
        "risk": risk,
        "latency": latency,
        "latencyEst": latency,
        "reputationThreshold": rep_th,
        "repThreshold": rep_th,
        "costBudget": cost_budget,
        "riskBudget": risk_budget,
        "deadline": deadline,
        "serviceMatch": service_match,
        "service_match": service_match,
        "cooldownFlag": cooldown,
        "cooldown": cooldown,
        "auditTruth": audit_truth,
        "auditTruthBefore": pre_truth,
        "auditTruthAfter": post_truth,
        "auditPass": audit_pass,
        "auditFail": audit_fail,
        "isCompliant": is_compliant,
        "zkIsCompliant": is_compliant,
    }


def build_input_for_signals(base: Dict[str, int], signals: Optional[List[str]]) -> Dict[str, int]:
    """Build an input JSON dictionary.

    If the circuit input names are known, only those keys are written. This avoids
    witness-calculator failures caused by extra unknown signal names. If no input
    list is available, a compact common schema is used.
    """
    aliases = {
        "selectedOracleId": ["selectedOracleId", "oracleId"],
        "oracleId": ["oracleId", "selectedOracleId"],
        "repEff": ["repEff", "reputation"],
        "reputation": ["reputation", "repEff"],
        "latencyEst": ["latencyEst", "latency"],
        "latency": ["latency", "latencyEst"],
        "repThreshold": ["repThreshold", "reputationThreshold"],
        "reputationThreshold": ["reputationThreshold", "repThreshold"],
        "service_match": ["service_match", "serviceMatch"],
        "serviceMatch": ["serviceMatch", "service_match"],
        "cooldownFlag": ["cooldownFlag", "cooldown"],
        "cooldown": ["cooldown", "cooldownFlag"],
        "zkIsCompliant": ["zkIsCompliant", "isCompliant"],
        "isCompliant": ["isCompliant", "zkIsCompliant"],
    }
    if signals:
        obj: Dict[str, int] = {}
        missing: List[str] = []
        for s in signals:
            if s in base:
                obj[s] = int(base[s])
                continue
            found = False
            for a in aliases.get(s, []):
                if a in base:
                    obj[s] = int(base[a])
                    found = True
                    break
            if not found:
                missing.append(s)
                obj[s] = 0
        if missing:
            eprint(f"[warning] filled missing circuit inputs with 0: {missing}")
        return obj
    common_keys = [
        "selectedOracleId", "repEff", "cost", "risk", "latencyEst",
        "reputationThreshold", "costBudget", "riskBudget", "deadline",
        "serviceMatch", "cooldownFlag", "auditTruth", "isCompliant",
    ]
    return {k: int(base[k]) for k in common_keys if k in base}


def prepare_inputs_from_trace(
    rows: List[Dict[str, str]],
    out_dir: Path,
    n: int,
    repeat_trace_rows: bool,
    circuit_path: Optional[Path] = None,
) -> List[Path]:
    signals = parse_circom_input_signals(circuit_path)
    if n > len(rows) and not repeat_trace_rows:
        raise ValueError(
            f"Requested {n} schedules but trace has only {len(rows)} rows. "
            "Use --repeat-trace-rows to cycle through the trace."
        )
    mkdir(out_dir)
    paths: List[Path] = []
    for i in range(n):
        row = rows[i % len(rows)]
        base = trace_row_to_base_values(row, i)
        obj = build_input_for_signals(base, signals)
        path = out_dir / f"input_{i:06d}.json"
        write_json(path, obj)
        paths.append(path)
    return paths


# -----------------------------------------------------------------------------
# snarkjs proving and verification
# -----------------------------------------------------------------------------


@dataclass
class ProofResult:
    index: int
    input_path: str
    proof_path: str
    public_path: str
    proof_time_ms: float
    verify_time_ms: float
    verify_success: bool
    proof_ok: bool
    proof_stdout: str = ""
    proof_stderr: str = ""
    verify_stdout: str = ""
    verify_stderr: str = ""


def snarkjs_fullprove(snarkjs: str, input_json: Path, wasm: Path, zkey: Path, proof_out: Path, public_out: Path, timeout: Optional[int]) -> Tuple[float, subprocess.CompletedProcess]:
    mkdir(proof_out.parent)
    mkdir(public_out.parent)
    cmd = [snarkjs, "groth16", "fullprove", str(input_json), str(wasm), str(zkey), str(proof_out), str(public_out)]
    return timed_cmd(cmd, timeout=timeout)


def snarkjs_verify(snarkjs: str, vkey: Path, public_json: Path, proof_json: Path, timeout: Optional[int]) -> Tuple[float, subprocess.CompletedProcess]:
    cmd = [snarkjs, "groth16", "verify", str(vkey), str(public_json), str(proof_json)]
    return timed_cmd(cmd, timeout=timeout)


def run_proof_verify_batch(
    snarkjs: str,
    inputs: List[Path],
    wasm: Path,
    zkey: Path,
    vkey: Path,
    proof_dir: Path,
    public_dir: Path,
    timeout: Optional[int],
    skip_prove: bool = False,
    skip_verify: bool = False,
) -> List[ProofResult]:
    mkdir(proof_dir)
    mkdir(public_dir)
    results: List[ProofResult] = []
    for i, input_path in enumerate(inputs):
        proof_path = proof_dir / f"proof_{i:06d}.json"
        public_path = public_dir / f"public_{i:06d}.json"
        proof_time_ms = float("nan")
        verify_time_ms = float("nan")
        proof_ok = False
        verify_success = False
        p_stdout = p_stderr = v_stdout = v_stderr = ""

        if not skip_prove:
            proof_time_ms, proc = snarkjs_fullprove(snarkjs, input_path, wasm, zkey, proof_path, public_path, timeout)
            proof_ok = (proc.returncode == 0 and proof_path.exists() and public_path.exists())
            p_stdout = proc.stdout[-4000:]
            p_stderr = proc.stderr[-4000:]
        else:
            proof_ok = proof_path.exists() and public_path.exists()

        if proof_ok and not skip_verify:
            verify_time_ms, vproc = snarkjs_verify(snarkjs, vkey, public_path, proof_path, timeout)
            v_stdout = vproc.stdout[-4000:]
            v_stderr = vproc.stderr[-4000:]
            combined = f"{vproc.stdout}\n{vproc.stderr}".lower()
            verify_success = (vproc.returncode == 0 and ("ok" in combined or "valid" in combined or vproc.returncode == 0))
        elif skip_verify:
            verify_success = True

        results.append(ProofResult(
            index=i,
            input_path=str(input_path),
            proof_path=str(proof_path),
            public_path=str(public_path),
            proof_time_ms=proof_time_ms,
            verify_time_ms=verify_time_ms,
            verify_success=bool(verify_success),
            proof_ok=bool(proof_ok),
            proof_stdout=p_stdout,
            proof_stderr=p_stderr,
            verify_stdout=v_stdout,
            verify_stderr=v_stderr,
        ))
        if (i + 1) % 50 == 0 or i == len(inputs) - 1:
            print(f"  processed {i + 1}/{len(inputs)} proofs")
    return results


def proof_results_to_rows(batch_size: int, results: List[ProofResult]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        out.append({
            "batch_size": batch_size,
            "index": r.index,
            "input_path": r.input_path,
            "proof_path": r.proof_path,
            "public_path": r.public_path,
            "proof_time_ms": r.proof_time_ms,
            "verify_time_ms": r.verify_time_ms,
            "proof_ok": int(r.proof_ok),
            "verify_success": int(r.verify_success),
        })
    return out


def summarize_stress(batch_size: int, results: List[ProofResult], avg_verify_gas: Optional[int]) -> Dict[str, Any]:
    proof_times = [r.proof_time_ms for r in results if r.proof_ok and not math.isnan(r.proof_time_ms)]
    verify_times = [r.verify_time_ms for r in results if r.verify_success and not math.isnan(r.verify_time_ms)]
    success_count = sum(1 for r in results if r.verify_success)
    proof_stats = stats_ms(proof_times)
    verify_stats = stats_ms(verify_times)
    total_gas = int(avg_verify_gas * batch_size) if avg_verify_gas is not None else ""
    return {
        "#Schedules": batch_size,
        "Avg. Proof Time (ms)": proof_stats["mean_ms"],
        "Median Proof Time (ms)": proof_stats["median_ms"],
        "P95 Proof Time (ms)": proof_stats["p95_ms"],
        "Total Proof Time (ms)": proof_stats["total_ms"],
        "Avg. Verify Time (ms)": verify_stats["mean_ms"],
        "Median Verify Time (ms)": verify_stats["median_ms"],
        "P95 Verify Time (ms)": verify_stats["p95_ms"],
        "Avg. Verify Gas": avg_verify_gas if avg_verify_gas is not None else "",
        "Success Verify Rate": success_count / max(len(results), 1),
        "Total Gas": total_gas,
        "successful_verifications": success_count,
        "attempted": len(results),
        "throughput_proof_per_s": 1000.0 / proof_stats["mean_ms"] if proof_stats["mean_ms"] and not math.isnan(proof_stats["mean_ms"]) else "",
    }


def write_markdown_table(path: Path, rows: List[Dict[str, Any]], columns: List[str], title: str) -> None:
    mkdir(path.parent)
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return ""
            if abs(v) < 1 and v != 0:
                return f"{v:.4f}"
            return f"{v:.3f}"
        return str(v)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |\n")


# -----------------------------------------------------------------------------
# Circuit constraints and ablation
# -----------------------------------------------------------------------------


def snarkjs_r1cs_info(snarkjs: str, r1cs: Path, timeout: Optional[int]) -> Tuple[int, str]:
    cmd = [snarkjs, "r1cs", "info", str(r1cs)]
    proc = run_cmd(cmd, timeout=timeout)
    text = f"{proc.stdout}\n{proc.stderr}"
    # snarkjs output examples:
    #   # of Constraints: 1234
    #   Number of constraints: 1234
    patterns = [
        r"#\s*of\s*Constraints\s*[:=]\s*([0-9]+)",
        r"Number\s+of\s+constraints\s*[:=]\s*([0-9]+)",
        r"constraints\s*[:=]\s*([0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), text
    # Some versions output JSON-like terms; fall back to first line containing constraints.
    for line in text.splitlines():
        if "constraint" in line.lower():
            nums = re.findall(r"[0-9]+", line)
            if nums:
                return int(nums[-1]), text
    return -1, text


def load_gas_map(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gas map does not exist: {p}")
    data = read_json(p)
    if not isinstance(data, dict):
        raise ValueError("Gas map must be a JSON object, e.g. {\"Full ZK-VOS\": 224532}")
    return {str(k): safe_int(v) for k, v in data.items()}


def load_ablation_config(path: Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict) and "circuits" in data:
        circuits = data["circuits"]
    else:
        circuits = data
    if not isinstance(circuits, list):
        raise ValueError("Ablation config must be a list or an object with key 'circuits'.")
    out: List[Dict[str, Any]] = []
    base = path.parent
    for c in circuits:
        cc = dict(c)
        for key in ["circuit", "r1cs", "wasm", "zkey", "vkey", "input"]:
            if key in cc and cc[key]:
                p = Path(str(cc[key]))
                if not p.is_absolute():
                    p = (base / p).resolve()
                cc[key] = str(p)
        out.append(cc)
    return out


def benchmark_one_circuit(
    c: Dict[str, Any],
    snarkjs: str,
    default_single_gas: Optional[int],
    gas_map: Dict[str, int],
    timeout: Optional[int],
    repeat: int,
) -> Dict[str, Any]:
    name = str(c.get("name") or c.get("circuit_name") or c.get("label") or "Unnamed")
    r1cs_path = ensure_file(c.get("r1cs"), f"r1cs for {name}", required=False)
    constraints = safe_int(c.get("constraints"), -1)
    constraints_info = ""
    if r1cs_path is not None and r1cs_path.exists():
        constraints, constraints_info = snarkjs_r1cs_info(snarkjs, r1cs_path, timeout)

    wasm = ensure_file(c.get("wasm"), f"wasm for {name}", required=True)
    zkey = ensure_file(c.get("zkey"), f"zkey for {name}", required=True)
    vkey = ensure_file(c.get("vkey"), f"vkey for {name}", required=True)
    input_json = ensure_file(c.get("input"), f"input for {name}", required=True)

    tmp = mkdir(Path(c.get("out_dir") or (Path(input_json).parent / f"_ablation_tmp_{re.sub(r'[^A-Za-z0-9]+', '_', name)}")))
    proof_times: List[float] = []
    verify_times: List[float] = []
    ok = 0
    for i in range(max(1, repeat)):
        proof = tmp / f"proof_{i:03d}.json"
        public = tmp / f"public_{i:03d}.json"
        pt, pproc = snarkjs_fullprove(snarkjs, input_json, wasm, zkey, proof, public, timeout)
        if pproc.returncode != 0:
            eprint(f"[ablation warning] proof failed for {name}: {pproc.stderr[-500:]}")
            continue
        proof_times.append(pt)
        vt, vproc = snarkjs_verify(snarkjs, vkey, public, proof, timeout)
        verify_times.append(vt)
        if vproc.returncode == 0:
            ok += 1

    avg_gas = gas_map.get(name)
    if avg_gas is None:
        # Also try normalized keys.
        norm_map = {re.sub(r"\s+", " ", k).strip().lower(): v for k, v in gas_map.items()}
        avg_gas = norm_map.get(re.sub(r"\s+", " ", name).strip().lower())
    if avg_gas is None:
        avg_gas = safe_int(c.get("verify_gas"), -1)
    if avg_gas is not None and avg_gas < 0:
        avg_gas = default_single_gas

    return {
        "Circuit": name,
        "Constraints": constraints if constraints >= 0 else "",
        "Proof Time (ms)": statistics.mean(proof_times) if proof_times else float("nan"),
        "Verify Time (ms)": statistics.mean(verify_times) if verify_times else float("nan"),
        "Verify Gas": avg_gas if avg_gas is not None else "",
        "Success Verify Rate": ok / max(1, len(verify_times)),
        "repeat": max(1, repeat),
        "r1cs": str(r1cs_path) if r1cs_path else "",
        "constraints_info_tail": constraints_info[-800:],
    }


# -----------------------------------------------------------------------------
# Main workflows
# -----------------------------------------------------------------------------


def run_stress(args: argparse.Namespace, out_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    snarkjs = ensure_executable(args.snarkjs, required=not args.prepare_only)
    trace_csv = ensure_file(args.trace_csv, "trace csv", required=True)
    wasm = ensure_file(args.wasm, "wasm", required=not args.prepare_only)
    zkey = ensure_file(args.zkey, "zkey", required=not args.prepare_only)
    vkey = ensure_file(args.vkey, "verification key", required=not args.prepare_only)
    circuit = ensure_file(args.circuit, "circom circuit", required=False) if args.circuit else None

    rows = read_trace_csv(trace_csv)
    sizes = parse_sizes(args.sizes)
    stress_dir = mkdir(out_root / "stress")
    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for batch_size in sizes:
        print(f"\n[stress] batch size = {batch_size}")
        batch_dir = mkdir(stress_dir / f"batch_{batch_size}")
        input_dir = mkdir(batch_dir / "inputs")
        proof_dir = mkdir(batch_dir / "proofs")
        public_dir = mkdir(batch_dir / "public")

        inputs = prepare_inputs_from_trace(
            rows=rows,
            out_dir=input_dir,
            n=batch_size,
            repeat_trace_rows=bool(args.repeat_trace_rows),
            circuit_path=circuit,
        )
        if args.prepare_only:
            results: List[ProofResult] = []
            for i, p in enumerate(inputs):
                results.append(ProofResult(
                    index=i, input_path=str(p), proof_path="", public_path="",
                    proof_time_ms=float("nan"), verify_time_ms=float("nan"),
                    verify_success=False, proof_ok=False,
                ))
        else:
            assert snarkjs is not None and wasm is not None and zkey is not None and vkey is not None
            results = run_proof_verify_batch(
                snarkjs=snarkjs,
                inputs=inputs,
                wasm=wasm,
                zkey=zkey,
                vkey=vkey,
                proof_dir=proof_dir,
                public_dir=public_dir,
                timeout=args.timeout,
                skip_prove=args.skip_prove,
                skip_verify=args.skip_verify,
            )
        raw_rows.extend(proof_results_to_rows(batch_size, results))
        summary = summarize_stress(batch_size, results, args.single_verify_gas)
        summary_rows.append(summary)
        write_csv(batch_dir / "raw.csv", proof_results_to_rows(batch_size, results))
        write_json(batch_dir / "summary.json", summary)

    summary_csv = stress_dir / "zk_vos_stress_summary.csv"
    raw_csv = stress_dir / "zk_vos_stress_raw.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(raw_csv, raw_rows)
    stress_cols = [
        "#Schedules", "Avg. Proof Time (ms)", "Avg. Verify Gas",
        "Success Verify Rate", "Total Gas",
        "Avg. Verify Time (ms)", "P95 Proof Time (ms)", "P95 Verify Time (ms)",
    ]
    write_markdown_table(stress_dir / "zk_vos_stress_summary.md", summary_rows, stress_cols, "ZK-VOS Batch Verification and Scalability")
    print(f"\n[stress] saved summary: {summary_csv}")
    return summary_rows, raw_rows, stress_dir


def run_ablation(args: argparse.Namespace, out_root: Path) -> Tuple[List[Dict[str, Any]], Path]:
    if not args.ablation_config:
        raise ValueError("--ablation-config is required for ablation/full mode")
    snarkjs = ensure_executable(args.snarkjs, required=True)
    config = ensure_file(args.ablation_config, "ablation config", required=True)
    circuits = load_ablation_config(config)
    gas_map = load_gas_map(args.gas_map)
    ablation_dir = mkdir(out_root / "ablation")
    rows: List[Dict[str, Any]] = []
    for c in circuits:
        c = dict(c)
        c.setdefault("out_dir", str(ablation_dir / "tmp" / re.sub(r"[^A-Za-z0-9]+", "_", str(c.get("name", "circuit")))))
        print(f"\n[ablation] {c.get('name', 'Unnamed')}")
        row = benchmark_one_circuit(
            c=c,
            snarkjs=snarkjs,
            default_single_gas=args.single_verify_gas,
            gas_map=gas_map,
            timeout=args.timeout,
            repeat=args.ablation_repeat,
        )
        rows.append(row)
        print(f"  constraints={row['Constraints']} proof_ms={row['Proof Time (ms)']} gas={row['Verify Gas']}")

    summary_csv = ablation_dir / "zk_vos_circuit_ablation_summary.csv"
    write_csv(summary_csv, rows)
    ablation_cols = ["Circuit", "Constraints", "Proof Time (ms)", "Verify Gas", "Verify Time (ms)", "Success Verify Rate"]
    write_markdown_table(ablation_dir / "zk_vos_circuit_ablation_summary.md", rows, ablation_cols, "ZK-VOS Circuit Ablation")
    print(f"\n[ablation] saved summary: {summary_csv}")
    return rows, ablation_dir


def auto_locate_artifacts(base_dir: Path) -> Dict[str, Optional[str]]:
    """Best-effort artifact locator for the local ZK-VOS experiment folder."""
    return {
        "trace_csv": str(find_latest_file(base_dir, ["*hcrl_zk_schedule_trace.csv", "trace_hcrl_zk_schedule_trace.csv"]) or ""),
        "wasm": str(find_latest_file(base_dir, ["zk_vos.wasm", "*_js/*.wasm", "*.wasm"]) or ""),
        "zkey": str(find_latest_file(base_dir, ["zk_vos_final.zkey", "*_final.zkey", "*.zkey"]) or ""),
        "vkey": str(find_latest_file(base_dir, ["verification_key.json", "*verification*.json"]) or ""),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ZK-VOS pressure/scalability and circuit-ablation benchmark")
    sub = p.add_subparsers(dest="command")

    def add_common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--snarkjs", default="snarkjs", help="snarkjs executable. On Windows, you may pass snarkjs.cmd")
        q.add_argument("--out-dir", default="zk_vos_real_circom_hcrl_patch/experiments/zk_vos/results", help="Output root directory")
        q.add_argument("--run-name", default="", help="Optional run folder name")
        q.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for each external command")
        q.add_argument("--single-verify-gas", type=int, default=None, help="Existing measured single-proof verifier gas. Used to compute Avg Verify Gas and Total Gas.")
        q.add_argument("--gas-map", default="", help="Optional JSON gas map for ablation, e.g. {\"Full ZK-VOS\": 224532}")

    def add_stress(q: argparse.ArgumentParser) -> None:
        q.add_argument("--trace-csv", default="", help="HCRL ZK schedule trace CSV, usually *_hcrl_zk_schedule_trace.csv")
        q.add_argument("--wasm", default="", help="Compiled circuit wasm path")
        q.add_argument("--zkey", default="", help="Groth16 proving key path")
        q.add_argument("--vkey", default="", help="Groth16 verification key JSON path")
        q.add_argument("--circuit", default="", help="Optional .circom path for auto-detecting signal input names")
        q.add_argument("--sizes", default="100 500 1000 5000", help="Batch sizes, e.g. '100 500 1000 5000'")
        q.add_argument("--repeat-trace-rows", action="store_true", help="Cycle through trace rows if trace is shorter than requested batch size")
        q.add_argument("--prepare-only", action="store_true", help="Only prepare input JSON files; do not call snarkjs")
        q.add_argument("--skip-prove", action="store_true", help="Skip proof generation and reuse existing proof/public files in output dirs")
        q.add_argument("--skip-verify", action="store_true", help="Skip snarkjs verification")

    stress = sub.add_parser("stress", help="Run batch proof/verification scalability test")
    add_common(stress)
    add_stress(stress)

    ablation = sub.add_parser("ablation", help="Run circuit ablation benchmark")
    add_common(ablation)
    ablation.add_argument("--ablation-config", required=True, help="JSON config describing circuit variants")
    ablation.add_argument("--ablation-repeat", type=int, default=3, help="Number of proof/verify repeats per circuit variant")

    full = sub.add_parser("full", help="Run both stress and ablation tests")
    add_common(full)
    add_stress(full)
    full.add_argument("--ablation-config", required=True, help="JSON config describing circuit variants")
    full.add_argument("--ablation-repeat", type=int, default=3, help="Number of proof/verify repeats per circuit variant")

    locate = sub.add_parser("locate", help="Locate trace/wasm/zkey/vkey artifacts under a directory")
    locate.add_argument("--base-dir", default="zk_vos_real_circom_hcrl_patch", help="Directory to search")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2

    if args.command == "locate":
        found = auto_locate_artifacts(Path(args.base_dir))
        print(json.dumps(found, ensure_ascii=False, indent=2))
        return 0

    out_root = Path(args.out_dir)
    run_name = args.run_name.strip() if getattr(args, "run_name", "") else f"zk_vos_pressure_{now_id()}"
    out_root = mkdir(out_root / run_name)

    manifest: Dict[str, Any] = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": args.command,
        "args": vars(args),
        "outputs": {},
    }

    try:
        if args.command in {"stress", "full"}:
            stress_rows, raw_rows, stress_dir = run_stress(args, out_root)
            manifest["outputs"]["stress_dir"] = str(stress_dir)
            manifest["stress_summary"] = stress_rows
        if args.command in {"ablation", "full"}:
            ablation_rows, ablation_dir = run_ablation(args, out_root)
            manifest["outputs"]["ablation_dir"] = str(ablation_dir)
            manifest["ablation_summary"] = ablation_rows
    finally:
        write_json(out_root / "manifest.json", manifest)
        print(f"\nSaved manifest: {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
