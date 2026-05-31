#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed ZK-VOS pressure/scalability benchmark.

Why this version exists
-----------------------
The previous pressure-test run produced 0 successful proofs because the generated
input JSON did not match the real `zk_vos_full.circom` witness schema.  The full
circuit requires Poseidon-derived public inputs and Merkle membership data:

  selectedOracleHash = Poseidon(oracleId)
  leaf = Poseidon(oracleId, oracleServiceType, repEff, cost, risk, latencyEst, cooldown)
  oraclePoolRoot = MerkleRoot(leaf, pathElements, pathIndices)

The old script filled missing fields with 0 when `--circuit` was not passed, so
the constraints in the full circuit could not be satisfied.  This patched script
uses the accompanying Node.js helper `zk_vos_enrich_full_inputs.js` to generate
valid full-circuit inputs with circomlibjs Poseidon.

Main outputs
------------
1) ZK-VOS batch scalability:
   #Schedules | Avg. Proof Time | Avg. Verify Time | Avg. Verify Gas | Success Verify Rate | Total Gas

2) Optional circuit ablation:
   Circuit | Constraints | Proof Time | Verify Time | Verify Gas | Success Verify Rate

Recommended command
-------------------
python zk_vos_pressure_test.py stress ^
  --snarkjs E:\\OpenClaw\\snarkjs.cmd ^
  --trace-csv zk_vos_real_circom_hcrl_patch\\experiments\\zk_vos\\data\\trace_hcrl_zk_schedule_trace.csv ^
  --wasm zk_vos_real_circom_hcrl_patch\\experiments\\zk_vos\\build\\zk_vos_full_js\\zk_vos_full.wasm ^
  --zkey zk_vos_real_circom_hcrl_patch\\experiments\\zk_vos\\build\\zk_vos_full_final.zkey ^
  --vkey zk_vos_real_circom_hcrl_patch\\experiments\\zk_vos\\build\\verification_key.json ^
  --sizes "100 500 1000 5000" ^
  --single-verify-gas 272132 ^
  --repeat-trace-rows

For overhead benchmarking, the default `--valid-witness-mode clamp` keeps trace-derived
values but clamps invalid rows into a satisfiable witness.  Use `--valid-witness-mode strict`
to preserve raw trace compliance exactly; invalid trace rows may fail proof generation.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ----------------------------- generic helpers -----------------------------


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("No batch sizes were provided")
    return out


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


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
    if os.name == "nt" and not str(path_or_name).lower().endswith(".cmd"):
        found = shutil.which(f"{path_or_name}.cmd")
        if found:
            return found
    if required:
        raise FileNotFoundError(
            f"Executable not found in PATH: {path_or_name}. "
            "Install it or pass an explicit path, e.g. --snarkjs E:/OpenClaw/snarkjs.cmd."
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
    return subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        shell=False,
    )


def timed_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Tuple[float, subprocess.CompletedProcess]:
    t0 = time.perf_counter()
    proc = run_cmd(cmd, cwd=cwd, timeout=timeout)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, proc


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


def find_latest_file(root: Path, patterns: Sequence[str]) -> Optional[Path]:
    if not root.exists():
        return None
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(root.rglob(pat))
    files = [p for p in candidates if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


# ----------------------------- trace conversion -----------------------------


def read_trace_csv(trace_csv: Path) -> List[Dict[str, str]]:
    with open(trace_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Trace CSV is empty: {trace_csv}")
    return rows


def row_first(row: Dict[str, Any], keys: Sequence[str], default: Any = 0) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def trace_row_to_base_values(row: Dict[str, Any], index: int = 0, valid_witness_mode: str = "clamp") -> Dict[str, int]:
    rep = safe_int(row_first(row, ["repEff_scaled", "rep_scaled", "reputation_scaled"], None), -1)
    if rep < 0:
        rep = safe_int(safe_float(row_first(row, ["repEff", "rep", "reputation"], 0.0)) * 10000)

    cost = safe_int(row_first(row, ["cost_scaled"], None), -1)
    if cost < 0:
        c = safe_float(row_first(row, ["cost"], 0.0))
        cost = safe_int(c * 1000) if c < 100.0 else safe_int(c)

    risk = safe_int(row_first(row, ["risk_scaled"], None), -1)
    if risk < 0:
        r = safe_float(row_first(row, ["risk"], 0.0))
        risk = safe_int(r * 10000) if r <= 1.0 else safe_int(r)

    latency = safe_int(row_first(row, ["latencyEst_scaled", "latency_scaled"], None), -1)
    if latency < 0:
        lt = safe_float(row_first(row, ["latencyEst", "latency", "final_duration"], 0.0))
        latency = safe_int(lt * 1000) if lt < 100.0 else safe_int(lt)

    rep_th = safe_int(row_first(row, ["reputationThreshold", "repThreshold", "reputation_threshold"], 6000))
    cost_budget = safe_int(row_first(row, ["costBudget", "cost_budget"], 1000))
    risk_budget = safe_int(row_first(row, ["riskBudget", "risk_budget"], 600))
    deadline = safe_int(row_first(row, ["deadline", "deadline_scaled"], 6000))

    selected = safe_int(row_first(row, ["selectedOracleId", "oracleId", "selected_oracle_id"], index))
    request_id = safe_int(row_first(row, ["request_id", "requestId"], index))

    request_service_type = safe_int(row_first(row, ["request_service_type", "requestServiceType"], 0))
    oracle_service_type = safe_int(row_first(row, ["oracleServiceType", "oracle_service_type"], request_service_type))
    cooldown = safe_int(row_first(row, ["cooldown_flag", "cooldownFlag", "cooldown"], 0))

    # For benchmark mode, ensure the witness satisfies the circuit so proof time and
    # verification time can be measured reliably.  Strict mode preserves raw values.
    if valid_witness_mode == "clamp":
        rep = max(rep, rep_th)
        cost = min(cost, cost_budget)
        risk = min(risk, risk_budget)
        latency = min(latency, deadline)
        cooldown = 0
        oracle_service_type = request_service_type
    elif valid_witness_mode == "filter":
        # filtering happens before conversion; here keep raw values
        pass
    elif valid_witness_mode == "strict":
        pass
    else:
        raise ValueError(f"Unknown valid_witness_mode: {valid_witness_mode}")

    return {
        "requestId": request_id,
        "oracleId": selected,
        "selectedOracleId": selected,
        "requestServiceType": request_service_type,
        "oracleServiceType": oracle_service_type,
        "repEff": rep,
        "cost": cost,
        "risk": risk,
        "latencyEst": latency,
        "cooldown": cooldown,
        "reputationThreshold": rep_th,
        "costBudget": cost_budget,
        "riskBudget": risk_budget,
        "deadline": deadline,
    }


def is_trace_row_satisfiable(row: Dict[str, Any]) -> bool:
    b = trace_row_to_base_values(row, 0, valid_witness_mode="strict")
    return (
        b["repEff"] >= b["reputationThreshold"]
        and b["cost"] <= b["costBudget"]
        and b["risk"] <= b["riskBudget"]
        and b["latencyEst"] <= b["deadline"]
        and b["cooldown"] == 0
        and b["oracleServiceType"] == b["requestServiceType"]
    )


def helper_path_near_script() -> Path:
    return Path(__file__).resolve().parent / "zk_vos_enrich_full_inputs.js"


def prepare_inputs_from_trace(
    rows: List[Dict[str, str]],
    out_dir: Path,
    n: int,
    repeat_trace_rows: bool,
    valid_witness_mode: str,
    node: str,
    helper_js: Path,
    timeout: Optional[int],
) -> List[Path]:
    if valid_witness_mode == "filter":
        filtered = [r for r in rows if is_trace_row_satisfiable(r)]
        if not filtered:
            raise ValueError("No satisfiable rows found in trace under --valid-witness-mode filter.")
        rows = filtered

    if n > len(rows) and not repeat_trace_rows:
        raise ValueError(
            f"Requested {n} schedules but trace has only {len(rows)} usable rows. "
            "Use --repeat-trace-rows to cycle through the trace."
        )

    mkdir(out_dir)
    base_jsonl = out_dir / "_base_inputs.jsonl"
    with open(base_jsonl, "w", encoding="utf-8") as f:
        for i in range(n):
            row = rows[i % len(rows)]
            obj = trace_row_to_base_values(row, i, valid_witness_mode=valid_witness_mode)
            obj["index"] = i
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if not helper_js.exists():
        raise FileNotFoundError(f"Missing helper JS: {helper_js}")

    cmd = [
        node,
        str(helper_js),
        "--jsonl", str(base_jsonl),
        "--out-dir", str(out_dir),
        "--levels", "8",
    ]
    proc = run_cmd(cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to enrich full-circuit inputs with Poseidon/Merkle fields.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}\n"
            "Make sure `npm install` has been run in experiments/zk_vos so circomlibjs is available."
        )

    paths = [out_dir / f"input_{i:06d}.json" for i in range(n)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Input helper did not create all expected files. Missing first entries: {missing[:5]}")
    return paths


# ----------------------------- snarkjs execution ----------------------------


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
            verify_success = (vproc.returncode == 0)
        elif skip_verify:
            verify_success = True

        if not proof_ok and (i < 3 or (i + 1) % 50 == 0):
            eprint(f"[proof failed] index={i} input={input_path}")
            if p_stderr:
                eprint(p_stderr[-1000:])
            elif p_stdout:
                eprint(p_stdout[-1000:])

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


def proof_results_to_rows(batch_size: int, results: List[ProofResult], include_logs: bool = True) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        row = {
            "batch_size": batch_size,
            "index": r.index,
            "input_path": r.input_path,
            "proof_path": r.proof_path,
            "public_path": r.public_path,
            "proof_time_ms": r.proof_time_ms,
            "verify_time_ms": r.verify_time_ms,
            "proof_ok": int(r.proof_ok),
            "verify_success": int(r.verify_success),
        }
        if include_logs:
            row["proof_error_tail"] = (r.proof_stderr or r.proof_stdout)[-1000:]
            row["verify_error_tail"] = (r.verify_stderr or r.verify_stdout)[-1000:]
        out.append(row)
    return out


def summarize_stress(batch_size: int, results: List[ProofResult], avg_verify_gas: Optional[int]) -> Dict[str, Any]:
    proof_times = [r.proof_time_ms for r in results if r.proof_ok and not math.isnan(r.proof_time_ms)]
    verify_times = [r.verify_time_ms for r in results if r.verify_success and not math.isnan(r.verify_time_ms)]
    success_count = sum(1 for r in results if r.verify_success)
    proof_count = sum(1 for r in results if r.proof_ok)
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
        "Proof Success Rate": proof_count / max(len(results), 1),
        "Total Gas": total_gas,
        "successful_proofs": proof_count,
        "successful_verifications": success_count,
        "attempted": len(results),
        "throughput_proof_per_s": 1000.0 / proof_stats["mean_ms"] if proof_stats["mean_ms"] and not math.isnan(proof_stats["mean_ms"]) else "",
    }


# ----------------------------- ablation support -----------------------------


def snarkjs_r1cs_info(snarkjs: str, r1cs: Path, timeout: Optional[int]) -> Tuple[int, str]:
    proc = run_cmd([snarkjs, "r1cs", "info", str(r1cs)], timeout=timeout)
    text = f"{proc.stdout}\n{proc.stderr}"
    for pat in [
        r"#\s*of\s*Constraints\s*[:=]\s*([0-9]+)",
        r"Number\s+of\s+constraints\s*[:=]\s*([0-9]+)",
        r"constraints\s*[:=]\s*([0-9]+)",
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), text
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
        raise ValueError('Gas map must be a JSON object, e.g. {"Full ZK-VOS": 272132}')
    return {str(k): safe_int(v) for k, v in data.items()}


def load_ablation_config(path: Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    circuits = data["circuits"] if isinstance(data, dict) and "circuits" in data else data
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
            eprint(f"[ablation warning] proof failed for {name}: {(pproc.stderr or pproc.stdout)[-1000:]}")
            continue
        proof_times.append(pt)
        vt, vproc = snarkjs_verify(snarkjs, vkey, public, proof, timeout)
        verify_times.append(vt)
        if vproc.returncode == 0:
            ok += 1

    norm_map = {re.sub(r"\s+", " ", k).strip().lower(): v for k, v in gas_map.items()}
    avg_gas = gas_map.get(name)
    if avg_gas is None:
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


# ----------------------------- workflows -----------------------------


def run_stress(args: argparse.Namespace, out_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
    snarkjs = ensure_executable(args.snarkjs, required=not args.prepare_only)
    node = ensure_executable(args.node, required=True)
    trace_csv = ensure_file(args.trace_csv, "trace csv", required=True)
    wasm = ensure_file(args.wasm, "wasm", required=not args.prepare_only)
    zkey = ensure_file(args.zkey, "zkey", required=not args.prepare_only)
    vkey = ensure_file(args.vkey, "verification key", required=not args.prepare_only)
    helper_js = ensure_file(args.input_helper, "input helper JS", required=True)

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
            valid_witness_mode=args.valid_witness_mode,
            node=node,
            helper_js=helper_js,
            timeout=args.timeout,
        )

        if args.prepare_only:
            results = [
                ProofResult(i, str(p), "", "", float("nan"), float("nan"), False, False)
                for i, p in enumerate(inputs)
            ]
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

        batch_raw = proof_results_to_rows(batch_size, results, include_logs=True)
        raw_rows.extend(batch_raw)
        summary = summarize_stress(batch_size, results, args.single_verify_gas)
        summary_rows.append(summary)
        write_csv(batch_dir / "raw.csv", batch_raw)
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


def auto_locate_artifacts(base_dir: Path) -> Dict[str, str]:
    return {
        "trace_csv": str(find_latest_file(base_dir, ["*hcrl_zk_schedule_trace.csv", "trace_hcrl_zk_schedule_trace.csv"]) or ""),
        "wasm": str(find_latest_file(base_dir, ["zk_vos_full.wasm", "zk_vos.wasm", "*_js/*.wasm", "*.wasm"]) or ""),
        "zkey": str(find_latest_file(base_dir, ["zk_vos_full_final.zkey", "zk_vos_final.zkey", "*_final.zkey", "*.zkey"]) or ""),
        "vkey": str(find_latest_file(base_dir, ["verification_key.json", "*verification*.json"]) or ""),
        "helper": str(find_latest_file(base_dir, ["zk_vos_enrich_full_inputs.js"]) or ""),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed ZK-VOS pressure/scalability and circuit-ablation benchmark")
    sub = p.add_subparsers(dest="command")

    def add_common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--snarkjs", default="snarkjs", help="snarkjs executable. On Windows, pass E:/OpenClaw/snarkjs.cmd if needed.")
        q.add_argument("--node", default="node", help="Node.js executable")
        q.add_argument("--out-dir", default="zk_vos_real_circom_hcrl_patch/experiments/zk_vos/results", help="Output root directory")
        q.add_argument("--run-name", default="", help="Optional run folder name")
        q.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for each external command")
        q.add_argument("--single-verify-gas", type=int, default=None, help="Measured submitSchedule gas; used to compute Avg Verify Gas and Total Gas.")
        q.add_argument("--gas-map", default="", help='Optional JSON gas map, e.g. {"Full ZK-VOS": 272132}')

    def add_stress(q: argparse.ArgumentParser) -> None:
        q.add_argument("--trace-csv", default="", help="HCRL ZK schedule trace CSV, usually *_hcrl_zk_schedule_trace.csv")
        q.add_argument("--wasm", default="", help="Compiled full circuit wasm path")
        q.add_argument("--zkey", default="", help="Groth16 proving key path")
        q.add_argument("--vkey", default="", help="Groth16 verification key JSON path")
        q.add_argument("--input-helper", default=str(helper_path_near_script()), help="Node helper used to compute Poseidon/Merkle full-circuit inputs")
        q.add_argument("--sizes", default="100 500 1000 5000", help="Batch sizes, e.g. '100 500 1000 5000'")
        q.add_argument("--repeat-trace-rows", action="store_true", help="Cycle through trace rows if trace is shorter than requested batch size")
        q.add_argument("--valid-witness-mode", choices=["clamp", "filter", "strict"], default="clamp",
                       help="clamp: make each trace-derived row satisfiable for overhead benchmark; filter: use only already satisfiable rows; strict: preserve raw trace values.")
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

    locate = sub.add_parser("locate", help="Locate trace/wasm/zkey/vkey/helper artifacts under a directory")
    locate.add_argument("--base-dir", default="zk_vos_real_circom_hcrl_patch", help="Directory to search")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2

    if args.command == "locate":
        print(json.dumps(auto_locate_artifacts(Path(args.base_dir)), ensure_ascii=False, indent=2))
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
        print(f"Saved manifest: {out_root / 'manifest.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
