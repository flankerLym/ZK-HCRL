#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZK-VOS batch proof/verification scalability experiment.

This utility measures the overhead of ZK-VOS scheduling proof generation and/or
Groth16 verification for batch sizes such as 100, 500, 1000, 2000 and 5000.

It is intentionally independent from the HCRL training loop. The expected flow is:

1) Run HCRL-Oracle and export `<run_id>_hcrl_zk_schedule_trace.csv`.
2) Use this script to convert trace rows into circuit input JSON files.
3) Run snarkjs fullprove and/or groth16 verify repeatedly.
4) Save raw timing records, summary CSV/Markdown and optional plots.

The default field mapping matches the HCRLTraceExporter CSV schema. If your
Circom circuit uses different input signal names, pass --field-map-json.
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

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas may be unavailable in minimal envs
    pd = None


DEFAULT_SIZES = [100, 500, 1000, 2000, 5000]

# Map Circom input signal -> candidate CSV columns from HCRLTraceExporter.
# Override with --field-map-json when the real circuit uses different names.
DEFAULT_FIELD_MAP: Dict[str, List[str]] = {
    "selectedOracleId": ["selectedOracleId", "selected_oracle_id"],
    "requestServiceType": ["request_service_type", "requestServiceType"],
    "oracleServiceType": ["oracleServiceType", "oracle_service_type"],
    "repEff": ["repEff_scaled", "repEff"],
    "reputationThreshold": ["reputationThreshold"],
    "cost": ["cost_scaled", "cost"],
    "costBudget": ["costBudget"],
    "risk": ["risk_scaled", "risk"],
    "riskBudget": ["riskBudget"],
    "latency": ["latencyEst_scaled", "latencyEst", "latency"],
    "deadline": ["deadline"],
    "cooldownFlag": ["cooldown_flag", "cooldownFlag", "cooldown"],
}

# A minimal proof/public naming convention created by this script.
PROOF_RE = re.compile(r"proof_(\d+)\.json$")
PUBLIC_RE = re.compile(r"public_(\d+)\.json$")
INPUT_RE = re.compile(r"input_(\d+)\.json$")


@dataclass
class CommandResult:
    ok: bool
    elapsed_ms: float
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class Pair:
    proof: Path
    public: Path


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def natural_key(path: Path) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def ensure_tool(path_or_name: str, required: bool = True) -> str:
    if not path_or_name:
        if required:
            raise FileNotFoundError("Empty executable path")
        return ""
    if os.sep in path_or_name or (os.altsep and os.altsep in path_or_name):
        p = Path(path_or_name).expanduser()
        if p.exists():
            return str(p)
        if required:
            raise FileNotFoundError(f"Executable not found: {p}")
        return str(p)
    found = shutil.which(path_or_name)
    if found:
        return found
    if required:
        raise FileNotFoundError(f"Executable not found in PATH: {path_or_name}")
    return path_or_name


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> CommandResult:
    t0 = time.perf_counter_ns()
    try:
        proc = subprocess.run(
            list(map(str, cmd)),
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        return CommandResult(proc.returncode == 0, elapsed_ms, proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        return CommandResult(False, elapsed_ms, 124, exc.stdout or "", exc.stderr or "timeout")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if pd is not None:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_field_map(path: Optional[Path]) -> Dict[str, List[str]]:
    if not path:
        return dict(DEFAULT_FIELD_MAP)
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for signal, source in data.items():
        if isinstance(source, list):
            out[str(signal)] = [str(x) for x in source]
        else:
            out[str(signal)] = [str(source)]
    return out


def pick_value(row: Dict[str, Any], candidates: Sequence[str], signal: str) -> Any:
    for col in candidates:
        if col in row and row[col] is not None and str(row[col]).strip() != "":
            return row[col]
    raise KeyError(f"Cannot find input signal '{signal}'. Tried CSV columns: {list(candidates)}")


def to_int_like(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0
        return int(round(value))
    s = str(value).strip()
    if s == "":
        return 0
    return int(round(float(s)))


def make_circuit_input(row: Dict[str, Any], field_map: Dict[str, List[str]], as_strings: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for signal, candidates in field_map.items():
        v = to_int_like(pick_value(row, candidates, signal))
        out[signal] = str(v) if as_strings else v
    return out


def expand_rows(rows: List[Dict[str, Any]], n: int, repeat: bool) -> List[Dict[str, Any]]:
    if len(rows) >= n:
        return rows[:n]
    if not repeat:
        raise ValueError(f"Trace has only {len(rows)} rows, but {n} rows are required. Use --repeat-trace-rows to repeat rows.")
    if not rows:
        raise ValueError("Trace CSV contains no rows.")
    out: List[Dict[str, Any]] = []
    while len(out) < n:
        need = n - len(out)
        out.extend(rows[:need])
    return out


def generate_input_jsons(
    trace_csv: Path,
    input_dir: Path,
    n: int,
    field_map: Dict[str, List[str]],
    repeat_rows: bool,
    as_strings: bool,
) -> List[Path]:
    rows = read_csv_rows(trace_csv)
    rows = expand_rows(rows, n, repeat_rows)
    input_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, row in enumerate(rows):
        p = input_dir / f"input_{i:05d}.json"
        p.write_text(json.dumps(make_circuit_input(row, field_map, as_strings), ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(p)
    return paths


def find_indexed_files(directory: Path, pattern: str) -> List[Path]:
    return sorted(directory.glob(pattern), key=natural_key)


def find_proof_public_pairs(proof_dir: Path, public_dir: Path) -> List[Pair]:
    proofs = find_indexed_files(proof_dir, "*.json")
    publics = find_indexed_files(public_dir, "*.json")
    if not proofs:
        raise FileNotFoundError(f"No proof JSON files found under {proof_dir}")
    if not publics:
        raise FileNotFoundError(f"No public JSON files found under {public_dir}")

    # Prefer matching names generated by this script: proof_00000.json <-> public_00000.json.
    public_by_idx: Dict[str, Path] = {}
    for p in publics:
        m = PUBLIC_RE.search(p.name)
        if m:
            public_by_idx[m.group(1)] = p
    pairs: List[Pair] = []
    for proof in proofs:
        m = PROOF_RE.search(proof.name)
        if m and m.group(1) in public_by_idx:
            pairs.append(Pair(proof=proof, public=public_by_idx[m.group(1)]))
    if pairs:
        return sorted(pairs, key=lambda x: natural_key(x.proof))

    # Fallback: pair by sorted order.
    return [Pair(proof=p, public=q) for p, q in zip(proofs, publics)]


def expand_pairs(pairs: List[Pair], n: int, repeat: bool) -> List[Pair]:
    if len(pairs) >= n:
        return pairs[:n]
    if not repeat:
        raise ValueError(f"Only {len(pairs)} proof/public pairs available, but batch size {n} is requested. Use --repeat-fixtures.")
    out: List[Pair] = []
    while len(out) < n:
        need = n - len(out)
        out.extend(pairs[:need])
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize(records: List[Dict[str, Any]], sizes: Sequence[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    phases = sorted({str(r["phase"]) for r in records})
    for phase in phases:
        for size in sizes:
            subset = [r for r in records if r["phase"] == phase and int(r["batch_size"]) == int(size)]
            if not subset:
                continue
            vals = [float(r["elapsed_ms"]) for r in subset]
            total = sum(vals)
            unique_inputs = len({str(r.get("input_path", "")) for r in subset if r.get("input_path")})
            unique_proofs = len({str(r.get("proof_path", "")) for r in subset if r.get("proof_path")})
            out.append({
                "phase": phase,
                "batch_size": int(size),
                "num_ops": len(vals),
                "total_ms": round(total, 6),
                "mean_ms": round(statistics.mean(vals), 6),
                "median_ms": round(percentile(vals, 0.50), 6),
                "p95_ms": round(percentile(vals, 0.95), 6),
                "std_ms": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 6),
                "throughput_ops_per_s": round((len(vals) / total * 1000.0) if total > 0 else 0.0, 6),
                "success_rate": round(sum(1 for r in subset if int(r.get("ok", 0)) == 1) / len(subset), 6),
                "unique_inputs_used": unique_inputs,
                "unique_proofs_used": unique_proofs,
                "paper_usable": int(all(int(r.get("paper_usable", 1)) == 1 for r in subset)),
            })
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_summary_md(path: Path, rows: List[Dict[str, Any]], command_line: str) -> None:
    lines = [
        "# ZK-VOS Batch Verification / Scalability Summary",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Command",
        "",
        "```bash",
        command_line,
        "```",
        "",
        "## Results",
        "",
    ]
    if not rows:
        lines.append("No rows were generated.")
    else:
        headers = ["phase", "batch_size", "total_ms", "mean_ms", "median_ms", "p95_ms", "throughput_ops_per_s", "success_rate", "paper_usable"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    lines.append("")
    lines.append("`paper_usable=0` means dry-run timing was used and must not be reported as real ZK overhead.")
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_plot(summary_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    phases = sorted({str(r["phase"]) for r in summary_rows})
    for phase in phases:
        rows = sorted([r for r in summary_rows if r["phase"] == phase], key=lambda x: int(x["batch_size"]))
        if not rows:
            continue
        x = [int(r["batch_size"]) for r in rows]
        y_total = [float(r["total_ms"]) for r in rows]
        y_mean = [float(r["mean_ms"]) for r in rows]

        plt.figure()
        plt.plot(x, y_total, marker="o")
        plt.xlabel("Batch size (# scheduling proofs)")
        plt.ylabel("Total time (ms)")
        plt.title(f"ZK-VOS {phase}: total overhead")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"zk_vos_{phase}_total_ms.png", dpi=300)
        plt.close()

        plt.figure()
        plt.plot(x, y_mean, marker="o")
        plt.xlabel("Batch size (# scheduling proofs)")
        plt.ylabel("Mean time per proof (ms)")
        plt.title(f"ZK-VOS {phase}: per-proof overhead")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"zk_vos_{phase}_mean_ms.png", dpi=300)
        plt.close()


def prove_inputs(
    input_paths: List[Path],
    proof_dir: Path,
    public_dir: Path,
    wasm: Path,
    zkey: Path,
    snarkjs: str,
    timeout: Optional[int],
    dry_run: bool,
) -> List[Dict[str, Any]]:
    proof_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for i, input_path in enumerate(input_paths):
        proof_path = proof_dir / f"proof_{i:05d}.json"
        public_path = public_dir / f"public_{i:05d}.json"
        if dry_run:
            t0 = time.perf_counter_ns()
            # Deterministic tiny computation only for checking the pipeline.
            _ = hash(input_path.read_text(encoding="utf-8"))
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            proof_path.write_text(json.dumps({"dry_run": True, "i": i}), encoding="utf-8")
            public_path.write_text(json.dumps([str(i)]), encoding="utf-8")
            res = CommandResult(True, elapsed_ms, 0, "", "")
        else:
            cmd = [snarkjs, "groth16", "fullprove", str(input_path), str(wasm), str(zkey), str(proof_path), str(public_path)]
            res = run_cmd(cmd, timeout=timeout)
        records.append({
            "phase": "prove",
            "op_index": i,
            "input_path": str(input_path),
            "proof_path": str(proof_path),
            "public_path": str(public_path),
            "elapsed_ms": round(res.elapsed_ms, 6),
            "ok": int(res.ok),
            "returncode": res.returncode,
            "stderr_tail": (res.stderr or "")[-300:].replace("\n", " "),
            "paper_usable": int(not dry_run),
        })
        if not res.ok:
            eprint(f"[prove failed] index={i}, rc={res.returncode}, stderr={res.stderr[-500:]}")
            break
    return records


def verify_pairs(
    pairs: List[Pair],
    batch_size: int,
    vkey: Path,
    snarkjs: str,
    repeat_fixtures: bool,
    timeout: Optional[int],
    dry_run: bool,
) -> List[Dict[str, Any]]:
    expanded = expand_pairs(pairs, batch_size, repeat_fixtures)
    records: List[Dict[str, Any]] = []
    for i, pair in enumerate(expanded):
        if dry_run:
            t0 = time.perf_counter_ns()
            _ = pair.proof.name + pair.public.name + str(vkey)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            res = CommandResult(True, elapsed_ms, 0, "", "")
        else:
            cmd = [snarkjs, "groth16", "verify", str(vkey), str(pair.public), str(pair.proof)]
            res = run_cmd(cmd, timeout=timeout)
        records.append({
            "phase": "verify",
            "batch_size": int(batch_size),
            "op_index": i,
            "proof_path": str(pair.proof),
            "public_path": str(pair.public),
            "elapsed_ms": round(res.elapsed_ms, 6),
            "ok": int(res.ok),
            "returncode": res.returncode,
            "stderr_tail": (res.stderr or "")[-300:].replace("\n", " "),
            "paper_usable": int(not dry_run),
        })
        if not res.ok:
            eprint(f"[verify failed] batch={batch_size}, index={i}, rc={res.returncode}, stderr={res.stderr[-500:]}")
            break
    return records


def parse_sizes(text: str) -> List[int]:
    out = [int(x.strip()) for x in text.replace(",", " ").split() if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("At least one batch size is required")
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("Batch sizes must be positive")
    return sorted(dict.fromkeys(out))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ZK-VOS batch proof/verification scalability experiment")
    p.add_argument("--mode", choices=["verify", "prove", "full", "prepare-inputs"], default="verify",
                   help="verify: verify existing proof/public files; prove: generate proofs; full: prove then verify; prepare-inputs: only export circuit inputs")
    p.add_argument("--sizes", type=parse_sizes, default=DEFAULT_SIZES,
                   help="Batch sizes, e.g. '100 500 1000 2000 5000' or '100,500,1000,2000,5000'")
    p.add_argument("--trace-csv", type=Path, default=None,
                   help="HCRL trace CSV, usually *_hcrl_zk_schedule_trace.csv")
    p.add_argument("--field-map-json", type=Path, default=None,
                   help="Optional JSON mapping from circuit signal names to trace CSV columns")
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Directory for generated/reused circuit input JSON files")
    p.add_argument("--proof-dir", type=Path, default=None, help="Directory containing/generated proof JSON files")
    p.add_argument("--public-dir", type=Path, default=None, help="Directory containing/generated public signal JSON files")
    p.add_argument("--wasm", type=Path, default=None, help="Compiled circuit wasm, required for prove/full")
    p.add_argument("--zkey", type=Path, default=None, help="Circuit final zkey, required for prove/full")
    p.add_argument("--vkey", type=Path, default=None, help="verification_key.json, required for verify/full")
    p.add_argument("--snarkjs", default="snarkjs", help="snarkjs executable path/name")
    p.add_argument("--out-dir", type=Path, default=Path("output/zk_vos_scalability"), help="Output directory")
    p.add_argument("--repeat-trace-rows", action="store_true",
                   help="Repeat trace rows when trace CSV has fewer rows than max batch size")
    p.add_argument("--repeat-fixtures", action="store_true",
                   help="Repeat proof/public fixtures when fewer unique proofs than a requested batch size")
    p.add_argument("--numeric-json", action="store_true",
                   help="Write circuit input values as JSON numbers instead of strings")
    p.add_argument("--timeout", type=int, default=None, help="Per snarkjs command timeout in seconds")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not call snarkjs. Only checks data flow; results are NOT paper-usable")
    p.add_argument("--no-plots", action="store_true", help="Disable matplotlib plot generation")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    sizes: List[int] = args.sizes if isinstance(args.sizes, list) else DEFAULT_SIZES
    max_n = max(sizes)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"zk_vos_batch_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = args.input_dir or (out_dir / "inputs")
    proof_dir = args.proof_dir or (out_dir / "proofs")
    public_dir = args.public_dir or (out_dir / "public")

    snarkjs = args.snarkjs
    if not args.dry_run:
        snarkjs = ensure_tool(args.snarkjs, required=True)

    raw_records: List[Dict[str, Any]] = []

    # 1) Prepare inputs when needed.
    if args.mode in ["prepare-inputs", "prove", "full"]:
        if not args.trace_csv:
            raise SystemExit("--trace-csv is required for prepare-inputs/prove/full")
        field_map = load_field_map(args.field_map_json)
        input_paths = generate_input_jsons(
            trace_csv=args.trace_csv,
            input_dir=input_dir,
            n=max_n,
            field_map=field_map,
            repeat_rows=args.repeat_trace_rows,
            as_strings=not args.numeric_json,
        )
        print(f"[inputs] wrote {len(input_paths)} circuit input JSON files to {input_dir}")
        if args.mode == "prepare-inputs":
            manifest = {
                "mode": args.mode,
                "sizes": sizes,
                "num_inputs": len(input_paths),
                "input_dir": str(input_dir),
                "trace_csv": str(args.trace_csv),
                "field_map": field_map,
            }
            (out_dir / "zk_vos_batch_scalability_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[done] manifest: {out_dir / 'zk_vos_batch_scalability_manifest.json'}")
            return 0
    else:
        input_paths = []

    # 2) Prove once up to max_n. Summary is computed for each prefix size.
    if args.mode in ["prove", "full"]:
        if not args.dry_run:
            if not args.wasm or not args.wasm.exists():
                raise SystemExit(f"--wasm is required and must exist for prove/full: {args.wasm}")
            if not args.zkey or not args.zkey.exists():
                raise SystemExit(f"--zkey is required and must exist for prove/full: {args.zkey}")
        prove_records_all = prove_inputs(
            input_paths=input_paths,
            proof_dir=proof_dir,
            public_dir=public_dir,
            wasm=args.wasm or Path("dry_run.wasm"),
            zkey=args.zkey or Path("dry_run.zkey"),
            snarkjs=snarkjs,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
        # Attach batch_size for prefix-based reporting.
        for size in sizes:
            for r in prove_records_all[:size]:
                rr = dict(r)
                rr["batch_size"] = size
                raw_records.append(rr)
        print(f"[prove] completed {sum(int(r['ok']) for r in prove_records_all)}/{len(prove_records_all)} proofs")

    # 3) Verify for each requested batch size.
    if args.mode in ["verify", "full"]:
        if not args.dry_run:
            if not args.vkey or not args.vkey.exists():
                raise SystemExit(f"--vkey is required and must exist for verify/full: {args.vkey}")
        vkey = args.vkey or Path("dry_run_verification_key.json")
        pairs = find_proof_public_pairs(proof_dir, public_dir)
        print(f"[verify] found {len(pairs)} proof/public pairs")
        for size in sizes:
            recs = verify_pairs(
                pairs=pairs,
                batch_size=size,
                vkey=vkey,
                snarkjs=snarkjs,
                repeat_fixtures=args.repeat_fixtures,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            raw_records.extend(recs)
            ok = sum(int(r["ok"]) for r in recs)
            print(f"[verify] batch={size}: {ok}/{len(recs)} ok")

    # 4) Save output.
    raw_csv = out_dir / "zk_vos_batch_scalability_raw.csv"
    summary_csv = out_dir / "zk_vos_batch_scalability_summary.csv"
    summary_md = out_dir / "zk_vos_batch_scalability_summary.md"
    manifest_json = out_dir / "zk_vos_batch_scalability_manifest.json"

    summary_rows = summarize(raw_records, sizes)
    write_csv(raw_csv, raw_records)
    write_csv(summary_csv, summary_rows)
    write_summary_md(summary_md, summary_rows, " ".join([Path(sys.argv[0]).name] + sys.argv[1:]))
    if not args.no_plots:
        maybe_plot(summary_rows, out_dir)

    manifest = {
        "mode": args.mode,
        "sizes": sizes,
        "dry_run": bool(args.dry_run),
        "paper_usable": int(not args.dry_run),
        "trace_csv": str(args.trace_csv) if args.trace_csv else None,
        "input_dir": str(input_dir),
        "proof_dir": str(proof_dir),
        "public_dir": str(public_dir),
        "vkey": str(args.vkey) if args.vkey else None,
        "wasm": str(args.wasm) if args.wasm else None,
        "zkey": str(args.zkey) if args.zkey else None,
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "num_raw_records": len(raw_records),
        "num_summary_rows": len(summary_rows),
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[saved] raw:     {raw_csv}")
    print(f"[saved] summary: {summary_csv}")
    print(f"[saved] md:      {summary_md}")
    print(f"[saved] manifest:{manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
