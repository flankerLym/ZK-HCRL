from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def find_final_csvs(root: Path):
    return sorted(root.rglob("*_final_results.csv"), key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="experiments_real_trace/output")
    ap.add_argument("--out", type=str, default="experiments_real_trace/output/real_trace_matched_summary.csv")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for p in find_final_csvs(root):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        run_dir = p.parent.name
        phase = "test" if "test" in str(p).lower() else ("train" if "train" in str(p).lower() else "unknown")
        for _, r in df.iterrows():
            rec = r.to_dict()
            rec["phase"] = phase
            rec["run_dir"] = run_dir
            rec["result_csv"] = str(p)
            rows.append(rec)
    if not rows:
        print(f"No final result CSVs found under {root}")
        return
    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path} rows={len(out)}")
    cols = [c for c in ["phase", "method", "success_rate", "success_time_rate", "avg_responseT", "cost_per_success", "malicious_rate", "trusted_rate", "primary_malicious_rate", "any_selected_malicious_rate", "primary_trusted_rate", "any_selected_trusted_rate", "backup_recovery_rate", "audit_fail_rate", "run_dir"] if c in out.columns]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
