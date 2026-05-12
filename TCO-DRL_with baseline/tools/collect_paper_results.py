"""Collect final result CSV files from output folders into one summary CSV."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="TCO-DRL_with baseline/output")
    parser.add_argument("--out_csv", default="paper_results_summary.csv")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    files = sorted(out_dir.rglob("*_final_results.csv")) if out_dir.exists() else []
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.insert(0, "run_folder", f.parent.name)
            df.insert(1, "source_file", str(f))
            rows.append(df)
        except Exception as exc:
            print(f"Skip {f}: {exc}")
    if not rows:
        print("No final result CSV files found.")
        return
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv} with {len(summary)} rows from {len(files)} files.")


if __name__ == "__main__":
    main()
