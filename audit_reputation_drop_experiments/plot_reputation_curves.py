"""Plot malicious/trusted reputation curves and mark attack onset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="audit_reputation_drop_experiments/output/audit_reputation_curve.csv")
    ap.add_argument("--out", default="audit_reputation_drop_experiments/output/plots")
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for scenario, g in df.groupby("scenario"):
        mean = g.groupby("step")[ ["malicious_rep_mean", "trusted_rep_mean", "reputation_gap"] ].mean().reset_index()
        onset = int(g["attack_onset_step"].iloc[0]) if "attack_onset_step" in g.columns else None
        end = int(g["attack_end_step"].iloc[0]) if "attack_end_step" in g.columns else None

        plt.figure(figsize=(7, 4.2))
        plt.plot(mean["step"], mean["malicious_rep_mean"], label="Malicious reputation")
        plt.plot(mean["step"], mean["trusted_rep_mean"], label="Trusted reputation")
        if onset is not None:
            plt.axvline(onset, linestyle="--", linewidth=1.0, label="Attack onset")
        if end is not None and end < mean["step"].max():
            plt.axvline(end, linestyle=":", linewidth=1.0, label="Attack end")
        plt.xlabel("Request step")
        plt.ylabel("Mean effective reputation")
        plt.title(scenario)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / (scenario + "_reputation_curve.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4.2))
        plt.plot(mean["step"], mean["reputation_gap"], label="Trusted - malicious")
        if onset is not None:
            plt.axvline(onset, linestyle="--", linewidth=1.0, label="Attack onset")
        if end is not None and end < mean["step"].max():
            plt.axvline(end, linestyle=":", linewidth=1.0, label="Attack end")
        plt.xlabel("Request step")
        plt.ylabel("Reputation gap")
        plt.title(scenario + " reputation gap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / (scenario + "_reputation_gap.png"), dpi=200)
        plt.close()

    print("[Done] plots written to %s" % out)


if __name__ == "__main__":
    main()
