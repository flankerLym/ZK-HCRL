#!/usr/bin/env python3
"""
Convert HCRL schedule logs into the simplified ZK-VOS CSV format.

Expected input columns can be flexible. The script looks for common names:

- request_id or requestId
- selected_oracle_id or selectedOracleId or oracle_id
- request_service_type or requestServiceType or service_type
- reputation_threshold or reputationThreshold
- cost_budget or costBudget
- risk_budget or riskBudget
- deadline

It writes:

requestId,selectedOracleId,requestServiceType,reputationThreshold,costBudget,riskBudget,deadline

This script does not generate ZK Merkle paths. It prepares schedule records that can be combined
with an oracle pool file by scripts/generate_inputs.js or a custom input builder.
"""
import argparse
import pandas as pd

ALIASES = {
    "requestId": ["requestId", "request_id", "req_id", "id"],
    "selectedOracleId": ["selectedOracleId", "selected_oracle_id", "oracle_id", "selected_oracle"],
    "requestServiceType": ["requestServiceType", "request_service_type", "service_type", "req_type"],
    "reputationThreshold": ["reputationThreshold", "reputation_threshold", "rep_threshold", "tau_rep"],
    "costBudget": ["costBudget", "cost_budget", "budget_cost", "B_cost"],
    "riskBudget": ["riskBudget", "risk_budget", "budget_risk", "B_risk"],
    "deadline": ["deadline", "latency_budget", "B_latency"]
}

DEFAULTS = {
    "reputationThreshold": 7000,
    "costBudget": 500,
    "riskBudget": 300,
    "deadline": 120
}

def pick_column(df, canonical):
    for c in ALIASES[canonical]:
        if c in df.columns:
            return df[c]
    if canonical in DEFAULTS:
        return pd.Series([DEFAULTS[canonical]] * len(df))
    raise ValueError(f"Missing required column for {canonical}. Tried: {ALIASES[canonical]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out = pd.DataFrame({canonical: pick_column(df, canonical) for canonical in ALIASES})
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(out)} rows")

if __name__ == "__main__":
    main()
