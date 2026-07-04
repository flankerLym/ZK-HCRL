# ZK-VOS Real Circom Proof + HCRL/Audit Integration Guide

This guide completes the two missing parts:

1. **Real Circom proof**: generate a real Groth16 proof from `zk_vos_full.circom`.
2. **Solidity Groth16 verifier**: export `contracts/Verifier.sol` from snarkjs and verify the proof on Hardhat.

It also explains how to connect your existing HCRL/Audit outputs.

---

## 1. What ZK-VOS verifies

ZK-VOS verifies **scheduling compliance**, not HCRL optimality.

For the selected oracle, the circuit proves:

| Constraint | Meaning |
|---|---|
| `oracleServiceType == requestServiceType` | the selected oracle supports the request type |
| `repEff >= reputationThreshold` | audit-adjusted reputation is above threshold |
| `cost <= costBudget` | cost is within budget |
| `risk <= riskBudget` | risk is within budget |
| `latencyEst <= deadline` | estimated latency satisfies the deadline |
| `cooldown == 0` | the oracle is not in audit cooldown |
| `oracle leaf ∈ oraclePoolRoot` | the selected oracle state belongs to the committed oracle pool |

The proof hides the private oracle attributes and audit-derived values while exposing only public thresholds and commitments.

---

## 2. Required software

The mock verifier test only needs Node.js and Hardhat. You already completed this.

The real proof pipeline needs:

- Node.js 18+
- npm dependencies: `npm install`
- `circom` 2.x in your PATH
- `snarkjs` from npm, already included in `package.json` and callable through `npx snarkjs`

Check:

```powershell
node -v
npm -v
circom --version
npx snarkjs --version
```

If `circom` is not recognized on Windows, install Circom first, then reopen PowerShell.

---

## 3. Run the real proof pipeline

From this folder:

```powershell
cd E:\keyan\code\TCO\TCO-DRL\zk_vos_root_patch\experiments\zk_vos
.\scripts\run_real_proof_pipeline.bat
```

Linux/macOS/Git Bash:

```bash
cd experiments/zk_vos
bash scripts/run_real_proof_pipeline.sh
```

The pipeline will:

1. Generate ZK inputs from `data/sample_oracle_pool.json` and `data/sample_schedules.csv`.
2. Compile `circuits/zk_vos_full.circom`.
3. Run demo Groth16 trusted setup.
4. Generate witness and proof for a valid HCRL-like schedule.
5. Verify the proof off-chain.
6. Confirm invalid schedules fail witness generation.
7. Export `contracts/Verifier.sol`.
8. Export Solidity calldata.
9. Run Hardhat real verifier test.
10. Collect proof artifact metrics.

Outputs:

```text
inputs/metadata.json
inputs/valid_schedule.json
inputs/invalid_*.json
proof/valid_proof.json
proof/valid_public.json
proof/valid_calldata.json
contracts/Verifier.sol
results/offchain_verify_valid.txt
results/invalid_witness_checks.csv
results/real_proof_file_metrics.json
```

---

## 4. How to connect HCRL and Audit

Your original HCRL/Audit experiment should export two things.

### 4.1 Oracle pool snapshot

Create or export a JSON file with the selected-time oracle state:

```json
{
  "oracles": [
    {
      "oracleId": 1,
      "oracleServiceType": 1,
      "repEff": 8200,
      "cost": 120,
      "risk": 80,
      "latencyEst": 40,
      "cooldown": 0
    }
  ]
}
```

Field meanings:

| Field | Source in your method |
|---|---|
| `oracleId` | oracle/node id |
| `oracleServiceType` | service type supported by the oracle |
| `repEff` | audit-adjusted effective reputation; scale to integer, e.g. 0–10000 |
| `cost` | oracle cost; scale to integer |
| `risk` | validation risk or malicious-risk score; scale to integer |
| `latencyEst` | expected response time or latency estimate; integer |
| `cooldown` | audit cooldown flag; 0 means available, 1 means blocked |

The most important bridge is `repEff`: it comes from your audit-aware reputation module.

### 4.2 HCRL schedule output CSV

Export one row per selected schedule:

```csv
requestId,selectedOracleId,requestServiceType,reputationThreshold,costBudget,riskBudget,deadline
1001,1,1,7000,200,150,60
```

Field meanings:

| Field | Source |
|---|---|
| `requestId` | request id / time step id |
| `selectedOracleId` | oracle selected by HCRL |
| `requestServiceType` | request service type |
| `reputationThreshold` | public compliance threshold |
| `costBudget` | public cost budget |
| `riskBudget` | public risk budget |
| `deadline` | public latency/deadline constraint |

Then run:

```powershell
node scripts\generate_inputs.js --pool data\sample_oracle_pool.json --schedules data\sample_schedules.csv --row 0 --levels 8
```

To use your own HCRL files:

```powershell
node scripts\generate_inputs.js --pool path\to\your_oracle_pool.json --schedules path\to\your_hcrl_schedules.csv --row 0 --levels 8
```

---

## 5. If your HCRL log has different column names

Use the converter:

```powershell
python scripts\export_hcrl_schedules_to_zk.py --input path\to\hcrl_log.csv --output data\sample_schedules.csv
```

It accepts common aliases such as:

- `request_id`, `requestId`
- `selected_oracle_id`, `selectedOracleId`, `oracle_id`
- `service_type`, `request_service_type`
- `rep_threshold`, `tau_rep`
- `cost_budget`, `B_cost`
- `risk_budget`, `B_risk`
- `deadline`, `latency_budget`

---

## 6. Paper wording

Use this wording:

> HCRL-Audit-GNN generates off-chain scheduling decisions. ZK-VOS converts a selected schedule into a zero-knowledge compliance proof. The circuit proves that the selected oracle satisfies service matching, audit-adjusted reputation, cost, risk, deadline, cooldown, and pool-membership constraints. The smart contract verifies the proof and accepts only compliant schedules. This establishes a closed loop of off-chain intelligent scheduling, on-chain compliance verification, and audit-based feedback update.

Do not claim:

- ZK-VOS proves HCRL is globally optimal.
- ZK-VOS proves the full neural network inference.

