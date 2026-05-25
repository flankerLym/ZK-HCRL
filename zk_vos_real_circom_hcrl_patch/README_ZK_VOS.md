# ZK-VOS: Zero-Knowledge Verifiable Oracle Scheduling Experiment

This patch adds a standalone **ZK-VOS** experiment folder for the `hcrl` branch of `TCO-DRL`.
It is designed to be extracted into the **repository root**.

ZK-VOS does **not** replace the existing HCRL/Audit/GNN experiments. It adds a deployment-oriented layer:

> HCRL schedules off-chain, while ZK-VOS proves on-chain that the selected oracle satisfies public safety constraints.

## What ZK-VOS proves

For a selected oracle, the zero-knowledge circuit proves the following compliance constraints:

| Constraint | Meaning |
|---|---|
| `oracleServiceType == requestServiceType` | the selected oracle supports the request type |
| `repEff >= reputationThreshold` | audit-adjusted reputation is above threshold |
| `cost <= costBudget` | cost is within budget |
| `risk <= riskBudget` | risk is within budget |
| `latencyEst <= deadline` | expected latency meets the deadline |
| `cooldown == 0` | oracle is not in audit cooldown |
| `oracle state leaf ∈ oraclePoolRoot` | selected oracle state is committed in the oracle pool Merkle root |

The circuit does **not** prove that HCRL is globally optimal. It only proves that the submitted schedule is compliant.

## Directory layout

```text
experiments/zk_vos/
  circuits/zk_vos_full.circom              # Circom circuit
  contracts/OracleScheduleRegistry.sol     # On-chain registry + verifier call
  contracts/MockVerifier.sol               # Mock verifier for Hardhat smoke test
  scripts/generate_inputs.js               # Build valid/invalid ZK inputs
  scripts/format_snarkjs_calldata.js       # Convert snarkjs calldata to JSON
  scripts/run_full_pipeline.sh             # Circom/snarkjs full proof pipeline
  scripts/run_full_pipeline.bat            # Windows helper
  scripts/export_hcrl_schedules_to_zk.py   # Convert HCRL logs to ZK schedule CSV
  test/mock_registry.test.js               # Hardhat mock verifier test
  test/real_proof_registry.test.js         # Optional real proof verifier test
  data/sample_oracle_pool.json             # Synthetic oracle pool
  data/sample_schedules.csv                # Sample HCRL-like schedule records
  inputs/                                  # Generated inputs
  results/                                 # Gas/proof reports
scripts/run_zk_vos_experiment.sh
run_zk_vos_experiment.bat
```

## Quick start: mock on-chain prototype only

This mode does not require Circom. It checks the Solidity registry logic using a mock verifier.

```bash
cd experiments/zk_vos
npm install
npx hardhat test test/mock_registry.test.js
```

Windows PowerShell:

```powershell
cd experiments\zk_vos
npm install
npx hardhat test test\mock_registry.test.js
```

## Full ZK proof pipeline

Requirements:

- Node.js 18+
- `circom` 2.x installed and available in `PATH`
- `snarkjs`
- npm dependencies from `package.json`

Run:

```bash
cd experiments/zk_vos
npm install
bash scripts/run_full_pipeline.sh
```

Windows:

```powershell
cd experiments\zk_vos
npm install
scripts\run_full_pipeline.bat
```

The full pipeline will:

1. Generate valid and invalid ZK inputs.
2. Compile the Circom circuit.
3. Run Groth16 trusted setup for the demo circuit.
4. Generate a valid proof.
5. Verify the proof off-chain.
6. Export a Solidity verifier.
7. Run Hardhat tests.

## Using your own HCRL schedule logs

If your HCRL experiment exports records such as selected oracle, reputation, cost, risk, latency, deadline, cooldown, and service type, convert them using:

```bash
python experiments/zk_vos/scripts/export_hcrl_schedules_to_zk.py \
  --input path/to/hcrl_schedule_log.csv \
  --output experiments/zk_vos/data/hcrl_schedules_for_zk.csv
```

Expected columns are documented in `experiments/zk_vos/scripts/export_hcrl_schedules_to_zk.py`.

## How to report the experiment in the paper

Suggested new experiment section:

**ZK-VOS On-chain Verifiability Experiment**

Report:

- valid proof acceptance rate
- invalid schedule rejection rate
- witness/proof generation time
- proof size
- Solidity verification gas
- transaction gas for `submitSchedule`
- scalability under different oracle pool sizes

This supports the claim:

> Complex scheduling is kept off-chain, while smart contracts can verify scheduling compliance on-chain through lightweight zero-knowledge proofs.

## Real Circom proof and HCRL/Audit integration update

This updated patch adds:

```text
experiments/zk_vos/README_ZK_VOS_REAL_PROOF_AND_HCRL.md
experiments/zk_vos/scripts/run_real_proof_pipeline.bat
experiments/zk_vos/scripts/run_real_proof_pipeline.sh
experiments/zk_vos/scripts/check_invalid_witnesses.js
experiments/zk_vos/scripts/collect_real_proof_metrics.js
```

After your mock verifier smoke test succeeds, run the real proof pipeline:

```powershell
cd experiments\zk_vos
.\scripts\run_real_proof_pipeline.bat
```

This generates a real Circom witness, Groth16 proof, Solidity `Verifier.sol`, calldata, and Hardhat real verifier test. The script also checks that invalid schedules fail circuit constraints.

To connect your own HCRL/Audit outputs, export:

1. an oracle pool snapshot containing `repEff`, `cost`, `risk`, `latencyEst`, `cooldown`, and service type;
2. an HCRL schedule CSV containing `selectedOracleId`, request type, and public thresholds.

See `experiments/zk_vos/README_ZK_VOS_REAL_PROOF_AND_HCRL.md` for the exact schema.
