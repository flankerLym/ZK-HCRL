# ZK-VOS+: Two-Layer Commitment-Bound ZK Verification and Public-Testnet Deployment

This overlay adds a stronger ZK layer for the ZK-HCRL Oracle project. It is designed to be extracted into the repository root and used independently of the original Truffle/Ganache code.

## What this patch adds

**Layer 1: Compliance validity**

The circuit verifies that the selected oracle schedule satisfies:

- `risk <= riskBudget`
- `reputation >= reputationThreshold`
- `cooldown == 0`
- `totalCost <= costBudget`
- `latency <= latencyBudget`
- `serviceType == requestType`

For backup-enabled schedules, the same reputation/risk/cooldown/service/latency checks are applied to the backup oracle, and `primaryOracleId != backupOracleId` is enforced.

**Layer 2: Identity-state consistency**

The circuit verifies that the values used in Layer 1 are not free-floating private numbers. They are bound to committed state roots:

- `oraclePoolRoot`: Merkle root of registered oracle attributes.
- `auditStateRoot`: Merkle root of dynamic audit/trust states.

The proof enforces:

```text
selectedOracleId == oraclePoolLeaf.oracleId == auditStateLeaf.oracleId
```

This prevents forged-source attacks where a scheduler selects a bad oracle but privately injects a good oracle's reputation/risk/cooldown values.

## Directory layout

```text
zk_vos_plus/
├── circuits/ZKVOSPlus.circom              # Two-layer ZK-VOS+ circuit
├── contracts/OracleScheduleRegistry.sol   # On-chain registry calling Groth16 verifier
├── contracts/IZKScheduleVerifier.sol      # Interface for snarkjs-generated verifier
├── contracts/MockZKScheduleVerifier.sol   # Local smoke-test verifier only
├── scripts/compile_circuit.sh             # circom/snarkjs compile + verifier export
├── scripts/build_sample_witness.js        # Generates valid and forged witness inputs
├── scripts/deploy_registry.js             # Deploy registry to Polygon Amoy/Sepolia
├── scripts/run_public_replay.js           # Generate proof and submit to testnet registry
├── scripts/test_forged_source_mismatch.js # Negative forged-source test
└── data/testnet_replay_sample.json         # Minimal replay list
```

## Installation

```bash
cd zk_vos_plus
npm install
npm install -g snarkjs
# install circom separately if it is not installed on your machine
```

## Compile circuit and export Solidity verifier

```bash
cd zk_vos_plus
bash scripts/compile_circuit.sh
npm run compile:sol
```

This produces:

```text
testnet_artifacts/ZKVOSPlus_js/ZKVOSPlus.wasm
testnet_artifacts/ZKVOSPlus_final.zkey
contracts/ZKVOSPlusVerifier.sol
```

## Generate valid and forged inputs

```bash
npm run input:sample
npm run input:forged
```

The forged input intentionally mismatches the primary oracle identity and audit state. It should fail proof generation.

```bash
npm run test:forged
```

## Deploy to Polygon Amoy or Sepolia

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Fill:

```text
PRIVATE_KEY=0x...
AMOY_RPC_URL=https://rpc-amoy.polygon.technology/
SEPOLIA_RPC_URL=https://rpc.sepolia.org
```

First deploy the snarkjs-generated verifier contract `contracts/ZKVOSPlusVerifier.sol`:

```bash
npm run deploy:verifier:amoy
# or
npm run deploy:verifier:sepolia
```

Copy the printed verifier address and set:

```text
VERIFIER_ADDRESS=0xYourGeneratedVerifier
```

Deploy registry:

```bash
npm run deploy:amoy
# or
npm run deploy:sepolia
```

Set the printed registry address:

```text
REGISTRY_ADDRESS=0xYourOracleScheduleRegistry
```

Run public-testnet replay:

```bash
npm run input:sample
npm run run:replay:amoy
```

The replay script writes transaction hashes, gas usage, proof-generation time, confirmation time, and accepted status to:

```text
testnet_artifacts/polygonAmoy/public_replay_*.json
```

## Paper experiment to add

Add a new experiment named:

**Public-Testnet Deployment of Commitment-Bound ZK-VOS+**

Report at least:

| Metric | Local Hardhat | Polygon Amoy or Sepolia |
|---|---:|---:|
| Circuit constraints | | |
| Proof generation time | | |
| Verification gas | | |
| Transaction confirmation time | | |
| End-to-end latency | | |
| Accepted legal schedules | | |
| Rejected forged-source schedules | | |

The most important negative test is:

```text
selectedOracleId = o_bad
risk / reputation / cooldown = o_good
```

The new circuit must reject this because it enforces identity consistency between the selected oracle, oracle-pool leaf, and audit-state leaf.
