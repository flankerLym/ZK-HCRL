# ZK-VOS Experiment Folder

This folder implements **Zero-Knowledge Verifiable Oracle Scheduling** for HCRL-Oracle.

## Main idea

- HCRL, Audit, and GNN run off-chain.
- ZK-VOS proves on-chain that the selected oracle satisfies scheduling safety constraints.
- The proof hides private oracle attributes while exposing only public thresholds and commitments.

## Fast smoke test

```bash
npm install
npx hardhat test test/mock_registry.test.js
```

## Full proof demo

```bash
npm install
bash scripts/run_full_pipeline.sh
```

The full proof demo requires `circom` in your PATH.

## Paper wording

> We introduce ZK-VOS, a zero-knowledge verifiable scheduling layer. Instead of proving the global optimality of the HCRL policy, ZK-VOS proves that a selected oracle satisfies service-type, audit reputation, budget, latency, cooldown, and membership constraints. This enables smart contracts to verify off-chain scheduling compliance with low on-chain overhead.
