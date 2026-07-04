# ZK-VOS PPT Talking Points

## Core innovation

**Zero-Knowledge Verifiable Oracle Scheduling Layer**

HCRL performs intelligent off-chain scheduling, while ZK-VOS proves on-chain that the selected oracle satisfies safety constraints.

## Why it is needed

- HCRL/GNN/Audit are too complex to run directly on-chain.
- Fully off-chain scheduling lacks verifiability.
- ZK-VOS bridges this gap: off-chain computation, on-chain compliance verification.

## What the proof verifies

- Service type matches the request.
- Audit-adjusted reputation is above threshold.
- Cost is within budget.
- Risk is within budget.
- Latency estimate meets the deadline.
- Oracle is not in cooldown.
- Oracle state belongs to the committed oracle pool.

## What it does not claim

- It does not prove HCRL is globally optimal.
- It does not prove the entire neural network inference process.
- It proves scheduling compliance.

## Closed loop

Request → HCRL off-chain scheduling → ZK proof → on-chain verification → oracle execution → audit feedback → next scheduling.
