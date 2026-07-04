# Paper Text Insert: Commitment-Bound ZK-VOS+

## Suggested replacement contribution

We design ZK-VOS+, a commitment-bound zero-knowledge compliance verification mechanism for audit-aware DeFi oracle scheduling. Unlike a plain constraint checker that only proves private numerical values satisfy thresholds, ZK-VOS+ verifies both schedule compliance and identity-state consistency. Each selected oracle is bound to a registered oracle-pool leaf and an audit-state leaf through Merkle membership proofs. The circuit enforces that the selected oracle identity is consistent across the submitted schedule, registered service attributes, and audit-derived trust states, and then verifies service compatibility, cost, latency, risk, reputation, and cooldown constraints.

## Suggested Section 6 text

### 6.X Commitment-Bound Two-Layer ZK-VOS+

ZK-VOS+ is designed to verify the legality of an off-chain oracle schedule without proving the full HCRL inference process. The verification target is decomposed into two layers. The first layer checks schedule compliance: the selected oracle must satisfy the service-type requirement, cost budget, latency deadline, risk budget, reputation threshold, and cooldown constraint. The second layer checks identity-state consistency: the risk, reputation, cooldown, cost, latency, and service attributes used in the compliance check must be bound to the selected oracle identity.

To achieve this, ZK-VOS+ introduces two public commitments. The oracle-pool commitment `oraclePoolRoot` is a Merkle root over registered oracle attributes, including oracle identity, address/public-key hash, service type, cost, latency, capacity, and stake. The audit-state commitment `auditStateRoot` is a Merkle root over dynamic trust states, including oracle identity, reputation, risk score, cooldown state, audit pass/failure counters, and update epoch. For each selected primary or backup oracle, the prover supplies private Merkle membership paths for both roots. The circuit enforces that

```text
selectedOracleId = oraclePoolLeaf.oracleId = auditStateLeaf.oracleId.
```

This rule prevents a forged-source attack where the scheduler selects a high-risk oracle while privately using another oracle's reputation or risk values as witness data. After identity-state consistency is established, ZK-VOS+ verifies the compliance constraints:

```text
risk <= riskBudget,
reputation >= reputationThreshold,
cooldown == 0,
totalCost <= costBudget,
latency <= latencyBudget,
serviceType == requestType.
```

For backup-enabled schedules, the proof additionally verifies backup membership, backup audit-state membership, primary-backup identity separation, and backup compliance constraints. The public output is a schedule commitment bound to the request identifier, execution mode, backup flag, selected primary and backup identities, oracle-pool root, and audit-state root. Therefore, the on-chain verifier learns that the submitted schedule is valid under committed oracle and audit states, without observing private risk scores, audit evidence, or policy outputs.

## Suggested experiment paragraph

We further evaluate ZK-VOS+ on a public EVM-compatible testnet. The Groth16 verifier and OracleScheduleRegistry contracts are deployed on Polygon Amoy or Sepolia, while the scheduler, oracle workers, witness generator, and proof generator run off-chain. For each replayed DeFi price request, the scheduler constructs a commitment-bound witness, generates a ZK-VOS+ proof, and submits the proof to the on-chain registry. We report proof-generation time, on-chain verification gas, transaction confirmation time, and end-to-end latency from request arrival to on-chain schedule acceptance. We also evaluate forged-source rejection by intentionally selecting one oracle while using another oracle's audit state in the witness; ZK-VOS+ rejects such proofs due to identity-state inconsistency.

## Suggested limitation wording

ZK-VOS+ verifies schedule compliance and identity-state consistency under committed oracle-pool and audit-state roots. It does not prove the full HCRL neural inference process. This design deliberately avoids the high overhead of ZKML-style neural-network verification and focuses on the business-critical legality of submitted oracle schedules. Future work may extend the proof layer to audit-state transition correctness and lightweight deterministic HCRL inference verification.
