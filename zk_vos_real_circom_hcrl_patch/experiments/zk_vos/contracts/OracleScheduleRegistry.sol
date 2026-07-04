// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IGroth16Verifier8 {
    function verifyProof(
        uint[2] calldata _pA,
        uint[2][2] calldata _pB,
        uint[2] calldata _pC,
        uint[8] calldata _pubSignals
    ) external view returns (bool);
}

contract OracleScheduleRegistry {
    IGroth16Verifier8 public verifier;

    event ScheduleAccepted(uint256 indexed requestId, uint256 indexed selectedOracleHash, uint256 oraclePoolRoot);
    event ScheduleRejected(uint256 indexed requestId, uint256 indexed selectedOracleHash, string reason);

    constructor(address verifierAddress) {
        verifier = IGroth16Verifier8(verifierAddress);
    }

    function submitSchedule(
        uint[2] calldata pA,
        uint[2][2] calldata pB,
        uint[2] calldata pC,
        uint[8] calldata pubSignals
    ) external returns (bool) {
        bool ok = verifier.verifyProof(pA, pB, pC, pubSignals);
        require(ok, "ZK_VOS_INVALID_SCHEDULE_PROOF");

        // pubSignals order follows circuits/zk_vos_full.circom:
        // [requestId, selectedOracleHash, oraclePoolRoot, reputationThreshold,
        //  costBudget, riskBudget, deadline, requestServiceType]
        emit ScheduleAccepted(pubSignals[0], pubSignals[1], pubSignals[2]);
        return true;
    }
}
