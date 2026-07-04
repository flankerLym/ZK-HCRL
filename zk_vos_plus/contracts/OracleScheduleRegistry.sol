// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./IZKScheduleVerifier.sol";

contract OracleScheduleRegistry {
    struct Proof {
        uint[2] a;
        uint[2][2] b;
        uint[2] c;
    }

    IZKScheduleVerifier public immutable verifier;

    mapping(uint256 => bytes32) public acceptedScheduleCommitment;
    mapping(uint256 => uint256) public acceptedAtBlock;

    event ScheduleAccepted(
        uint256 indexed requestId,
        bytes32 indexed scheduleCommitment,
        bytes32 oraclePoolRoot,
        bytes32 auditStateRoot,
        uint256 gasUsed,
        address indexed submitter
    );

    event ScheduleRejected(
        uint256 indexed requestId,
        bytes32 indexed scheduleCommitment,
        address indexed submitter
    );

    constructor(address verifier_) {
        require(verifier_ != address(0), "verifier is zero");
        verifier = IZKScheduleVerifier(verifier_);
    }

    function submitSchedule(Proof calldata proof, uint[9] calldata pubSignals) external returns (bool) {
        uint256 gasStart = gasleft();
        bool ok = verifier.verifyProof(proof.a, proof.b, proof.c, pubSignals);
        uint256 gasUsed = gasStart - gasleft();

        uint256 requestId = pubSignals[0];
        bytes32 scheduleCommitment = bytes32(pubSignals[8]);

        if (!ok) {
            emit ScheduleRejected(requestId, scheduleCommitment, msg.sender);
            return false;
        }

        acceptedScheduleCommitment[requestId] = scheduleCommitment;
        acceptedAtBlock[requestId] = block.number;

        emit ScheduleAccepted(
            requestId,
            scheduleCommitment,
            bytes32(pubSignals[6]),
            bytes32(pubSignals[7]),
            gasUsed,
            msg.sender
        );
        return true;
    }
}
