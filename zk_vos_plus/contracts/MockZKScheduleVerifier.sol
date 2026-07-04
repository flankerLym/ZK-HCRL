// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Local smoke-test verifier only. Do NOT use this contract for paper experiments.
contract MockZKScheduleVerifier {
    bool public result = true;

    function setResult(bool result_) external {
        result = result_;
    }

    function verifyProof(
        uint[2] calldata,
        uint[2][2] calldata,
        uint[2] calldata,
        uint[9] calldata
    ) external view returns (bool) {
        return result;
    }
}
