// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MockVerifier {
    bool public shouldAccept = true;

    function setShouldAccept(bool value) external {
        shouldAccept = value;
    }

    function verifyProof(
        uint[2] calldata,
        uint[2][2] calldata,
        uint[2] calldata,
        uint[8] calldata
    ) external view returns (bool) {
        return shouldAccept;
    }
}
