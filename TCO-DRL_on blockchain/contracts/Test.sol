pragma solidity ^0.5.12;

contract Selection {
    struct SelectionResult {
        uint requestId;
        uint methodId;
        uint8 mode;
        uint primaryOracle;
        uint backupOracle;
        bytes32 policyHash;
        address submitter;
        uint timestamp;
    }

    mapping(uint => SelectionResult) public results;

    event SelectionSubmitted(
        uint indexed requestId,
        uint indexed methodId,
        uint8 mode,
        uint primaryOracle,
        uint backupOracle,
        bytes32 policyHash,
        address submitter,
        uint timestamp
    );

    function submitSelection(
        uint requestId,
        uint methodId,
        uint8 mode,
        uint primaryOracle,
        uint backupOracle,
        bytes32 policyHash
    )
        public
        returns (bool)
    {
        results[requestId] = SelectionResult(
            requestId,
            methodId,
            mode,
            primaryOracle,
            backupOracle,
            policyHash,
            msg.sender,
            now
        );

        emit SelectionSubmitted(
            requestId,
            methodId,
            mode,
            primaryOracle,
            backupOracle,
            policyHash,
            msg.sender,
            now
        );

        return true;
    }

    function getSelection(uint requestId)
        public
        view
        returns (
            uint methodId,
            uint8 mode,
            uint primaryOracle,
            uint backupOracle,
            bytes32 policyHash,
            address submitter,
            uint timestamp
        )
    {
        SelectionResult storage r = results[requestId];

        return (
            r.methodId,
            r.mode,
            r.primaryOracle,
            r.backupOracle,
            r.policyHash,
            r.submitter,
            r.timestamp
        );
    }

    // Backward-compatible interface for the original code.
    uint action;

    function ReAction(uint ac) public returns(uint) {
        action = ac;
        return action;
    }

    function SeAction() public view returns(uint) {
        return action;
    }
}