pragma solidity ^0.5.12;

/**
 * Selection contract for offline-trained oracle-selection policies.
 *
 * Backward compatibility:
 * - ReAction(uint) / SeAction() are kept for the original DQN-only demo.
 *
 * New deployment path:
 * - submitSelection(...) records the full inference result produced by a
 *   chain-off Python policy service that loads trained .npz checkpoints.
 */
contract Selection {
    uint public action;

    struct SelectionResult {
        uint requestId;
        uint methodId;
        uint8 mode;
        uint primaryOracle;
        uint backupOracle;
        bytes32 policyHash;
        address submitter;
        uint timestamp;
        bool exists;
    }

    mapping(uint => SelectionResult) private results;

    event LegacyActionSubmitted(
        uint action,
        address indexed submitter,
        uint timestamp
    );

    event SelectionSubmitted(
        uint indexed requestId,
        uint indexed methodId,
        uint8 mode,
        uint primaryOracle,
        uint backupOracle,
        bytes32 policyHash,
        address indexed submitter,
        uint timestamp
    );

    function ReAction(uint ac) public returns(uint) {
        action = ac;
        emit LegacyActionSubmitted(ac, msg.sender, block.timestamp);
        return action;
    }

    function SeAction() public view returns(uint) {
       return action;
    }

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
        action = primaryOracle;

        results[requestId] = SelectionResult(
            requestId,
            methodId,
            mode,
            primaryOracle,
            backupOracle,
            policyHash,
            msg.sender,
            block.timestamp,
            true
        );

        emit SelectionSubmitted(
            requestId,
            methodId,
            mode,
            primaryOracle,
            backupOracle,
            policyHash,
            msg.sender,
            block.timestamp
        );

        return true;
    }

    function hasSelection(uint requestId) public view returns (bool) {
        return results[requestId].exists;
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
}
