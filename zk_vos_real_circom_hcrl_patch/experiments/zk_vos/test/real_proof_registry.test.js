const fs = require("fs");
const path = require("path");
const { expect } = require("chai");
const { ethers } = require("hardhat");

// This test intentionally avoids hardhat-chai-matchers, so it works with the
// stable Hardhat 2 + ethers v5 stack. It checks the receipt manually instead of
// using expect(tx).to.emit(...), which requires @nomicfoundation/hardhat-chai-matchers.

describe("OracleScheduleRegistry with real snarkjs verifier", function () {
  it("submits a real valid proof if generated verifier/calldata exist", async function () {
    const verifierArtifactPath = path.join(__dirname, "../contracts/Verifier.sol");
    const calldataPath = path.join(__dirname, "../proof/valid_calldata.json");

    if (!fs.existsSync(verifierArtifactPath) || !fs.existsSync(calldataPath)) {
      this.skip();
    }

    const Verifier = await ethers.getContractFactory("Groth16Verifier");
    const verifier = await Verifier.deploy();
    const verifierDeployReceipt = await verifier.deployTransaction.wait();
    console.log("Groth16Verifier deployment gasUsed:", verifierDeployReceipt.gasUsed.toString());

    const Registry = await ethers.getContractFactory("OracleScheduleRegistry");
    const registry = await Registry.deploy(verifier.address);
    const registryDeployReceipt = await registry.deployTransaction.wait();
    console.log("OracleScheduleRegistry deployment gasUsed:", registryDeployReceipt.gasUsed.toString());

    const calldata = JSON.parse(fs.readFileSync(calldataPath, "utf8"));
    const tx = await registry.submitSchedule(
      calldata.pA,
      calldata.pB,
      calldata.pC,
      calldata.pubSignals
    );
    const receipt = await tx.wait();
    console.log("submitSchedule with real Groth16 verifier gasUsed:", receipt.gasUsed.toString());

    expect(receipt.status).to.equal(1);

    const accepted = (receipt.events || []).find((e) => e.event === "ScheduleAccepted");
    if (!accepted) {
      const eventNames = (receipt.events || []).map((e) => e.event || e.eventSignature || "unknown");
      throw new Error(`ScheduleAccepted event not found. Events seen: ${eventNames.join(", ")}`);
    }
  });
});
