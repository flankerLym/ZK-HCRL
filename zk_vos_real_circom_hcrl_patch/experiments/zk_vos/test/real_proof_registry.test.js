const fs = require("fs");
const path = require("path");
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("OracleScheduleRegistry with real snarkjs verifier", function () {
  it("submits a real valid proof if generated verifier/calldata exist", async function () {
    const verifierArtifactPath = path.join(__dirname, "../contracts/Verifier.sol");
    const calldataPath = path.join(__dirname, "../proof/valid_calldata.json");

    if (!fs.existsSync(verifierArtifactPath) || !fs.existsSync(calldataPath)) {
      this.skip();
    }

    const Verifier = await ethers.getContractFactory("Groth16Verifier");
    const verifier = await Verifier.deploy();
    await verifier.deployed();

    const Registry = await ethers.getContractFactory("OracleScheduleRegistry");
    const registry = await Registry.deploy(verifier.address);
    await registry.deployed();

    const calldata = JSON.parse(fs.readFileSync(calldataPath, "utf8"));
    await expect(registry.submitSchedule(calldata.pA, calldata.pB, calldata.pC, calldata.pubSignals))
      .to.emit(registry, "ScheduleAccepted");
  });
});
