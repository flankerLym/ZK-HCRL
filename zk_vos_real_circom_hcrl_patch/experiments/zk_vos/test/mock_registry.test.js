const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("OracleScheduleRegistry with MockVerifier", function () {
  async function deployFixture() {
    const MockVerifier = await ethers.getContractFactory("MockVerifier");
    const verifier = await MockVerifier.deploy();
    await verifier.deployed();

    const Registry = await ethers.getContractFactory("OracleScheduleRegistry");
    const registry = await Registry.deploy(verifier.address);
    await registry.deployed();
    return { verifier, registry };
  }

  const pA = [0, 0];
  const pB = [[0, 0], [0, 0]];
  const pC = [0, 0];
  const pubSignals = [1, 12345, 67890, 7000, 500, 300, 120, 1];

  it("accepts a schedule when verifier returns true", async function () {
    const { registry } = await deployFixture();
    const tx = await registry.submitSchedule(pA, pB, pC, pubSignals);
    const receipt = await tx.wait();

    console.log("Mock submitSchedule accepted gasUsed:", receipt.gasUsed.toString());
    expect(receipt.status).to.equal(1);

    const accepted = receipt.events.find((e) => e.event === "ScheduleAccepted");
    expect(accepted).to.not.equal(undefined);
    expect(accepted.args.requestId.toString()).to.equal("1");
    expect(accepted.args.selectedOracleHash.toString()).to.equal("12345");
    expect(accepted.args.oraclePoolRoot.toString()).to.equal("67890");
  });

  it("rejects a schedule when verifier returns false", async function () {
    const { verifier, registry } = await deployFixture();
    await verifier.setShouldAccept(false);

    let reverted = false;
    try {
      const tx = await registry.submitSchedule(pA, pB, pC, pubSignals);
      await tx.wait();
    } catch (err) {
      reverted = true;
      expect(String(err.message)).to.include("ZK_VOS_INVALID_SCHEDULE_PROOF");
    }
    expect(reverted).to.equal(true);
  });
});
