const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("OracleScheduleRegistry with MockVerifier", function () {
  async function deployFixture() {
    const MockVerifier = await ethers.getContractFactory("MockVerifier");
    const verifier = await MockVerifier.deploy();
    await verifier.waitForDeployment();

    const Registry = await ethers.getContractFactory("OracleScheduleRegistry");
    const registry = await Registry.deploy(await verifier.getAddress());
    await registry.waitForDeployment();
    return { verifier, registry };
  }

  const pA = [0, 0];
  const pB = [[0, 0], [0, 0]];
  const pC = [0, 0];
  const pubSignals = [1, 12345, 67890, 7000, 500, 300, 120, 1];

  it("accepts a schedule when verifier returns true", async function () {
    const { registry } = await deployFixture();
    await expect(registry.submitSchedule(pA, pB, pC, pubSignals))
      .to.emit(registry, "ScheduleAccepted")
      .withArgs(1, 12345, 67890);
  });

  it("rejects a schedule when verifier returns false", async function () {
    const { verifier, registry } = await deployFixture();
    await verifier.setShouldAccept(false);
    await expect(registry.submitSchedule(pA, pB, pC, pubSignals))
      .to.be.revertedWith("ZK_VOS_INVALID_SCHEDULE_PROOF");
  });
});
