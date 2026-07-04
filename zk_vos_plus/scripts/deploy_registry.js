const hre = require("hardhat");

async function main() {
  const network = hre.network.name;
  const useMock = process.env.USE_MOCK_VERIFIER === "1";
  let verifierAddress = process.env.VERIFIER_ADDRESS;

  if (useMock) {
    const Mock = await hre.ethers.getContractFactory("MockZKScheduleVerifier");
    const mock = await Mock.deploy();
    await mock.waitForDeployment();
    verifierAddress = await mock.getAddress();
    console.log(`[deploy] MockZKScheduleVerifier: ${verifierAddress}`);
  }

  if (!verifierAddress) {
    throw new Error("VERIFIER_ADDRESS is required. Generate contracts/ZKVOSPlusVerifier.sol with snarkjs and deploy it first, or set USE_MOCK_VERIFIER=1 for local smoke tests only.");
  }

  const Registry = await hre.ethers.getContractFactory("OracleScheduleRegistry");
  const registry = await Registry.deploy(verifierAddress);
  await registry.waitForDeployment();

  const out = {
    network,
    chainId: Number((await hre.ethers.provider.getNetwork()).chainId),
    verifierAddress,
    registryAddress: await registry.getAddress(),
    deployedAt: new Date().toISOString()
  };
  console.log(JSON.stringify(out, null, 2));
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
