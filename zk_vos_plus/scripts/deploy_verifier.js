const hre = require("hardhat");

async function main() {
  const network = hre.network.name;
  const Verifier = await hre.ethers.getContractFactory("Groth16Verifier");
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();

  const out = {
    network,
    chainId: Number((await hre.ethers.provider.getNetwork()).chainId),
    verifierAddress: await verifier.getAddress(),
    deployedAt: new Date().toISOString()
  };
  console.log(JSON.stringify(out, null, 2));
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
