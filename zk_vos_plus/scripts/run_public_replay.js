const fs = require("fs");
const path = require("path");
const snarkjs = require("snarkjs");
const hre = require("hardhat");
const { groth16ToSolidityCalldata, normalizePublicSignals } = require("./proof_utils");

async function main() {
  const registryAddress = process.env.REGISTRY_ADDRESS;
  if (!registryAddress) throw new Error("REGISTRY_ADDRESS is required in .env");

  const wasmPath = process.env.WASM_PATH || "testnet_artifacts/ZKVOSPlus_js/ZKVOSPlus.wasm";
  const zkeyPath = process.env.ZKEY_PATH || "testnet_artifacts/ZKVOSPlus_final.zkey";
  const replayPath = process.env.REPLAY_JSON || "data/testnet_replay_sample.json";

  const replay = JSON.parse(fs.readFileSync(replayPath, "utf8"));
  const registry = await hre.ethers.getContractAt("OracleScheduleRegistry", registryAddress);
  const rows = [];

  for (const [idx, item] of replay.entries()) {
    const inputFile = item.input || "testnet_artifacts/sample_input.json";
    const input = JSON.parse(fs.readFileSync(inputFile, "utf8"));

    const t0 = Date.now();
    const { proof, publicSignals } = await snarkjs.groth16.fullProve(input, wasmPath, zkeyPath);
    const proofMs = Date.now() - t0;

    const calldataProof = groth16ToSolidityCalldata(proof);
    const pubSignals = normalizePublicSignals(publicSignals);

    const sendAt = Date.now();
    const tx = await registry.submitSchedule(calldataProof, pubSignals);
    const receipt = await tx.wait();
    const confirmMs = Date.now() - sendAt;

    rows.push({
      idx,
      requestId: pubSignals[0],
      txHash: receipt.hash,
      blockNumber: receipt.blockNumber,
      gasUsed: receipt.gasUsed.toString(),
      proofMs,
      confirmationMs: confirmMs,
      accepted: receipt.status === 1
    });
    console.log(JSON.stringify(rows[rows.length - 1]));
  }

  const outDir = path.join("testnet_artifacts", hre.network.name);
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `public_replay_${Date.now()}.json`);
  fs.writeFileSync(outPath, JSON.stringify(rows, null, 2));
  console.log(`Saved replay metrics to ${outPath}`);
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
