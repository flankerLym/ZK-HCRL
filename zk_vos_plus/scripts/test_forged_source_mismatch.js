const fs = require("fs");
const { execFileSync } = require("child_process");

function run(cmd, args, opts = {}) {
  return execFileSync(cmd, args, { stdio: "pipe", env: { ...process.env, ...(opts.env || {}) } }).toString();
}

try {
  const forged = run("node", ["scripts/build_sample_witness.js"], { env: { FORGE_SOURCE_MISMATCH: "1" } });
  fs.mkdirSync("testnet_artifacts", { recursive: true });
  fs.writeFileSync("testnet_artifacts/forged_source_mismatch_input.json", forged);
  console.log("Forged input written to testnet_artifacts/forged_source_mismatch_input.json");
  console.log("Now trying snarkjs fullprove; it should fail.");

  try {
    run("snarkjs", [
      "groth16", "fullprove",
      "testnet_artifacts/forged_source_mismatch_input.json",
      "testnet_artifacts/ZKVOSPlus_js/ZKVOSPlus.wasm",
      "testnet_artifacts/ZKVOSPlus_final.zkey",
      "testnet_artifacts/forged_proof.json",
      "testnet_artifacts/forged_public.json"
    ]);
    console.error("ERROR: forged-source mismatch unexpectedly produced a proof.");
    process.exit(1);
  } catch (err) {
    console.log("PASS: forged-source mismatch was rejected by circuit constraints.");
  }
} catch (err) {
  console.error(err.stdout?.toString() || err.stderr?.toString() || err.message);
  process.exit(1);
}
