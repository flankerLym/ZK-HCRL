const fs = require("fs");
const path = require("path");

const base = path.join(__dirname, "..");
const build = path.join(base, "build");
const proof = path.join(base, "proof");
const results = path.join(base, "results");
fs.mkdirSync(results, { recursive: true });

function size(file) {
  return fs.existsSync(file) ? fs.statSync(file).size : 0;
}

function readOneLine(file) {
  return fs.existsSync(file) ? fs.readFileSync(file, "utf8").trim().split(/\r?\n/).slice(-1)[0] : "";
}

const metrics = {
  r1cs_bytes: size(path.join(build, "zk_vos_full.r1cs")),
  wasm_bytes: size(path.join(build, "zk_vos_full_js", "zk_vos_full.wasm")),
  zkey_bytes: size(path.join(build, "zk_vos_full_final.zkey")),
  proof_json_bytes: size(path.join(proof, "valid_proof.json")),
  public_json_bytes: size(path.join(proof, "valid_public.json")),
  calldata_json_bytes: size(path.join(proof, "valid_calldata.json")),
  offchain_verify_result: readOneLine(path.join(results, "offchain_verify_valid.txt"))
};
fs.writeFileSync(path.join(results, "real_proof_file_metrics.json"), JSON.stringify(metrics, null, 2));
fs.writeFileSync(path.join(results, "real_proof_file_metrics.csv"), Object.entries(metrics).map(([k,v]) => `${k},${v}`).join("\n"));
console.log(metrics);
