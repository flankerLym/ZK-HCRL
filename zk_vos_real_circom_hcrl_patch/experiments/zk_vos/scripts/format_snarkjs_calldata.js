const fs = require("fs");

// Usage:
//   node scripts/format_snarkjs_calldata.js proof/valid_calldata_raw.txt proof/valid_calldata.json
// snarkjs generates calldata as a comma-separated JS-like list:
//   ["a0","a1"],[["b00","b01"],["b10","b11"]],["c0","c1"],["pub0",...]

const input = process.argv[2];
const output = process.argv[3];
if (!input || !output) {
  console.error("Usage: node scripts/format_snarkjs_calldata.js <raw_calldata.txt> <output.json>");
  process.exit(1);
}

const raw = fs.readFileSync(input, "utf8").trim();
const parsed = JSON.parse(`[${raw}]`);
const obj = {
  pA: parsed[0],
  pB: parsed[1],
  pC: parsed[2],
  pubSignals: parsed[3]
};
fs.writeFileSync(output, JSON.stringify(obj, null, 2));
console.log(`Wrote ${output}`);
