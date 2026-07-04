const fs = require("fs");
const path = require("path");
function findFirstFile(dataDir, contains, suffix) {
  if (!fs.existsSync(dataDir)) return undefined;
  const files = fs.readdirSync(dataDir).filter(f => f.toLowerCase().includes(contains.toLowerCase()) && f.toLowerCase().endsWith(suffix.toLowerCase())).sort();
  return files.length ? path.join(dataDir, files[0]) : undefined;
}
const base = path.join(__dirname, "..");
const dataDir = path.join(base, "data");
const schedule = findFirstFile(dataDir, "hcrl_zk_schedule_trace", ".csv");
const snapshot = findFirstFile(dataDir, "oracle_pool_snapshot", ".jsonl");
console.log("[Trace data inspection]");
console.log("Data dir:", dataDir);
console.log("Schedule trace:", schedule || "NOT FOUND");
console.log("Oracle snapshot:", snapshot || "NOT FOUND");
if (schedule) {
  const lines = fs.readFileSync(schedule, "utf8").split(/\r?\n/).filter(Boolean);
  console.log("Schedule rows:", Math.max(lines.length - 1, 0));
  console.log("Schedule header:", lines[0]);
}
if (snapshot) {
  const lines = fs.readFileSync(snapshot, "utf8").split(/\r?\n/).filter(Boolean);
  console.log("Snapshot rows:", lines.length);
  console.log("First snapshot:", lines[0]?.slice(0, 500));
}
if (!schedule || !snapshot) process.exit(1);
