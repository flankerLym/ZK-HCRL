# ZK-VOS 压力测试与电路消融补丁

本补丁用于补充论文实验中的两张表：

1. **ZK-VOS batch verification / scalability**

| #Schedules | Avg. Proof Time | Avg. Verify Gas | Success Verify Rate | Total Gas |
|---:|---:|---:|---:|---:|
| 100 | ... | ... | ... | ... |
| 500 | ... | ... | ... | ... |
| 1000 | ... | ... | ... | ... |
| 5000 | ... | ... | ... | ... |

2. **ZK-VOS circuit ablation**

| Circuit | Constraints | Proof Time | Verify Gas |
|---|---:|---:|---:|
| Membership only | ... | ... | ... |
| + Cost/Latency | ... | ... | ... |
| + Risk | ... | ... | ... |
| + Audit update | ... | ... | ... |
| Full ZK-VOS | ... | ... | ... |

---

## 1. 覆盖位置

把 zip 解压到仓库根目录 `TCO-DRL/`，会新增/覆盖以下文件：

```text
zk_vos_real_circom_hcrl_patch/
└── experiments/
    └── zk_vos/
        ├── tools/
        │   └── zk_vos_pressure_test.py
        └── pressure_test/
            ├── circuits/
            │   ├── membership_only.circom
            │   ├── cost_latency.circom
            │   ├── risk.circom
            │   ├── audit_update.circom
            │   └── full_zk_vos.circom
            ├── inputs/
            │   ├── membership_only_input.json
            │   ├── cost_latency_input.json
            │   ├── risk_input.json
            │   ├── audit_update_input.json
            │   └── full_zk_vos_input.json
            ├── configs/
            │   ├── ablation_config_template.json
            │   └── gas_map_template.json
            ├── compile_zk_vos_ablation.ps1
            └── compile_zk_vos_ablation.sh

TCO-DRL_with baseline/
├── tools/
│   └── zk_vos_pressure_test.py
└── scripts/
    ├── run_zk_vos_pressure_test.ps1
    └── run_zk_vos_pressure_test.sh
```

其中 `TCO-DRL_with baseline/tools/zk_vos_pressure_test.py` 只是兼容旧路径的 wrapper，真正代码在：

```text
zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py
```

---

## 2. 环境检查

PowerShell：

```powershell
node -v
npm -v
snarkjs --version
```

如果找不到 `snarkjs`：

```powershell
npm install -g snarkjs
```

如果你的 Node 在 NVM 目录，例如：

```text
E:\Develop\node\nvm\v20.9.0
```

当前 PowerShell 临时加入 PATH：

```powershell
$env:Path = "E:\Develop\node\nvm\v20.9.0;$env:APPDATA\npm;$env:Path"
node -v
npm -v
snarkjs --version
```

---

## 3. 自动定位 ZK-VOS 文件

先在仓库根目录运行：

```powershell
python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" locate `
  --base-dir "zk_vos_real_circom_hcrl_patch"
```

它会尝试找到：

```text
trace_csv
wasm
zkey
vkey
```

---

## 4. 跑压力测试表

你已经有单次 on-chain verifier gas 时，推荐直接传入 `--single-verify-gas`。例如你的单次 verifier gas 是 `224532`，命令为：

```powershell
python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" stress `
  --trace-csv "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/data/trace_hcrl_zk_schedule_trace.csv" `
  --wasm "zk_vos_real_circom_hcrl_patch/circuits/zk_vos_js/zk_vos.wasm" `
  --zkey "zk_vos_real_circom_hcrl_patch/circuits/zk_vos_final.zkey" `
  --vkey "zk_vos_real_circom_hcrl_patch/circuits/verification_key.json" `
  --sizes "100 500 1000 5000" `
  --single-verify-gas 224532 `
  --repeat-trace-rows
```

如果你的真实文件路径不同，把 `--wasm`、`--zkey`、`--vkey` 改成 `locate` 输出的路径。

输出位置：

```text
zk_vos_real_circom_hcrl_patch/experiments/zk_vos/results/zk_vos_pressure_YYYYMMDD_HHMMSS/stress/
├── zk_vos_stress_summary.csv
├── zk_vos_stress_summary.md
├── zk_vos_stress_raw.csv
└── batch_*/
```

论文主要使用：

```text
zk_vos_stress_summary.csv
```

---

## 5. 跑电路消融表

### 5.1 编译消融电路

消融电路需要 `circom`、`snarkjs` 和 Powers of Tau 文件。

PowerShell：

```powershell
cd "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/pressure_test"

.\compile_zk_vos_ablation.ps1 `
  -PowersOfTau "path\to\pot12_final.ptau" `
  -Circom "circom" `
  -Snarkjs "snarkjs"
```

Linux/macOS：

```bash
cd zk_vos_real_circom_hcrl_patch/experiments/zk_vos/pressure_test
bash compile_zk_vos_ablation.sh path/to/pot12_final.ptau
```

编译完成后会生成：

```text
pressure_test/build/<circuit_name>/
├── <circuit_name>.r1cs
├── <circuit_name>_js/<circuit_name>.wasm
├── <circuit_name>_final.zkey
└── verification_key.json
```

### 5.2 运行电路消融 benchmark

```powershell
python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" ablation `
  --ablation-config "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/pressure_test/configs/ablation_config_template.json" `
  --single-verify-gas 224532 `
  --ablation-repeat 3
```

输出位置：

```text
zk_vos_real_circom_hcrl_patch/experiments/zk_vos/results/zk_vos_pressure_YYYYMMDD_HHMMSS/ablation/
├── zk_vos_circuit_ablation_summary.csv
└── zk_vos_circuit_ablation_summary.md
```

如果你有每个消融电路的真实 verifier gas，可以填写：

```text
zk_vos_real_circom_hcrl_patch/experiments/zk_vos/pressure_test/configs/gas_map_template.json
```

然后运行：

```powershell
python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" ablation `
  --ablation-config "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/pressure_test/configs/ablation_config_template.json" `
  --gas-map "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/pressure_test/configs/gas_map_template.json" `
  --ablation-repeat 3
```

---

## 6. 一次性跑 stress + ablation

```powershell
python "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/tools/zk_vos_pressure_test.py" full `
  --trace-csv "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/data/trace_hcrl_zk_schedule_trace.csv" `
  --wasm "zk_vos_real_circom_hcrl_patch/circuits/zk_vos_js/zk_vos.wasm" `
  --zkey "zk_vos_real_circom_hcrl_patch/circuits/zk_vos_final.zkey" `
  --vkey "zk_vos_real_circom_hcrl_patch/circuits/verification_key.json" `
  --sizes "100 500 1000 5000" `
  --single-verify-gas 224532 `
  --repeat-trace-rows `
  --ablation-config "zk_vos_real_circom_hcrl_patch/experiments/zk_vos/pressure_test/configs/ablation_config_template.json" `
  --ablation-repeat 3
```

---

## 7. 论文写作建议

压力测试表建议写成：

> To evaluate the scalability of the ZK-VOS verification layer, we measured proof generation and verification overhead under increasing numbers of scheduling decisions. The batch size was varied from 100 to 5000 schedules. For each schedule, a proof was generated from the exported HCRL scheduling trace and verified using the same Groth16 verification key. The total gas was computed by multiplying the measured single-proof verifier gas by the number of verified schedules.

电路消融表建议写成：

> We further conducted a circuit-level ablation to quantify the overhead introduced by each ZK-VOS constraint group. Starting from membership verification, we incrementally added cost/latency constraints, risk constraints, audit-update consistency, and the full ZK-VOS rule set. We report the number of R1CS constraints, proof generation time, and verifier gas for each circuit variant.
