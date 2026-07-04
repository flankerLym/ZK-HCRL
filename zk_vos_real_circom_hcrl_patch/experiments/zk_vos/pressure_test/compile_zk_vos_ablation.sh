#!/usr/bin/env bash
set -euo pipefail

PTAU=${1:-${POWERS_OF_TAU:-}}
CIRCOM=${CIRCOM:-circom}
SNARKJS=${SNARKJS:-snarkjs}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIRCUITS="$ROOT/circuits"
BUILD="$ROOT/build"
mkdir -p "$BUILD"

if [[ -z "$PTAU" || ! -f "$PTAU" ]]; then
  echo "Usage: $0 path/to/potXX_final.ptau" >&2
  exit 1
fi

for name in membership_only cost_latency risk audit_update full_zk_vos; do
  file="$CIRCUITS/${name}.circom"
  out="$BUILD/$name"
  mkdir -p "$out"
  echo "[compile] $name"
  "$CIRCOM" "$file" --r1cs --wasm --sym -o "$out"
  "$SNARKJS" groth16 setup "$out/$name.r1cs" "$PTAU" "$out/${name}_0000.zkey"
  "$SNARKJS" zkey contribute "$out/${name}_0000.zkey" "$out/${name}_final.zkey" --name="zk-vos-ablation-$name" -v -e="zk-vos-ablation-fixed-entropy-$name"
  "$SNARKJS" zkey export verificationkey "$out/${name}_final.zkey" "$out/verification_key.json"
  "$SNARKJS" r1cs info "$out/$name.r1cs"
done

echo "Ablation circuits compiled under: $BUILD"
