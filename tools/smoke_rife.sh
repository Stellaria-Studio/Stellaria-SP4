#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$ROOT/Models/RIFE-safetensors/flownet.safetensors"
ASSET="$ROOT/build/rife_cpp.sp4"

if [[ ! -f "$MODEL" ]]; then
  echo "missing $MODEL"
  echo "copy it with: cp -R \"/Users/minsawa/Documents/Stellaria Motion/Models/RIFE-safetensors\" \"$ROOT/Models/\""
  exit 2
fi

cd "$ROOT"
cmake -S . -B build-sdk
cmake --build build-sdk
./build-sdk/sp4c compile --input "$MODEL" --output "$ASSET" --adapter rife --target metal-m4pro --block-size auto --quality-profile adaptive-rife --residual-ratio 0.02
./build-sdk/sp4inspect-cpp "$ASSET"
python3 -m sp4_toolchain validate "$ASSET" --teacher "$MODEL"
./build-sdk/sp4quantbench "$MODEL" "$ASSET"
