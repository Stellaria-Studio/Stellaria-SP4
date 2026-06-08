#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$ROOT/Models/RIFE-safetensors/flownet.safetensors"
ASSET="${1:-$ROOT/build/rife_cpp.sp4}"

if [[ ! -f "$MODEL" ]]; then
  echo "missing $MODEL"
  exit 2
fi

cd "$ROOT"
cmake -S . -B build-sdk
cmake --build build-sdk

./build-sdk/sp4c compile \
  --input "$MODEL" \
  --output "$ASSET" \
  --adapter rife \
  --target metal-m4pro \
  --block-size auto \
  --quality-profile "${SP4_QUALITY_PROFILE:-adaptive-rife}" \
  --residual-ratio "${SP4_RESIDUAL_RATIO:-0.02}"

python3 -m sp4_toolchain validate "$ASSET" --teacher "$MODEL"
./build-sdk/sp4metal-validate "$ASSET"

xcrun -sdk macosx metal -I shaders/metal \
  -c shaders/metal/sp4_reconstruct.metal \
  -o build-sdk/sp4_reconstruct.air
xcrun -sdk macosx metallib build-sdk/sp4_reconstruct.air \
  -o build-sdk/sp4_reconstruct.metallib
xcrun -sdk macosx metal -I shaders/metal \
  -c shaders/metal/sp4_block_w4a8.metal \
  -o build-sdk/sp4_block_w4a8.air
xcrun -sdk macosx metallib build-sdk/sp4_block_w4a8.air \
  -o build-sdk/sp4_block_w4a8.metallib
xcrun -sdk macosx metal -I shaders/metal \
  -c shaders/metal/sp4_compare_kernels.metal \
  -o build-sdk/sp4_compare_kernels.air
xcrun -sdk macosx metallib build-sdk/sp4_compare_kernels.air \
  -o build-sdk/sp4_compare_kernels.metallib
xcrun -sdk macosx metal -I shaders/metal \
  -c shaders/metal/sp4_mlp_ssm.metal \
  -o build-sdk/sp4_mlp_ssm.air
xcrun -sdk macosx metallib build-sdk/sp4_mlp_ssm.air \
  -o build-sdk/sp4_mlp_ssm.metallib

slangc shaders/slang/sp4_reconstruct.slang \
  -I shaders/slang \
  -entry sp4ReconstructA1P \
  -target spirv \
  -o build-sdk/sp4_reconstruct.spv
slangc shaders/slang/sp4_reconstruct.slang \
  -I shaders/slang \
  -entry sp4ReconstructA1P \
  -target metal \
  -o build-sdk/sp4_reconstruct_from_slang.metal
xcrun -sdk macosx metal -c build-sdk/sp4_reconstruct_from_slang.metal \
  -o build-sdk/sp4_reconstruct_from_slang.air
xcrun -sdk macosx metallib build-sdk/sp4_reconstruct_from_slang.air \
  -o build-sdk/sp4_reconstruct_from_slang.metallib

slangc shaders/slang/sp4_block_w4a8.slang \
  -I shaders/slang \
  -entry sp4Conv2DBlockW4A8 \
  -target spirv \
  -o build-sdk/sp4_block_w4a8.spv
slangc shaders/slang/sp4_block_w4a8.slang \
  -I shaders/slang \
  -entry sp4Conv2DBlockW4A8 \
  -target metal \
  -o build-sdk/sp4_block_w4a8_from_slang.metal
xcrun -sdk macosx metal -c build-sdk/sp4_block_w4a8_from_slang.metal \
  -o build-sdk/sp4_block_w4a8_from_slang.air
xcrun -sdk macosx metallib build-sdk/sp4_block_w4a8_from_slang.air \
  -o build-sdk/sp4_block_w4a8_from_slang.metallib

slangc shaders/slang/sp4_mlp_ssm.slang \
  -I shaders/slang \
  -entry sp4LinearTile16W4A8 \
  -target spirv \
  -o build-sdk/sp4_mlp_linear.spv
slangc shaders/slang/sp4_mlp_ssm.slang \
  -I shaders/slang \
  -entry sp4MlpGatedTile16W4A8 \
  -target spirv \
  -o build-sdk/sp4_mlp_gate.spv
slangc shaders/slang/sp4_mlp_ssm.slang \
  -I shaders/slang \
  -entry sp4SsmProjectionW4A8 \
  -target spirv \
  -o build-sdk/sp4_ssm_projection.spv
slangc shaders/slang/sp4_mlp_ssm.slang \
  -I shaders/slang \
  -entry sp4SsmScanFp16 \
  -target spirv \
  -o build-sdk/sp4_ssm_scan.spv

./build-sdk/sp4quantbench "$MODEL" "$ASSET"
