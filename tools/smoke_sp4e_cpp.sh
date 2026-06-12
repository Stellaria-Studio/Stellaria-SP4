#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-"$ROOT/build-sdk"}"

cmake -S "$ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"

python3 - <<'PY'
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

build = Path("build-sdk")
build.mkdir(exist_ok=True)
rng = np.random.default_rng(20260612)
save_file({
    "block0.conv0.0.0.weight": rng.normal(0, 0.08, (4, 3, 3, 3)).astype("float32"),
    "block0.conv0.0.0.bias": rng.normal(0, 0.02, (4,)).astype("float32"),
    "mlp.up_proj.weight": rng.normal(0, 0.05, (8, 4)).astype("float32"),
    "mlp.up_proj.bias": rng.normal(0, 0.01, (8,)).astype("float32"),
}, str(build / "sp4e_smoke.safetensors"))
PY

"$BUILD_DIR/sp4ec" compile \
  --input "$ROOT/build-sdk/sp4e_smoke.safetensors" \
  --output "$ROOT/build-sdk/sp4e_smoke.sp4e" \
  --adapter generic \
  --residual-ratio 0.08

"$BUILD_DIR/sp4einspect-cpp" "$ROOT/build-sdk/sp4e_smoke.sp4e"
