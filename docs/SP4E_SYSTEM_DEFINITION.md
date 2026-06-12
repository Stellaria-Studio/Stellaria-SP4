# Stellaria SP4E A1P System Definition

SP4E, **Stellaria Split Precision 4 Extended**, is the matrix-accelerated extension layer for SP4. It is not a replacement for SP4. SP4 remains the default path for ordinary GPU ALU and CPU fallback. SP4E is enabled only when a backend exposes native FP4 matrix acceleration, strong low-bit tensor throughput, or profile-proven speedup over SP4.

The current preview ABI is:

```text
SP4E-F4I4block16(A1P)
SP4E-F4I4tile16x16(A1P)
```

The core implementation lives in the C++ SDK:

```text
sdk/include/sp4/SP4EFormat.h
sdk/include/sp4/SP4ECompiler.h
sdk/include/sp4/SP4ERuntime.h
sdk/src/SP4ECompiler.cpp
sdk/src/SP4ERuntime.cpp
sdk/tools/sp4ec.cpp
sdk/tools/sp4einspect.cpp
```

Python remains a research and regression helper for SP4. SP4E core compile/load/reference behavior is C++ because this is HPC/runtime infrastructure.

## Runtime Gate

SP4E must not run just because an asset exists. The loader and backend choose:

```text
if native FP4 TensorOps are present:
  SP4E_TENSOR_PATH
else if low-bit ALU throughput is at least 2.0x FP16 baseline:
  SP4E_LOWQ_ALU_PATH
else if profiling proves SP4E faster than SP4:
  SP4E_PROFILE_GUIDED_PATH
else:
  SP4E_SP4_SUBSET_PATH
```

On this machine the local probe intentionally reports no public FP4 matrix acceleration, so generated assets record the SP4-compatible partial fallback path:

```text
runtime_gate=SP4E_SP4_SUBSET_PATH enabled=true
```

## A1P Container

```text
magic:          8 bytes, "STSP4E1P"
container_ver:  uint32 little endian, currently 1
manifest_len:   uint32 little endian
manifest_json:  UTF-8 JSON
payload:        concatenated binary sections
```

Sections:

```text
qweight_fp4          uint8 packed selected 4-bit main-plane nibbles; FP4, INT4, or SP4 zero-point codes by mode
qweight_int4         optional uint8 packed INT4 side-plane nibbles for future bridge sidecars; empty in compact A1P assets
scale_fp4            float16 selected main scale per block
scale_int4           optional float16 side scale per block for future bridge sidecars; empty in compact A1P assets
bias                 float16 per output channel
mode                 packed 4-bit mode control, two blocks per byte
aux                  packed 4-bit per-block zero-point/control nibble, two blocks per byte
bridge               reserved bridge sidecar; empty in compact A1P assets
residual_index       uint8 sparse residual local index within block16
residual_value       float16 sparse residual value
residual_table       int32 sparse pairs {block+count,start}; loaders expand to dense tables internally
activation_precision uint8 per layer
tile_plan            reserved for backend-native tensor tile planning
low_rank_u/v         active FP16 rank-4 low-rank residual sidecar for structured error
channel_correction   active FP16 per-output-channel task-aware gain correction
```

## Tools

```bash
cmake -S . -B build-sdk
cmake --build build-sdk

./build-sdk/sp4ec compile \
  --input Models/RIFE-safetensors/flownet.safetensors \
  --output build/rife.sp4e \
  --adapter rife \
  --target cpu-reference \
  --residual-ratio 0.01

./build-sdk/sp4einspect-cpp build/rife.sp4e
```

The CPU reference path is for correctness and asset inspection, not performance claims.

## Local Validation Result

The local SP4 backend validation flow now also compiles and inspects SP4E:

```bash
./tools/validate_backends.sh
python3 tools/generate_comparison_charts.py
```

On the current host, SP4E correctly reports partial SP4-subset gating:

```text
runtime_gate=SP4E_SP4_SUBSET_PATH enabled=true reason=sp4_subset_available_on_generic_backend
```

The C++ CPU reference backend supports:

```text
ReconstructSP4ELayer
ExecuteSP4EConv2DNCHW
ExecuteSP4ELinear
```

This proves the engineering chain can compile `.sp4e`, load it, reconstruct weights, run CPU reference operators, and measure local precision even without matrix acceleration hardware.

Generated comparison artifacts:

```text
docs/assets/sp4e_accuracy_comparison.svg
docs/assets/sp4e_backend_gate.svg
docs/assets/sp4_comparison_metrics.json
```

For the local RIFE safetensors model, the current generated metrics report:

```text
SP4 A1P sparse residual:
  compression=2.87x rmse=0.007060 task_acc=90.78%

SP4E A1P balanced CPU reference, residual_ratio=0.01:
  compression=2.12x rmse=0.004722 task_acc=96.36%

SP4E A1P int8-chase CPU reference, effective residual_ratio=0.01:
  compression=2.12x rmse=0.004722 task_acc=96.36%

SP4E A1P max-accuracy CPU reference, full correction profile:
  compression=0.55x rmse=0.001681 task_acc=98.38%
```

SP4E A1P compact assets store a selected 4-bit main plane rather than both FP4 and INT4 candidates. Most generic-host blocks route through the SP4 subset as asymmetric INT4 zero-point blocks; FP4 blocks remain available for TensorOps-capable backends. The 2x-constrained path now combines packed control planes, compact residual-table pairs, L2-optimized block scales, task-ranked sparse residuals, rank-4 FP16 low-rank correction, and per-channel gain correction. `int8-chase` is now a compression-preserving profile; only `max-accuracy` is allowed to spend past the 2x asset budget.
