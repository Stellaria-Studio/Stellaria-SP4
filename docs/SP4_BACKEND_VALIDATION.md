# SP4 Backend Validation

One-command backend validation:

```bash
./tools/validate_backends.sh build/rife_cpp.sp4
```

This command verifies:

```text
C++ sp4c compiler
Python asset validator
Metal GPU A1P reconstruct backend
Metal fused W4A8 shader compilation
Metal MLP/SSM specialized shader compilation
Metal MLP/SSM synthetic dispatch benchmark and numeric reference
Slang reconstruct shader compilation to SPIR-V
Slang fused W4A8 shader compilation to SPIR-V
Slang MLP/SSM shader entry compilation to SPIR-V
Metal throughput comparison kernels for FP16 / INT8 / INT4 / NF4-like / W4A8
local quantization comparison table
```

## Current Machine

```text
GPU: Apple M4 Pro
Metal toolchain: xcrun metal / metallib
Slang compiler: /usr/local/bin/slangc
```

## Current RIFE Result

```text
asset: build/rife_cpp.sp4
runtime layers: 42
excluded teacher/training layers: 14
blocks: 35,907
residual entries: 70,332
compression: 2.86908x
Python validator: ok
worst validator layer: block2.conv2.2.weight
worst validator RMSE: 0.022455
conv probe RMSE: 0.008781
mode distribution: LinearW4A8=9,352, SparseResidual=1,215, AdaptiveProject=14,138, LearnedProject=11,202
```

Metal backend:

```text
SP4 Metal backend validate: ok
device: Apple M4 Pro
layers: 42
GPU reconstruct: 23.9303 ms
worst GPU-vs-CPU abs error: 0
SP4Conv2D_W4A8_Block32 avg/peak: 156.844 / 203.932 GOPS
SP4Conv2D_W4A8_Block64 avg/peak: 538.47 / 1556.89 GOPS
SP4Linear_Tile16_W4A8 avg/peak:  1183.49 / 1784.28 GOPS, max_abs=0.0271912
SP4SSM_Projection_W4A8 avg/peak: 689.873 / 823.983 GOPS, max_abs=0.0271912
SP4MLP_Gated_Tile16_W4A8 avg/peak: 21.8584 / 24.4625 GOPS, max_abs=0.000242591
SP4MLP_Gated_Fast4_W4A8 avg/peak: 30.8254 / 34.6517 GOPS, max_abs=0.000242591
SP4MLP_Gated_HardSiLU4_W4A8 avg/peak: 17.1791 / 37.9575 GOPS, max_abs=0.0165738
SP4MLP_FusedProjectionGateSiLU_W4A8 avg/peak: 1615.11 / 2434.79 GOPS, max_abs=0
SP4SSM_Scan_FP16 serial avg/peak: 1.37958 / 1.97643 GOPS, max_abs=0.0000592768
SP4SSM_Scan_Parallel256_FP16 avg/peak: 4.30233 / 6.68593 GOPS, max_abs=0.0000592768
SP4SSM_Scan_Parallel512_FP16 avg/peak: 5.69827 / 6.20001 GOPS, max_abs=0.0000592768
FP16 synthetic conv avg/peak:       454.825 / 566.153 GOPS
INT8 per-output avg/peak:          402.582 / 511.007 GOPS
INT4 per-output avg/peak:          269.64 / 328.687 GOPS
NF4-like avg/peak:                 114.033 / 137.355 GOPS
W4A8 block avg/peak:               124.478 / 151.978 GOPS
SP4 INT32x8 avg/peak:              359.792 / 454.372 GOPS
```

High-accuracy zero-point trial:

```text
asset: build/rife_int8_chase.sp4
profile: int8-chase
compression: 2.80935x
residual entries: 82,472
modes: LinearW4A8=2,233, LinearW4A8_ZP=21,743, SparseResidual=257, AdaptiveProject=6,471, LearnedProject=5,203
SP4 sparse residual task_acc: 91.28%
SP4 sparse residual RMSE: 0.006678
after optimized ZP scale search and learned codebook: task_acc=91.28%, RMSE=0.006678
Python validator: ok
Metal backend validator: ok
```

Metal shader compilation:

```text
sp4_reconstruct.metallib: ok
sp4_block_w4a8.metallib: ok
sp4_compare_kernels.metallib: ok
sp4_mlp_ssm.metallib: ok
```

Slang shader compilation:

```text
sp4_reconstruct.spv: ok
sp4_block_w4a8.spv: ok
sp4_mlp_linear.spv: ok
sp4_mlp_gate.spv: ok
sp4_ssm_projection.spv: ok
sp4_ssm_scan.spv: ok
sp4_reconstruct_from_slang.metallib: ok
sp4_block_w4a8_from_slang.metallib: ok
```

The executed Metal validation verifies the core SP4 A1P ABI: INT4 unpack, zero-point W4A8 restore, learned/codebook restore, FP16 scale restore, mode dispatch, sparse residual lookup, SoA buffer binding, plus real dispatch for the canonical `SP4Conv2D_W4A8_Block32`, `SP4Conv2D_W4A8_Block64`, `SP4Linear_Tile16_W4A8`, `SP4SSM_Projection_W4A8`, `SP4MLP_Gated_Tile16_W4A8`, `SP4MLP_FusedProjectionGateSiLU_W4A8`, `SP4SSM_Scan_FP16`, and `SP4SSM_Scan_Parallel256/512_FP16` kernel families.

`SP4SSM_Scan_FP16` is retained as a serial recurrence reference. `SP4SSM_Scan_Parallel256_FP16` and `SP4SSM_Scan_Parallel512_FP16` use chunked affine prefix scan, so they are the preferred SSM backend paths when the graph can expose scan dependencies. The faster chunk size can vary with token length, channel count, and current GPU scheduling noise.

The throughput comparison kernels are synthetic same-shape microbenchmarks. They compare arithmetic/memory behavior of common quantization schemes on this machine, not full end-to-end video interpolation latency. The SP4 Conv2D rows are executed backend kernels with 16-pixel tiling, INT32 packed-dot accumulation, and block-boundary FP correction. Short Metal kernel timings can vary by run; use `docs/assets/sp4_comparison_metrics.json` as the latest generated result.
