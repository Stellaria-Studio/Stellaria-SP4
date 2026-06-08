# RIFE SP4 A1P Validation

Validation target:

```text
Models/RIFE-safetensors/flownet.safetensors
```

Smoke command:

```bash
./tools/smoke_rife.sh
```

C++ SDK compiler smoke:

```bash
cmake -S . -B build-sdk
cmake --build build-sdk
./build-sdk/sp4c compile --input Models/RIFE-safetensors/flownet.safetensors --output build/rife_cpp.sp4 --adapter rife --target metal-m4pro --quality-profile adaptive-rife
python3 -m sp4_toolchain validate build/rife_cpp.sp4 --teacher Models/RIFE-safetensors/flownet.safetensors
```

Current result on this workspace:

```text
asset: build/rife_cpp.sp4
adapter: rife
runtime layers: 42
excluded non-runtime teacher layers: 14
graph tensors in source: 160
blocks: 35,907
residual entries: 70,332
teacher FP16 weight bytes: 4,543,830
SP4 bytes: 1,583,724
compression: 2.86908x
```

Mode distribution:

```text
SP4_MODE_LINEAR_W4A8:      9,352
SP4_MODE_SPARSE_RESIDUAL:  1,215
SP4_MODE_ADAPTIVE_PROJECT: 14,138
SP4_MODE_LEARNED_PROJECT:  11,202
```

Validator result:

```text
SP4 validate: ok
worst layer: block2.conv2.2.weight
worst RMSE: 0.022455
worst relative L2: 0.079383
conv probe: block0.conv0.0.0.weight
conv probe RMSE: 0.008781
conv probe MAE: 0.006877
```

C++ `sp4c` result:

```text
asset: build/rife_cpp.sp4
runtime layers: 42
excluded non-runtime teacher layers: 14
blocks: 35,907
residual entries: 70,332
compression: 2.86908x
validator: ok
worst layer: block2.conv2.2.weight
worst RMSE: 0.022455
conv probe RMSE: 0.008781
conv probe MAE: 0.006877
```

Quantization comparison result:

```text
SP4 A1P sparse residual: 2.87x compression, RMSE 0.007060, task_acc 90.78%
NF4-like block codebook: 3.76x compression, RMSE 0.008428, task_acc 88.76%
INT8 per-output: 1.99x compression, RMSE 0.000942, task_acc 98.79%
```

The RIFE adapter excludes `block_tea.*` tensors by default because they are teacher/training tensors, not runtime interpolation blocks. Use `--include-nonruntime-layers` to compile them intentionally.
