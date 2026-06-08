# Stellaria SP4 SDK Overview

SP4 SDK turns the A1P prototype into an embeddable compiler/runtime/backend stack.

```text
Python tools
  research, PAC calibration, visualization, regression automation

C++ SDK
  official compiler core, asset container, runtime loader, validator contracts, CPU reference

Metal backend
  M4 Pro first high-performance backend for W4A8 fused kernels

Slang backend
  portable shader contract for DX12, Vulkan, CUDA-like, and future SP4E paths
```

The important boundary is that users consume `.sp4` assets through runtime APIs. They do not directly edit Metal or Slang shader code unless they are backend developers.

## SDK Layout

```text
sdk/include/sp4/
  SP4Format.h       A1P format enums and layer metadata
  SP4Asset.h        asset writer contract
  SP4Compiler.h     C++ compiler entry
  SP4Runtime.h      asset loader and CPU reference reconstruction
  SP4Backend.h      C ABI backend plugin interface

sdk/src/
  C++ compiler, safetensors reader, asset writer, runtime loader

sdk/tools/
  sp4c              C++ compiler CLI
  sp4inspect-cpp    C++ asset loader smoke tool

shaders/metal/
  native MSL A1P kernels

shaders/slang/
  portable A1P kernel contract
```

## Language Roles

Python remains useful, but no longer owns the official runtime surface.

```text
Python:
  SP4-PAC, RIFE experiments, charts, automated quality analysis

C++:
  sp4c compiler core, asset ABI, runtime loader, CPU reference, engine integration

Metal:
  M4 Pro high-performance backend, small-kernel batching, out-of-order dispatch contract

Slang:
  portable backend source for DX12/Vulkan and future SP4E work
```

## Supported Compile Surface

The C++ compiler now compiles the common weight tensors it can execute or hand to a specialized backend:

```text
4D .weight tensors: Conv2D / pointwise / depthwise-style weights
2D .weight tensors: Linear / MLP / FFN / SSM projection weights
input dtypes: F32, F16, BF16 safetensors
unsupported graph ops: left in the manifest plan for FP16 or external optimized kernels
```

This means most ordinary model checkpoints can have their supported dense weights compiled into SP4 even when the full graph still needs FP16 fallback ops, custom kernels, or an engine-native operator.

High-accuracy profiles can emit `SP4_MODE_LINEAR_W4A8_ZP` for skewed blocks. This is still an SP4 main-path mode: the backend computes the dot as `sum(u4 * int8) - zero_point * sum(int8)` and applies the FP scale once at the block/tile boundary.

## Backend Scheduling Contract

The backend plugin ABI includes batch dispatch and fusion metadata:

```text
SP4KernelBatchDesc
SP4DispatchItem.batchKey
SP4DispatchItem.dependencyMask
SP4_FUSION_CONV_ACT_REQUANT
SP4_FUSION_LINEAR_GATE_SILU
SP4_FUSION_SSM_PROJECTION_SCAN
```

Backends use this to merge many identical small kernels, encode unrelated kernels out of order, and apply automatic operator fusion when tensor lifetimes and precision rules make it safe.

## Validation Contract

The C++ compiler writes the same `.sp4` A1P container consumed by the existing Python validator. This gives us a cross-language check:

```bash
cmake -S . -B build-sdk
cmake --build build-sdk
./build-sdk/sp4c compile --input Models/RIFE-safetensors/flownet.safetensors --output build/rife_cpp.sp4 --adapter rife --target metal-m4pro --quality-profile adaptive-rife
python3 -m sp4_toolchain validate build/rife_cpp.sp4 --teacher Models/RIFE-safetensors/flownet.safetensors
```
