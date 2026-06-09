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

### Realtime Presentation Scheduler

Stellaria Motion 的 24fps 到 60fps 实时插帧暴露了一个可复用规律：低精度后端的平均 kernel 速度不是唯一瓶颈，CPU/GPU 同步边界、重复 flow 前向、满分辨率中间纹理写回和 presentation cadence 都会让观感退回“像 24fps”。

SP4 runtime cache 因此把这些调度意图显式写入 `RuntimePrepareOptions` 和 `PreparedRuntimeCache`：

```text
targetFps / frameBudgetMs
multiTOutputCount
preferPresentationReuse
preferCommandBufferBatch
preferBlendRefineFusion
preferTiledGraphExecutor
recommendedMaxFlowHeight
```

后端实现应优先遵循以下顺序：

```text
1. 一次 flow/feature 前向，复用到多个 t 输出。
2. 同一组 t 输出编码进同一个 native command buffer，只在批尾同步。
3. 将 blend、residual refine、overshoot guard 合并成单个 presentation kernel。
4. 对大 graph 使用 tile/block 级 dequant cache，避免每像素重复解码相同 SP4 权重。
5. 让 scheduler 依据 p95/p99 和 display cadence 调整 flow height，而不是只看 avg。
6. presentation 输出纹理必须使用 generation-owned texture，完成后再原子切换，避免显示端与生成端同时 blit 同一纹理。
```

这不是 Stellaria Motion 专属技巧，而是 SP4 compiler/scheduler 的通用策略：compiler 负责暴露可融合的 op 和 tile/reuse key，runtime cache 给出预算与复用意图，backend 在自己的命令队列里完成 tiled/fused graph execution。

Stellaria Motion 当前把这套策略落成两个实时档：

```text
60fps:  quiet 360p, balanced 432p, quality 544p
120fps: quiet 216p, balanced 360p, quality 432p
```

这些档位不是格式限制，而是当前 M4 Pro + RIFE SP4 A1P hot-subgraph executor 的调度建议。后续 tiled/fused graph executor 扩大覆盖后，scheduler 可以把 quality 档继续推向 720p flow。

## Validation Contract

The C++ compiler writes the same `.sp4` A1P container consumed by the existing Python validator. This gives us a cross-language check:

```bash
cmake -S . -B build-sdk
cmake --build build-sdk
./build-sdk/sp4c compile --input Models/RIFE-safetensors/flownet.safetensors --output build/rife_cpp.sp4 --adapter rife --target metal-m4pro --quality-profile adaptive-rife
python3 -m sp4_toolchain validate build/rife_cpp.sp4 --teacher Models/RIFE-safetensors/flownet.safetensors
```
