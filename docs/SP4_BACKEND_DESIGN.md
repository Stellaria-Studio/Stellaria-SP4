# SP4 High-Performance Backend Design

SP4 backends are plugins behind `libsp4runtime`. Users load assets and bind resources; backend developers implement the kernel contract.

## Engine-Inspired Principles

The Stellaria engine already uses native Metal paths, Slang mirrors, queue occupancy reporting, and fixed ABI structs for rendering subsystems. SP4 follows the same engineering style:

```text
stable ABI structs
SoA memory layout
fixed backend plan
GPU occupancy as a measurable signal
queue/resource binding isolated behind runtime wrappers
Metal native first, Slang portable second
```

## Metal First

The Metal backend is the first performance backend because the immediate target is M4 Pro.

Metal backend responsibilities:

```text
load metallib
create pipeline variants for block32 and block64
bind SoA asset buffers
bind external textures/buffers
select fixed kernel plan
enqueue command buffers
report backend capability and validation errors
```

## Canonical IR

The backend does not understand model names such as RIFE, AnimePBR, or SuperRes. The compiler and project adapter normalize model graphs into a small SP4 canonical IR:

```text
SP4OpConv2D
SP4OpDepthwiseConv2D
SP4OpPointwiseConv2D
SP4OpLinear
SP4OpElementwise
SP4OpActivation
SP4OpRequantize
SP4OpResidualCorrect
SP4OpFallbackFP16
SP4OpWarpGridSample
SP4OpBlend
SP4OpExternal
```

A1P focuses on the common high-frequency families:

```text
SP4Conv2D_W4A8_Block32
SP4Conv2D_W4A8_Block64
SP4PointwiseConv1x1_W4A8
SP4Linear_Tile16_W4A8
SP4MLP_Gated_Tile16_W4A8
SP4SSM_Projection_W4A8
SP4SSM_Scan_FP16
SP4ResidualCorrection
SP4Requantize
SP4FP16Fallback
SP4WarpGridSampleFP16
SP4BlendFP16
SP4ExternalOptimized
```

Low-frequency, visually sensitive, or irregular ops should use FP16/platform fallback instead of forcing every model-specific op into the high-performance backend.

## Graph Runtime Plan

The current SDK emits a canonical static graph plan in every `.sp4` manifest:

```text
graph_runtime: canonical_a1p_static_plan
fallback_ops: warp_grid_sample_fp16, blend_fp16, resize_fp16, external_texture_fp16
```

This is the boundary that makes SP4 usable outside one model family:

```text
quantized hot path: Conv2D / Depthwise / Pointwise / Linear / Elementwise / Activation / Requantize
precision fallback: warp, grid-sample, blend, resize, external texture interop
asset policy: each layer keeps mode/aux/residual metadata in SoA sections
backend policy: each device reports caps before pipeline creation
```

RIFE-style interpolation uses FP16 fallback for warping/blending stages while keeping the dense convolution path in W4A8/SP4 kernels. Other graph adapters can reuse the same canonical ops and only specialize adapter policy, not backend ABI.

Kernel responsibilities:

```text
packed INT4 load
sign extension
INT8 activation load
INT32/FP32 accumulation
FP16 scale restore
sparse FP16 residual correction only on marked blocks
FP16 fallback path when supported
requantize or FP16 output
```

## Current Metal Fast Path

The current Metal Conv2D backend uses a tiled INT32 ALU path:

```text
16 output pixels per thread for weight reuse in the current Conv2D validation path
8 packed INT4 values unpacked per inner group
INT32 partial sums before FP16/FP32 scale restore
zero-point blocks use sum(u4*a) - zp*sum(a), keeping the correction at block boundary
block-level FP16 residual correction after the integer dot
AdaptiveProject codebook blocks routed through a slower FP path only when offline validation proves they are worth it
```

This keeps the ordinary W4A8 blocks on INT32 ALUs and moves FP16/FP32 correction to the block boundary instead of stalling every nibble multiply.

## Scheduling, Batching, And Fusion

Small kernels can become launch-bound before they become math-bound. The C backend ABI now exposes:

```text
SP4DispatchItem
SP4KernelBatchDesc
SP4_SCHEDULE_BATCH_IDENTICAL
SP4_SCHEDULE_OUT_OF_ORDER
SP4_SCHEDULE_FUSE_WHEN_SAFE
```

The intended backend policy is:

```text
identical small kernels:
  group by batchKey and encode them into one command buffer or indirect dispatch packet

different independent kernels:
  use dependencyMask to allow out-of-order encoding up to the backend's maxOutOfOrderDepth

FP correction and residual work:
  schedule as late epilogue work or side-queue work when it does not feed the next integer dot

operator fusion:
  fuse Conv/Linear + Activation + Requantize when tensor lifetime and precision contract allow it
  fuse Linear + Gate + SiLU for MLP/FFN blocks
  fuse SSM projection + scan only when recurrence/state hazards are explicit in the dependency mask
```

The compiler manifest advertises these fusion patterns, but backends still decide at pipeline creation time based on their caps. This keeps graph-level optimization automatic without hard-coding every future device into the asset.

The MLP gate kernel is not a good standalone GOPS showcase because exact SiLU spends time in special-function `exp` and often runs as a small launch-bound epilogue. The backend therefore exposes:

```text
SP4MLP_Gated_Tile16_W4A8: exact reference path
SP4MLP_Gated_Fast4_W4A8: vectorized fast-exp path
SP4MLP_Gated_HardSiLU4_W4A8: approximate no-exp path for tolerant graphs
SP4MLP_FusedProjectionGateSiLU_W4A8: preferred fused projection + gate + SiLU route
```

Production MLP/FFN plans should prefer `SP4MLP_FusedProjectionGateSiLU_W4A8` when the graph has paired `up_proj` and `gate_proj` shapes. It removes one gate epilogue launch and avoids writing the intermediate up/gate tensors. Standalone gate kernels remain useful as references or when a graph backend cannot fuse the producer projections.

The SSM scan path has a true recurrence dependency. The serial reference path assigns one long chain per channel, which leaves little latency hiding. The high-throughput path uses chunked affine prefix scan:

```text
state_t = A_t * state_0 + B_t
compose((A2,B2),(A1,B1)) = (A2*A1, A2*B1+B2)
```

`SP4SSM_Scan_Parallel256_FP16` and `SP4SSM_Scan_Parallel512_FP16` scan each channel in a threadgroup and carry only one state between chunks. The scheduler should choose chunk width by token length/channel count and may run independent SSM blocks out of order.

## Mixed Precision And Kernel Extension

SP4 does not require every graph op to be forced through W4A8. The manifest now carries a mixed precision plan:

```text
Conv2D / PointwiseConv2D: SP4_W4A8_or_AdaptiveProject, Metal INT32 tile16 route
Linear / MLPProjection / SSMProjection: SP4_W4A8_or_ZeroPoint_or_AdaptiveProject, Metal INT32 tile16 route
WarpGridSample / Blend / Resize: FP16 platform or custom kernel route
External: backend registered optimized kernel
```

Backend plugins can expose a kernel registry entry with `SP4KernelDesc`, including op kind, precision kind, kernel family, native pipeline, and a semantic tag. This lets an engine use a specialized warp, grid-sample, optical-flow, denoise, or post-process kernel without weakening the SP4 asset/runtime contract.

## Model-Family Acceleration Targets

The current kernel map prioritizes three families:

```text
RIFE / video interpolation:
  Conv2D block32/block64 tile16 W4A8
  FP16 warp/grid-sample/blend route
  project-weighted motion-boundary validation

MLP / FFN:
  Linear tile16 W4A8 projection
  exact and fast gated MLP entries: up * SiLU(gate)
  fused projection+gate+SiLU entry to avoid small epilogue launches
  approximate hard-SiLU entry when task validation allows it
  optional external optimized activation kernels

SSM:
  W4A8 projection kernel for in/out/x/dt projection weights
  serial FP16 scan reference
  parallel affine-prefix scan kernels for high-throughput recurrence/state update
  external optimized scan can replace the reference scan through kernel registry
```

The practical rule is: compile dense supported weights into SP4, then let the graph plan route irregular or recurrence-heavy pieces to FP16/custom kernels until a dedicated SP4 backend kernel exists.

## Slang Portable Backend

Slang is the second-stage backend source, not the M4 Pro first path. It is useful for:

```text
DX12
Vulkan
future SP4E cooperative vector paths
shared kernel contract testing
backend parity work
MLP/SSM portable entrypoint bring-up
```

## Occupancy And Latency Hiding

SP4 should not chase compression alone. Backend kernels should maximize useful work:

```text
use compact qweight bandwidth
keep metadata streams separate
avoid residual reads for ordinary blocks
split accumulation into independent chains
prefer predictable branches from compile-time modes
group work so active lanes remain high
track queue occupancy and useful-work ratio in integration tests
```

This is why SP4 stores project-adaptive policy at compile time. The runtime sees fixed metadata and can keep dispatch predictable.
