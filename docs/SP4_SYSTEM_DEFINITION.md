# Stellaria SP4 总体系定义与工具链说明稿

## 1. 总体定义

**SP4**，全称为 **Stellaria Split Precision 4**，是 Stellaria 面向实时视觉推理、神经渲染、视频插帧与运行时神经模块设计的私有 4-bit 分离精度量化体系。

SP4 的核心目标是在普通 GPU ALU 上实现接近 INT4 的推理效率，同时通过 FP16 缩放、残差修正、敏感路径保护与项目自适应校准，获得接近 FP16 的视觉稳定性。

SP4 不是单纯的 W4A8、W4A16 或 weight-only quantization，而是一套包含 **数据格式、block ABI、量化编译器、项目校准工具、运行时加载器、后端 kernel、验证工具与质量基准** 的完整低精度推理系统。

SP4 的基本思想是：

```text
INT4 / INT8 主路径：
  负责主要压缩率、主要计算量与主要运行效率。

FP16 校正路径：
  负责 scale、bias、动态范围恢复、residual、sensitive fallback 与视觉稳定性维护。

Project-Adaptive Calibration：
  在离线校准与资产编译阶段自动分析项目特征，生成基本固定的 block-level policy。

Runtime：
  在运行时读取固定 metadata 并执行对应 kernel path，保持稳定、可预测、低分支开销。
```

SP4 的目标不是数学意义上的 0 损耗，而是实时视觉任务中的：

```text
视觉近无损
时序稳定
低带宽
低延迟
运行时可预测
普通 ALU 可执行
未来矩阵加速器可映射
```

---

## 2. 系统组成

SP4 体系由以下部分组成：

```text
SP4 Format Spec
SP4 Block ABI
SP4 Asset Container
SP4 Compiler
SP4 Project-Adaptive Calibration
SP4 Project Adapter
SP4 Runtime Loader
SP4 Backend Kernel Library
SP4 CPU Reference Backend
SP4 Inspector
SP4 Validator
SP4 Benchmark & Quality Suite
SP4 Version & Compatibility System
```

其中，SP4 格式本体负责定义低精度数据如何存储；SP4 Compiler 负责将 FP16 / FP32 模型编译成 SP4 asset；SP4 Runtime Loader 负责加载并绑定运行时资源；SP4 Backend Kernel Library 负责在不同硬件后端执行 SP4 计算；SP4-PAC 负责根据项目特征生成固定或基本固定的 block policy。

SP4 的核心设计原则是：

```text
格式稳定
策略可插拔
项目可自定义
运行时固定
后端可替换
调试可观察
质量可验证
版本可兼容
```

---

## 3. 版本命名规则

SP4 的版本命名遵循：

```text
SP + 近似量化位数 + block 大小 + 架构版本号 + 状态标识
```

其中：

```text
SP:
  Stellaria Precision / Split Precision 产品名前缀。

4:
  近似 4-bit 主量化精度。

block32 / block64:
  block 分组大小。

A1:
  架构版本号。

P:
  Preview，预览版。

S:
  Stable，正式版。
```

第一版 SP4 格式命名为：

```text
SP4block32(A1P)
SP4block64(A1P)
```

含义分别为：

```text
SP4block32(A1P):
  32-value block 的 SP4 A1 预览版格式。

SP4block64(A1P):
  64-value block 的 SP4 A1 预览版格式。
```

对应正式版命名为：

```text
SP4block32(A1S)
SP4block64(A1S)
```

若后续 block ABI、metadata 语义、scale 策略、residual 存储方式或资产容器结构发生不兼容变化，则进入新的架构版本，例如：

```text
SP4block32(A2P)
SP4block64(A2P)
```

SP4E 表示 **Stellaria Split Precision 4 Extended**，用于承载更复杂的 block 策略、矩阵加速器映射、多精度组合与复杂 runtime guard。

---

## 4. 默认精度形态

SP4 A1P 默认采用：

```text
W4A8
```

即：

```text
Weight: INT4
Activation: INT8
Scale: FP16
Bias: FP16
Main Accumulation: INT32
Restore Precision: FP16 / FP32
Sensitive Fallback: FP16
```

基础计算形式为：

```text
Y ≈ scale_w_fp16 × scale_a_fp16 × dot(Qw_int4, Qa_int8)
```

其中：

```text
Qw_int4:
  INT4 权重主编码。

Qa_int8:
  INT8 激活主编码。

scale_w_fp16:
  权重 block 的 FP16 缩放因子。

scale_a_fp16:
  激活的 FP16 缩放因子。

dot:
  INT4 × INT8 主计算路径。

Y:
  缩放恢复后的输出结果。
```

当 block 启用 residual 时，SP4 的权重表示扩展为：

```text
W_sp4 ≈ scale_w_fp16 × Qw_int4 + R_sparse_fp16
```

对应输出形式为：

```text
Y ≈ scale_w_fp16 × scale_a_fp16 × dot(Qw_int4, Qa_int8)
  + residual_correction_fp16
```

其中 `R_sparse_fp16` 是稀疏 FP16 残差，仅存在于被标记的敏感 block、敏感通道或项目指定路径中。

---

## 5. SP4block32(A1P)

**SP4block32(A1P)** 是 SP4 的基础 block 格式，以 32 个权重为一个量化组。

逻辑结构为：

```text
SP4block32(A1P):
  qweight:       32 × int4
  scale:         FP16
  mode:          uint8
  aux:           uint8
  residual_ref:  optional
```

字段含义如下：

```text
qweight:
  32 个 INT4 权重编码。

scale:
  FP16 block scale，用于恢复该 block 的动态范围。

mode:
  描述该 block 的行为模式。

aux:
  项目级辅助字段，可由不同项目定义具体语义。

residual_ref:
  可选 residual 索引或 residual block 引用。
```

在不启用 residual 的情况下，其基础数据规模约为：

```text
32 × INT4 = 128 bit = 16 bytes
FP16 scale = 2 bytes
mode + aux = 2 bytes
总计约 20 bytes
```

对比 FP16 权重：

```text
32 × FP16 = 64 bytes
```

理论压缩比例约为：

```text
64 / 20 ≈ 3.2x
```

SP4block32(A1P) 的特点是粒度较细、误差更容易控制，适合对视觉稳定性、细节边缘、运动边界和局部 artifact 更敏感的区域。

---

## 6. SP4block64(A1P)

**SP4block64(A1P)** 是 SP4 的大 block 格式，以 64 个权重为一个量化组。

逻辑结构为：

```text
SP4block64(A1P):
  qweight:       64 × int4
  scale0:        FP16
  scale1:        FP16 optional
  mode:          uint8
  aux:           uint8
  residual_ref:  optional
```

字段含义如下：

```text
qweight:
  64 个 INT4 权重编码。

scale0:
  主 FP16 block scale。

scale1:
  可选二级 FP16 scale，用于更细的误差控制或项目级自定义。

mode:
  描述该 block 的行为模式。

aux:
  项目级辅助字段。

residual_ref:
  可选 residual 索引或 residual block 引用。
```

在不启用 residual 的情况下，其基础数据规模约为：

```text
64 × INT4 = 256 bit = 32 bytes
FP16 scale0 = 2 bytes
FP16 scale1 = 2 bytes optional
mode + aux = 2 bytes
总计约 34~36 bytes
```

对比 FP16 权重：

```text
64 × FP16 = 128 bytes
```

理论压缩比例约为：

```text
128 / 36 ≈ 3.56x
```

SP4block64(A1P) 的特点是 metadata 占比更低、带宽效率更高，更适合较大通道数、较低敏感度或更偏向吞吐的计算区域。

---

## 7. Block Mode

SP4 的 `mode` 字段用于描述 block 行为。SP4 A1P 的基础 mode 定义如下：

```text
0 = SP4_MODE_LINEAR_W4A8
    默认 W4A8 线性量化 block。

1 = SP4_MODE_LINEAR_W4A8_ZP
    带 zero-point 的非对称 W4A8 block。

2 = SP4_MODE_RESIDUAL
    W4A8 主路径 + FP16 residual correction。

3 = SP4_MODE_SPARSE_RESIDUAL
    W4A8 主路径 + 更稀疏的 FP16 residual correction。

4 = SP4_MODE_SENSITIVE_FP16
    敏感 block 或敏感路径，回退 FP16。

5 = SP4_MODE_ADAPTIVE_PROJECT
    项目自适应校准生成的项目级 block。

6 = SP4_MODE_RESERVED_MATRIX
    预留给矩阵加速器映射。

7 = SP4_MODE_RESERVED_SP4E
    预留给 SP4E 扩展。
```

SP4 的 mode 字段不是用户侧运行时画质档位，而是由项目校准、资产编译和 block 分析结果写入的 block metadata。

---

## 8. Aux Metadata

`aux` 字段是 SP4 项目级自定义能力的核心入口。

SP4 本身只规定 `aux` 是项目级辅助字段，不强制规定所有 bit 的统一语义。不同项目可以在不破坏 SP4 基础 ABI 的前提下定义 `aux` 的局部含义。

例如，在视频插帧任务中，`aux` 可以表达：

```text
motion_boundary_sensitive
flow_branch
mask_branch
occlusion_sensitive
IFBlock stage id
```

在神经渲染任务中，`aux` 可以表达：

```text
toon_edge_sensitive
flat_color_region
specular_response
shadow_ramp_sensitive
material group id
```

因此，SP4 不只是一种固定量化算法，而是一种：

```text
统一的 4-bit split precision block runtime ABI
```

---

## 9. Residual 与 Sparse Residual

SP4 本体包含 residual 与 sparse residual 能力。Residual 不属于 SP4E 的专属扩展，而是 SP4block32(A1P) 与 SP4block64(A1P) 的基础能力之一。

普通 residual 形式为：

```text
W_sp4 = scale × Q4 + R16
```

其中：

```text
scale × Q4:
  INT4 主路径权重表示。

R16:
  FP16 残差修正。
```

Sparse residual 是 residual 的轻量形式：

```text
W_sp4 = scale × Q4 + sparse(R16)
```

其结构通常表现为：

```text
index + FP16 value
```

或：

```text
channel_id + correction value
```

SP4 的 residual 只服务于被标记的敏感 block、敏感通道或项目指定区域。普通 block 不承担 residual 访存成本，也不默认访问 residual buffer。

Residual 的存在使 SP4 不再只是传统 W4A8，而是形成了：

```text
INT4 主表达
FP16 动态范围恢复
FP16 局部误差校正
block 级敏感路径保护
```

这种分离式表示。

---

## 10. 项目自适应校准

SP4 的项目级自定义与动态自适应统一归入：

```text
SP4 Project-Adaptive Calibration
SP4 项目自适应校准
```

SP4 的“自适应”主要发生在离线量化、项目校准和资产编译阶段，而不是在普通用户侧运行时进行全量动态切换。

项目自适应校准的作用是分析具体项目的模型结构、误差分布、视觉敏感区域和 artifact 风险，并生成基本固定的 block-level policy。

其结果体现为：

```text
固定的 block mode
固定的 aux metadata
固定的 residual_ref
固定的 scale policy
固定的 fallback policy
固定的 backend execution policy
```

运行时只读取这些 metadata 并执行对应路径，而不是每帧重新分析或大范围改变精度策略。

因此，SP4 的运行时特征是：

```text
deterministic
stable
cache-friendly
branch-controlled
artifact-safe
```

中文可描述为：

```text
确定
稳定
缓存友好
分支可控
artifact 安全
```

---

## 11. SP4 Format Spec

**SP4 Format Spec** 是 SP4 系统的基础规范文件。

它定义：

```text
SP4block32(A1P)
SP4block64(A1P)
mode 字段语义
aux 字段使用边界
scale 存储方式
activation scale 存储方式
residual_ref 语义
sparse residual buffer 格式
tensor layout
alignment 要求
section 布局
版本兼容规则
```

SP4 Format Spec 的作用是保证所有项目、工具、runtime 和 backend 对 SP4 asset 有一致理解。

没有 SP4 Format Spec，SP4 只会变成某个模型或某个 shader 的临时量化实现；有了 Spec，SP4 才能成为 Stellaria 内部稳定的低精度 runtime ABI。

---

## 12. SP4 Asset Container

**SP4 Asset Container** 是 SP4 模型资产的容器格式，可采用 `.sp4` 或并入 Stellaria 的统一资产包中。

一个 SP4 asset 应包含：

```text
header
version
model graph metadata
tensor layout metadata
qweight buffer
scale buffer
activation scale table
mode buffer
aux buffer
residual_ref buffer
sparse residual buffer
fallback layer list
backend kernel plan
checksum
debug / compiler report metadata
```

SP4 asset 的基本逻辑是：

```text
FP16 / FP32 model
  ↓
SP4 Compiler
  ↓
SP4 asset
  ↓
SP4 Runtime Loader
  ↓
Backend Kernel Execution
```

SP4 Asset Container 的目标是让 SP4 权重、metadata、scale、residual 和 backend plan 作为统一资产被加载，而不是分散成多个无法追踪的临时 buffer。

---

## 13. SP4 Compiler

**SP4 Compiler** 是 SP4 工具链的核心组件。

它负责将 FP16 / FP32 模型转换为 SP4 asset。

SP4 Compiler 的输入包括：

```text
FP16 / FP32 模型
模型结构信息
项目配置
目标 backend
校准数据
项目 adapter
SP4 Format Spec
```

SP4 Compiler 的输出包括：

```text
SP4 qweight buffer
FP16 scale buffer
INT8 activation scale table
mode buffer
aux buffer
residual_ref buffer
sparse residual buffer
fallback layer list
backend kernel plan
SP4 asset container
compiler report
```

SP4 Compiler 不只是普通量化脚本，而是一个离线模型到 runtime asset 的编译系统。它的职责包括：

```text
图结构分析
层级敏感度分析
W4A8 量化
block32 / block64 分配
scale 生成
activation scale 校准
residual / sparse residual 分配
fallback 路径标记
mode / aux metadata 写入
backend kernel plan 生成
资产容器导出
```

SP4 Compiler 可以简称为：

```text
sp4c
```

其本质是：

```text
Stellaria 的低精度视觉推理资产编译器
```

---

## 14. SP4 Project Adapter

**SP4 Project Adapter** 是 SP4 通用化的关键。

不同项目的敏感区域、artifact 风险、层结构和质量目标不同，因此 SP4 Compiler 不应把所有项目策略写死在核心逻辑里。

Project Adapter 负责描述项目语义，例如：

```text
哪些层适合 block32
哪些层适合 block64
哪些层允许 residual
哪些层允许 sparse residual
哪些路径必须 fallback FP16
aux 字段如何解释
artifact 指标如何定义
calibration 数据如何组织
backend plan 如何偏好
```

典型 adapter 包括：

```text
RIFEAdapter
AnimePBRAdapter
NeuralTextureAdapter
SuperResolutionAdapter
DenoiseAdapter
```

Project Adapter 让 SP4 保持通用格式，同时允许每个项目在自己的视觉任务中获得专用策略。

---

## 15. SP4-PAC：Project-Adaptive Calibration

**SP4-PAC** 是 SP4 的项目自适应校准系统。

SP4-PAC 的输入包括：

```text
FP16 / FP32 teacher model
目标模型
校准数据集
项目 adapter
目标 backend
artifact 评估规则
```

SP4-PAC 的输出不是运行时动态系统，而是固定或基本固定的 block-level policy：

```text
block32 / block64 分配
mode metadata
aux metadata
residual_ref
sparse residual table
fallback layer list
scale policy
backend execution policy
```

SP4-PAC 的核心原则是：

```text
校准期聪明
运行期固定
```

也就是说，SP4 的自适应发生在编译与校准阶段，运行时只执行已经确定的策略。

这可以避免用户侧全量动态量化带来的：

```text
性能不稳定
分支膨胀
cache 行为不可预测
时序 artifact
调试困难
```

---

## 16. SP4 Runtime Loader

**SP4 Runtime Loader** 负责在 Stellaria runtime 中加载 SP4 asset。

它的职责包括：

```text
读取 SP4 asset
检查版本
检查 block ABI
检查 backend 支持能力
绑定 qweight buffer
绑定 scale buffer
绑定 activation scale table
绑定 mode / aux / residual_ref buffer
绑定 sparse residual buffer
构建 tensor view
选择 backend kernel path
提交 runtime dispatch
```

Runtime Loader 的目标是让 SP4 asset 作为一种标准神经推理资产进入 Stellaria runtime。

对引擎而言，SP4 不应该表现为某个模型专用 shader，而应该表现为：

```text
可加载
可验证
可绑定
可调度
可跨项目复用
```

的 runtime neural asset。

---

## 17. SP4 Backend Kernel Library

**SP4 Backend Kernel Library** 是 SP4 的执行后端集合。

SP4 的格式应与后端解耦。同一份 SP4 asset 可以在不同后端上选择不同执行路径。

基础后端包括：

```text
CPU Reference Backend
Metal Backend
DX12 Stub / Future Backend
Vulkan Stub / Future Backend
Matrix Accelerator Backend Interface
```

其中：

```text
CPU Reference Backend:
  用于正确性验证、调试、golden output 对比，不追求速度。

Metal Backend:
  用于普通 GPU ALU 上的 SP4 高性能执行，是 A1P 的主要执行路径。

Future Backend Interface:
  为 Cooperative Vector、Tensor Core、INT4 / FP4 matrix path 等后续扩展预留。
```

SP4 普通 ALU 路径可以概括为：

```text
packed INT4 qweight load
INT8 activation load
INT4 unpack in register
INT4 × INT8 main dot / conv
INT32 accumulation
FP16 / FP32 scale restore
mode-controlled residual correction
mode-controlled sensitive fallback
activation / requantize
```

SP4 在普通 ALU 上的效率来源包括：

```text
权重带宽下降
cache 命中率提升
INT 主路径计算成本降低
减少中间 FP16 feature 写回
unpack / dot / scale / activation / requantize 融合
普通 block 不承担 residual 访存成本
```

SP4 后端调度器不应只做“逐 op 逐 command buffer”的直接翻译。实时视觉项目中更有效的执行单元是 **tiled / fused graph executor**：

```text
SP4 graph / hot subgraph
-> compiler emits tile, fusion, dependency, reuse keys
-> runtime prepare cache expands hot metadata once
-> backend batches compatible dispatches into one native command buffer
-> tile/block dequant cache feeds multiple pixels / channels / t outputs
-> fused presentation or fused epilogue writes the final tensor / texture
```

Stellaria Motion 的实时插帧接入给出的通用经验是：

```text
多 t presentation 先复用 flow，再生成多个输出，而不是重复完整前向。
blend / refine / guard 这类 presentation epilogue 应融合，避免满分辨率中间纹理写回。
调度质量必须看 p95 / p99 长尾和 display cadence，不能只看平均 kernel 时间。
display-owned texture 与 generation-owned texture 分离，完成后原子切换，避免闪烁与崩溃。
flow height 由 targetFps、frameBudgetMs、multiTOutputCount 共同决定。
```

因此，SP4 backend caps 与 dispatch item 需要承载：

```text
command-buffer batch
blend/refine fusion
tiled graph executor
dequant cache
presentation reuse
tile size
reuse cache key
output slot
```

---

## 18. SP4 Kernel Contract

**SP4 Kernel Contract** 是 Compiler 与 Backend Kernel 之间的契约。

它规定：

```text
qweight 如何排列
activation 如何排列
scale 如何索引
block id 如何计算
mode 如何读取
aux 如何解释
residual_ref 如何解析
sparse residual buffer 如何访问
输出是 INT8 还是 FP16
requantize 如何执行
alignment 要求
threadgroup tile 如何对应 block
```

Kernel Contract 的作用是确保：

```text
Compiler 导出的 asset
Runtime Loader 绑定的 buffer
Backend Kernel 读取的数据
```

三者完全一致。

没有 Kernel Contract，SP4 Compiler 和 shader backend 很容易出现布局不一致、scale 对不齐、residual_ref 错位或 mode 解释错误等问题。

---

## 19. SP4 Inspector

**SP4 Inspector** 是 SP4 的可视化与检查工具。

它用于观察 SP4 asset 的内部结构，包括：

```text
每层使用 block32 还是 block64
每层 scale 分布
每层 activation scale 分布
每层 mode 分布
哪些 block 启用 residual
哪些 block 启用 sparse residual
哪些路径 fallback FP16
INT4 权重分布
量化误差分布
residual 分布
项目级 aux metadata 分布
```

SP4 Inspector 的作用是让 SP4 不成为黑盒量化格式，而是成为可观察、可解释、可调试的 runtime asset。

---

## 20. SP4 Validator

**SP4 Validator** 负责验证 SP4 asset 的合法性和正确性。

格式验证包括：

```text
header 是否正确
version 是否支持
block ABI 是否匹配
scale 数量是否匹配
mode 值是否合法
aux 是否符合项目定义
residual_ref 是否越界
sparse residual buffer 是否完整
tensor layout 是否满足 alignment
checksum 是否正确
backend 是否支持该 asset
```

数值验证包括：

```text
FP16 teacher output 对比
SP4 output 对比
INT4 output 对比
误差范围统计
residual 修正效果
fallback 路径一致性
```

SP4 Validator 的目标是保证 SP4 asset 在进入 runtime 前已经满足格式、数值和 backend 约束。

---

## 21. SP4 Benchmark & Quality Suite

**SP4 Benchmark & Quality Suite** 用于评估 SP4 的性能与视觉质量。

性能指标包括：

```text
latency
throughput
GPU time
memory bandwidth
peak memory
power
SP4 vs INT4 speed ratio
SP4 vs FP16 speed ratio
```

质量指标包括：

```text
PSNR
SSIM
LPIPS
temporal flicker
motion boundary error
ghosting score
edge stability
flat color contamination
subjective visual inspection
```

SP4 的评估目标不是单一 PSNR 或单一速度指标，而是综合衡量：

```text
是否接近 INT4 效率
是否接近 FP16 画质
是否减少纯 INT4 的视觉损伤
是否保持时序稳定
是否符合项目 artifact 约束
```

Benchmark & Quality Suite 是 SP4 能够长期演进的基础。

---

## 22. SP4 Version & Compatibility System

SP4 需要明确版本兼容规则。

基本规则如下：

```text
A1P:
  Preview 版本，不保证长期 ABI 稳定。

A1S:
  Stable 版本，同架构内保证加载兼容。

A2P:
  block ABI、metadata 语义、scale 策略或 residual 存储方式出现不兼容变化。

SP4E:
  可读取 SP4 基础 asset 的部分语义，但扩展 mode、矩阵 tile 或新精度组合不一定向下兼容。
```

Runtime Loader 需要根据版本信息判断：

```text
是否支持该 asset
是否需要 fallback
是否拒绝加载
是否需要 migration tool
是否可使用兼容模式执行
```

版本兼容系统保证 SP4 不会因为快速迭代而破坏已有项目资产。

---

## 23. SP4 与 SP4E

SP4 与 SP4E 的边界如下：

```text
SP4:
  默认 W4A8。
  支持 SP4block32(A1P) / SP4block64(A1P)。
  支持 FP16 scale。
  支持 residual / sparse residual。
  支持 block-level project metadata。
  支持项目自适应校准。
  支持 sensitive FP16 fallback。
  优先面向普通 ALU 高效运行。

SP4E:
  SP4 Extended。
  扩展复杂 block 策略。
  扩展多级 residual。
  扩展 runtime guard。
  扩展矩阵加速器支持。
  扩展 Cooperative Vector / Tensor Core 映射。
  扩展 W4A4 / W4A16。
  扩展 NF4 / LUT / codebook。
  扩展 FP4 / NVFP4 / MXFP4 backend mapping。
  扩展原生矩阵 tile layout。
```

SP4 是基础体系，SP4E 是增强体系。SP4E 不替代 SP4，而是在 SP4 的 block ABI、metadata 语义和 split precision 思想基础上继续扩展。

---

## 24. 与传统量化方法的区别

SP4 与传统 W4A8 / W4A16 / weight-only quantization 的区别主要体现在：

```text
1. SP4 是 runtime block format，而不是单纯量化算法。
2. SP4 面向实时视觉推理，而不是通用 LLM 量化。
3. SP4 默认包含 residual / sparse residual 能力。
4. SP4 通过项目自适应校准生成固定 block policy。
5. SP4 以 artifact 控制、时序稳定和视觉一致性作为重要质量目标。
6. SP4 同时面向普通 ALU 与未来矩阵加速器。
7. SP4 支持项目级 metadata 与私有 runtime ABI。
8. SP4 包含 Compiler、Runtime、Backend、Validator、Inspector 与 Benchmark 工具链。
```

传统量化通常关注：

```text
模型大小
perplexity
classification accuracy
通用 GEMM 性能
```

SP4 更关注：

```text
实时性
视觉稳定性
时序一致性
边缘 artifact
运动区域稳定性
shader runtime 可控性
项目级可定制性
```

---

## 25. 对外描述

SP4 可以对外描述为：

```text
SP4 is Stellaria's split-precision 4-bit block quantization system for real-time visual inference.
The first preview release, SP4block32(A1P) / SP4block64(A1P), adopts W4A8 as its default computation path and supports FP16 scaling, sparse residual correction, project-adaptive block metadata, and sensitivity-aware FP16 fallback.
SP4 is not only a quantization format, but also a runtime-oriented toolchain including a compiler, asset container, runtime loader, backend kernels, validator, inspector, and benchmark suite.
It is designed to run efficiently on ordinary shader ALUs while leaving room for future SP4E matrix-accelerated extensions.
```

中文描述为：

```text
SP4 是 Stellaria 面向实时视觉推理的 4-bit 分离精度 block 量化系统。
第一版 SP4block32(A1P) / SP4block64(A1P) 默认采用 W4A8 主路径，
并支持 FP16 缩放、稀疏残差校正、项目自适应 block metadata 与敏感路径 FP16 回退。
SP4 不只是一个量化格式，而是一套面向 runtime 的工具链，
包含 Compiler、Asset Container、Runtime Loader、Backend Kernel、Validator、Inspector 与 Benchmark Suite。
它优先面向普通 shader ALU 高效运行，同时为后续 SP4E 的矩阵加速扩展预留空间。
```

---

## 26. 总结

SP4 是 Stellaria 为实时视觉推理设计的 W4A8 分离精度 block 量化系统。

它以：

```text
INT4 权重
INT8 激活
FP16 scale
INT32 accumulate
FP16 / FP32 restore
residual / sparse residual
project-adaptive metadata
sensitive FP16 fallback
```

构成基础格式。

它以：

```text
SP4 Format Spec
SP4 Asset Container
SP4 Compiler
SP4 Project Adapter
SP4-PAC
SP4 Runtime Loader
SP4 Backend Kernel Library
SP4 Kernel Contract
SP4 Inspector
SP4 Validator
SP4 Benchmark & Quality Suite
SP4 Version System
```

构成完整工具链。

SP4 的第一版格式为：

```text
SP4block32(A1P)
SP4block64(A1P)
```

SP4 的核心价值在于：它不是单纯追求低比特压缩，而是在普通 ALU 可高效执行的前提下，通过分离精度、block metadata、稀疏残差和项目自适应校准，使实时视觉推理获得接近 INT4 的效率与接近 FP16 的视觉稳定性。

SP4 的最终定位是：

```text
一个面向 Stellaria runtime 的私有低精度神经推理资产体系。
```
