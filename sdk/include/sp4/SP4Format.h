#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sp4 {

inline constexpr char kMagicA1P[8] = {'S', 'T', 'S', 'P', '4', 'A', '1', 'P'};
inline constexpr uint32_t kContainerVersion = 1;

enum class Mode : uint8_t {
    LinearW4A8 = 0,
    LinearW4A8ZeroPoint = 1,
    Residual = 2,
    SparseResidual = 3,
    SensitiveFp16 = 4,
    AdaptiveProject = 5,
    LearnedProject = 6,
    ReservedSp4E = 7,
};

enum class BackendKind : uint32_t {
    CpuReference = 0,
    Metal = 1,
    SlangPortable = 2,
    Dx12 = 3,
    Vulkan = 4,
};

enum class PrecisionKind : uint32_t {
    Sp4W4A8 = 0,
    AdaptiveProject = 1,
    Int8 = 2,
    Fp16 = 3,
    Fp32 = 4,
    ExternalOptimized = 5,
    Sp4eFp4Main = 6,
    Sp4eInt4Main = 7,
    Sp4eFp4Int4Bridge = 8,
    Sp4eSparseR16 = 9,
};

enum class OpKind : uint32_t {
    Conv2D = 0,
    DepthwiseConv2D = 1,
    PointwiseConv2D = 2,
    Linear = 3,
    Elementwise = 4,
    Activation = 5,
    Requantize = 6,
    ResidualCorrect = 7,
    FallbackFp16 = 8,
    External = 9,
    WarpGridSample = 10,
    Blend = 11,
};

enum class KernelFamily : uint32_t {
    Conv2DW4A8Block32 = 0,
    Conv2DW4A8Block64 = 1,
    PointwiseConv1x1W4A8 = 2,
    ResidualCorrection = 3,
    Requantize = 4,
    Fp16Fallback = 5,
    WarpGridSampleFp16 = 6,
    BlendFp16 = 7,
    ExternalOptimized = 8,
    CustomBackend = 9,
    LinearTile16W4A8 = 10,
    MlpGatedTile16W4A8 = 11,
    SsmProjectionW4A8 = 12,
    SsmScanFp16 = 13,
    MlpFusedProjectionGateSiluW4A8 = 14,
    SsmScanParallelFp16 = 15,
    Sp4eFp4TensorTile16x16 = 16,
    Sp4eInt4TensorTile16x16 = 17,
    Sp4eFp4Int4BridgeTile16x16 = 18,
    Sp4eSparseR16Correction = 19,
    Sp4eSp4Fallback = 20,
};

struct SectionDesc {
    std::string name;
    size_t offset = 0;
    size_t length = 0;
};

struct LayerDesc {
    std::string name;
    std::vector<uint32_t> shape;
    uint32_t blockSize = 32;
    uint32_t numBlocks = 0;
    uint32_t paddedValues = 0;
    uint32_t qweightOffsetBytes = 0;
    uint32_t qweightLengthBytes = 0;
    uint32_t scaleOffset = 0;
    uint32_t biasOffset = 0;
    uint32_t biasCount = 0;
    uint32_t modeOffset = 0;
    uint32_t auxOffset = 0;
    uint32_t residualIndexOffset = 0;
    uint32_t residualValueOffset = 0;
    uint32_t residualTableOffset = 0;
    uint32_t residualEntries = 0;
    uint32_t activationScaleOffset = 0;
    uint32_t auxPolicy = 0;
    bool sensitive = false;
    bool fallbackFp16 = false;
    float rmse = 0.0f;
    float mae = 0.0f;
    float maxAbs = 0.0f;
};

struct AssetManifest {
    std::string version = "A1P";
    std::string adapter = "generic";
    std::string source;
    uint32_t graphTensorCount = 0;
    std::vector<LayerDesc> layers;
    std::vector<std::string> excludedLayers;
    uint64_t teacherWeightBytes = 0;
    uint64_t sp4WeightBytes = 0;
    uint64_t blocks = 0;
    uint64_t residualEntries = 0;
    double compileMs = 0.0;
};

} // namespace sp4
