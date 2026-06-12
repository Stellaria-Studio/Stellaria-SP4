#pragma once

#include "sp4/SP4Format.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sp4 {

inline constexpr char kSP4EMagicA1P[8] = {'S', 'T', 'S', 'P', '4', 'E', '1', 'P'};
inline constexpr uint32_t kSP4EContainerVersion = 1;
inline constexpr uint32_t kSP4EBlockSizeA1P = 16;

enum class SP4EMode : uint8_t {
    Fp4Main = 0,
    Int4Main = 1,
    Fp4WithR16 = 2,
    Int4WithR16 = 3,
    F4I4Bridge = 4,
    LowRankR16 = 5,
    ChannelCorrection = 6,
    SensitiveFp16 = 7,
    SensitiveBf16 = 8,
    ProjectAdaptive = 9,
    BackendNative = 10,
    Sp4Fallback = 11,
};

enum class SP4ERuntimePath : uint32_t {
    Sp4Fallback = 0,
    TensorPath = 1,
    LowQAluPath = 2,
    ProfileGuidedPath = 3,
    Sp4SubsetPath = 4,
};

struct SP4ECapabilityProfile {
    bool nativeFp4TensorSupport = false;
    bool nativeInt4TensorSupport = false;
    bool tensorTileSupport = false;
    bool cooperativeMatrixSupport = false;
    float lowPrecisionAluSpeedup = 1.0f;
    float sp4eVsSp4Speedup = 0.0f;
    std::string backend = "cpu_reference";
    std::string device = "unknown";
};

struct SP4ERuntimeDecision {
    bool enabled = false;
    SP4ERuntimePath path = SP4ERuntimePath::Sp4Fallback;
    std::string reason = "no_profitable_fp4_or_lowq_acceleration";
};

struct SP4ELayerDesc {
    std::string name;
    std::vector<uint32_t> shape;
    uint32_t blockSize = kSP4EBlockSizeA1P;
    uint32_t numBlocks = 0;
    uint32_t paddedValues = 0;
    uint32_t qweightFp4OffsetBytes = 0;
    uint32_t qweightInt4OffsetBytes = 0;
    uint32_t qweightLengthBytes = 0;
    uint32_t scaleFp4Offset = 0;
    uint32_t scaleInt4Offset = 0;
    uint32_t biasOffset = 0;
    uint32_t biasCount = 0;
    uint32_t modeOffset = 0;
    uint32_t auxOffset = 0;
    uint32_t bridgeOffset = 0;
    uint32_t residualIndexOffset = 0;
    uint32_t residualValueOffset = 0;
    uint32_t residualTableOffset = 0;
    uint32_t residualEntries = 0;
    uint32_t activationPrecisionOffset = 0;
    uint32_t channelCorrectionOffset = 0;
    uint32_t channelCorrectionCount = 0;
    uint32_t lowRankUOffset = 0;
    uint32_t lowRankVOffset = 0;
    uint32_t lowRankRank = 0;
    uint32_t auxPolicy = 0;
    bool sensitive = false;
    uint32_t fp4Blocks = 0;
    uint32_t int4Blocks = 0;
    uint32_t bridgeBlocks = 0;
    uint32_t sp4SubsetBlocks = 0;
    uint32_t fp16FallbackBlocks = 0;
    float rmse = 0.0f;
    float mae = 0.0f;
    float maxAbs = 0.0f;
};

struct SP4EManifest {
    std::string version = "A1P";
    std::string adapter = "generic";
    std::string source;
    std::string activationPrecision = "fp16";
    uint32_t graphTensorCount = 0;
    std::vector<SP4ELayerDesc> layers;
    std::vector<std::string> excludedLayers;
    uint64_t teacherWeightBytes = 0;
    uint64_t sp4eWeightBytes = 0;
    uint64_t blocks = 0;
    uint64_t fp4Blocks = 0;
    uint64_t int4Blocks = 0;
    uint64_t bridgeBlocks = 0;
    uint64_t sp4SubsetBlocks = 0;
    uint64_t fp16FallbackBlocks = 0;
    uint64_t residualEntries = 0;
    double compileMs = 0.0;
    SP4ECapabilityProfile capability;
    SP4ERuntimeDecision runtimeDecision;
};

struct SP4ESections {
    std::vector<uint8_t> qweightFp4;
    std::vector<uint8_t> qweightInt4;
    std::vector<uint16_t> scaleFp16Fp4;
    std::vector<uint16_t> scaleFp16Int4;
    std::vector<uint16_t> biasFp16;
    std::vector<uint8_t> mode;
    std::vector<uint8_t> aux;
    std::vector<uint8_t> bridge;
    std::vector<uint8_t> residualIndex;
    std::vector<uint16_t> residualValueFp16;
    std::vector<int32_t> residualTable;
    std::vector<uint8_t> activationPrecision;
    std::vector<uint8_t> tilePlan;
    std::vector<uint16_t> lowRankU;
    std::vector<uint16_t> lowRankV;
    std::vector<uint16_t> channelCorrection;
};

} // namespace sp4
