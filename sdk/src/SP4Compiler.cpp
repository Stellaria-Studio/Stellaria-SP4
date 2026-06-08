#include "sp4/SP4Compiler.h"

#include "sp4/SP4Asset.h"
#include "sp4/SafeTensorsReader.h"
#include "SP4Common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

namespace sp4 {

namespace {

struct LayerPolicy {
    uint32_t blockSize = 32;
    uint8_t aux = 0;
    float residualRatio = 0.02f;
    bool sensitive = false;
    bool fallbackFp16 = false;
    bool optimizeScale = false;
    bool taskAdaptiveResidual = false;
    bool enableZeroPoint = false;
};

uint32_t Product(const std::vector<uint32_t>& shape) {
    uint64_t result = 1;
    for (uint32_t dim : shape) result *= dim;
    return static_cast<uint32_t>(result);
}

bool IsSupportedRuntimeWeight(const TensorInfo& info) {
    return (info.dtype == "F32" || info.dtype == "F16" || info.dtype == "BF16") &&
           (info.shape.size() == 2 || info.shape.size() == 4);
}

bool IsRifeRuntimeLayer(const std::string& name) {
    return name.rfind("block_tea.", 0) != 0;
}

uint32_t StageId(const std::string& name) {
    if (name.rfind("block", 0) != 0) return 0;
    const size_t dot = name.find('.');
    if (dot == std::string::npos) return 0;
    try {
        return static_cast<uint32_t>(std::stoul(name.substr(5, dot - 5)));
    } catch (...) {
        return 0;
    }
}

LayerPolicy PolicyFor(const std::string& adapter,
                      const std::string& name,
                      const std::vector<uint32_t>& shape,
                      const std::string& requestedBlockSize,
                      const std::string& qualityProfile,
                      float residualRatio) {
    LayerPolicy policy;
    const uint32_t valuesPerOutput = shape.size() == 4 ? shape[1] * shape[2] * shape[3] : shape.back();
    if (requestedBlockSize == "32" || requestedBlockSize == "block32") {
        policy.blockSize = 32;
    } else if (requestedBlockSize == "64" || requestedBlockSize == "block64") {
        policy.blockSize = 64;
    } else {
        policy.blockSize = valuesPerOutput <= 128 ? 32 : 64;
    }
    policy.residualRatio = std::max(0.0f, residualRatio);
    policy.optimizeScale = qualityProfile == "adaptive-rife" || qualityProfile == "max-accuracy" || qualityProfile == "int8-chase";
    policy.taskAdaptiveResidual = qualityProfile == "adaptive-rife" || qualityProfile == "max-accuracy" || qualityProfile == "int8-chase";
    policy.enableZeroPoint = qualityProfile == "max-accuracy" || qualityProfile == "int8-chase";

    if (adapter == "rife") {
        const uint32_t stage = StageId(name);
        policy.aux = static_cast<uint8_t>((stage & 0x0fu) << 4u);
        if (name.find(".conv1.") != std::string::npos) {
            policy.aux |= 0b0001u | 0b1000u;
        }
        if (name.find(".conv2.") != std::string::npos) {
            policy.aux |= 0b0010u | 0b1000u;
        }
        if (name.find(".conv0.") != std::string::npos || stage == 0) {
            policy.aux |= 0b0100u;
            policy.sensitive = true;
        }
        if (name.find("convblock") != std::string::npos) {
            policy.sensitive = stage <= 1;
        }
        if (policy.sensitive) {
            policy.residualRatio = std::max(policy.residualRatio, residualRatio * 1.5f);
        }
        if (name.find(".conv1.") != std::string::npos || name.find(".conv2.") != std::string::npos) {
            policy.residualRatio = std::max(policy.residualRatio, residualRatio * 1.25f);
        }
        if (policy.taskAdaptiveResidual) {
            const float stageBoost = stage <= 1 ? 1.25f : (stage >= 3 ? 0.75f : 1.0f);
            const float branchBoost =
                name.find(".conv0.") != std::string::npos ? 1.35f :
                name.find(".conv1.") != std::string::npos ? 1.20f :
                name.find(".conv2.") != std::string::npos ? 1.15f : 0.90f;
            policy.residualRatio = std::max(0.0f, policy.residualRatio * stageBoost * branchBoost);
        }
    }
    if (qualityProfile == "max-accuracy" || qualityProfile == "int8-chase") {
        policy.residualRatio = std::max(policy.residualRatio, residualRatio * 1.75f);
    }
    if (shape.size() == 2 && qualityProfile == "adaptive-rife") {
        policy.optimizeScale = true;
        policy.taskAdaptiveResidual = true;
        policy.residualRatio = std::max(policy.residualRatio * 0.65f, 0.0f);
    }
    const bool isSsmProjection = name.find("in_proj") != std::string::npos ||
                                 name.find("x_proj") != std::string::npos ||
                                 name.find("dt_proj") != std::string::npos ||
                                 name.find("out_proj") != std::string::npos ||
                                 name.find("ssm") != std::string::npos;
    if (isSsmProjection) {
        policy.aux |= 0b0010u;
        policy.blockSize = 64;
        if (qualityProfile != "balanced") {
            policy.residualRatio = std::max(policy.residualRatio, residualRatio * 0.75f);
        }
    }
    const bool isMlp = name.find("mlp") != std::string::npos ||
                       name.find("ffn") != std::string::npos ||
                       name.find("gate_proj") != std::string::npos ||
                       name.find("up_proj") != std::string::npos ||
                       name.find("down_proj") != std::string::npos;
    if (isMlp) {
        policy.aux |= 0b0001u;
        if (qualityProfile != "balanced") {
            policy.residualRatio = std::max(policy.residualRatio, residualRatio * 0.5f);
        }
    }
    return policy;
}

uint8_t PackNibble(int q) {
    return static_cast<uint8_t>(q) & 0x0fu;
}

float CodebookValue(uint32_t code) {
    static constexpr float kCodebook[16] = {
        -1.000000f, -0.696193f, -0.525073f, -0.394917f,
        -0.284441f, -0.184773f, -0.091050f,  0.000000f,
         0.079580f,  0.160930f,  0.246112f,  0.337915f,
         0.440710f,  0.562617f,  0.722956f,  1.000000f,
    };
    return kCodebook[code & 0x0fu];
}

float LearnedCodebookValue(uint32_t code) {
    static constexpr float kCodebook[16] = {
        -1.000000f, -0.770166f, -0.607256f, -0.472373f,
        -0.356107f, -0.252558f, -0.158382f, -0.070901f,
         0.012902f,  0.098159f,  0.189695f,  0.291085f,
         0.407779f,  0.548442f,  0.727477f,  1.000000f,
    };
    return kCodebook[code & 0x0fu];
}

struct QuantizedLayer {
    std::vector<uint8_t> qweight;
    std::vector<uint16_t> scales;
    std::vector<uint8_t> modes;
    std::vector<uint8_t> aux;
    std::vector<uint8_t> residualIndex;
    std::vector<uint16_t> residualValue;
    std::vector<int32_t> residualTable;
    uint32_t numBlocks = 0;
    uint32_t paddedValues = 0;
    float rmse = 0.0f;
    float mae = 0.0f;
    float maxAbs = 0.0f;
};

float QuantErrorForScale(const std::vector<float>& values, uint32_t base, uint32_t valid, float scale) {
    double sq = 0.0;
    for (uint32_t i = 0; i < valid; ++i) {
        const float value = values[base + i];
        const int quant = std::max(-8, std::min(7, static_cast<int>(std::lrint(value / scale))));
        const double err = static_cast<double>(value) - static_cast<double>(quant) * scale;
        sq += err * err;
    }
    return valid ? static_cast<float>(sq / valid) : 0.0f;
}

float OptimizedBlockScale(const std::vector<float>& values, uint32_t base, uint32_t valid, float maxAbsScale) {
    if (valid == 0) return 1.0e-8f;
    float best = maxAbsScale;
    float bestErr = QuantErrorForScale(values, base, valid, best);
    float scale = maxAbsScale;
    for (uint32_t iter = 0; iter < 4; ++iter) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (uint32_t i = 0; i < valid; ++i) {
            const float value = values[base + i];
            const int quant = std::max(-8, std::min(7, static_cast<int>(std::lrint(value / scale))));
            numerator += static_cast<double>(value) * static_cast<double>(quant);
            denominator += static_cast<double>(quant) * static_cast<double>(quant);
        }
        if (denominator > 0.0) {
            scale = std::max(1.0e-8f, static_cast<float>(std::fabs(numerator) / denominator));
        }
        const float err = QuantErrorForScale(values, base, valid, scale);
        if (err < bestErr) {
            best = scale;
            bestErr = err;
        }
    }
    static constexpr float kMultipliers[] = {0.72f, 0.80f, 0.90f, 1.0f, 1.10f, 1.22f, 1.36f};
    for (float m : kMultipliers) {
        const float candidate = std::max(1.0e-8f, best * m);
        const float err = QuantErrorForScale(values, base, valid, candidate);
        if (err < bestErr) {
            best = candidate;
            bestErr = err;
        }
    }
    return best;
}

struct ZeroPointCandidate {
    std::vector<int> q;
    std::vector<float> recon;
    double sq = std::numeric_limits<double>::infinity();
    float scale = 1.0e-8f;
    int zeroPoint = 0;
};

double EvaluateZeroPointBlock(const std::vector<float>& values,
                              uint32_t base,
                              uint32_t valid,
                              uint32_t blockSize,
                              float scale,
                              int zeroPoint,
                              ZeroPointCandidate* out) {
    double sq = 0.0;
    if (out) {
        out->q.assign(blockSize, 0);
        out->recon.assign(blockSize, 0.0f);
    }
    for (uint32_t i = 0; i < valid; ++i) {
        const float value = values[base + i];
        const int quant = std::max(0, std::min(15, static_cast<int>(std::lrint(value / scale + static_cast<float>(zeroPoint)))));
        const float recon = (static_cast<float>(quant) - static_cast<float>(zeroPoint)) * scale;
        const double err = static_cast<double>(value) - static_cast<double>(recon);
        sq += err * err;
        if (out) {
            out->q[i] = quant;
            out->recon[i] = recon;
        }
    }
    return sq;
}

ZeroPointCandidate OptimizedZeroPointBlock(const std::vector<float>& values,
                                           uint32_t base,
                                           uint32_t valid,
                                           uint32_t blockSize,
                                           float minValue,
                                           float maxValue) {
    ZeroPointCandidate best;
    best.q.assign(blockSize, 0);
    best.recon.assign(blockSize, 0.0f);
    if (valid == 0 || maxValue <= minValue) {
        return best;
    }

    const float rangeScale = std::max(1.0e-8f, (maxValue - minValue) / 15.0f);
    for (int zp = 0; zp <= 15; ++zp) {
        const float positiveScale = (maxValue > 0.0f && zp < 15) ? maxValue / static_cast<float>(15 - zp) : 0.0f;
        const float negativeScale = (minValue < 0.0f && zp > 0) ? -minValue / static_cast<float>(zp) : 0.0f;
        float scale = std::max({1.0e-8f, rangeScale, positiveScale, negativeScale});
        double localBestSq = EvaluateZeroPointBlock(values, base, valid, blockSize, scale, zp, nullptr);
        float localBestScale = scale;

        for (uint32_t iter = 0; iter < 5; ++iter) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (uint32_t i = 0; i < valid; ++i) {
                const float value = values[base + i];
                const int quant = std::max(0, std::min(15, static_cast<int>(std::lrint(value / scale + static_cast<float>(zp)))));
                const int centered = quant - zp;
                numerator += static_cast<double>(value) * static_cast<double>(centered);
                denominator += static_cast<double>(centered) * static_cast<double>(centered);
            }
            if (denominator > 0.0) {
                const float solved = static_cast<float>(numerator / denominator);
                if (solved > 1.0e-8f && std::isfinite(solved)) {
                    scale = solved;
                }
            }
            const double sq = EvaluateZeroPointBlock(values, base, valid, blockSize, scale, zp, nullptr);
            if (sq < localBestSq) {
                localBestSq = sq;
                localBestScale = scale;
            }
        }

        static constexpr float kMultipliers[] = {0.78f, 0.86f, 0.94f, 1.0f, 1.06f, 1.14f, 1.28f};
        for (float m : kMultipliers) {
            const float candidateScale = std::max(1.0e-8f, localBestScale * m);
            const double sq = EvaluateZeroPointBlock(values, base, valid, blockSize, candidateScale, zp, nullptr);
            if (sq < localBestSq) {
                localBestSq = sq;
                localBestScale = candidateScale;
            }
        }

        if (localBestSq < best.sq) {
            best.sq = localBestSq;
            best.scale = localBestScale;
            best.zeroPoint = zp;
        }
    }
    EvaluateZeroPointBlock(values, base, valid, blockSize, best.scale, best.zeroPoint, &best);
    return best;
}

QuantizedLayer Quantize(const std::vector<float>& values,
                        uint32_t blockSize,
                        uint8_t auxValue,
                        float residualRatio,
                        bool optimizeScale,
                        bool taskAdaptiveResidual,
                        bool enableZeroPoint) {
    QuantizedLayer ql;
    const uint32_t originalCount = static_cast<uint32_t>(values.size());
    ql.numBlocks = (originalCount + blockSize - 1u) / blockSize;
    ql.paddedValues = ql.numBlocks * blockSize;
    ql.qweight.assign((ql.paddedValues + 1u) / 2u, 0u);
    ql.scales.resize(ql.numBlocks);
    ql.modes.assign(ql.numBlocks, static_cast<uint8_t>(Mode::LinearW4A8));
    ql.aux.assign(ql.numBlocks, auxValue);
    std::vector<int32_t> denseResidualTable(ql.numBlocks * 2u, 0);

    std::vector<float> errors(ql.paddedValues, 0.0f);
    std::vector<float> blockScores(ql.numBlocks, 0.0f);
    std::vector<uint32_t> validCounts(ql.numBlocks, 0u);

    for (uint32_t block = 0; block < ql.numBlocks; ++block) {
        const uint32_t base = block * blockSize;
        const uint32_t valid = std::min(blockSize, originalCount > base ? originalCount - base : 0u);
        validCounts[block] = valid;
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) {
            maxAbs = std::max(maxAbs, std::fabs(values[base + i]));
        }
        const float maxAbsScale = maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f;
        const float linearScale = optimizeScale ? OptimizedBlockScale(values, base, valid, maxAbsScale) : maxAbsScale;
        const float codebookScale = maxAbs > 1.0e-8f ? maxAbs : 1.0e-8f;
        std::vector<int> linearQ(blockSize, 0);
        std::vector<int> codebookQ(blockSize, 0);
        std::vector<int> learnedQ(blockSize, 0);
        std::vector<float> linearRecon(blockSize, 0.0f);
        std::vector<float> codebookRecon(blockSize, 0.0f);
        std::vector<float> learnedRecon(blockSize, 0.0f);
        double linearSq = 0.0;
        double codebookSq = 0.0;
        double learnedSq = 0.0;
        float minValue = 0.0f;
        float maxValue = 0.0f;
        if (valid > 0) {
            minValue = values[base];
            maxValue = values[base];
            for (uint32_t i = 1; i < valid; ++i) {
                minValue = std::min(minValue, values[base + i]);
                maxValue = std::max(maxValue, values[base + i]);
            }
        }
        const ZeroPointCandidate zeroPointCandidate = enableZeroPoint
            ? OptimizedZeroPointBlock(values, base, valid, blockSize, minValue, maxValue)
            : ZeroPointCandidate{};
        for (uint32_t i = 0; i < valid; ++i) {
            const float value = values[base + i];
            linearQ[i] = std::max(-8, std::min(7, static_cast<int>(std::lrint(value / linearScale))));
            linearRecon[i] = static_cast<float>(linearQ[i]) * linearScale;
            double err = static_cast<double>(value) - linearRecon[i];
            linearSq += err * err;

            int best = 0;
            float bestErr = std::fabs(value - CodebookValue(0) * codebookScale);
            for (int c = 1; c < 16; ++c) {
                const float candidateErr = std::fabs(value - CodebookValue(static_cast<uint32_t>(c)) * codebookScale);
                if (candidateErr < bestErr) {
                    best = c;
                    bestErr = candidateErr;
                }
            }
            codebookQ[i] = best;
            codebookRecon[i] = CodebookValue(static_cast<uint32_t>(best)) * codebookScale;
            err = static_cast<double>(value) - codebookRecon[i];
            codebookSq += err * err;

            best = 0;
            bestErr = std::fabs(value - LearnedCodebookValue(0) * codebookScale);
            for (int c = 1; c < 16; ++c) {
                const float candidateErr = std::fabs(value - LearnedCodebookValue(static_cast<uint32_t>(c)) * codebookScale);
                if (candidateErr < bestErr) {
                    best = c;
                    bestErr = candidateErr;
                }
            }
            learnedQ[i] = best;
            learnedRecon[i] = LearnedCodebookValue(static_cast<uint32_t>(best)) * codebookScale;
            err = static_cast<double>(value) - learnedRecon[i];
            learnedSq += err * err;
        }
        enum class BlockChoice { Linear, ZeroPoint, Codebook, LearnedCodebook };
        BlockChoice choice = BlockChoice::Linear;
        double bestSq = linearSq;
        if (enableZeroPoint && zeroPointCandidate.sq < bestSq * 0.985) {
            choice = BlockChoice::ZeroPoint;
            bestSq = zeroPointCandidate.sq;
        }
        if (optimizeScale && codebookSq < bestSq * 0.92) {
            choice = BlockChoice::Codebook;
            bestSq = codebookSq;
        }
        if (optimizeScale && learnedSq < bestSq * 0.92) {
            choice = BlockChoice::LearnedCodebook;
            bestSq = learnedSq;
        }
        const bool useCodebook = choice == BlockChoice::Codebook;
        const bool useLearnedCodebook = choice == BlockChoice::LearnedCodebook;
        const bool useZeroPoint = choice == BlockChoice::ZeroPoint;
        const float scale = (useCodebook || useLearnedCodebook) ? codebookScale : (useZeroPoint ? zeroPointCandidate.scale : linearScale);
        ql.scales[block] = FloatToHalf(scale);
        if (useCodebook) {
            ql.modes[block] = static_cast<uint8_t>(Mode::AdaptiveProject);
        } else if (useLearnedCodebook) {
            ql.modes[block] = static_cast<uint8_t>(Mode::LearnedProject);
        } else if (useZeroPoint) {
            ql.modes[block] = static_cast<uint8_t>(Mode::LinearW4A8ZeroPoint);
        } else {
            ql.modes[block] = static_cast<uint8_t>(Mode::LinearW4A8);
        }
        if (useZeroPoint) {
            ql.aux[block] = static_cast<uint8_t>((auxValue & 0xf0u) | static_cast<uint8_t>(zeroPointCandidate.zeroPoint & 0x0f));
        }
        double score = 0.0;
        double maxScore = 0.0;
        for (uint32_t i = 0; i < blockSize; ++i) {
            const uint32_t index = base + i;
            const float value = index < originalCount ? values[index] : 0.0f;
            const int quant = useCodebook ? codebookQ[i] :
                (useLearnedCodebook ? learnedQ[i] : (useZeroPoint ? zeroPointCandidate.q[i] : linearQ[i]));
            const uint32_t packedOffset = index / 2u;
            if ((index & 1u) == 0u) {
                ql.qweight[packedOffset] = PackNibble(quant);
            } else {
                ql.qweight[packedOffset] |= static_cast<uint8_t>(PackNibble(quant) << 4u);
            }
            const float recon = useCodebook ? codebookRecon[i] :
                (useLearnedCodebook ? learnedRecon[i] : (useZeroPoint ? zeroPointCandidate.recon[i] : linearRecon[i]));
            const float err = value - recon;
            errors[index] = err;
            if (i < valid) {
                const double weighted = static_cast<double>(err) * static_cast<double>(err);
                score += weighted;
                maxScore = std::max(maxScore, std::fabs(static_cast<double>(err)));
            }
        }
        blockScores[block] = valid > 0
            ? static_cast<float>(std::sqrt(score / static_cast<double>(valid)) + (taskAdaptiveResidual ? maxScore * 0.35 : 0.0))
            : 0.0f;
    }

    if (residualRatio > 0.0f) {
        const uint32_t totalBudget = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(originalCount) * residualRatio)));
        const float perBlockRatio = std::min(taskAdaptiveResidual ? 0.1875f : 0.125f,
                                             std::max(residualRatio * (taskAdaptiveResidual ? 5.0f : 4.0f), residualRatio));
        const uint32_t entriesPerSelectedBlock = std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(blockSize) * perBlockRatio)));
        std::vector<uint32_t> candidates(ql.numBlocks);
        std::iota(candidates.begin(), candidates.end(), 0u);
        std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
            return blockScores[a] > blockScores[b];
        });
        uint32_t used = 0;
        for (uint32_t block : candidates) {
            if (used >= totalBudget || validCounts[block] == 0 || blockScores[block] <= 0.0f) continue;
            const uint32_t count = std::min({validCounts[block], entriesPerSelectedBlock, totalBudget - used});
            std::vector<uint32_t> indices(validCounts[block]);
            std::iota(indices.begin(), indices.end(), 0u);
            const uint32_t base = block * blockSize;
            std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&](uint32_t a, uint32_t b) {
                return std::fabs(errors[base + a]) > std::fabs(errors[base + b]);
            });
            std::sort(indices.begin(), indices.begin() + count);
            const int32_t start = static_cast<int32_t>(ql.residualIndex.size());
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t local = indices[i];
                ql.residualIndex.push_back(static_cast<uint8_t>(local));
                ql.residualValue.push_back(FloatToHalf(errors[base + local]));
                errors[base + local] = 0.0f;
            }
            denseResidualTable[block * 2u + 0u] = start;
            denseResidualTable[block * 2u + 1u] = static_cast<int32_t>(count);
            if (ql.modes[block] == static_cast<uint8_t>(Mode::LinearW4A8)) {
                ql.modes[block] = static_cast<uint8_t>(Mode::SparseResidual);
            }
            used += count;
        }
    }

    for (uint32_t block = 0; block < ql.numBlocks; ++block) {
        const int32_t start = denseResidualTable[block * 2u + 0u];
        const int32_t count = denseResidualTable[block * 2u + 1u];
        if (count > 0) {
            ql.residualTable.push_back(static_cast<int32_t>(block));
            ql.residualTable.push_back(start);
            ql.residualTable.push_back(count);
        }
    }

    double sq = 0.0;
    double ab = 0.0;
    float maxAbsErr = 0.0f;
    for (uint32_t i = 0; i < originalCount; ++i) {
        sq += static_cast<double>(errors[i]) * static_cast<double>(errors[i]);
        ab += std::fabs(errors[i]);
        maxAbsErr = std::max(maxAbsErr, std::fabs(errors[i]));
    }
    ql.rmse = originalCount ? static_cast<float>(std::sqrt(sq / originalCount)) : 0.0f;
    ql.mae = originalCount ? static_cast<float>(ab / originalCount) : 0.0f;
    ql.maxAbs = maxAbsErr;
    return ql;
}

uint64_t SectionBytes(const AssetSections& sections) {
    return sections.qweight.size() +
           sections.scaleFp16.size() * sizeof(uint16_t) +
           sections.biasFp16.size() * sizeof(uint16_t) +
           sections.mode.size() +
           sections.aux.size() +
           sections.residualIndex.size() +
           sections.residualValueFp16.size() * sizeof(uint16_t) +
           sections.residualTable.size() * sizeof(int32_t) +
           sections.activationScaleFp16.size() * sizeof(uint16_t);
}

} // namespace

CompileResult CompileSafetensorsToSp4(const CompileOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    CompileResult result;
    SafeTensorsReader reader;
    std::string error;
    if (!reader.Open(options.inputPath, &error)) {
        result.message = error;
        return result;
    }

    AssetSections sections;
    AssetManifest manifest;
    manifest.adapter = options.adapter;
    manifest.source = options.inputPath;
    manifest.graphTensorCount = static_cast<uint32_t>(reader.Tensors().size());

    const std::vector<std::string> weightNames = options.includeLinearLayers ? reader.SupportedWeightNames() : reader.ConvWeightNames();
    for (const std::string& name : weightNames) {
        if (options.adapter == "rife" && !options.includeNonRuntimeLayers && !IsRifeRuntimeLayer(name)) {
            manifest.excludedLayers.push_back(name);
            continue;
        }
        const auto tensorIt = reader.Tensors().find(name);
        if (tensorIt == reader.Tensors().end()) continue;
        if (!IsSupportedRuntimeWeight(tensorIt->second)) continue;
        std::vector<float> weights;
        if (!reader.ReadF32(name, weights, &error)) {
            result.message = error;
            return result;
        }
        const LayerPolicy policy = PolicyFor(options.adapter, name, tensorIt->second.shape, options.blockSize, options.qualityProfile, options.residualRatio);
        QuantizedLayer ql = Quantize(weights, policy.blockSize, policy.aux, policy.residualRatio, policy.optimizeScale, policy.taskAdaptiveResidual, policy.enableZeroPoint);

        LayerDesc layer;
        layer.name = name;
        layer.shape = tensorIt->second.shape;
        layer.blockSize = policy.blockSize;
        layer.numBlocks = ql.numBlocks;
        layer.paddedValues = ql.paddedValues;
        layer.qweightOffsetBytes = static_cast<uint32_t>(sections.qweight.size());
        layer.qweightLengthBytes = static_cast<uint32_t>(ql.qweight.size());
        layer.scaleOffset = static_cast<uint32_t>(sections.scaleFp16.size());
        layer.biasOffset = static_cast<uint32_t>(sections.biasFp16.size());
        layer.modeOffset = static_cast<uint32_t>(sections.mode.size());
        layer.auxOffset = static_cast<uint32_t>(sections.aux.size());
        layer.residualIndexOffset = static_cast<uint32_t>(sections.residualIndex.size());
        layer.residualValueOffset = static_cast<uint32_t>(sections.residualValueFp16.size());
        layer.residualTableOffset = static_cast<uint32_t>(sections.residualTable.size() / 3u);
        layer.residualEntries = static_cast<uint32_t>(ql.residualIndex.size());
        layer.activationScaleOffset = static_cast<uint32_t>(sections.activationScaleFp16.size());
        layer.auxPolicy = policy.aux;
        layer.sensitive = policy.sensitive;
        layer.fallbackFp16 = policy.fallbackFp16;
        layer.rmse = ql.rmse;
        layer.mae = ql.mae;
        layer.maxAbs = ql.maxAbs;

        const std::string biasName = BasenameNoExt(name) + ".bias";
        std::vector<float> biasF32;
        if (!reader.ReadF32(biasName, biasF32, nullptr)) {
            biasF32.assign(layer.shape.empty() ? 0u : layer.shape[0], 0.0f);
        }
        layer.biasCount = static_cast<uint32_t>(biasF32.size());

        sections.qweight.insert(sections.qweight.end(), ql.qweight.begin(), ql.qweight.end());
        sections.scaleFp16.insert(sections.scaleFp16.end(), ql.scales.begin(), ql.scales.end());
        for (float value : biasF32) sections.biasFp16.push_back(FloatToHalf(value));
        sections.mode.insert(sections.mode.end(), ql.modes.begin(), ql.modes.end());
        sections.aux.insert(sections.aux.end(), ql.aux.begin(), ql.aux.end());
        sections.residualIndex.insert(sections.residualIndex.end(), ql.residualIndex.begin(), ql.residualIndex.end());
        sections.residualValueFp16.insert(sections.residualValueFp16.end(), ql.residualValue.begin(), ql.residualValue.end());
        sections.residualTable.insert(sections.residualTable.end(), ql.residualTable.begin(), ql.residualTable.end());
        sections.activationScaleFp16.push_back(FloatToHalf(1.0f / 127.0f));

        manifest.teacherWeightBytes += static_cast<uint64_t>(weights.size()) * sizeof(uint16_t);
        manifest.blocks += ql.numBlocks;
        manifest.residualEntries += ql.residualIndex.size();
        manifest.layers.push_back(std::move(layer));
    }

    manifest.sp4WeightBytes = SectionBytes(sections);
    manifest.compileMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    if (!WriteAsset(options.outputPath, manifest, sections, &error)) {
        result.message = error;
        return result;
    }
    result.ok = true;
    result.message = "ok";
    result.manifest = std::move(manifest);
    return result;
}

} // namespace sp4
