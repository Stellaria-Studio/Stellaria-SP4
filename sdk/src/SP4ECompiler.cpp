#include "sp4/SP4ECompiler.h"

#include "sp4/SafeTensorsReader.h"
#include "sp4/SP4ERuntime.h"
#include "SP4Common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

namespace sp4 {

namespace {

constexpr float kFp4E2M1Codebook[16] = {
    -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.0f,
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
};

struct LayerPolicy {
    uint8_t aux = 0;
    float residualRatio = 0.02f;
    bool sensitive = false;
};

struct QuantizedSP4ELayer {
    std::vector<uint8_t> qweightFp4;
    std::vector<uint8_t> qweightInt4;
    std::vector<uint16_t> scaleFp4;
    std::vector<uint16_t> scaleInt4;
    std::vector<uint8_t> mode;
    std::vector<uint8_t> aux;
    std::vector<uint8_t> bridge;
    std::vector<uint8_t> residualIndex;
    std::vector<uint16_t> residualValue;
    std::vector<int32_t> residualTable;
    std::vector<uint16_t> channelCorrection;
    std::vector<uint16_t> lowRankU;
    std::vector<uint16_t> lowRankV;
    uint32_t numBlocks = 0;
    uint32_t paddedValues = 0;
    uint32_t fp4Blocks = 0;
    uint32_t int4Blocks = 0;
    uint32_t bridgeBlocks = 0;
    uint32_t sp4SubsetBlocks = 0;
    uint32_t fp16FallbackBlocks = 0;
    float rmse = 0.0f;
    float mae = 0.0f;
    float maxAbs = 0.0f;
};

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
                      float residualRatio,
                      const std::string& qualityProfile) {
    LayerPolicy policy;
    policy.residualRatio = std::max(0.0f, residualRatio);
    if (adapter == "rife") {
        const uint32_t stage = StageId(name);
        policy.aux = static_cast<uint8_t>((stage & 0x0fu) << 4u);
        if (name.find(".conv1.") != std::string::npos) policy.aux |= 0b0001u | 0b1000u;
        if (name.find(".conv2.") != std::string::npos) policy.aux |= 0b0010u | 0b1000u;
        if (name.find(".conv0.") != std::string::npos || stage == 0) {
            policy.aux |= 0b0100u;
            policy.sensitive = true;
        }
        if (name.find("convblock") != std::string::npos) policy.sensitive = stage <= 1;
        if (policy.sensitive) policy.residualRatio = std::max(policy.residualRatio, residualRatio * 1.5f);
        if (name.find(".conv1.") != std::string::npos || name.find(".conv2.") != std::string::npos) {
            policy.residualRatio = std::max(policy.residualRatio, residualRatio * 1.25f);
        }
        if (qualityProfile == "max-accuracy") {
            float taskMultiplier = 0.35f;
            if (stage <= 1) taskMultiplier = 1.55f;
            if (name.find(".conv0.") != std::string::npos) taskMultiplier = std::max(taskMultiplier, 2.25f);
            if (name.find(".conv1.") != std::string::npos) taskMultiplier = std::max(taskMultiplier, 1.70f);
            if (name.find(".conv2.") != std::string::npos) taskMultiplier = std::max(taskMultiplier, 1.60f);
            if (name.find("convblock") != std::string::npos && stage <= 1) taskMultiplier = std::max(taskMultiplier, 1.75f);
            policy.residualRatio = std::min(1.0f, residualRatio * taskMultiplier);
        }
    }
    return policy;
}

uint8_t PackNibble(int value) {
    return static_cast<uint8_t>(value) & 0x0fu;
}

void Pack4(const std::vector<int>& values, std::vector<uint8_t>& out) {
    out.assign((values.size() + 1u) / 2u, 0u);
    for (size_t i = 0; i < values.size(); ++i) {
        if ((i & 1u) == 0u) out[i / 2u] = PackNibble(values[i]);
        else out[i / 2u] |= static_cast<uint8_t>(PackNibble(values[i]) << 4u);
    }
}

std::vector<uint8_t> PackNibbles(const std::vector<uint8_t>& values) {
    std::vector<uint8_t> out((values.size() + 1u) / 2u, 0u);
    for (size_t i = 0; i < values.size(); ++i) {
        if ((i & 1u) == 0u) out[i / 2u] = PackNibble(values[i]);
        else out[i / 2u] |= static_cast<uint8_t>(PackNibble(values[i]) << 4u);
    }
    return out;
}

uint64_t PackedControlBytes(uint32_t blocks) {
    return (static_cast<uint64_t>(blocks) + 1u) / 2u;
}

std::vector<int8_t> RifeProbeActivation(uint32_t count, uint32_t probe) {
    std::vector<int8_t> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        int v = 0;
        if (probe == 0) {
            v = (static_cast<int>(i * 13u + 7u) % 127) - 63;
        } else if (probe == 1) {
            v = (i & 1u) ? 63 : -64;
        } else if (probe == 2) {
            v = static_cast<int>((i * 9u + (i / 3u) * 17u) % 127u) - 63;
        } else if (probe == 3) {
            v = (i % 11u == 0u || i % 13u == 0u) ? 96 : ((i & 1u) ? -12 : 12);
        } else if (probe == 4) {
            v = static_cast<int>((i % 5u) * 28u) - 56;
        } else {
            const int sign = ((i / 7u) & 1u) ? -1 : 1;
            v = sign * (24 + static_cast<int>((i * 19u) % 72u));
        }
        values[i] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
    }
    return values;
}

std::vector<float> ResidualPriorityScores(const std::vector<float>& errors,
                                          const std::vector<uint32_t>& shape,
                                          uint32_t paddedValues) {
    std::vector<float> scores(paddedValues, 0.0f);
    if (shape.size() != 4u || shape[0] == 0u) {
        for (size_t i = 0; i < errors.size() && i < scores.size(); ++i) scores[i] = std::fabs(errors[i]);
        return scores;
    }
    const uint32_t outputChannels = shape[0];
    const uint32_t valuesPerOutput = shape[1] * shape[2] * shape[3];
    if (valuesPerOutput == 0u || static_cast<uint64_t>(outputChannels) * valuesPerOutput > errors.size()) {
        for (size_t i = 0; i < errors.size() && i < scores.size(); ++i) scores[i] = std::fabs(errors[i]);
        return scores;
    }
    std::vector<std::vector<int8_t>> probes;
    for (uint32_t probe = 0; probe < 6u; ++probe) probes.push_back(RifeProbeActivation(valuesPerOutput, probe));
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        const uint32_t base = oc * valuesPerOutput;
        for (uint32_t probe = 0; probe < probes.size(); ++probe) {
            const std::vector<int8_t>& activation = probes[probe];
            const double probeWeight = probe == 1u ? 1.55 : (probe == 3u ? 1.45 : (probe >= 4u ? 1.25 : 1.0));
            double projectionErr = 0.0;
            for (uint32_t i = 0; i < valuesPerOutput; ++i) {
                projectionErr += static_cast<double>(errors[base + i]) * static_cast<double>(activation[i]);
            }
            for (uint32_t i = 0; i < valuesPerOutput; ++i) {
                const double contribution = static_cast<double>(errors[base + i]) * static_cast<double>(activation[i]);
                const double improvement = 2.0 * projectionErr * contribution - contribution * contribution;
                if (improvement > 0.0) scores[base + i] += static_cast<float>(probeWeight * improvement);
            }
        }
        for (uint32_t i = 0; i < valuesPerOutput; ++i) {
            scores[base + i] += std::fabs(errors[base + i]) * 1.0e-6f;
        }
    }
    return scores;
}

bool ApplyRank1LowRankCorrection(const std::vector<float>& values,
                                 const std::vector<uint32_t>& shape,
                                 std::vector<float>& reconstruction,
                                 std::vector<float>& errors,
                                 std::vector<uint16_t>& lowRankU,
                                 std::vector<uint16_t>& lowRankV) {
    if (shape.size() != 4u || shape[0] == 0u) return false;
    const uint32_t outputChannels = shape[0];
    const uint32_t valuesPerOutput = shape[1] * shape[2] * shape[3];
    if (valuesPerOutput == 0u || static_cast<uint64_t>(outputChannels) * valuesPerOutput > values.size()) return false;

    std::vector<float> v(valuesPerOutput, 1.0f / std::sqrt(static_cast<float>(valuesPerOutput)));
    std::vector<float> u(outputChannels, 0.0f);
    for (uint32_t iter = 0; iter < 8u; ++iter) {
        std::fill(u.begin(), u.end(), 0.0f);
        for (uint32_t oc = 0; oc < outputChannels; ++oc) {
            const uint32_t base = oc * valuesPerOutput;
            double dot = 0.0;
            for (uint32_t k = 0; k < valuesPerOutput; ++k) dot += static_cast<double>(errors[base + k]) * v[k];
            u[oc] = static_cast<float>(dot);
        }
        double uNorm = 0.0;
        for (float x : u) uNorm += static_cast<double>(x) * x;
        uNorm = std::sqrt(uNorm);
        if (uNorm <= 1.0e-20) return false;
        for (float& x : u) x = static_cast<float>(x / uNorm);

        std::fill(v.begin(), v.end(), 0.0f);
        for (uint32_t oc = 0; oc < outputChannels; ++oc) {
            const uint32_t base = oc * valuesPerOutput;
            for (uint32_t k = 0; k < valuesPerOutput; ++k) v[k] += errors[base + k] * u[oc];
        }
        double vNorm = 0.0;
        for (float x : v) vNorm += static_cast<double>(x) * x;
        vNorm = std::sqrt(vNorm);
        if (vNorm <= 1.0e-20) return false;
        for (float& x : v) x = static_cast<float>(x / vNorm);
    }

    std::vector<float> uScaled(outputChannels, 0.0f);
    double sigmaSq = 0.0;
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        const uint32_t base = oc * valuesPerOutput;
        double dot = 0.0;
        for (uint32_t k = 0; k < valuesPerOutput; ++k) dot += static_cast<double>(errors[base + k]) * v[k];
        uScaled[oc] = static_cast<float>(dot);
        sigmaSq += dot * dot;
    }
    if (sigmaSq <= 1.0e-20) return false;

    double currentSq = 0.0;
    double correctedSq = 0.0;
    std::vector<float> uHalf(outputChannels, 0.0f);
    std::vector<float> vHalf(valuesPerOutput, 0.0f);
    for (uint32_t oc = 0; oc < outputChannels; ++oc) uHalf[oc] = HalfToFloat(FloatToHalf(uScaled[oc]));
    for (uint32_t k = 0; k < valuesPerOutput; ++k) vHalf[k] = HalfToFloat(FloatToHalf(v[k]));
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        const uint32_t base = oc * valuesPerOutput;
        for (uint32_t k = 0; k < valuesPerOutput; ++k) {
            const float err = errors[base + k];
            const float correction = uHalf[oc] * vHalf[k];
            currentSq += static_cast<double>(err) * err;
            const double after = static_cast<double>(err) - correction;
            correctedSq += after * after;
        }
    }
    if (correctedSq >= currentSq * 0.995) return false;

    const size_t uOffset = lowRankU.size();
    const size_t vOffset = lowRankV.size();
    lowRankU.resize(uOffset + outputChannels);
    lowRankV.resize(vOffset + valuesPerOutput);
    for (uint32_t oc = 0; oc < outputChannels; ++oc) lowRankU[uOffset + oc] = FloatToHalf(uScaled[oc]);
    for (uint32_t k = 0; k < valuesPerOutput; ++k) lowRankV[vOffset + k] = FloatToHalf(v[k]);
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        const uint32_t base = oc * valuesPerOutput;
        for (uint32_t k = 0; k < valuesPerOutput; ++k) {
            const float correction = uHalf[oc] * vHalf[k];
            reconstruction[base + k] += correction;
            errors[base + k] -= correction;
        }
    }
    return true;
}

int BestFp4Code(float normalized) {
    int best = 0;
    float bestErr = std::fabs(normalized - kFp4E2M1Codebook[0]);
    for (int i = 1; i < 16; ++i) {
        const float err = std::fabs(normalized - kFp4E2M1Codebook[i]);
        if (err < bestErr) {
            best = i;
            bestErr = err;
        }
    }
    return best;
}

struct ZeroPointBlock {
    std::vector<int> codes;
    std::vector<float> recon;
    float scale = 1.0e-8f;
    int zeroPoint = 0;
    double sq = std::numeric_limits<double>::infinity();
};

ZeroPointBlock ChooseZeroPointBlock(const std::vector<float>& values,
                                    uint32_t base,
                                    uint32_t valid) {
    ZeroPointBlock best;
    best.codes.assign(kSP4EBlockSizeA1P, 0);
    best.recon.assign(kSP4EBlockSizeA1P, 0.0f);
    if (valid == 0) return best;
    float minValue = values[base];
    float maxValue = values[base];
    for (uint32_t i = 1; i < valid; ++i) {
        minValue = std::min(minValue, values[base + i]);
        maxValue = std::max(maxValue, values[base + i]);
    }
    if (maxValue <= minValue) return best;
    for (int zp = 0; zp <= 15; ++zp) {
        const float positiveScale = (maxValue > 0.0f && zp < 15) ? maxValue / static_cast<float>(15 - zp) : 0.0f;
        const float negativeScale = (minValue < 0.0f && zp > 0) ? -minValue / static_cast<float>(zp) : 0.0f;
        float scale = std::max({1.0e-8f, positiveScale, negativeScale, (maxValue - minValue) / 15.0f});
        double localBestSq = std::numeric_limits<double>::infinity();
        float localBestScale = scale;
        for (uint32_t iter = 0; iter < 5; ++iter) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (uint32_t i = 0; i < valid; ++i) {
                const int code = std::max(0, std::min(15, static_cast<int>(std::lrint(values[base + i] / scale + static_cast<float>(zp)))));
                const int centered = code - zp;
                numerator += static_cast<double>(values[base + i]) * centered;
                denominator += static_cast<double>(centered) * centered;
            }
            if (denominator > 0.0) {
                const float solved = static_cast<float>(numerator / denominator);
                if (solved > 1.0e-8f && std::isfinite(solved)) scale = solved;
            }
            double sq = 0.0;
            for (uint32_t i = 0; i < valid; ++i) {
                const int code = std::max(0, std::min(15, static_cast<int>(std::lrint(values[base + i] / scale + static_cast<float>(zp)))));
                const float recon = static_cast<float>(code - zp) * scale;
                const double err = static_cast<double>(values[base + i]) - recon;
                sq += err * err;
            }
            if (sq < localBestSq) {
                localBestSq = sq;
                localBestScale = scale;
            }
        }
        if (localBestSq < best.sq) {
            best.sq = localBestSq;
            best.scale = localBestScale;
            best.zeroPoint = zp;
        }
    }
    for (uint32_t i = 0; i < valid; ++i) {
        const int code = std::max(0, std::min(15, static_cast<int>(std::lrint(values[base + i] / best.scale + static_cast<float>(best.zeroPoint)))));
        best.codes[i] = code;
        best.recon[i] = static_cast<float>(code - best.zeroPoint) * best.scale;
    }
    return best;
}

struct ScalarQuantBlock {
    std::vector<int> codes;
    std::vector<float> recon;
    float scale = 1.0e-8f;
    double sq = std::numeric_limits<double>::infinity();
};

ScalarQuantBlock ChooseFp4Block(const std::vector<float>& values,
                                uint32_t base,
                                uint32_t valid,
                                float initialScale) {
    ScalarQuantBlock best;
    best.codes.assign(kSP4EBlockSizeA1P, 0);
    best.recon.assign(kSP4EBlockSizeA1P, 0.0f);
    if (valid == 0) return best;
    float scale = std::max(1.0e-8f, initialScale);
    for (uint32_t iter = 0; iter < 6; ++iter) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (uint32_t i = 0; i < valid; ++i) {
            const int code = BestFp4Code(values[base + i] / scale);
            const float basis = kFp4E2M1Codebook[code];
            numerator += static_cast<double>(values[base + i]) * basis;
            denominator += static_cast<double>(basis) * basis;
        }
        if (denominator > 0.0) {
            const float solved = static_cast<float>(numerator / denominator);
            if (solved > 1.0e-8f && std::isfinite(solved)) scale = solved;
        }
    }
    best.scale = scale;
    best.sq = 0.0;
    for (uint32_t i = 0; i < valid; ++i) {
        const int code = BestFp4Code(values[base + i] / scale);
        best.codes[i] = code;
        best.recon[i] = kFp4E2M1Codebook[code] * scale;
        const double err = static_cast<double>(values[base + i]) - best.recon[i];
        best.sq += err * err;
    }
    return best;
}

ScalarQuantBlock ChooseInt4Block(const std::vector<float>& values,
                                 uint32_t base,
                                 uint32_t valid,
                                 float initialScale) {
    ScalarQuantBlock best;
    best.codes.assign(kSP4EBlockSizeA1P, 0);
    best.recon.assign(kSP4EBlockSizeA1P, 0.0f);
    if (valid == 0) return best;
    float scale = std::max(1.0e-8f, initialScale);
    for (uint32_t iter = 0; iter < 6; ++iter) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (uint32_t i = 0; i < valid; ++i) {
            const int code = std::max(-8, std::min(7, static_cast<int>(std::lrint(values[base + i] / scale))));
            numerator += static_cast<double>(values[base + i]) * code;
            denominator += static_cast<double>(code) * code;
        }
        if (denominator > 0.0) {
            const float solved = static_cast<float>(numerator / denominator);
            if (solved > 1.0e-8f && std::isfinite(solved)) scale = solved;
        }
    }
    best.scale = scale;
    best.sq = 0.0;
    for (uint32_t i = 0; i < valid; ++i) {
        const int code = std::max(-8, std::min(7, static_cast<int>(std::lrint(values[base + i] / scale))));
        best.codes[i] = code;
        best.recon[i] = static_cast<float>(code) * scale;
        const double err = static_cast<double>(values[base + i]) - best.recon[i];
        best.sq += err * err;
    }
    return best;
}

uint8_t ActivationPrecisionCode(const std::string& precision) {
    if (precision == "bf16") return 1;
    if (precision == "fp8") return 2;
    if (precision == "fp4") return 3;
    return 0;
}

QuantizedSP4ELayer QuantizeSP4E(const std::vector<float>& values,
                                const std::vector<uint32_t>& shape,
                                uint8_t auxValue,
                                float residualRatio,
                                float fp16FallbackRatio,
                                float maxResidualFractionPerBlock) {
    QuantizedSP4ELayer ql;
    const uint32_t originalCount = static_cast<uint32_t>(values.size());
    ql.numBlocks = (originalCount + kSP4EBlockSizeA1P - 1u) / kSP4EBlockSizeA1P;
    ql.paddedValues = ql.numBlocks * kSP4EBlockSizeA1P;
    ql.mode.assign(ql.numBlocks, static_cast<uint8_t>(SP4EMode::Fp4Main));
    ql.aux.assign(ql.numBlocks, auxValue);
    ql.scaleFp4.resize(ql.numBlocks);
    ql.residualTable.assign(ql.numBlocks * 2u, 0);

    std::vector<int> fp4Codes(ql.paddedValues, 0);
    std::vector<int> int4Codes(ql.paddedValues, 0);
    std::vector<int> mainCodes(ql.paddedValues, 0);
    std::vector<float> reconstruction(ql.paddedValues, 0.0f);
    std::vector<float> errors(ql.paddedValues, 0.0f);
    std::vector<float> blockScores(ql.numBlocks, 0.0f);
    std::vector<uint32_t> validCounts(ql.numBlocks, 0u);

    for (uint32_t block = 0; block < ql.numBlocks; ++block) {
        const uint32_t base = block * kSP4EBlockSizeA1P;
        const uint32_t valid = std::min(kSP4EBlockSizeA1P, originalCount > base ? originalCount - base : 0u);
        validCounts[block] = valid;
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) maxAbs = std::max(maxAbs, std::fabs(values[base + i]));
        const ScalarQuantBlock fp4Block = ChooseFp4Block(values, base, valid, maxAbs > 1.0e-8f ? maxAbs / 6.0f : 1.0e-8f);
        const ScalarQuantBlock int4Block = ChooseInt4Block(values, base, valid, maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f);
        const ZeroPointBlock zpBlock = ChooseZeroPointBlock(values, base, valid);
        for (uint32_t i = 0; i < valid; ++i) {
            fp4Codes[base + i] = fp4Block.codes[i];
            int4Codes[base + i] = int4Block.codes[i];
        }

        enum class Choice { Fp4, Int4, Zp };
        Choice choice = fp4Block.sq <= int4Block.sq ? Choice::Fp4 : Choice::Int4;
        double bestSq = std::min(fp4Block.sq, int4Block.sq);
        if (zpBlock.sq < bestSq * 0.995) {
            choice = Choice::Zp;
            bestSq = zpBlock.sq;
        }
        if (choice == Choice::Fp4) {
            ql.mode[block] = static_cast<uint8_t>(SP4EMode::Fp4Main);
            ql.scaleFp4[block] = FloatToHalf(fp4Block.scale);
            ql.fp4Blocks++;
        } else if (choice == Choice::Int4) {
            ql.mode[block] = static_cast<uint8_t>(SP4EMode::Int4Main);
            ql.scaleFp4[block] = FloatToHalf(int4Block.scale);
            ql.int4Blocks++;
        } else {
            ql.mode[block] = static_cast<uint8_t>(SP4EMode::Sp4Fallback);
            ql.scaleFp4[block] = FloatToHalf(zpBlock.scale);
            ql.aux[block] = static_cast<uint8_t>((auxValue & 0xf0u) | static_cast<uint8_t>(zpBlock.zeroPoint & 0x0f));
            ql.int4Blocks++;
        }

        double score = 0.0;
        for (uint32_t i = 0; i < kSP4EBlockSizeA1P; ++i) {
            const uint32_t index = base + i;
            const float value = index < originalCount ? values[index] : 0.0f;
            float recon = fp4Block.recon[i];
            mainCodes[index] = fp4Codes[index];
            if (ql.mode[block] == static_cast<uint8_t>(SP4EMode::Int4Main)) {
                recon = int4Block.recon[i];
                mainCodes[index] = int4Codes[index];
            } else if (ql.mode[block] == static_cast<uint8_t>(SP4EMode::Sp4Fallback)) {
                recon = zpBlock.recon[i];
                mainCodes[index] = zpBlock.codes[i];
            }
            reconstruction[index] = recon;
            errors[index] = value - recon;
            if (i < valid) score += static_cast<double>(errors[index]) * errors[index];
        }
        blockScores[block] = valid ? static_cast<float>(std::sqrt(score / valid)) : 0.0f;
    }

    const std::vector<float> residualPriority = ResidualPriorityScores(errors, shape, ql.paddedValues);
    for (uint32_t block = 0; block < ql.numBlocks; ++block) {
        const uint32_t base = block * kSP4EBlockSizeA1P;
        float score = 0.0f;
        for (uint32_t i = 0; i < validCounts[block]; ++i) score += residualPriority[base + i];
        blockScores[block] = score;
    }

    std::vector<uint8_t> blockHasFp16Fallback(ql.numBlocks, 0u);
    if (fp16FallbackRatio > 0.0f && originalCount > 0) {
        const uint32_t fallbackBlocks = std::min<uint32_t>(
            ql.numBlocks,
            std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(ql.numBlocks) * fp16FallbackRatio))));
        std::vector<uint32_t> candidates(ql.numBlocks);
        std::iota(candidates.begin(), candidates.end(), 0u);
        std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) { return blockScores[a] > blockScores[b]; });
        for (uint32_t rank = 0; rank < fallbackBlocks; ++rank) {
            const uint32_t block = candidates[rank];
            if (validCounts[block] == 0 || blockScores[block] <= 0.0f) continue;
            const uint32_t base = block * kSP4EBlockSizeA1P;
            const int32_t start = static_cast<int32_t>(ql.residualIndex.size());
            for (uint32_t i = 0; i < validCounts[block]; ++i) {
                ql.residualIndex.push_back(static_cast<uint8_t>(i));
                ql.residualValue.push_back(FloatToHalf(errors[base + i]));
                reconstruction[base + i] += errors[base + i];
                errors[base + i] = 0.0f;
            }
            ql.residualTable[block * 2u + 0u] = start;
            ql.residualTable[block * 2u + 1u] = static_cast<int32_t>(validCounts[block]);
            ql.mode[block] = static_cast<uint8_t>(SP4EMode::SensitiveFp16);
            blockHasFp16Fallback[block] = 1u;
            ql.fp16FallbackBlocks++;
        }
    }

    if (residualRatio > 0.0f && originalCount > 0) {
        const uint32_t totalBudget = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(originalCount) * residualRatio)));
        const uint32_t entriesPerSelectedBlock = std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(kSP4EBlockSizeA1P) * std::min(maxResidualFractionPerBlock, std::max(residualRatio * 4.0f, residualRatio)))));
        std::vector<uint32_t> candidates(ql.numBlocks);
        std::iota(candidates.begin(), candidates.end(), 0u);
        std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) { return blockScores[a] > blockScores[b]; });
        uint32_t used = 0;
        for (uint32_t block : candidates) {
            if (blockHasFp16Fallback[block]) continue;
            if (used >= totalBudget || validCounts[block] == 0 || blockScores[block] <= 0.0f) continue;
            const uint32_t count = std::min({validCounts[block], entriesPerSelectedBlock, totalBudget - used});
            std::vector<uint32_t> indices(validCounts[block]);
            std::iota(indices.begin(), indices.end(), 0u);
            const uint32_t base = block * kSP4EBlockSizeA1P;
            std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&](uint32_t a, uint32_t b) {
                return residualPriority[base + a] > residualPriority[base + b];
            });
            std::sort(indices.begin(), indices.begin() + count);
            const int32_t start = static_cast<int32_t>(ql.residualIndex.size());
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t local = indices[i];
                ql.residualIndex.push_back(static_cast<uint8_t>(local));
                ql.residualValue.push_back(FloatToHalf(errors[base + local]));
                reconstruction[base + local] += errors[base + local];
                errors[base + local] = 0.0f;
            }
            ql.residualTable[block * 2u + 0u] = start;
            ql.residualTable[block * 2u + 1u] = static_cast<int32_t>(count);
            if (ql.mode[block] == static_cast<uint8_t>(SP4EMode::Fp4Main)) {
                ql.mode[block] = static_cast<uint8_t>(SP4EMode::Fp4WithR16);
            } else if (ql.mode[block] == static_cast<uint8_t>(SP4EMode::Int4Main)) {
                ql.mode[block] = static_cast<uint8_t>(SP4EMode::Int4WithR16);
            }
            used += count;
        }
    }

    for (uint32_t rank = 0; rank < 4u; ++rank) {
        if (!ApplyRank1LowRankCorrection(values, shape, reconstruction, errors, ql.lowRankU, ql.lowRankV)) break;
    }

    for (uint8_t mode : ql.mode) {
        if (mode == static_cast<uint8_t>(SP4EMode::Int4Main) ||
            mode == static_cast<uint8_t>(SP4EMode::Int4WithR16) ||
            mode == static_cast<uint8_t>(SP4EMode::Sp4Fallback)) {
            ql.sp4SubsetBlocks++;
        }
    }

    std::vector<int32_t> compactResidualTable;
    for (uint32_t block = 0; block < ql.numBlocks; ++block) {
        const int32_t start = ql.residualTable[block * 2u + 0u];
        const int32_t count = ql.residualTable[block * 2u + 1u];
        if (count > 0) {
            compactResidualTable.push_back(static_cast<int32_t>((block << 5u) | (static_cast<uint32_t>(count) & 0x1fu)));
            compactResidualTable.push_back(start);
        }
    }
    ql.residualTable = std::move(compactResidualTable);

    Pack4(mainCodes, ql.qweightFp4);
    ql.qweightInt4.clear();
    ql.scaleInt4.clear();

    uint32_t outputChannels = shape.empty() ? 0u : shape[0];
    if (outputChannels > 0u && originalCount > 0u && originalCount % outputChannels == 0u) {
        const uint32_t valuesPerOutput = originalCount / outputChannels;
        ql.channelCorrection.assign(outputChannels, FloatToHalf(1.0f));
        std::vector<std::vector<int8_t>> taskProbes;
        if (shape.size() == 4u &&
            shape[1] * shape[2] * shape[3] == valuesPerOutput &&
            valuesPerOutput > 0u) {
            for (uint32_t probe = 0; probe < 6u; ++probe) {
                taskProbes.push_back(RifeProbeActivation(valuesPerOutput, probe));
            }
        }
        for (uint32_t oc = 0; oc < outputChannels; ++oc) {
            const uint32_t base = oc * valuesPerOutput;
            double dot = 0.0;
            double denom = 0.0;
            double currentSq = 0.0;
            for (uint32_t i = 0; i < valuesPerOutput; ++i) {
                const float target = values[base + i];
                const float recon = reconstruction[base + i];
                dot += static_cast<double>(target) * recon;
                denom += static_cast<double>(recon) * recon;
                const double err = static_cast<double>(target) - recon;
                currentSq += err * err;
            }
            double currentTaskSq = currentSq;
            if (!taskProbes.empty()) {
                dot = 0.0;
                denom = 0.0;
                currentTaskSq = 0.0;
                for (const std::vector<int8_t>& activation : taskProbes) {
                    double targetProjection = 0.0;
                    double reconProjection = 0.0;
                    for (uint32_t i = 0; i < valuesPerOutput; ++i) {
                        const double a = static_cast<double>(activation[i]);
                        targetProjection += static_cast<double>(values[base + i]) * a;
                        reconProjection += static_cast<double>(reconstruction[base + i]) * a;
                    }
                    dot += targetProjection * reconProjection;
                    denom += reconProjection * reconProjection;
                    const double err = targetProjection - reconProjection;
                    currentTaskSq += err * err;
                }
            }
            float gain = 1.0f;
            if (denom > 1.0e-20) {
                gain = static_cast<float>(dot / denom);
                gain = std::max(0.75f, std::min(1.25f, gain));
            }
            double correctedSq = 0.0;
            for (uint32_t i = 0; i < valuesPerOutput; ++i) {
                const double err = static_cast<double>(values[base + i]) -
                                   static_cast<double>(reconstruction[base + i]) * gain;
                correctedSq += err * err;
            }
            double correctedTaskSq = correctedSq;
            if (!taskProbes.empty()) {
                correctedTaskSq = 0.0;
                for (const std::vector<int8_t>& activation : taskProbes) {
                    double targetProjection = 0.0;
                    double reconProjection = 0.0;
                    for (uint32_t i = 0; i < valuesPerOutput; ++i) {
                        const double a = static_cast<double>(activation[i]);
                        targetProjection += static_cast<double>(values[base + i]) * a;
                        reconProjection += static_cast<double>(reconstruction[base + i]) * gain * a;
                    }
                    const double err = targetProjection - reconProjection;
                    correctedTaskSq += err * err;
                }
            }
            const bool improvesTask = correctedTaskSq < currentTaskSq;
            const bool keepsWeightsStable = correctedSq <= currentSq * (taskProbes.empty() ? 1.0 : 1.15);
            if (improvesTask && keepsWeightsStable) {
                ql.channelCorrection[oc] = FloatToHalf(gain);
                for (uint32_t i = 0; i < valuesPerOutput; ++i) reconstruction[base + i] *= gain;
            }
        }
    }

    double sq = 0.0;
    double ab = 0.0;
    float maxAbsErr = 0.0f;
    for (uint32_t i = 0; i < originalCount; ++i) {
        const float err = values[i] - reconstruction[i];
        sq += static_cast<double>(err) * err;
        ab += std::fabs(err);
        maxAbsErr = std::max(maxAbsErr, std::fabs(err));
    }
    ql.rmse = originalCount ? static_cast<float>(std::sqrt(sq / originalCount)) : 0.0f;
    ql.mae = originalCount ? static_cast<float>(ab / originalCount) : 0.0f;
    ql.maxAbs = maxAbsErr;
    return ql;
}

uint64_t SectionBytes(const QuantizedSP4ELayer& ql, uint32_t biasCount) {
    return ql.qweightFp4.size() +
           ql.qweightInt4.size() +
           ql.scaleFp4.size() * sizeof(uint16_t) +
           ql.scaleInt4.size() * sizeof(uint16_t) +
           biasCount * sizeof(uint16_t) +
           PackedControlBytes(ql.numBlocks) +
           PackedControlBytes(ql.numBlocks) +
           ql.bridge.size() +
           ql.residualIndex.size() +
           ql.residualValue.size() * sizeof(uint16_t) +
           ql.residualTable.size() * sizeof(int32_t) +
           ql.channelCorrection.size() * sizeof(uint16_t) +
           ql.lowRankU.size() * sizeof(uint16_t) +
           ql.lowRankV.size() * sizeof(uint16_t) +
           1u;
}

float DefaultFp16FallbackRatio(const std::string& qualityProfile) {
    (void)qualityProfile;
    return 0.0f;
}

float EffectiveResidualRatio(float requested, const std::string& qualityProfile) {
    if (qualityProfile == "int8-chase") return std::max(requested, 0.01f);
    if (qualityProfile == "max-accuracy") return std::max(requested, 1.0f);
    if (qualityProfile == "quality") return std::max(requested, 0.025f);
    return requested;
}

float MaxResidualFractionPerBlock(const std::string& qualityProfile) {
    if (qualityProfile == "max-accuracy") return 1.0f;
    if (qualityProfile == "int8-chase") return 0.50f;
    if (qualityProfile == "quality") return 0.25f;
    return 0.125f;
}

} // namespace

SP4ERuntimeDecision DecideSP4ERuntimePath(const SP4ECapabilityProfile& profile) {
    SP4ERuntimeDecision decision;
    if (profile.nativeFp4TensorSupport) {
        decision.enabled = true;
        decision.path = SP4ERuntimePath::TensorPath;
        decision.reason = "native_fp4_tensor_support";
    } else if (profile.lowPrecisionAluSpeedup >= 2.0f) {
        decision.enabled = true;
        decision.path = SP4ERuntimePath::LowQAluPath;
        decision.reason = "low_precision_alu_speedup";
    } else if (profile.sp4eVsSp4Speedup > 1.0f) {
        decision.enabled = true;
        decision.path = SP4ERuntimePath::ProfileGuidedPath;
        decision.reason = "profiled_sp4e_faster_than_sp4";
    } else {
        decision.enabled = true;
        decision.path = SP4ERuntimePath::Sp4SubsetPath;
        decision.reason = "sp4_subset_available_on_generic_backend";
    }
    return decision;
}

SP4ECapabilityProfile LocalSP4ECapabilityProfile(const std::string& backend, float profileSpeedup) {
    SP4ECapabilityProfile profile;
    profile.backend = backend;
#if defined(__APPLE__) && defined(__aarch64__)
    profile.device = "apple_arm64_no_public_fp4_tensor_probe";
#elif defined(__x86_64__)
    profile.device = "x86_64_no_fp4_tensor_probe";
#else
    profile.device = "generic_no_fp4_tensor_probe";
#endif
    profile.sp4eVsSp4Speedup = profileSpeedup;
    return profile;
}

SP4ECompileResult CompileSafetensorsToSp4e(const SP4ECompileOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    SP4ECompileResult result;
    SafeTensorsReader reader;
    std::string error;
    if (!reader.Open(options.inputPath, &error)) {
        result.message = error;
        return result;
    }

    SP4ESections sections;
    SP4EManifest manifest;
    manifest.adapter = options.adapter;
    manifest.source = options.inputPath;
    manifest.activationPrecision = options.activationPrecision;
    manifest.graphTensorCount = static_cast<uint32_t>(reader.Tensors().size());
    manifest.capability = LocalSP4ECapabilityProfile(options.target, options.profileSpeedup);
    manifest.runtimeDecision = DecideSP4ERuntimePath(manifest.capability);

    for (const std::string& name : reader.SupportedWeightNames()) {
        if (options.adapter == "rife" && !options.includeNonRuntimeLayers && !IsRifeRuntimeLayer(name)) {
            manifest.excludedLayers.push_back(name);
            continue;
        }
        const auto tensorIt = reader.Tensors().find(name);
        if (tensorIt == reader.Tensors().end()) continue;
        std::vector<float> weights;
        if (!reader.ReadF32(name, weights, &error)) {
            result.message = error;
            return result;
        }

        const float residualRatio = EffectiveResidualRatio(options.residualRatio, options.qualityProfile);
        const float fallbackRatio = options.fp16FallbackRatio >= 0.0f
            ? options.fp16FallbackRatio
            : DefaultFp16FallbackRatio(options.qualityProfile);
        const float maxResidualFraction = MaxResidualFractionPerBlock(options.qualityProfile);
        const LayerPolicy policy = PolicyFor(options.adapter, name, residualRatio, options.qualityProfile);
        QuantizedSP4ELayer ql = QuantizeSP4E(weights, tensorIt->second.shape, policy.aux, policy.residualRatio, fallbackRatio, maxResidualFraction);

        SP4ELayerDesc layer;
        layer.name = name;
        layer.shape = tensorIt->second.shape;
        layer.numBlocks = ql.numBlocks;
        layer.paddedValues = ql.paddedValues;
        layer.qweightFp4OffsetBytes = static_cast<uint32_t>(sections.qweightFp4.size());
        layer.qweightInt4OffsetBytes = static_cast<uint32_t>(sections.qweightInt4.size());
        layer.qweightLengthBytes = static_cast<uint32_t>(ql.qweightFp4.size());
        layer.scaleFp4Offset = static_cast<uint32_t>(sections.scaleFp16Fp4.size());
        layer.scaleInt4Offset = static_cast<uint32_t>(sections.scaleFp16Int4.size());
        layer.biasOffset = static_cast<uint32_t>(sections.biasFp16.size());
        layer.modeOffset = static_cast<uint32_t>(sections.mode.size());
        layer.auxOffset = static_cast<uint32_t>(sections.aux.size());
        layer.bridgeOffset = static_cast<uint32_t>(sections.bridge.size());
        layer.residualIndexOffset = static_cast<uint32_t>(sections.residualIndex.size());
        layer.residualValueOffset = static_cast<uint32_t>(sections.residualValueFp16.size());
        layer.residualTableOffset = static_cast<uint32_t>(sections.residualTable.size() / 2u);
        layer.residualEntries = static_cast<uint32_t>(ql.residualIndex.size());
        layer.activationPrecisionOffset = static_cast<uint32_t>(sections.activationPrecision.size());
        layer.channelCorrectionOffset = static_cast<uint32_t>(sections.channelCorrection.size());
        layer.channelCorrectionCount = static_cast<uint32_t>(ql.channelCorrection.size());
        layer.lowRankUOffset = static_cast<uint32_t>(sections.lowRankU.size());
        layer.lowRankVOffset = static_cast<uint32_t>(sections.lowRankV.size());
        const uint32_t lowRankOutputChannels = layer.shape.empty() ? 0u : layer.shape[0];
        layer.lowRankRank = lowRankOutputChannels == 0u ? 0u : static_cast<uint32_t>(ql.lowRankU.size() / lowRankOutputChannels);
        layer.auxPolicy = policy.aux;
        layer.sensitive = policy.sensitive;
        layer.fp4Blocks = ql.fp4Blocks;
        layer.int4Blocks = ql.int4Blocks;
        layer.bridgeBlocks = ql.bridgeBlocks;
        layer.sp4SubsetBlocks = ql.sp4SubsetBlocks;
        layer.fp16FallbackBlocks = ql.fp16FallbackBlocks;
        layer.rmse = ql.rmse;
        layer.mae = ql.mae;
        layer.maxAbs = ql.maxAbs;

        std::vector<float> biasF32;
        if (!reader.ReadF32(BasenameNoExt(name) + ".bias", biasF32, nullptr)) {
            biasF32.assign(layer.shape.empty() ? 0u : layer.shape[0], 0.0f);
        }
        layer.biasCount = static_cast<uint32_t>(biasF32.size());

        sections.qweightFp4.insert(sections.qweightFp4.end(), ql.qweightFp4.begin(), ql.qweightFp4.end());
        sections.qweightInt4.insert(sections.qweightInt4.end(), ql.qweightInt4.begin(), ql.qweightInt4.end());
        sections.scaleFp16Fp4.insert(sections.scaleFp16Fp4.end(), ql.scaleFp4.begin(), ql.scaleFp4.end());
        sections.scaleFp16Int4.insert(sections.scaleFp16Int4.end(), ql.scaleInt4.begin(), ql.scaleInt4.end());
        for (float value : biasF32) sections.biasFp16.push_back(FloatToHalf(value));
        const std::vector<uint8_t> packedMode = PackNibbles(ql.mode);
        const std::vector<uint8_t> packedAux = PackNibbles(ql.aux);
        sections.mode.insert(sections.mode.end(), packedMode.begin(), packedMode.end());
        sections.aux.insert(sections.aux.end(), packedAux.begin(), packedAux.end());
        sections.bridge.insert(sections.bridge.end(), ql.bridge.begin(), ql.bridge.end());
        sections.residualIndex.insert(sections.residualIndex.end(), ql.residualIndex.begin(), ql.residualIndex.end());
        sections.residualValueFp16.insert(sections.residualValueFp16.end(), ql.residualValue.begin(), ql.residualValue.end());
        sections.residualTable.insert(sections.residualTable.end(), ql.residualTable.begin(), ql.residualTable.end());
        sections.activationPrecision.push_back(ActivationPrecisionCode(options.activationPrecision));
        sections.channelCorrection.insert(sections.channelCorrection.end(), ql.channelCorrection.begin(), ql.channelCorrection.end());
        sections.lowRankU.insert(sections.lowRankU.end(), ql.lowRankU.begin(), ql.lowRankU.end());
        sections.lowRankV.insert(sections.lowRankV.end(), ql.lowRankV.begin(), ql.lowRankV.end());

        manifest.layers.push_back(layer);
        manifest.teacherWeightBytes += static_cast<uint64_t>(weights.size()) * sizeof(uint16_t);
        manifest.sp4eWeightBytes += SectionBytes(ql, layer.biasCount);
        manifest.blocks += ql.numBlocks;
        manifest.fp4Blocks += ql.fp4Blocks;
        manifest.int4Blocks += ql.int4Blocks;
        manifest.bridgeBlocks += ql.bridgeBlocks;
        manifest.sp4SubsetBlocks += ql.sp4SubsetBlocks;
        manifest.fp16FallbackBlocks += ql.fp16FallbackBlocks;
        manifest.residualEntries += ql.residualIndex.size();
    }

    const auto end = std::chrono::steady_clock::now();
    manifest.compileMs = std::chrono::duration<double, std::milli>(end - start).count();
    if (!WriteSP4EAsset(options.outputPath, manifest, sections, &error)) {
        result.message = error;
        return result;
    }
    result.ok = true;
    result.message = "ok";
    result.manifest = std::move(manifest);
    return result;
}

} // namespace sp4
