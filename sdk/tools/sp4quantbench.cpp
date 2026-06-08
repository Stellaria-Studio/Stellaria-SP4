#include "sp4/SP4Runtime.h"
#include "sp4/SafeTensorsReader.h"

#include "../src/SP4Common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Metrics {
    std::string name;
    uint64_t bytes = 0;
    double compression = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    double maxAbs = 0.0;
    double taskRelativeL2 = 0.0;
    double taskAccuracyPct = 0.0;
    double ms = 0.0;
};

enum class TaskProfile {
    Generic,
    RifeMotionBoundaryV2,
};

TaskProfile ParseTaskProfile(const std::string& value) {
    if (value == "generic") return TaskProfile::Generic;
    return TaskProfile::RifeMotionBoundaryV2;
}

const char* TaskProfileName(TaskProfile profile) {
    return profile == TaskProfile::Generic ? "generic" : "rife_motion_boundary_v2";
}

struct Accum {
    double sq = 0.0;
    double ab = 0.0;
    double maxAbs = 0.0;
    uint64_t count = 0;

    void Add(float expected, float actual) {
        const double err = static_cast<double>(expected) - static_cast<double>(actual);
        sq += err * err;
        ab += std::fabs(err);
        maxAbs = std::max(maxAbs, std::fabs(err));
        count += 1;
    }

    void Finish(Metrics& m) const {
        m.rmse = std::sqrt(sq / std::max<uint64_t>(1, count));
        m.mae = ab / std::max<uint64_t>(1, count);
        m.maxAbs = maxAbs;
    }
};

uint64_t Product(const std::vector<uint32_t>& shape) {
    uint64_t v = 1;
    for (uint32_t dim : shape) v *= dim;
    return v;
}

std::vector<int8_t> SyntheticActivation(uint32_t count) {
    std::vector<int8_t> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        values[i] = static_cast<int8_t>((static_cast<int>(i * 13u + 7u) % 127) - 63);
    }
    return values;
}

struct TaskAccum {
    TaskProfile profile = TaskProfile::RifeMotionBoundaryV2;
    double errSq = 0.0;
    double refSq = 0.0;
    double edgeErrSq = 0.0;
    double edgeRefSq = 0.0;

    explicit TaskAccum(TaskProfile p = TaskProfile::RifeMotionBoundaryV2) : profile(p) {}

    double LayerWeight(const std::string& name) const {
        if (profile == TaskProfile::Generic) return 1.0;
        double w = 1.0;
        if (name.find(".conv0.") != std::string::npos) w *= 1.65;
        if (name.find(".conv1.") != std::string::npos) w *= 1.35;
        if (name.find(".conv2.") != std::string::npos) w *= 1.30;
        if (name.rfind("block0.", 0) == 0 || name.rfind("block1.", 0) == 0) w *= 1.35;
        if (name.find("convblock") != std::string::npos) w *= 1.15;
        return w;
    }

    static std::vector<int8_t> ProbeActivation(uint32_t count, uint32_t probe, TaskProfile profile) {
        std::vector<int8_t> values(count);
        for (uint32_t i = 0; i < count; ++i) {
            int v = 0;
            if (probe == 0) {
                v = (static_cast<int>(i * 13u + 7u) % 127) - 63;
            } else if (probe == 1) {
                v = (i & 1u) ? 63 : -64;
            } else if (probe == 2) {
                v = (static_cast<int>((i * 9u + (i / 3u) * 17u) % 127u) - 63);
            } else if (probe == 3) {
                v = (i % 11u == 0u || i % 13u == 0u) ? 96 : ((i & 1u) ? -12 : 12);
            } else if (probe == 4) {
                v = static_cast<int>((i % 5u) * 28u) - 56;
            } else {
                const int sign = ((i / 7u) & 1u) ? -1 : 1;
                v = sign * (24 + static_cast<int>((i * 19u) % 72u));
            }
            if (profile == TaskProfile::Generic && probe > 2) {
                v /= 2;
            }
            values[i] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
        }
        return values;
    }

    void AddProbe(const std::string& name,
                  const std::vector<float>& teacher,
                  const std::vector<float>& actual,
                  const std::vector<uint32_t>& shape) {
        if (shape.size() != 4 || teacher.size() != actual.size()) return;
        const uint32_t outputChannels = shape[0];
        const uint32_t kernelValues = shape[1] * shape[2] * shape[3];
        const double layerWeight = LayerWeight(name);
        const uint32_t probeCount = profile == TaskProfile::Generic ? 3u : 6u;
        for (uint32_t probe = 0; probe < probeCount; ++probe) {
            const std::vector<int8_t> activation = ProbeActivation(kernelValues, probe, profile);
            double probeWeight = 1.0;
            if (profile == TaskProfile::RifeMotionBoundaryV2) {
                probeWeight = probe == 1 ? 1.55 : (probe == 3 ? 1.45 : (probe >= 4 ? 1.25 : 1.0));
            }
            probeWeight *= layerWeight;
            for (uint32_t oc = 0; oc < outputChannels; ++oc) {
                double ref = 0.0;
                double got = 0.0;
                for (uint32_t k = 0; k < kernelValues; ++k) {
                    const double a = static_cast<double>(activation[k]);
                    ref += static_cast<double>(teacher[oc * kernelValues + k]) * a;
                    got += static_cast<double>(actual[oc * kernelValues + k]) * a;
                }
                const double e = ref - got;
                errSq += probeWeight * e * e;
                refSq += probeWeight * ref * ref;
                if (probe == 1 || probe == 3 || probe == 5) {
                    edgeErrSq += probeWeight * e * e;
                    edgeRefSq += probeWeight * ref * ref;
                }
            }
        }
    }

    void Finish(Metrics& m) const {
        const double rel = std::sqrt(errSq / std::max(1.0e-20, refSq));
        const double edgeRel = std::sqrt(edgeErrSq / std::max(1.0e-20, edgeRefSq));
        const double artifactPenalty = profile == TaskProfile::Generic ? 0.0 : std::min(0.20, edgeRel * 0.35);
        m.taskRelativeL2 = rel;
        m.taskAccuracyPct = std::max(0.0, 100.0 * (1.0 - rel - artifactPenalty));
    }
};

std::vector<float> ReconstructFp16(const std::vector<float>& values) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = sp4::HalfToFloat(sp4::FloatToHalf(values[i]));
    }
    return out;
}

std::vector<float> ReconstructInt8PerOutput(const std::vector<float>& values, const std::vector<uint32_t>& shape) {
    std::vector<float> out(values.size());
    const uint32_t oc = shape[0];
    const uint32_t perOutput = static_cast<uint32_t>(values.size() / oc);
    for (uint32_t o = 0; o < oc; ++o) {
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < perOutput; ++i) maxAbs = std::max(maxAbs, std::fabs(values[o * perOutput + i]));
        const float scale = maxAbs > 1.0e-8f ? maxAbs / 127.0f : 1.0e-8f;
        for (uint32_t i = 0; i < perOutput; ++i) {
            const float value = values[o * perOutput + i];
            const int q = std::max(-128, std::min(127, static_cast<int>(std::lrint(value / scale))));
            out[o * perOutput + i] = static_cast<float>(q) * scale;
        }
    }
    return out;
}

std::vector<float> ReconstructInt4PerOutput(const std::vector<float>& values, const std::vector<uint32_t>& shape) {
    std::vector<float> out(values.size());
    const uint32_t oc = shape[0];
    const uint32_t perOutput = static_cast<uint32_t>(values.size() / oc);
    for (uint32_t o = 0; o < oc; ++o) {
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < perOutput; ++i) maxAbs = std::max(maxAbs, std::fabs(values[o * perOutput + i]));
        const float scale = maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f;
        for (uint32_t i = 0; i < perOutput; ++i) {
            const float value = values[o * perOutput + i];
            const int q = std::max(-8, std::min(7, static_cast<int>(std::lrint(value / scale))));
            out[o * perOutput + i] = static_cast<float>(q) * scale;
        }
    }
    return out;
}

std::vector<float> ReconstructW4A8Block(const std::vector<float>& values, uint32_t blockSize) {
    std::vector<float> out(values.size());
    const uint32_t blocks = static_cast<uint32_t>((values.size() + blockSize - 1u) / blockSize);
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t base = b * blockSize;
        const uint32_t valid = static_cast<uint32_t>(std::min<uint64_t>(blockSize, values.size() - std::min<uint64_t>(base, values.size())));
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) maxAbs = std::max(maxAbs, std::fabs(values[base + i]));
        const float scale = maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f;
        for (uint32_t i = 0; i < valid; ++i) {
            const float value = values[base + i];
            const int q = std::max(-8, std::min(7, static_cast<int>(std::lrint(value / scale))));
            out[base + i] = static_cast<float>(q) * scale;
        }
    }
    return out;
}

std::vector<float> ReconstructNF4LikeBlock(const std::vector<float>& values, uint32_t blockSize) {
    static constexpr float kCodebook[16] = {
        -1.000000f, -0.696193f, -0.525073f, -0.394917f,
        -0.284441f, -0.184773f, -0.091050f,  0.000000f,
         0.079580f,  0.160930f,  0.246112f,  0.337915f,
         0.440710f,  0.562617f,  0.722956f,  1.000000f,
    };
    std::vector<float> out(values.size());
    const uint32_t blocks = static_cast<uint32_t>((values.size() + blockSize - 1u) / blockSize);
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t base = b * blockSize;
        const uint32_t valid = static_cast<uint32_t>(std::min<uint64_t>(blockSize, values.size() - std::min<uint64_t>(base, values.size())));
        float absMax = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) absMax = std::max(absMax, std::fabs(values[base + i]));
        const float scale = absMax > 1.0e-8f ? absMax : 1.0e-8f;
        for (uint32_t i = 0; i < valid; ++i) {
            int best = 0;
            float bestErr = std::fabs(values[base + i] - kCodebook[0] * scale);
            for (int c = 1; c < 16; ++c) {
                const float err = std::fabs(values[base + i] - kCodebook[c] * scale);
                if (err < bestErr) {
                    bestErr = err;
                    best = c;
                }
            }
            out[base + i] = kCodebook[best] * scale;
        }
    }
    return out;
}

void AddInt4PerOutput(const std::vector<float>& values, const std::vector<uint32_t>& shape, Accum& acc, uint64_t& bytes) {
    const uint32_t oc = shape[0];
    const uint32_t perOutput = static_cast<uint32_t>(values.size() / oc);
    bytes += (values.size() + 1u) / 2u + oc * sizeof(uint16_t);
    for (uint32_t o = 0; o < oc; ++o) {
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < perOutput; ++i) {
            maxAbs = std::max(maxAbs, std::fabs(values[o * perOutput + i]));
        }
        const float scale = maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f;
        for (uint32_t i = 0; i < perOutput; ++i) {
            const float value = values[o * perOutput + i];
            const int q = std::max(-8, std::min(7, static_cast<int>(std::lrint(value / scale))));
            acc.Add(value, static_cast<float>(q) * scale);
        }
    }
}

void AddInt8PerOutput(const std::vector<float>& values, const std::vector<uint32_t>& shape, Accum& acc, uint64_t& bytes) {
    const uint32_t oc = shape[0];
    const uint32_t perOutput = static_cast<uint32_t>(values.size() / oc);
    bytes += values.size() + oc * sizeof(uint16_t);
    for (uint32_t o = 0; o < oc; ++o) {
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < perOutput; ++i) {
            maxAbs = std::max(maxAbs, std::fabs(values[o * perOutput + i]));
        }
        const float scale = maxAbs > 1.0e-8f ? maxAbs / 127.0f : 1.0e-8f;
        for (uint32_t i = 0; i < perOutput; ++i) {
            const float value = values[o * perOutput + i];
            const int q = std::max(-128, std::min(127, static_cast<int>(std::lrint(value / scale))));
            acc.Add(value, static_cast<float>(q) * scale);
        }
    }
}

void AddW4A8Block(const std::vector<float>& values, uint32_t blockSize, Accum& acc, uint64_t& bytes) {
    const uint32_t blocks = static_cast<uint32_t>((values.size() + blockSize - 1u) / blockSize);
    bytes += ((blocks * blockSize) + 1u) / 2u + blocks * sizeof(uint16_t) + blocks * 2u;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t base = b * blockSize;
        const uint32_t valid = static_cast<uint32_t>(std::min<uint64_t>(blockSize, values.size() - std::min<uint64_t>(base, values.size())));
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) maxAbs = std::max(maxAbs, std::fabs(values[base + i]));
        const float scale = maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f;
        for (uint32_t i = 0; i < valid; ++i) {
            const float value = values[base + i];
            const int q = std::max(-8, std::min(7, static_cast<int>(std::lrint(value / scale))));
            acc.Add(value, static_cast<float>(q) * scale);
        }
    }
}

void AddNF4LikeBlock(const std::vector<float>& values, uint32_t blockSize, Accum& acc, uint64_t& bytes) {
    static constexpr float kCodebook[16] = {
        -1.000000f, -0.696193f, -0.525073f, -0.394917f,
        -0.284441f, -0.184773f, -0.091050f,  0.000000f,
         0.079580f,  0.160930f,  0.246112f,  0.337915f,
         0.440710f,  0.562617f,  0.722956f,  1.000000f,
    };
    const uint32_t blocks = static_cast<uint32_t>((values.size() + blockSize - 1u) / blockSize);
    bytes += ((blocks * blockSize) + 1u) / 2u + blocks * sizeof(uint16_t);
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t base = b * blockSize;
        const uint32_t valid = static_cast<uint32_t>(std::min<uint64_t>(blockSize, values.size() - std::min<uint64_t>(base, values.size())));
        float absMax = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) absMax = std::max(absMax, std::fabs(values[base + i]));
        const float scale = absMax > 1.0e-8f ? absMax : 1.0e-8f;
        for (uint32_t i = 0; i < valid; ++i) {
            const float value = values[base + i];
            int best = 0;
            float bestErr = std::fabs(value - kCodebook[0] * scale);
            for (int c = 1; c < 16; ++c) {
                const float err = std::fabs(value - kCodebook[c] * scale);
                if (err < bestErr) {
                    bestErr = err;
                    best = c;
                }
            }
            acc.Add(value, kCodebook[best] * scale);
        }
    }
}

bool IsRuntimeLayer(const std::string& name) {
    return name.rfind("block_tea.", 0) != 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string teacherPath = "Models/RIFE-safetensors/flownet.safetensors";
    std::string assetPath = "build/rife_cpp.sp4";
    TaskProfile taskProfile = TaskProfile::RifeMotionBoundaryV2;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--task-profile" && i + 1 < argc) {
            taskProfile = ParseTaskProfile(argv[++i]);
        } else if (arg.rfind("--task-profile=", 0) == 0) {
            taskProfile = ParseTaskProfile(arg.substr(std::string("--task-profile=").size()));
        } else if (positional == 0) {
            teacherPath = arg;
            ++positional;
        } else if (positional == 1) {
            assetPath = arg;
            ++positional;
        }
    }

    sp4::SafeTensorsReader reader;
    std::string error;
    if (!reader.Open(teacherPath, &error)) {
        std::cerr << "sp4quantbench: " << error << "\n";
        return 1;
    }

    std::vector<std::string> names;
    for (const std::string& name : reader.ConvWeightNames()) {
        if (IsRuntimeLayer(name)) names.push_back(name);
    }

    uint64_t teacherFp16Bytes = 0;
    Accum fp16Accum;
    Accum int8Accum;
    Accum int4Accum;
    Accum nf4Accum;
    Accum w4a8Accum;
    TaskAccum fp16Task(taskProfile);
    TaskAccum int8Task(taskProfile);
    TaskAccum int4Task(taskProfile);
    TaskAccum nf4Task(taskProfile);
    TaskAccum w4a8Task(taskProfile);
    uint64_t fp16Bytes = 0;
    uint64_t int8Bytes = 0;
    uint64_t int4Bytes = 0;
    uint64_t nf4Bytes = 0;
    uint64_t w4a8Bytes = 0;

    auto benchStart = std::chrono::steady_clock::now();
    for (const std::string& name : names) {
        const auto infoIt = reader.Tensors().find(name);
        if (infoIt == reader.Tensors().end()) continue;
        std::vector<float> values;
        if (!reader.ReadF32(name, values, &error)) {
            std::cerr << "sp4quantbench: " << error << "\n";
            return 1;
        }
        teacherFp16Bytes += values.size() * sizeof(uint16_t);
        fp16Bytes += values.size() * sizeof(uint16_t);
        for (float value : values) {
            fp16Accum.Add(value, sp4::HalfToFloat(sp4::FloatToHalf(value)));
        }
        AddInt8PerOutput(values, infoIt->second.shape, int8Accum, int8Bytes);
        AddInt4PerOutput(values, infoIt->second.shape, int4Accum, int4Bytes);
        const uint32_t perOutput = infoIt->second.shape[1] * infoIt->second.shape[2] * infoIt->second.shape[3];
        AddNF4LikeBlock(values, perOutput <= 128 ? 32u : 64u, nf4Accum, nf4Bytes);
        AddW4A8Block(values, perOutput <= 128 ? 32u : 64u, w4a8Accum, w4a8Bytes);

        fp16Task.AddProbe(name, values, ReconstructFp16(values), infoIt->second.shape);
        int8Task.AddProbe(name, values, ReconstructInt8PerOutput(values, infoIt->second.shape), infoIt->second.shape);
        int4Task.AddProbe(name, values, ReconstructInt4PerOutput(values, infoIt->second.shape), infoIt->second.shape);
        nf4Task.AddProbe(name, values, ReconstructNF4LikeBlock(values, perOutput <= 128 ? 32u : 64u), infoIt->second.shape);
        w4a8Task.AddProbe(name, values, ReconstructW4A8Block(values, perOutput <= 128 ? 32u : 64u), infoIt->second.shape);
    }
    const double localMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - benchStart).count();

    std::vector<Metrics> rows;
    rows.push_back({"FP16 baseline", fp16Bytes, 1.0, 0, 0, 0, 0, 0, localMs});
    fp16Accum.Finish(rows.back());
    fp16Task.Finish(rows.back());
    rows.push_back({"INT8 per-output", int8Bytes, static_cast<double>(teacherFp16Bytes) / std::max<uint64_t>(1, int8Bytes), 0, 0, 0, 0, 0, localMs});
    int8Accum.Finish(rows.back());
    int8Task.Finish(rows.back());
    rows.push_back({"INT4 weight-only per-output", int4Bytes, static_cast<double>(teacherFp16Bytes) / std::max<uint64_t>(1, int4Bytes), 0, 0, 0, 0, 0, localMs});
    int4Accum.Finish(rows.back());
    int4Task.Finish(rows.back());
    rows.push_back({"NF4-like block codebook", nf4Bytes, static_cast<double>(teacherFp16Bytes) / std::max<uint64_t>(1, nf4Bytes), 0, 0, 0, 0, 0, localMs});
    nf4Accum.Finish(rows.back());
    nf4Task.Finish(rows.back());
    rows.push_back({"W4A8 block no residual", w4a8Bytes, static_cast<double>(teacherFp16Bytes) / std::max<uint64_t>(1, w4a8Bytes), 0, 0, 0, 0, 0, localMs});
    w4a8Accum.Finish(rows.back());
    w4a8Task.Finish(rows.back());

    sp4::LoadedAsset asset;
    if (sp4::LoadAsset(assetPath, asset, &error)) {
        Accum sp4Accum;
        TaskAccum sp4Task(taskProfile);
        const auto sp4Start = std::chrono::steady_clock::now();
        for (const sp4::LayerDesc& layer : asset.manifest.layers) {
            std::vector<float> teacher;
            if (!reader.ReadF32(layer.name, teacher, &error)) {
                std::cerr << "sp4quantbench: " << error << "\n";
                return 1;
            }
            std::vector<float> actual = sp4::ReconstructLayer(asset, layer);
            for (size_t i = 0; i < teacher.size(); ++i) {
                sp4Accum.Add(teacher[i], actual[i]);
            }
            sp4Task.AddProbe(layer.name, teacher, actual, layer.shape);
        }
        Metrics sp4Row;
        sp4Row.name = "SP4 A1P sparse residual";
        sp4Row.bytes = asset.manifest.sp4WeightBytes ? asset.manifest.sp4WeightBytes :
                       asset.qweight.size() + asset.scaleFp16.size() * 2ull + asset.biasFp16.size() * 2ull +
                       asset.mode.size() + asset.aux.size() + asset.residualIndex.size() * 4ull +
                       asset.residualValueFp16.size() * 2ull + asset.residualTable.size() * 4ull +
                       asset.activationScaleFp16.size() * 2ull;
        sp4Row.compression = static_cast<double>(teacherFp16Bytes) / std::max<uint64_t>(1, sp4Row.bytes);
        sp4Row.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sp4Start).count();
        sp4Accum.Finish(sp4Row);
        sp4Task.Finish(sp4Row);
        rows.push_back(sp4Row);
    } else {
        std::cerr << "sp4quantbench: warning: " << error << "\n";
    }

    std::cout << "SP4 local quantization comparison\n";
    std::cout << "  teacher=" << teacherPath << "\n";
    std::cout << "  runtime_layers=" << names.size() << " teacher_fp16_bytes=" << teacherFp16Bytes
              << " task_profile=" << TaskProfileName(taskProfile) << "\n";
    std::cout << std::left << std::setw(30) << "scheme"
              << std::right << std::setw(12) << "bytes"
              << std::setw(12) << "compress"
              << std::setw(12) << "rmse"
              << std::setw(12) << "mae"
              << std::setw(12) << "max_abs"
              << std::setw(12) << "task_acc"
              << std::setw(12) << "cpu_ms" << "\n";
    for (const Metrics& row : rows) {
        std::cout << std::left << std::setw(30) << row.name
                  << std::right << std::setw(12) << row.bytes
                  << std::setw(11) << std::fixed << std::setprecision(2) << row.compression << "x"
                  << std::setw(12) << std::setprecision(6) << row.rmse
                  << std::setw(12) << row.mae
                  << std::setw(12) << row.maxAbs
                  << std::setw(11) << std::setprecision(2) << row.taskAccuracyPct << "%"
                  << std::setw(12) << std::setprecision(3) << row.ms << "\n";
    }
    return 0;
}
