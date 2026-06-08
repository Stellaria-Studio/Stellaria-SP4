#include "sp4/SP4Runtime.h"

#include "SP4Common.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>

namespace sp4 {

namespace {

template <typename T>
void CopySection(const std::vector<uint8_t>& payload, size_t offset, size_t length, std::vector<T>& out) {
    out.resize(length / sizeof(T));
    if (!out.empty()) {
        std::memcpy(out.data(), payload.data() + offset, out.size() * sizeof(T));
    }
}

bool FindSection(const std::string& manifest, const std::string& name, size_t& offset, size_t& length) {
    const std::regex re("\"" + name + "\"\\s*:\\s*\\{\\s*\"length\"\\s*:\\s*([0-9]+)\\s*,\\s*\"offset\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(manifest, match, re)) {
        return false;
    }
    length = static_cast<size_t>(std::stoull(match[1].str()));
    offset = static_cast<size_t>(std::stoull(match[2].str()));
    return true;
}

std::vector<uint32_t> ParseShapeList(const std::string& text) {
    std::vector<uint32_t> shape;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            shape.push_back(static_cast<uint32_t>(std::stoul(item)));
        }
    }
    return shape;
}

void ParseManifestPlan(const std::string& text, AssetManifest& manifest) {
    std::smatch match;
    if (std::regex_search(text, match, std::regex("\"version\"\\s*:\\s*\"(A[0-9][A-Z])\""))) {
        manifest.version = match[1].str();
    }
    if (std::regex_search(text, match, std::regex("\"adapter\"\\s*:\\s*\"([^\"]+)\""))) {
        manifest.adapter = match[1].str();
    }
    if (std::regex_search(text, match, std::regex("\"source\"\\s*:\\s*\"([^\"]*)\""))) {
        manifest.source = match[1].str();
    }
    if (std::regex_search(text, match, std::regex("\"graph_tensor_count\"\\s*:\\s*([0-9]+)"))) {
        manifest.graphTensorCount = static_cast<uint32_t>(std::stoul(match[1].str()));
    }
    if (std::regex_search(text, match, std::regex("\"blocks\"\\s*:\\s*([0-9]+)"))) {
        manifest.blocks = static_cast<uint64_t>(std::stoull(match[1].str()));
    }
    if (std::regex_search(text, match, std::regex("\"residual_entries\"\\s*:\\s*([0-9]+)"))) {
        manifest.residualEntries = static_cast<uint64_t>(std::stoull(match[1].str()));
    }
    if (std::regex_search(text, match, std::regex("\"sp4_weight_bytes\"\\s*:\\s*([0-9]+)"))) {
        manifest.sp4WeightBytes = static_cast<uint64_t>(std::stoull(match[1].str()));
    }
    if (std::regex_search(text, match, std::regex("\"teacher_weight_bytes\"\\s*:\\s*([0-9]+)"))) {
        manifest.teacherWeightBytes = static_cast<uint64_t>(std::stoull(match[1].str()));
    }
    const std::regex layerRegex(
        "\\{\"activation_scale_offset\":([0-9]+),\"aux_offset\":([0-9]+),\"aux_policy\":([0-9]+),"
        "\"bias_count\":([0-9]+),\"bias_offset\":([0-9]+),\"block_abi\":\"SP4block([0-9]+)\\(A1P\\)\","
        "\"block_size\":([0-9]+),\"fallback_fp16\":(true|false),"
        "\"metrics\":\\{\"mae\":([-+0-9.eE]+),\"max_abs\":([-+0-9.eE]+),\"rmse\":([-+0-9.eE]+)\\},"
        "\"mode_offset\":([0-9]+),\"name\":\"([^\"]+)\",\"num_blocks\":([0-9]+),"
        "\"padded_values\":([0-9]+),\"qweight_length_bytes\":([0-9]+),\"qweight_offset_bytes\":([0-9]+),"
        "\"residual_entries\":([0-9]+),\"residual_index_offset\":([0-9]+),\"residual_table_offset\":([0-9]+),"
        "\"residual_value_offset\":([0-9]+),\"scale_offset\":([0-9]+),\"sensitive\":(true|false),\"shape\":\\[([^\\]]*)\\]\\}");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), layerRegex);
         it != std::sregex_iterator();
         ++it) {
        const auto& m = *it;
        LayerDesc layer;
        layer.activationScaleOffset = static_cast<uint32_t>(std::stoul(m[1].str()));
        layer.auxOffset = static_cast<uint32_t>(std::stoul(m[2].str()));
        layer.auxPolicy = static_cast<uint32_t>(std::stoul(m[3].str()));
        layer.biasCount = static_cast<uint32_t>(std::stoul(m[4].str()));
        layer.biasOffset = static_cast<uint32_t>(std::stoul(m[5].str()));
        layer.blockSize = static_cast<uint32_t>(std::stoul(m[7].str()));
        layer.fallbackFp16 = m[8].str() == "true";
        layer.mae = std::stof(m[9].str());
        layer.maxAbs = std::stof(m[10].str());
        layer.rmse = std::stof(m[11].str());
        layer.modeOffset = static_cast<uint32_t>(std::stoul(m[12].str()));
        layer.name = m[13].str();
        layer.numBlocks = static_cast<uint32_t>(std::stoul(m[14].str()));
        layer.paddedValues = static_cast<uint32_t>(std::stoul(m[15].str()));
        layer.qweightLengthBytes = static_cast<uint32_t>(std::stoul(m[16].str()));
        layer.qweightOffsetBytes = static_cast<uint32_t>(std::stoul(m[17].str()));
        layer.residualEntries = static_cast<uint32_t>(std::stoul(m[18].str()));
        layer.residualIndexOffset = static_cast<uint32_t>(std::stoul(m[19].str()));
        layer.residualTableOffset = static_cast<uint32_t>(std::stoul(m[20].str()));
        layer.residualValueOffset = static_cast<uint32_t>(std::stoul(m[21].str()));
        layer.scaleOffset = static_cast<uint32_t>(std::stoul(m[22].str()));
        layer.sensitive = m[23].str() == "true";
        layer.shape = ParseShapeList(m[24].str());
        manifest.layers.push_back(std::move(layer));
    }
}

int8_t SignExtend4(uint8_t value) {
    value &= 0x0fu;
    return static_cast<int8_t>(value >= 8u ? static_cast<int>(value) - 16 : value);
}

std::vector<int8_t> UnpackInt4(const std::vector<uint8_t>& qweight, uint32_t offset, uint32_t length, uint32_t count) {
    std::vector<int8_t> out(count);
    uint32_t write = 0;
    for (uint32_t i = 0; i < length && write < count; ++i) {
        const uint8_t byte = qweight[offset + i];
        out[write++] = SignExtend4(byte & 0x0fu);
        if (write < count) {
            out[write++] = SignExtend4(byte >> 4u);
        }
    }
    return out;
}

std::vector<uint8_t> UnpackUInt4(const std::vector<uint8_t>& qweight, uint32_t offset, uint32_t length, uint32_t count) {
    std::vector<uint8_t> out(count);
    uint32_t write = 0;
    for (uint32_t i = 0; i < length && write < count; ++i) {
        const uint8_t byte = qweight[offset + i];
        out[write++] = byte & 0x0fu;
        if (write < count) {
            out[write++] = (byte >> 4u) & 0x0fu;
        }
    }
    return out;
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

} // namespace

bool LoadAsset(const std::string& path, LoadedAsset& outAsset, std::string* error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        if (error) *error = "failed to open SP4 asset: " + path;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size < 16) {
        if (error) *error = "SP4 asset is too small";
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(raw.data()), size);
    if (std::memcmp(raw.data(), kMagicA1P, 8) != 0) {
        if (error) *error = "invalid SP4 magic";
        return false;
    }
    if (ReadU32(raw.data() + 8) != kContainerVersion) {
        if (error) *error = "unsupported SP4 container version";
        return false;
    }
    const uint32_t manifestBytes = ReadU32(raw.data() + 12);
    if (16u + manifestBytes > raw.size()) {
        if (error) *error = "invalid SP4 manifest length";
        return false;
    }
    const std::string manifest(reinterpret_cast<const char*>(raw.data() + 16), manifestBytes);
    std::vector<uint8_t> payload(raw.begin() + 16 + manifestBytes, raw.end());
    ParseManifestPlan(manifest, outAsset.manifest);

    auto findRange = [&](const std::string& name, size_t& offset, size_t& length) -> bool {
        if (!FindSection(manifest, name, offset, length)) {
            if (error) *error = "missing SP4 section: " + name;
            return false;
        }
        if (offset + length > payload.size()) {
            if (error) *error = "SP4 section range invalid: " + name;
            return false;
        }
        return true;
    };

    auto load = [&](const std::string& name, auto& section) -> bool {
        size_t offset = 0;
        size_t length = 0;
        if (!findRange(name, offset, length)) return false;
        using Vec = std::decay_t<decltype(section)>;
        using T = typename Vec::value_type;
        CopySection(payload, offset, length, section);
        return true;
    };

    const bool loaded = load("qweight", outAsset.qweight) &&
                        load("scale", outAsset.scaleFp16) &&
                        load("bias", outAsset.biasFp16) &&
                        load("mode", outAsset.mode) &&
                        load("aux", outAsset.aux) &&
                        load("residual_value", outAsset.residualValueFp16) &&
                        load("residual_table", outAsset.residualTable) &&
                        load("activation_scale", outAsset.activationScaleFp16);
    if (!loaded) {
        return false;
    }

    size_t residualIndexOffset = 0;
    size_t residualIndexLength = 0;
    if (!findRange("residual_index", residualIndexOffset, residualIndexLength)) {
        return false;
    }
    if (residualIndexLength == outAsset.residualValueFp16.size()) {
        outAsset.residualIndex.resize(residualIndexLength);
        for (size_t i = 0; i < residualIndexLength; ++i) {
            outAsset.residualIndex[i] = static_cast<int32_t>(payload[residualIndexOffset + i]);
        }
    } else {
        CopySection(payload, residualIndexOffset, residualIndexLength, outAsset.residualIndex);
    }

    uint64_t totalBlocks = 0;
    for (const LayerDesc& layer : outAsset.manifest.layers) {
        totalBlocks += layer.numBlocks;
    }
    if (outAsset.residualTable.size() != totalBlocks * 2u) {
        const std::vector<int32_t> compact = outAsset.residualTable;
        std::vector<int32_t> dense;
        dense.reserve(static_cast<size_t>(totalBlocks) * 2u);
        for (size_t layerIndex = 0; layerIndex < outAsset.manifest.layers.size(); ++layerIndex) {
            LayerDesc& layer = outAsset.manifest.layers[layerIndex];
            const uint32_t compactStart = layer.residualTableOffset;
            const uint32_t compactEnd = layerIndex + 1u < outAsset.manifest.layers.size()
                ? outAsset.manifest.layers[layerIndex + 1u].residualTableOffset
                : static_cast<uint32_t>(compact.size() / 3u);
            const uint32_t denseOffset = static_cast<uint32_t>(dense.size() / 2u);
            dense.insert(dense.end(), layer.numBlocks * 2u, 0);
            for (uint32_t span = compactStart; span < compactEnd; ++span) {
                const size_t base = static_cast<size_t>(span) * 3u;
                if (base + 2u >= compact.size()) {
                    if (error) *error = "SP4 compact residual span out of range";
                    return false;
                }
                const uint32_t block = static_cast<uint32_t>(compact[base + 0u]);
                if (block >= layer.numBlocks) {
                    if (error) *error = "SP4 compact residual block out of range";
                    return false;
                }
                const size_t out = static_cast<size_t>(denseOffset + block) * 2u;
                dense[out + 0u] = compact[base + 1u];
                dense[out + 1u] = compact[base + 2u];
            }
            layer.residualTableOffset = denseOffset;
        }
        outAsset.residualTable = std::move(dense);
    }
    return true;
}

std::vector<float> ReconstructLayer(const LoadedAsset& asset, const LayerDesc& layer) {
    const uint32_t paddedValues = layer.paddedValues;
    std::vector<int8_t> q = UnpackInt4(asset.qweight, layer.qweightOffsetBytes, layer.qweightLengthBytes, paddedValues);
    std::vector<uint8_t> uq = UnpackUInt4(asset.qweight, layer.qweightOffsetBytes, layer.qweightLengthBytes, paddedValues);
    std::vector<float> out(paddedValues, 0.0f);
    for (uint32_t block = 0; block < layer.numBlocks; ++block) {
        const float scale = HalfToFloat(asset.scaleFp16[layer.scaleOffset + block]);
        const uint8_t mode = asset.mode[layer.modeOffset + block];
        const bool codebook = mode == static_cast<uint8_t>(Mode::AdaptiveProject);
        const bool learnedCodebook = mode == static_cast<uint8_t>(Mode::LearnedProject);
        const bool zeroPoint = mode == static_cast<uint8_t>(Mode::LinearW4A8ZeroPoint);
        const int zp = zeroPoint ? static_cast<int>(asset.aux[layer.auxOffset + block] & 0x0fu) : 0;
        for (uint32_t i = 0; i < layer.blockSize; ++i) {
            const uint32_t index = block * layer.blockSize + i;
            if (codebook) {
                out[index] = CodebookValue(uq[index]) * scale;
            } else if (learnedCodebook) {
                out[index] = LearnedCodebookValue(uq[index]) * scale;
            } else if (zeroPoint) {
                out[index] = (static_cast<float>(uq[index]) - static_cast<float>(zp)) * scale;
            } else {
                out[index] = static_cast<float>(q[index]) * scale;
            }
        }
        const uint32_t tableOffset = (layer.residualTableOffset + block) * 2u;
        const int32_t start = asset.residualTable[tableOffset + 0u];
        const int32_t count = asset.residualTable[tableOffset + 1u];
        for (int32_t r = 0; r < count; ++r) {
            const int32_t localIndex = asset.residualIndex[layer.residualIndexOffset + start + r];
            out[block * layer.blockSize + static_cast<uint32_t>(localIndex)] +=
                HalfToFloat(asset.residualValueFp16[layer.residualValueOffset + start + r]);
        }
    }
    uint32_t original = 1;
    for (uint32_t dim : layer.shape) {
        original *= dim;
    }
    out.resize(original);
    return out;
}

} // namespace sp4
