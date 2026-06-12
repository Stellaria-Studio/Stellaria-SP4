#include "sp4/SP4ERuntime.h"

#include "SP4Common.h"

#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <type_traits>
#include <unordered_map>

namespace sp4 {

namespace {

constexpr float kFp4E2M1Codebook[16] = {
    -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.0f,
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
};

template <typename T>
void AppendVectorBytes(std::vector<uint8_t>& payload, const std::vector<T>& values) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
    payload.insert(payload.end(), bytes, bytes + values.size() * sizeof(T));
}

void AddSection(std::vector<uint8_t>& payload,
                std::vector<SectionDesc>& descs,
                const std::string& name,
                const std::vector<uint8_t>& data) {
    SectionDesc desc;
    desc.name = name;
    desc.offset = payload.size();
    desc.length = data.size();
    payload.insert(payload.end(), data.begin(), data.end());
    descs.push_back(std::move(desc));
}

template <typename T>
void AddTypedSection(std::vector<uint8_t>& payload,
                     std::vector<SectionDesc>& descs,
                     const std::string& name,
                     const std::vector<T>& values) {
    std::vector<uint8_t> bytes;
    AppendVectorBytes(bytes, values);
    AddSection(payload, descs, name, bytes);
}

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

std::string RuntimeDecisionReason(SP4ERuntimePath path) {
    switch (path) {
    case SP4ERuntimePath::TensorPath: return "native_fp4_tensor_support";
    case SP4ERuntimePath::LowQAluPath: return "low_precision_alu_speedup";
    case SP4ERuntimePath::ProfileGuidedPath: return "profiled_sp4e_faster_than_sp4";
    case SP4ERuntimePath::Sp4SubsetPath: return "sp4_subset_available_on_generic_backend";
    case SP4ERuntimePath::Sp4Fallback: break;
    }
    return "no_profitable_fp4_or_lowq_acceleration";
}

std::string BoolText(bool value) {
    return value ? "true" : "false";
}

std::string SP4EManifestToJson(const SP4EManifest& manifest,
                               const std::vector<SectionDesc>& sections) {
    std::ostringstream out;
    out << "{";
    out << "\"adapter\":\"" << JsonEscape(manifest.adapter) << "\",";
    out << "\"architecture\":\"SP4E-F4I4block16(A1P)/SP4E-F4I4tile16x16(A1P)\",";
    out << "\"backend_plan\":{";
    out << "\"available\":[\"cpu_reference\",\"sp4_fallback\",\"metal_tensorops_future\",\"cuda_tensor_core_fp4_future\",\"dx12_cooperative_matrix_future\",\"vulkan_cooperative_matrix_future\"],";
    out << "\"fallback\":\"sp4_subset_per_block_then_fp16_correction\",";
    out << "\"preferred\":\"" << SP4ERuntimePathName(manifest.runtimeDecision.path) << "\",";
    out << "\"sp4_subset_blocks\":" << manifest.sp4SubsetBlocks << ",";
    out << "\"fp16_fallback_blocks\":" << manifest.fp16FallbackBlocks << ",";
    out << "\"tensor_layout\":\"SP4E tile16x16 with separated fp4/int4 scale planes\",";
    out << "\"dispatch\":\"fixed_backend_plan_from_asset\"";
    out << "},";
    out << "\"compiler\":{\"name\":\"sp4ec-cpp\",\"version\":\"0.1.0-preview\"},";
    out << "\"compile_ms\":" << manifest.compileMs << ",";
    out << "\"excluded_layers\":[";
    for (size_t i = 0; i < manifest.excludedLayers.size(); ++i) {
        if (i) out << ",";
        out << "{\"name\":\"" << JsonEscape(manifest.excludedLayers[i]) << "\",\"reason\":\"excluded_by_" << JsonEscape(manifest.adapter) << "_runtime_policy\"}";
    }
    out << "],";
    out << "\"fallback_plan\":{\"kind\":\"partial_sp4_subset\",\"reason\":\"INT4-compatible blocks route to SP4 backends on generic hardware; FP4/bridge/sensitive blocks use correction or FP16 fallback\"},";
    out << "\"format\":\"stellaria.sp4e\",";
    out << "\"graph_tensor_count\":" << manifest.graphTensorCount << ",";
    out << "\"hardware_gate\":{";
    out << "\"backend\":\"" << JsonEscape(manifest.capability.backend) << "\",";
    out << "\"cooperative_matrix_support\":" << BoolText(manifest.capability.cooperativeMatrixSupport) << ",";
    out << "\"device\":\"" << JsonEscape(manifest.capability.device) << "\",";
    out << "\"low_precision_alu_speedup\":" << manifest.capability.lowPrecisionAluSpeedup << ",";
    out << "\"native_fp4_tensor_support\":" << BoolText(manifest.capability.nativeFp4TensorSupport) << ",";
    out << "\"native_int4_tensor_support\":" << BoolText(manifest.capability.nativeInt4TensorSupport) << ",";
    out << "\"runtime_decision\":{";
    out << "\"enabled\":" << BoolText(manifest.runtimeDecision.enabled) << ",";
    out << "\"path\":\"" << SP4ERuntimePathName(manifest.runtimeDecision.path) << "\",";
    out << "\"reason\":\"" << RuntimeDecisionReason(manifest.runtimeDecision.path) << "\"";
    out << "},";
    out << "\"sp4e_vs_sp4_speedup\":" << manifest.capability.sp4eVsSp4Speedup << ",";
    out << "\"tensor_tile_support\":" << BoolText(manifest.capability.tensorTileSupport);
    out << "},";
    out << "\"layers\":[";
    for (size_t i = 0; i < manifest.layers.size(); ++i) {
        const SP4ELayerDesc& layer = manifest.layers[i];
        if (i) out << ",";
        out << "{";
        out << "\"activation_precision\":\"" << JsonEscape(manifest.activationPrecision) << "\",";
        out << "\"activation_precision_offset\":" << layer.activationPrecisionOffset << ",";
        out << "\"aux_offset\":" << layer.auxOffset << ",";
        out << "\"aux_policy\":" << layer.auxPolicy << ",";
        out << "\"bias_count\":" << layer.biasCount << ",";
        out << "\"bias_offset\":" << layer.biasOffset << ",";
        out << "\"block_abi\":\"SP4E-F4I4block16(A1P)\",";
        out << "\"block_size\":" << layer.blockSize << ",";
        out << "\"bridge_blocks\":" << layer.bridgeBlocks << ",";
        out << "\"bridge_offset\":" << layer.bridgeOffset << ",";
        out << "\"channel_correction_count\":" << layer.channelCorrectionCount << ",";
        out << "\"channel_correction_offset\":" << layer.channelCorrectionOffset << ",";
        out << "\"fp16_fallback_blocks\":" << layer.fp16FallbackBlocks << ",";
        out << "\"fp4_blocks\":" << layer.fp4Blocks << ",";
        out << "\"int4_blocks\":" << layer.int4Blocks << ",";
        out << "\"low_rank_rank\":" << layer.lowRankRank << ",";
        out << "\"low_rank_u_offset\":" << layer.lowRankUOffset << ",";
        out << "\"low_rank_v_offset\":" << layer.lowRankVOffset << ",";
        out << "\"metrics\":{\"mae\":" << layer.mae << ",\"max_abs\":" << layer.maxAbs << ",\"rmse\":" << layer.rmse << "},";
        out << "\"mode_offset\":" << layer.modeOffset << ",";
        out << "\"name\":\"" << JsonEscape(layer.name) << "\",";
        out << "\"num_blocks\":" << layer.numBlocks << ",";
        out << "\"padded_values\":" << layer.paddedValues << ",";
        out << "\"qweight_fp4_offset_bytes\":" << layer.qweightFp4OffsetBytes << ",";
        out << "\"qweight_int4_offset_bytes\":" << layer.qweightInt4OffsetBytes << ",";
        out << "\"qweight_length_bytes\":" << layer.qweightLengthBytes << ",";
        out << "\"residual_entries\":" << layer.residualEntries << ",";
        out << "\"residual_index_offset\":" << layer.residualIndexOffset << ",";
        out << "\"residual_table_offset\":" << layer.residualTableOffset << ",";
        out << "\"residual_value_offset\":" << layer.residualValueOffset << ",";
        out << "\"scale_fp4_offset\":" << layer.scaleFp4Offset << ",";
        out << "\"scale_int4_offset\":" << layer.scaleInt4Offset << ",";
        out << "\"sensitive\":" << BoolText(layer.sensitive) << ",";
        out << "\"sp4_subset_blocks\":" << layer.sp4SubsetBlocks << ",";
        out << "\"shape\":[";
        for (size_t s = 0; s < layer.shape.size(); ++s) {
            if (s) out << ",";
            out << layer.shape[s];
        }
        out << "],";
        out << "\"tile_abi\":\"SP4E-F4I4tile16x16(A1P)\"";
        out << "}";
    }
    out << "],";
    out << "\"precision\":{\"activation\":\"" << JsonEscape(manifest.activationPrecision)
        << "\",\"bias\":\"fp16\",\"correction\":[\"sparse_fp16\",\"low_rank_fp16_future\",\"channel_correction_future\"],"
        << "\"residual\":\"sparse_fp16\",\"scale\":\"fp16\",\"weight\":\"fp4_int4_mixed\"},";
    out << "\"residual_table_encoding\":\"pair_i32_block_count_start\",";
    out << "\"sections\":{";
    for (size_t i = 0; i < sections.size(); ++i) {
        if (i) out << ",";
        out << "\"" << sections[i].name << "\":{\"length\":" << sections[i].length << ",\"offset\":" << sections[i].offset << "}";
    }
    out << "},";
    out << "\"source\":\"" << JsonEscape(manifest.source) << "\",";
    out << "\"totals\":{\"blocks\":" << manifest.blocks
        << ",\"bridge_blocks\":" << manifest.bridgeBlocks
        << ",\"fp16_fallback_blocks\":" << manifest.fp16FallbackBlocks
        << ",\"fp4_blocks\":" << manifest.fp4Blocks
        << ",\"int4_blocks\":" << manifest.int4Blocks
        << ",\"residual_entries\":" << manifest.residualEntries
        << ",\"sp4_subset_blocks\":" << manifest.sp4SubsetBlocks
        << ",\"sp4e_weight_bytes\":" << manifest.sp4eWeightBytes
        << ",\"teacher_weight_bytes\":" << manifest.teacherWeightBytes << "},";
    out << "\"version\":\"" << manifest.version << "\"";
    out << "}";
    return out.str();
}

void ParseSP4EManifest(const std::string& text, SP4EManifest& manifest) {
    std::smatch match;
    if (std::regex_search(text, match, std::regex("\"version\"\\s*:\\s*\"(A[0-9][A-Z])\""))) manifest.version = match[1].str();
    if (std::regex_search(text, match, std::regex("\"adapter\"\\s*:\\s*\"([^\"]+)\""))) manifest.adapter = match[1].str();
    if (std::regex_search(text, match, std::regex("\"source\"\\s*:\\s*\"([^\"]*)\""))) manifest.source = match[1].str();
    if (std::regex_search(text, match, std::regex("\"graph_tensor_count\"\\s*:\\s*([0-9]+)"))) manifest.graphTensorCount = static_cast<uint32_t>(std::stoul(match[1].str()));
    if (std::regex_search(text, match, std::regex("\"totals\"\\s*:\\s*\\{\"blocks\"\\s*:\\s*([0-9]+),\"bridge_blocks\"\\s*:\\s*([0-9]+),\"fp16_fallback_blocks\"\\s*:\\s*([0-9]+),\"fp4_blocks\"\\s*:\\s*([0-9]+),\"int4_blocks\"\\s*:\\s*([0-9]+),\"residual_entries\"\\s*:\\s*([0-9]+),\"sp4_subset_blocks\"\\s*:\\s*([0-9]+),\"sp4e_weight_bytes\"\\s*:\\s*([0-9]+),\"teacher_weight_bytes\"\\s*:\\s*([0-9]+)\\}"))) {
        manifest.blocks = static_cast<uint64_t>(std::stoull(match[1].str()));
        manifest.bridgeBlocks = static_cast<uint64_t>(std::stoull(match[2].str()));
        manifest.fp16FallbackBlocks = static_cast<uint64_t>(std::stoull(match[3].str()));
        manifest.fp4Blocks = static_cast<uint64_t>(std::stoull(match[4].str()));
        manifest.int4Blocks = static_cast<uint64_t>(std::stoull(match[5].str()));
        manifest.residualEntries = static_cast<uint64_t>(std::stoull(match[6].str()));
        manifest.sp4SubsetBlocks = static_cast<uint64_t>(std::stoull(match[7].str()));
        manifest.sp4eWeightBytes = static_cast<uint64_t>(std::stoull(match[8].str()));
        manifest.teacherWeightBytes = static_cast<uint64_t>(std::stoull(match[9].str()));
    }
    if (std::regex_search(text, match, std::regex("\"path\"\\s*:\\s*\"(SP4E_TENSOR_PATH|SP4E_LOWQ_ALU_PATH|SP4E_PROFILE_GUIDED_PATH|SP4E_SP4_SUBSET_PATH|SP4_FALLBACK)\""))) {
        const std::string path = match[1].str();
        if (path == "SP4E_TENSOR_PATH") manifest.runtimeDecision.path = SP4ERuntimePath::TensorPath;
        else if (path == "SP4E_LOWQ_ALU_PATH") manifest.runtimeDecision.path = SP4ERuntimePath::LowQAluPath;
        else if (path == "SP4E_PROFILE_GUIDED_PATH") manifest.runtimeDecision.path = SP4ERuntimePath::ProfileGuidedPath;
        else if (path == "SP4E_SP4_SUBSET_PATH") manifest.runtimeDecision.path = SP4ERuntimePath::Sp4SubsetPath;
        else manifest.runtimeDecision.path = SP4ERuntimePath::Sp4Fallback;
        manifest.runtimeDecision.enabled = manifest.runtimeDecision.path != SP4ERuntimePath::Sp4Fallback;
    }
    const std::regex layerRegex(
        "\\{\"activation_precision\":\"([^\"]+)\",\"activation_precision_offset\":([0-9]+),\"aux_offset\":([0-9]+),\"aux_policy\":([0-9]+),"
        "\"bias_count\":([0-9]+),\"bias_offset\":([0-9]+),\"block_abi\":\"SP4E-F4I4block16\\(A1P\\)\","
        "\"block_size\":([0-9]+),\"bridge_blocks\":([0-9]+),\"bridge_offset\":([0-9]+),\"channel_correction_count\":([0-9]+),\"channel_correction_offset\":([0-9]+),\"fp16_fallback_blocks\":([0-9]+),\"fp4_blocks\":([0-9]+),\"int4_blocks\":([0-9]+),"
        "\"low_rank_rank\":([0-9]+),\"low_rank_u_offset\":([0-9]+),\"low_rank_v_offset\":([0-9]+),"
        "\"metrics\":\\{\"mae\":([-+0-9.eE]+),\"max_abs\":([-+0-9.eE]+),\"rmse\":([-+0-9.eE]+)\\},"
        "\"mode_offset\":([0-9]+),\"name\":\"([^\"]+)\",\"num_blocks\":([0-9]+),\"padded_values\":([0-9]+),"
        "\"qweight_fp4_offset_bytes\":([0-9]+),\"qweight_int4_offset_bytes\":([0-9]+),\"qweight_length_bytes\":([0-9]+),"
        "\"residual_entries\":([0-9]+),\"residual_index_offset\":([0-9]+),\"residual_table_offset\":([0-9]+),\"residual_value_offset\":([0-9]+),"
        "\"scale_fp4_offset\":([0-9]+),\"scale_int4_offset\":([0-9]+),\"sensitive\":(true|false),\"sp4_subset_blocks\":([0-9]+),\"shape\":\\[([^\\]]*)\\],\"tile_abi\":\"SP4E-F4I4tile16x16\\(A1P\\)\"\\}");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), layerRegex);
         it != std::sregex_iterator();
         ++it) {
        const auto& m = *it;
        SP4ELayerDesc layer;
        manifest.activationPrecision = m[1].str();
        layer.activationPrecisionOffset = static_cast<uint32_t>(std::stoul(m[2].str()));
        layer.auxOffset = static_cast<uint32_t>(std::stoul(m[3].str()));
        layer.auxPolicy = static_cast<uint32_t>(std::stoul(m[4].str()));
        layer.biasCount = static_cast<uint32_t>(std::stoul(m[5].str()));
        layer.biasOffset = static_cast<uint32_t>(std::stoul(m[6].str()));
        layer.blockSize = static_cast<uint32_t>(std::stoul(m[7].str()));
        layer.bridgeBlocks = static_cast<uint32_t>(std::stoul(m[8].str()));
        layer.bridgeOffset = static_cast<uint32_t>(std::stoul(m[9].str()));
        layer.channelCorrectionCount = static_cast<uint32_t>(std::stoul(m[10].str()));
        layer.channelCorrectionOffset = static_cast<uint32_t>(std::stoul(m[11].str()));
        layer.fp16FallbackBlocks = static_cast<uint32_t>(std::stoul(m[12].str()));
        layer.fp4Blocks = static_cast<uint32_t>(std::stoul(m[13].str()));
        layer.int4Blocks = static_cast<uint32_t>(std::stoul(m[14].str()));
        layer.lowRankRank = static_cast<uint32_t>(std::stoul(m[15].str()));
        layer.lowRankUOffset = static_cast<uint32_t>(std::stoul(m[16].str()));
        layer.lowRankVOffset = static_cast<uint32_t>(std::stoul(m[17].str()));
        layer.mae = std::stof(m[18].str());
        layer.maxAbs = std::stof(m[19].str());
        layer.rmse = std::stof(m[20].str());
        layer.modeOffset = static_cast<uint32_t>(std::stoul(m[21].str()));
        layer.name = m[22].str();
        layer.numBlocks = static_cast<uint32_t>(std::stoul(m[23].str()));
        layer.paddedValues = static_cast<uint32_t>(std::stoul(m[24].str()));
        layer.qweightFp4OffsetBytes = static_cast<uint32_t>(std::stoul(m[25].str()));
        layer.qweightInt4OffsetBytes = static_cast<uint32_t>(std::stoul(m[26].str()));
        layer.qweightLengthBytes = static_cast<uint32_t>(std::stoul(m[27].str()));
        layer.residualEntries = static_cast<uint32_t>(std::stoul(m[28].str()));
        layer.residualIndexOffset = static_cast<uint32_t>(std::stoul(m[29].str()));
        layer.residualTableOffset = static_cast<uint32_t>(std::stoul(m[30].str()));
        layer.residualValueOffset = static_cast<uint32_t>(std::stoul(m[31].str()));
        layer.scaleFp4Offset = static_cast<uint32_t>(std::stoul(m[32].str()));
        layer.scaleInt4Offset = static_cast<uint32_t>(std::stoul(m[33].str()));
        layer.sensitive = m[34].str() == "true";
        layer.sp4SubsetBlocks = static_cast<uint32_t>(std::stoul(m[35].str()));
        layer.shape = ParseShapeList(m[36].str());
        manifest.layers.push_back(std::move(layer));
    }
}

std::vector<uint8_t> UnpackUInt4(const std::vector<uint8_t>& packed, uint32_t offset, uint32_t length, uint32_t count) {
    std::vector<uint8_t> out(count);
    uint32_t write = 0;
    for (uint32_t i = 0; i < length && write < count; ++i) {
        const uint8_t byte = packed[offset + i];
        out[write++] = byte & 0x0fu;
        if (write < count) out[write++] = (byte >> 4u) & 0x0fu;
    }
    return out;
}

int8_t SignExtend4(uint8_t value) {
    value &= 0x0fu;
    return static_cast<int8_t>(value >= 8u ? static_cast<int>(value) - 16 : value);
}

std::vector<int8_t> UnpackInt4(const std::vector<uint8_t>& packed, uint32_t offset, uint32_t length, uint32_t count) {
    std::vector<int8_t> out(count);
    uint32_t write = 0;
    for (uint32_t i = 0; i < length && write < count; ++i) {
        const uint8_t byte = packed[offset + i];
        out[write++] = SignExtend4(byte & 0x0fu);
        if (write < count) out[write++] = SignExtend4(byte >> 4u);
    }
    return out;
}

uint8_t ReadPackedNibble(const std::vector<uint8_t>& packed, uint32_t offsetBytes, uint32_t index) {
    const uint32_t byteOffset = offsetBytes + index / 2u;
    if (byteOffset >= packed.size()) return 0u;
    const uint8_t byte = packed[byteOffset];
    return (index & 1u) == 0u ? byte & 0x0fu : (byte >> 4u) & 0x0fu;
}

} // namespace

const char* SP4ERuntimePathName(SP4ERuntimePath path) {
    switch (path) {
    case SP4ERuntimePath::TensorPath: return "SP4E_TENSOR_PATH";
    case SP4ERuntimePath::LowQAluPath: return "SP4E_LOWQ_ALU_PATH";
    case SP4ERuntimePath::ProfileGuidedPath: return "SP4E_PROFILE_GUIDED_PATH";
    case SP4ERuntimePath::Sp4SubsetPath: return "SP4E_SP4_SUBSET_PATH";
    case SP4ERuntimePath::Sp4Fallback: break;
    }
    return "SP4_FALLBACK";
}

const char* SP4EModeName(uint8_t mode) {
    switch (static_cast<SP4EMode>(mode)) {
    case SP4EMode::Fp4Main: return "SP4E_MODE_FP4_MAIN";
    case SP4EMode::Int4Main: return "SP4E_MODE_INT4_MAIN";
    case SP4EMode::Fp4WithR16: return "SP4E_MODE_FP4_WITH_R16";
    case SP4EMode::Int4WithR16: return "SP4E_MODE_INT4_WITH_R16";
    case SP4EMode::F4I4Bridge: return "SP4E_MODE_F4I4_BRIDGE";
    case SP4EMode::LowRankR16: return "SP4E_MODE_LOW_RANK_R16";
    case SP4EMode::ChannelCorrection: return "SP4E_MODE_CHANNEL_CORRECTION";
    case SP4EMode::SensitiveFp16: return "SP4E_MODE_SENSITIVE_FP16";
    case SP4EMode::SensitiveBf16: return "SP4E_MODE_SENSITIVE_BF16";
    case SP4EMode::ProjectAdaptive: return "SP4E_MODE_PROJECT_ADAPTIVE";
    case SP4EMode::BackendNative: return "SP4E_MODE_BACKEND_NATIVE";
    case SP4EMode::Sp4Fallback: return "SP4E_MODE_SP4_FALLBACK";
    }
    return "SP4E_MODE_INVALID";
}

bool WriteSP4EAsset(const std::string& path,
                    const SP4EManifest& manifest,
                    const SP4ESections& sections,
                    std::string* error) {
    std::vector<uint8_t> payload;
    std::vector<SectionDesc> descs;
    AddSection(payload, descs, "qweight_fp4", sections.qweightFp4);
    AddSection(payload, descs, "qweight_int4", sections.qweightInt4);
    AddTypedSection(payload, descs, "scale_fp4", sections.scaleFp16Fp4);
    AddTypedSection(payload, descs, "scale_int4", sections.scaleFp16Int4);
    AddTypedSection(payload, descs, "bias", sections.biasFp16);
    AddSection(payload, descs, "mode", sections.mode);
    AddSection(payload, descs, "aux", sections.aux);
    AddSection(payload, descs, "bridge", sections.bridge);
    AddSection(payload, descs, "residual_index", sections.residualIndex);
    AddTypedSection(payload, descs, "residual_value", sections.residualValueFp16);
    AddTypedSection(payload, descs, "residual_table", sections.residualTable);
    AddSection(payload, descs, "activation_precision", sections.activationPrecision);
    AddSection(payload, descs, "tile_plan", sections.tilePlan);
    AddTypedSection(payload, descs, "low_rank_u", sections.lowRankU);
    AddTypedSection(payload, descs, "low_rank_v", sections.lowRankV);
    AddTypedSection(payload, descs, "channel_correction", sections.channelCorrection);

    const std::string manifestJson = SP4EManifestToJson(manifest, descs);
    std::vector<uint8_t> file;
    file.insert(file.end(), kSP4EMagicA1P, kSP4EMagicA1P + 8);
    AppendU32(file, kSP4EContainerVersion);
    AppendU32(file, static_cast<uint32_t>(manifestJson.size()));
    file.insert(file.end(), manifestJson.begin(), manifestJson.end());
    file.insert(file.end(), payload.begin(), payload.end());

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) *error = "failed to open SP4E output asset: " + path;
        return false;
    }
    out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    if (!out.good()) {
        if (error) *error = "failed to write SP4E output asset: " + path;
        return false;
    }
    return true;
}

bool LoadSP4EAsset(const std::string& path, LoadedSP4EAsset& outAsset, std::string* error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        if (error) *error = "failed to open SP4E asset: " + path;
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size < 16) {
        if (error) *error = "SP4E asset is too small";
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(raw.data()), size);
    if (std::memcmp(raw.data(), kSP4EMagicA1P, 8) != 0) {
        if (error) *error = "invalid SP4E magic";
        return false;
    }
    if (ReadU32(raw.data() + 8) != kSP4EContainerVersion) {
        if (error) *error = "unsupported SP4E container version";
        return false;
    }
    const uint32_t manifestBytes = ReadU32(raw.data() + 12);
    if (16u + manifestBytes > raw.size()) {
        if (error) *error = "invalid SP4E manifest length";
        return false;
    }
    const std::string manifest(reinterpret_cast<const char*>(raw.data() + 16), manifestBytes);
    const std::vector<uint8_t> payload(raw.begin() + 16 + manifestBytes, raw.end());
    outAsset = LoadedSP4EAsset{};
    ParseSP4EManifest(manifest, outAsset.manifest);

    auto load = [&](const std::string& name, auto& section) -> bool {
        size_t offset = 0;
        size_t length = 0;
        if (!FindSection(manifest, name, offset, length)) {
            if (error) *error = "missing SP4E section: " + name;
            return false;
        }
        if (offset + length > payload.size()) {
            if (error) *error = "SP4E section range invalid: " + name;
            return false;
        }
        using Vec = std::decay_t<decltype(section)>;
        using T = typename Vec::value_type;
        CopySection(payload, offset, length, section);
        return true;
    };

    const bool loaded = load("qweight_fp4", outAsset.sections.qweightFp4) &&
                        load("qweight_int4", outAsset.sections.qweightInt4) &&
                        load("scale_fp4", outAsset.sections.scaleFp16Fp4) &&
                        load("scale_int4", outAsset.sections.scaleFp16Int4) &&
                        load("bias", outAsset.sections.biasFp16) &&
                        load("mode", outAsset.sections.mode) &&
                        load("aux", outAsset.sections.aux) &&
                        load("bridge", outAsset.sections.bridge) &&
                        load("residual_index", outAsset.sections.residualIndex) &&
                        load("residual_value", outAsset.sections.residualValueFp16) &&
                        load("residual_table", outAsset.sections.residualTable) &&
                        load("activation_precision", outAsset.sections.activationPrecision) &&
                        load("tile_plan", outAsset.sections.tilePlan) &&
                        load("low_rank_u", outAsset.sections.lowRankU) &&
                        load("low_rank_v", outAsset.sections.lowRankV) &&
                        load("channel_correction", outAsset.sections.channelCorrection);
    if (!loaded) return false;

    uint64_t totalBlocks = 0;
    for (const SP4ELayerDesc& layer : outAsset.manifest.layers) totalBlocks += layer.numBlocks;
    if (outAsset.sections.residualTable.size() != totalBlocks * 2u &&
        manifest.find("\"residual_table_encoding\":\"pair_i32_block_count_start\"") != std::string::npos &&
        outAsset.sections.residualTable.size() % 2u == 0u) {
        const std::vector<int32_t> compact = outAsset.sections.residualTable;
        std::vector<int32_t> dense;
        dense.reserve(static_cast<size_t>(totalBlocks) * 2u);
        for (size_t layerIndex = 0; layerIndex < outAsset.manifest.layers.size(); ++layerIndex) {
            SP4ELayerDesc& layer = outAsset.manifest.layers[layerIndex];
            const uint32_t compactStart = layer.residualTableOffset;
            const uint32_t compactEnd = layerIndex + 1u < outAsset.manifest.layers.size()
                ? outAsset.manifest.layers[layerIndex + 1u].residualTableOffset
                : static_cast<uint32_t>(compact.size() / 2u);
            const uint32_t denseOffset = static_cast<uint32_t>(dense.size() / 2u);
            dense.insert(dense.end(), layer.numBlocks * 2u, 0);
            for (uint32_t span = compactStart; span < compactEnd; ++span) {
                const size_t base = static_cast<size_t>(span) * 2u;
                const uint32_t packed = static_cast<uint32_t>(compact[base + 0u]);
                const uint32_t block = packed >> 5u;
                const int32_t count = static_cast<int32_t>(packed & 0x1fu);
                if (block >= layer.numBlocks) {
                    if (error) *error = "SP4E compact residual block out of range";
                    return false;
                }
                const size_t out = static_cast<size_t>(denseOffset + block) * 2u;
                dense[out + 0u] = compact[base + 1u];
                dense[out + 1u] = count;
            }
            layer.residualTableOffset = denseOffset;
        }
        outAsset.sections.residualTable = std::move(dense);
    } else if (outAsset.sections.residualTable.size() != totalBlocks * 2u &&
               outAsset.sections.residualTable.size() % 3u == 0u) {
        const std::vector<int32_t> compact = outAsset.sections.residualTable;
        std::vector<int32_t> dense;
        dense.reserve(static_cast<size_t>(totalBlocks) * 2u);
        for (size_t layerIndex = 0; layerIndex < outAsset.manifest.layers.size(); ++layerIndex) {
            SP4ELayerDesc& layer = outAsset.manifest.layers[layerIndex];
            const uint32_t compactStart = layer.residualTableOffset;
            const uint32_t compactEnd = layerIndex + 1u < outAsset.manifest.layers.size()
                ? outAsset.manifest.layers[layerIndex + 1u].residualTableOffset
                : static_cast<uint32_t>(compact.size() / 3u);
            const uint32_t denseOffset = static_cast<uint32_t>(dense.size() / 2u);
            dense.insert(dense.end(), layer.numBlocks * 2u, 0);
            for (uint32_t span = compactStart; span < compactEnd; ++span) {
                const size_t base = static_cast<size_t>(span) * 3u;
                const uint32_t block = static_cast<uint32_t>(compact[base + 0u]);
                if (block >= layer.numBlocks) {
                    if (error) *error = "SP4E compact residual block out of range";
                    return false;
                }
                const size_t out = static_cast<size_t>(denseOffset + block) * 2u;
                dense[out + 0u] = compact[base + 1u];
                dense[out + 1u] = compact[base + 2u];
            }
            layer.residualTableOffset = denseOffset;
        }
        outAsset.sections.residualTable = std::move(dense);
    }
    return true;
}

std::vector<float> ReconstructSP4ELayer(const LoadedSP4EAsset& asset, const SP4ELayerDesc& layer) {
    const std::vector<uint8_t> mainU4 = UnpackUInt4(asset.sections.qweightFp4,
                                                    layer.qweightFp4OffsetBytes,
                                                    layer.qweightLengthBytes,
                                                    layer.paddedValues);
    const std::vector<int8_t> mainI4 = UnpackInt4(asset.sections.qweightFp4,
                                                  layer.qweightFp4OffsetBytes,
                                                  layer.qweightLengthBytes,
                                                  layer.paddedValues);
    const bool hasSideInt4 = layer.qweightInt4OffsetBytes + layer.qweightLengthBytes <= asset.sections.qweightInt4.size();
    const std::vector<int8_t> sideInt4 = hasSideInt4
        ? UnpackInt4(asset.sections.qweightInt4, layer.qweightInt4OffsetBytes, layer.qweightLengthBytes, layer.paddedValues)
        : std::vector<int8_t>{};
    std::vector<float> out(layer.paddedValues, 0.0f);
    for (uint32_t block = 0; block < layer.numBlocks; ++block) {
        const uint8_t mode = ReadPackedNibble(asset.sections.mode, layer.modeOffset, block);
        const float scaleMain = HalfToFloat(asset.sections.scaleFp16Fp4[layer.scaleFp4Offset + block]);
        const float scaleSideInt4 = layer.scaleInt4Offset + block < asset.sections.scaleFp16Int4.size()
            ? HalfToFloat(asset.sections.scaleFp16Int4[layer.scaleInt4Offset + block])
            : scaleMain;
        for (uint32_t i = 0; i < layer.blockSize; ++i) {
            const uint32_t index = block * layer.blockSize + i;
            const float fp4Value = kFp4E2M1Codebook[mainU4[index] & 0x0fu] * scaleMain;
            const float int4Value = static_cast<float>(mainI4[index]) * scaleMain;
            const float zpValue = (static_cast<float>(mainU4[index]) -
                                   static_cast<float>(ReadPackedNibble(asset.sections.aux, layer.auxOffset, block))) * scaleMain;
            if (mode == static_cast<uint8_t>(SP4EMode::Int4Main) ||
                mode == static_cast<uint8_t>(SP4EMode::Int4WithR16)) {
                out[index] = int4Value;
            } else if (mode == static_cast<uint8_t>(SP4EMode::Sp4Fallback)) {
                out[index] = zpValue;
            } else if (mode == static_cast<uint8_t>(SP4EMode::F4I4Bridge)) {
                const float side = hasSideInt4 ? static_cast<float>(sideInt4[index]) * scaleSideInt4 : int4Value;
                out[index] = 0.75f * fp4Value + 0.25f * side;
            } else {
                out[index] = fp4Value;
            }
        }
        const uint32_t tableOffset = (layer.residualTableOffset + block) * 2u;
        const int32_t start = asset.sections.residualTable[tableOffset + 0u];
        const int32_t count = asset.sections.residualTable[tableOffset + 1u];
        for (int32_t r = 0; r < count; ++r) {
            const uint8_t localIndex = asset.sections.residualIndex[layer.residualIndexOffset + start + r];
            out[block * layer.blockSize + localIndex] +=
                HalfToFloat(asset.sections.residualValueFp16[layer.residualValueOffset + start + r]);
        }
    }
    uint32_t original = 1;
    for (uint32_t dim : layer.shape) original *= dim;
    if (layer.shape.size() == 4u &&
        layer.lowRankRank > 0u &&
        layer.shape[0] > 0u &&
        original % layer.shape[0] == 0u) {
        const uint32_t outputChannels = layer.shape[0];
        const uint32_t valuesPerOutput = original / outputChannels;
        const uint64_t uCount = static_cast<uint64_t>(layer.lowRankRank) * outputChannels;
        const uint64_t vCount = static_cast<uint64_t>(layer.lowRankRank) * valuesPerOutput;
        if (static_cast<uint64_t>(layer.lowRankUOffset) + uCount <= asset.sections.lowRankU.size() &&
            static_cast<uint64_t>(layer.lowRankVOffset) + vCount <= asset.sections.lowRankV.size()) {
            for (uint32_t rank = 0; rank < layer.lowRankRank; ++rank) {
                const uint32_t uBase = layer.lowRankUOffset + rank * outputChannels;
                const uint32_t vBase = layer.lowRankVOffset + rank * valuesPerOutput;
                for (uint32_t oc = 0; oc < outputChannels; ++oc) {
                    const float u = HalfToFloat(asset.sections.lowRankU[uBase + oc]);
                    const uint32_t outBase = oc * valuesPerOutput;
                    for (uint32_t k = 0; k < valuesPerOutput; ++k) {
                        out[outBase + k] += u * HalfToFloat(asset.sections.lowRankV[vBase + k]);
                    }
                }
            }
        }
    }
    if (!layer.shape.empty() &&
        layer.shape[0] > 0u &&
        layer.channelCorrectionCount >= layer.shape[0] &&
        layer.channelCorrectionOffset + layer.shape[0] <= asset.sections.channelCorrection.size() &&
        original % layer.shape[0] == 0u) {
        const uint32_t valuesPerOutput = original / layer.shape[0];
        for (uint32_t oc = 0; oc < layer.shape[0]; ++oc) {
            const float gain = HalfToFloat(asset.sections.channelCorrection[layer.channelCorrectionOffset + oc]);
            const uint32_t base = oc * valuesPerOutput;
            for (uint32_t i = 0; i < valuesPerOutput; ++i) out[base + i] *= gain;
        }
    }
    out.resize(original);
    return out;
}

bool ExecuteSP4EConv2DNCHW(const LoadedSP4EAsset& asset,
                           const SP4ELayerDesc& layer,
                           const std::vector<float>& input,
                           uint32_t batch,
                           uint32_t inputChannels,
                           uint32_t height,
                           uint32_t width,
                           uint32_t stride,
                           uint32_t padding,
                           std::vector<float>& output,
                           std::string* error) {
    if (layer.shape.size() != 4) {
        if (error) *error = "SP4E Conv2D requires a 4D weight layer";
        return false;
    }
    const uint32_t outputChannels = layer.shape[0];
    const uint32_t weightInputChannels = layer.shape[1];
    const uint32_t kernelH = layer.shape[2];
    const uint32_t kernelW = layer.shape[3];
    if (inputChannels != weightInputChannels) {
        if (error) *error = "SP4E Conv2D input channel mismatch";
        return false;
    }
    if (stride == 0 || height + 2u * padding < kernelH || width + 2u * padding < kernelW) {
        if (error) *error = "SP4E Conv2D invalid shape/stride/padding";
        return false;
    }
    const uint32_t outH = (height + 2u * padding - kernelH) / stride + 1u;
    const uint32_t outW = (width + 2u * padding - kernelW) / stride + 1u;
    const std::vector<float> weight = ReconstructSP4ELayer(asset, layer);
    output.assign(static_cast<size_t>(batch) * outputChannels * outH * outW, 0.0f);
    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t oc = 0; oc < outputChannels; ++oc) {
            const float bias = layer.biasOffset + oc < asset.sections.biasFp16.size()
                ? HalfToFloat(asset.sections.biasFp16[layer.biasOffset + oc])
                : 0.0f;
            for (uint32_t oy = 0; oy < outH; ++oy) {
                for (uint32_t ox = 0; ox < outW; ++ox) {
                    float sum = bias;
                    for (uint32_t ic = 0; ic < inputChannels; ++ic) {
                        for (uint32_t ky = 0; ky < kernelH; ++ky) {
                            const int32_t iy = static_cast<int32_t>(oy * stride + ky) - static_cast<int32_t>(padding);
                            if (iy < 0 || iy >= static_cast<int32_t>(height)) continue;
                            for (uint32_t kx = 0; kx < kernelW; ++kx) {
                                const int32_t ix = static_cast<int32_t>(ox * stride + kx) - static_cast<int32_t>(padding);
                                if (ix < 0 || ix >= static_cast<int32_t>(width)) continue;
                                const size_t inputIndex = ((static_cast<size_t>(n) * inputChannels + ic) * height + static_cast<uint32_t>(iy)) * width + static_cast<uint32_t>(ix);
                                const size_t weightIndex = ((static_cast<size_t>(oc) * inputChannels + ic) * kernelH + ky) * kernelW + kx;
                                sum += input[inputIndex] * weight[weightIndex];
                            }
                        }
                    }
                    output[((static_cast<size_t>(n) * outputChannels + oc) * outH + oy) * outW + ox] = sum;
                }
            }
        }
    }
    return true;
}

bool ExecuteSP4ELinear(const LoadedSP4EAsset& asset,
                       const SP4ELayerDesc& layer,
                       const std::vector<float>& input,
                       uint32_t rows,
                       uint32_t inputFeatures,
                       std::vector<float>& output,
                       std::string* error) {
    if (layer.shape.size() != 2) {
        if (error) *error = "SP4E Linear requires a 2D weight layer";
        return false;
    }
    const uint32_t outputFeatures = layer.shape[0];
    const uint32_t weightInputFeatures = layer.shape[1];
    if (inputFeatures != weightInputFeatures) {
        if (error) *error = "SP4E Linear input feature mismatch";
        return false;
    }
    if (input.size() != static_cast<size_t>(rows) * inputFeatures) {
        if (error) *error = "SP4E Linear input size mismatch";
        return false;
    }
    const std::vector<float> weight = ReconstructSP4ELayer(asset, layer);
    output.assign(static_cast<size_t>(rows) * outputFeatures, 0.0f);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t outFeature = 0; outFeature < outputFeatures; ++outFeature) {
            float sum = layer.biasOffset + outFeature < asset.sections.biasFp16.size()
                ? HalfToFloat(asset.sections.biasFp16[layer.biasOffset + outFeature])
                : 0.0f;
            for (uint32_t inFeature = 0; inFeature < inputFeatures; ++inFeature) {
                sum += input[static_cast<size_t>(row) * inputFeatures + inFeature] *
                       weight[static_cast<size_t>(outFeature) * inputFeatures + inFeature];
            }
            output[static_cast<size_t>(row) * outputFeatures + outFeature] = sum;
        }
    }
    return true;
}

} // namespace sp4
