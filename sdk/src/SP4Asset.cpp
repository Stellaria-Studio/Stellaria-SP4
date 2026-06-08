#include "sp4/SP4Asset.h"

#include "SP4Common.h"

#include <fstream>
#include <sstream>

namespace sp4 {

namespace {

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

} // namespace

std::string ManifestToJson(const AssetManifest& manifest,
                           const std::vector<SectionDesc>& sections) {
    std::ostringstream out;
    out << "{";
    out << "\"adapter\":\"" << JsonEscape(manifest.adapter) << "\",";
    out << "\"architecture\":\"SP4block32/64(A1P)\",";
    out << "\"backend_plan\":{";
    out << "\"available\":[\"cpu_reference\",\"metal_alu_w4a8\",\"slang_portable_w4a8\"],";
    out << "\"canonical_ir\":[\"SP4OpConv2D\",\"SP4OpDepthwiseConv2D\",\"SP4OpPointwiseConv2D\",\"SP4OpLinear\",\"SP4OpElementwise\",\"SP4OpActivation\",\"SP4OpResidualCorrect\",\"SP4OpRequantize\",\"SP4OpFP16Fallback\",\"SP4OpWarpGridSample\",\"SP4OpBlend\",\"SP4OpExternal\"],";
    out << "\"fallback\":\"cpu_reference\",";
    out << "\"fallback_ops\":[\"warp_grid_sample_fp16\",\"blend_fp16\",\"resize_fp16\",\"external_texture_fp16\"],";
    out << "\"graph_runtime\":\"canonical_a1p_static_plan\",";
    out << "\"kernel_families\":[\"SP4Conv2D_W4A8_Block32\",\"SP4Conv2D_W4A8_Block64\",\"SP4PointwiseConv1x1_W4A8\",\"SP4Linear_Tile16_W4A8\",\"SP4MLP_Gated_Tile16_W4A8\",\"SP4MLP_FusedProjectionGateSiLU_W4A8\",\"SP4SSM_Projection_W4A8\",\"SP4SSM_Scan_FP16\",\"SP4SSM_Scan_Parallel_FP16\",\"SP4ResidualCorrection\",\"SP4Requantize\",\"SP4FP16Fallback\",\"SP4WarpGridSampleFP16\",\"SP4BlendFP16\",\"SP4ExternalOptimized\",\"SP4CustomBackend\"],";
    out << "\"fusion_patterns\":[\"Conv2D+Activation+Requantize\",\"Linear+Activation+Requantize\",\"Linear+Gate+SiLU\",\"SSMProjection+Scan\",\"Pointwise+Blend\"],";
    out << "\"mixed_precision\":[";
    out << "{\"ops\":[\"Conv2D\",\"PointwiseConv2D\",\"Linear\",\"MLPProjection\",\"SSMProjection\"],\"precision\":\"SP4_W4A8_or_ZeroPoint_or_AdaptiveProject_or_LearnedProject\",\"route\":\"metal_int32_tile16\"},";
    out << "{\"ops\":[\"WarpGridSample\",\"Blend\",\"Resize\"],\"precision\":\"FP16\",\"route\":\"platform_or_custom_kernel\"},";
    out << "{\"ops\":[\"External\"],\"precision\":\"external_optimized\",\"route\":\"backend_registered_kernel\"}";
    out << "],";
    out << "\"preferred\":\"metal_alu_w4a8\",";
    out << "\"target\":\"metal-m4pro\",";
    out << "\"tensor_layout\":\"SoA_sections_qweight_scale_mode_aux_residual\",";
    out << "\"dispatch\":\"fixed_backend_plan_from_asset\",";
    out << "\"scheduling\":{\"small_kernel_batch\":\"batch_identical_by_batch_key\",\"different_kernel_policy\":\"out_of_order_dependency_mask\",\"max_ooo_depth\":8,\"async_fp_correction\":\"late_epilogue_or_side_queue\"}";
    out << "},";
    out << "\"compiler\":{\"name\":\"sp4c-cpp\",\"version\":\"0.1.0\",\"auto_fusion\":true,\"sp4_modes\":[\"LinearW4A8\",\"LinearW4A8ZeroPoint\",\"SparseResidual\",\"AdaptiveProject\",\"LearnedProject\"]},";
    out << "\"compile_ms\":" << manifest.compileMs << ",";
    out << "\"excluded_layers\":[";
    for (size_t i = 0; i < manifest.excludedLayers.size(); ++i) {
        if (i) out << ",";
        out << "{\"name\":\"" << JsonEscape(manifest.excludedLayers[i]) << "\",\"reason\":\"excluded_by_" << JsonEscape(manifest.adapter) << "_runtime_policy\"}";
    }
    out << "],";
    out << "\"format\":\"stellaria.sp4\",";
    out << "\"graph_tensor_count\":" << manifest.graphTensorCount << ",";
    out << "\"layers\":[";
    for (size_t i = 0; i < manifest.layers.size(); ++i) {
        const LayerDesc& layer = manifest.layers[i];
        if (i) out << ",";
        out << "{";
        out << "\"activation_scale_offset\":" << layer.activationScaleOffset << ",";
        out << "\"aux_offset\":" << layer.auxOffset << ",";
        out << "\"aux_policy\":" << layer.auxPolicy << ",";
        out << "\"bias_count\":" << layer.biasCount << ",";
        out << "\"bias_offset\":" << layer.biasOffset << ",";
        out << "\"block_abi\":\"SP4block" << layer.blockSize << "(A1P)\",";
        out << "\"block_size\":" << layer.blockSize << ",";
        out << "\"fallback_fp16\":" << (layer.fallbackFp16 ? "true" : "false") << ",";
        out << "\"metrics\":{\"mae\":" << layer.mae << ",\"max_abs\":" << layer.maxAbs << ",\"rmse\":" << layer.rmse << "},";
        out << "\"mode_offset\":" << layer.modeOffset << ",";
        out << "\"name\":\"" << JsonEscape(layer.name) << "\",";
        out << "\"num_blocks\":" << layer.numBlocks << ",";
        out << "\"padded_values\":" << layer.paddedValues << ",";
        out << "\"qweight_length_bytes\":" << layer.qweightLengthBytes << ",";
        out << "\"qweight_offset_bytes\":" << layer.qweightOffsetBytes << ",";
        out << "\"residual_entries\":" << layer.residualEntries << ",";
        out << "\"residual_index_offset\":" << layer.residualIndexOffset << ",";
        out << "\"residual_table_offset\":" << layer.residualTableOffset << ",";
        out << "\"residual_value_offset\":" << layer.residualValueOffset << ",";
        out << "\"scale_offset\":" << layer.scaleOffset << ",";
        out << "\"sensitive\":" << (layer.sensitive ? "true" : "false") << ",";
        out << "\"shape\":[";
        for (size_t s = 0; s < layer.shape.size(); ++s) {
            if (s) out << ",";
            out << layer.shape[s];
        }
        out << "]";
        out << "}";
    }
    out << "],";
    out << "\"precision\":{\"activation\":\"int8\",\"bias\":\"fp16\",\"residual\":\"sparse_fp16\",\"scale\":\"fp16\",\"weight\":\"int4_sp4_or_zero_point_or_codebook\"},";
    out << "\"residual_index_encoding\":\"u8_local_index_v1\",";
    out << "\"residual_table_encoding\":\"sparse_triplet_v1\",";
    out << "\"sections\":{";
    for (size_t i = 0; i < sections.size(); ++i) {
        if (i) out << ",";
        out << "\"" << sections[i].name << "\":{\"length\":" << sections[i].length << ",\"offset\":" << sections[i].offset << "}";
    }
    out << "},";
    out << "\"source\":\"" << JsonEscape(manifest.source) << "\",";
    out << "\"totals\":{\"blocks\":" << manifest.blocks
        << ",\"residual_entries\":" << manifest.residualEntries
        << ",\"sp4_weight_bytes\":" << manifest.sp4WeightBytes
        << ",\"teacher_weight_bytes\":" << manifest.teacherWeightBytes << "},";
    out << "\"version\":\"" << manifest.version << "\"";
    out << "}";
    return out.str();
}

bool WriteAsset(const std::string& path,
                const AssetManifest& manifest,
                const AssetSections& sections,
                std::string* error) {
    std::vector<uint8_t> payload;
    std::vector<SectionDesc> descs;
    AddSection(payload, descs, "qweight", sections.qweight);
    AddTypedSection(payload, descs, "scale", sections.scaleFp16);
    AddTypedSection(payload, descs, "bias", sections.biasFp16);
    AddSection(payload, descs, "mode", sections.mode);
    AddSection(payload, descs, "aux", sections.aux);
    AddSection(payload, descs, "residual_index", sections.residualIndex);
    AddTypedSection(payload, descs, "residual_value", sections.residualValueFp16);
    AddTypedSection(payload, descs, "residual_table", sections.residualTable);
    AddTypedSection(payload, descs, "activation_scale", sections.activationScaleFp16);

    const std::string manifestJson = ManifestToJson(manifest, descs);
    std::vector<uint8_t> file;
    file.insert(file.end(), kMagicA1P, kMagicA1P + 8);
    AppendU32(file, kContainerVersion);
    AppendU32(file, static_cast<uint32_t>(manifestJson.size()));
    file.insert(file.end(), manifestJson.begin(), manifestJson.end());
    file.insert(file.end(), payload.begin(), payload.end());

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        if (error) *error = "failed to open output asset: " + path;
        return false;
    }
    out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    if (!out.good()) {
        if (error) *error = "failed to write output asset: " + path;
        return false;
    }
    return true;
}

} // namespace sp4
