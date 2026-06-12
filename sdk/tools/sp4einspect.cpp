#include "sp4/SP4ERuntime.h"

#include <iostream>
#include <map>

namespace {

uint8_t ReadPackedNibble(const std::vector<uint8_t>& packed, uint32_t offsetBytes, uint32_t index) {
    const uint32_t byteOffset = offsetBytes + index / 2u;
    if (byteOffset >= packed.size()) return 0u;
    const uint8_t byte = packed[byteOffset];
    return (index & 1u) == 0u ? byte & 0x0fu : (byte >> 4u) & 0x0fu;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "usage: sp4einspect-cpp model.sp4e\n";
        return argc == 1 ? 0 : 2;
    }
    sp4::LoadedSP4EAsset asset;
    std::string error;
    if (!sp4::LoadSP4EAsset(argv[1], asset, &error)) {
        std::cerr << "sp4einspect-cpp failed: " << error << "\n";
        return 1;
    }
    std::map<uint8_t, uint64_t> modes;
    for (const sp4::SP4ELayerDesc& layer : asset.manifest.layers) {
        for (uint32_t block = 0; block < layer.numBlocks; ++block) {
            modes[ReadPackedNibble(asset.sections.mode, layer.modeOffset, block)]++;
        }
    }
    std::cout << "SP4E asset loaded: " << argv[1] << "\n";
    std::cout << "  adapter=" << asset.manifest.adapter
              << " version=" << asset.manifest.version
              << " layers=" << asset.manifest.layers.size() << "\n";
    std::cout << "  qweight_fp4_bytes=" << asset.sections.qweightFp4.size()
              << " qweight_int4_bytes=" << asset.sections.qweightInt4.size()
              << " scale_fp4_count=" << asset.sections.scaleFp16Fp4.size()
              << " scale_int4_count=" << asset.sections.scaleFp16Int4.size()
              << " residual_entries=" << asset.sections.residualIndex.size() << "\n";
    std::cout << "  blocks=" << asset.manifest.blocks
              << " fp4=" << asset.manifest.fp4Blocks
              << " int4=" << asset.manifest.int4Blocks
              << " bridge=" << asset.manifest.bridgeBlocks
              << " sp4_subset=" << asset.manifest.sp4SubsetBlocks
              << " fp16_fallback=" << asset.manifest.fp16FallbackBlocks << "\n";
    std::cout << "  runtime_gate=" << sp4::SP4ERuntimePathName(asset.manifest.runtimeDecision.path)
              << " enabled=" << (asset.manifest.runtimeDecision.enabled ? "true" : "false") << "\n";
    std::cout << "  modes:\n";
    for (const auto& [mode, count] : modes) {
        std::cout << "    " << sp4::SP4EModeName(mode) << ": " << count << "\n";
    }
    if (!asset.manifest.layers.empty()) {
        const sp4::SP4ELayerDesc& first = asset.manifest.layers.front();
        const std::vector<float> reconstructed = sp4::ReconstructSP4ELayer(asset, first);
        std::cout << "  first_layer=" << first.name
                  << " blocks=" << first.numBlocks
                  << " block_size=" << first.blockSize
                  << " rmse=" << first.rmse
                  << " reconstructed_values=" << reconstructed.size() << "\n";
        std::vector<float> opOutput;
        std::string error;
        if (first.shape.size() == 4) {
            std::vector<float> input(static_cast<size_t>(first.shape[1]) * 8u * 8u, 0.0f);
            for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i * 13u) % 17u) / 16.0f - 0.5f;
            if (sp4::ExecuteSP4EConv2DNCHW(asset, first, input, 1, first.shape[1], 8, 8, 1, 1, opOutput, &error)) {
                std::cout << "  cpu_operator=Conv2D_NCHW output_values=" << opOutput.size() << "\n";
            } else {
                std::cout << "  cpu_operator=Conv2D_NCHW unavailable: " << error << "\n";
            }
        } else if (first.shape.size() == 2) {
            std::vector<float> input(first.shape[1] * 2u, 0.0f);
            for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>((i * 7u) % 11u) / 10.0f - 0.5f;
            if (sp4::ExecuteSP4ELinear(asset, first, input, 2, first.shape[1], opOutput, &error)) {
                std::cout << "  cpu_operator=Linear output_values=" << opOutput.size() << "\n";
            } else {
                std::cout << "  cpu_operator=Linear unavailable: " << error << "\n";
            }
        }
    }
    return 0;
}
