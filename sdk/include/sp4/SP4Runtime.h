#pragma once

#include "sp4/SP4Format.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sp4 {

struct LoadedAsset {
    AssetManifest manifest;
    std::vector<uint8_t> qweight;
    std::vector<uint16_t> scaleFp16;
    std::vector<uint16_t> biasFp16;
    std::vector<uint8_t> mode;
    std::vector<uint8_t> aux;
    std::vector<int32_t> residualIndex;
    std::vector<uint16_t> residualValueFp16;
    std::vector<int32_t> residualTable;
    std::vector<uint16_t> activationScaleFp16;
};

struct PreparedLayerDesc {
    uint32_t sourceLayerIndex = 0;
    uint32_t weightOffset = 0;
    uint32_t weightCount = 0;
    uint32_t biasOffset = 0;
    uint32_t biasCount = 0;
    uint32_t inputChannels = 0;
    uint32_t outputChannels = 0;
    uint32_t kernelWidth = 1;
    uint32_t kernelHeight = 1;
    uint32_t flags = 0;
};

struct RuntimePrepareOptions {
    uint32_t maxLayers = 0;
    uint32_t maxWeightValues = 0;
    float targetFps = 60.0f;
    float frameBudgetMs = 16.67f;
    uint32_t multiTOutputCount = 1;
    uint32_t maxTilePixels = 0;
    bool includeBias = true;
    bool preferFp16WeightCache = true;
    bool preferPresentationReuse = true;
    bool preferCommandBufferBatch = true;
    bool preferBlendRefineFusion = true;
    bool preferTiledGraphExecutor = true;
};

struct PreparedRuntimeCache {
    std::vector<PreparedLayerDesc> layers;
    std::vector<uint16_t> weightFp16;
    std::vector<uint16_t> biasFp16;
    uint64_t sourceSp4Bytes = 0;
    uint64_t preparedBytes = 0;
    uint32_t recommendedMaxFlowHeight = 0;
    bool reuseFlowForMultiT = false;
    bool commandBufferBatch = false;
    bool blendRefineFusion = false;
    bool tiledGraphExecutor = false;
    std::string scheduleSummary;
};

bool LoadAsset(const std::string& path, LoadedAsset& outAsset, std::string* error = nullptr);
std::vector<float> ReconstructLayer(const LoadedAsset& asset, const LayerDesc& layer);
bool PrepareRuntimeCache(const LoadedAsset& asset,
                         const RuntimePrepareOptions& options,
                         PreparedRuntimeCache& outCache,
                         std::string* error = nullptr);

} // namespace sp4
