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

bool LoadAsset(const std::string& path, LoadedAsset& outAsset, std::string* error = nullptr);
std::vector<float> ReconstructLayer(const LoadedAsset& asset, const LayerDesc& layer);

} // namespace sp4
