#pragma once

#include "sp4/SP4Format.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sp4 {

struct AssetSections {
    std::vector<uint8_t> qweight;
    std::vector<uint16_t> scaleFp16;
    std::vector<uint16_t> biasFp16;
    std::vector<uint8_t> mode;
    std::vector<uint8_t> aux;
    std::vector<uint8_t> residualIndex;
    std::vector<uint16_t> residualValueFp16;
    std::vector<int32_t> residualTable;
    std::vector<uint16_t> activationScaleFp16;
};

bool WriteAsset(const std::string& path,
                const AssetManifest& manifest,
                const AssetSections& sections,
                std::string* error = nullptr);

std::string ManifestToJson(const AssetManifest& manifest,
                           const std::vector<SectionDesc>& sections);

} // namespace sp4
