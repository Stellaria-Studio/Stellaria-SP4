#pragma once

#include "sp4/SP4EFormat.h"

#include <string>
#include <vector>

namespace sp4 {

struct LoadedSP4EAsset {
    SP4EManifest manifest;
    SP4ESections sections;
};

bool WriteSP4EAsset(const std::string& path,
                    const SP4EManifest& manifest,
                    const SP4ESections& sections,
                    std::string* error = nullptr);
bool LoadSP4EAsset(const std::string& path, LoadedSP4EAsset& outAsset, std::string* error = nullptr);
std::vector<float> ReconstructSP4ELayer(const LoadedSP4EAsset& asset, const SP4ELayerDesc& layer);
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
                           std::string* error = nullptr);
bool ExecuteSP4ELinear(const LoadedSP4EAsset& asset,
                       const SP4ELayerDesc& layer,
                       const std::vector<float>& input,
                       uint32_t rows,
                       uint32_t inputFeatures,
                       std::vector<float>& output,
                       std::string* error = nullptr);

const char* SP4ERuntimePathName(SP4ERuntimePath path);
const char* SP4EModeName(uint8_t mode);

} // namespace sp4
