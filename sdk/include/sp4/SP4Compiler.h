#pragma once

#include "sp4/SP4Format.h"

#include <string>

namespace sp4 {

struct CompileOptions {
    std::string inputPath;
    std::string outputPath;
    std::string adapter = "generic";
    std::string target = "cpu-reference";
    std::string blockSize = "auto";
    std::string qualityProfile = "balanced";
    float residualRatio = 0.02f;
    bool includeNonRuntimeLayers = false;
    bool includeLinearLayers = true;
};

struct CompileResult {
    bool ok = false;
    std::string message;
    AssetManifest manifest;
};

CompileResult CompileSafetensorsToSp4(const CompileOptions& options);

} // namespace sp4
