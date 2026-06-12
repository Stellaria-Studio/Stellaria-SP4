#pragma once

#include "sp4/SP4EFormat.h"

#include <string>

namespace sp4 {

struct SP4ECompileOptions {
    std::string inputPath;
    std::string outputPath;
    std::string adapter = "generic";
    std::string target = "cpu-reference";
    std::string activationPrecision = "fp16";
    std::string qualityProfile = "balanced";
    float residualRatio = 0.02f;
    float fp16FallbackRatio = -1.0f;
    float profileSpeedup = 0.0f;
    bool includeNonRuntimeLayers = false;
};

struct SP4ECompileResult {
    bool ok = false;
    std::string message;
    SP4EManifest manifest;
};

SP4ERuntimeDecision DecideSP4ERuntimePath(const SP4ECapabilityProfile& profile);
SP4ECapabilityProfile LocalSP4ECapabilityProfile(const std::string& backend, float profileSpeedup = 0.0f);
SP4ECompileResult CompileSafetensorsToSp4e(const SP4ECompileOptions& options);

} // namespace sp4
