#include "sp4/SP4ECompiler.h"
#include "sp4/SP4ERuntime.h"

#include <algorithm>
#include <iostream>
#include <string>

namespace {

void PrintHelp() {
    std::cout << "usage: sp4ec compile --input model.safetensors --output model.sp4e "
                 "[--adapter rife] [--target cpu-reference] [--activation-precision fp16|bf16|fp8|fp4] "
                 "[--quality-profile balanced|quality|int8-chase|max-accuracy] "
                 "[--residual-ratio 0.02] [--fp16-fallback-ratio auto|0.08] "
                 "[--profile-speedup 0.0] [--include-nonruntime-layers]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "compile") {
        PrintHelp();
        return argc < 2 ? 0 : 2;
    }
    sp4::SP4ECompileOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value after " << arg << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--input") options.inputPath = next();
        else if (arg == "--output") options.outputPath = next();
        else if (arg == "--adapter" || arg == "--project") options.adapter = next();
        else if (arg == "--target") options.target = next();
        else if (arg == "--activation-precision") options.activationPrecision = next();
        else if (arg == "--quality-profile") options.qualityProfile = next();
        else if (arg == "--residual-ratio") options.residualRatio = std::stof(next());
        else if (arg == "--fp16-fallback-ratio") {
            const std::string value = next();
            options.fp16FallbackRatio = value == "auto" ? -1.0f : std::stof(value);
        }
        else if (arg == "--profile-speedup") options.profileSpeedup = std::stof(next());
        else if (arg == "--include-nonruntime-layers") options.includeNonRuntimeLayers = true;
        else if (arg == "--help" || arg == "-h") {
            PrintHelp();
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }
    if (options.inputPath.empty() || options.outputPath.empty()) {
        PrintHelp();
        return 2;
    }
    sp4::SP4ECompileResult result = sp4::CompileSafetensorsToSp4e(options);
    if (!result.ok) {
        std::cerr << "sp4ec failed: " << result.message << "\n";
        return 1;
    }
    const auto& m = result.manifest;
    const double compression = static_cast<double>(m.teacherWeightBytes) /
        static_cast<double>(std::max<uint64_t>(1, m.sp4eWeightBytes));
    std::cout << "wrote " << options.outputPath << "\n";
    std::cout << "layers=" << m.layers.size()
              << " excluded=" << m.excludedLayers.size()
              << " blocks=" << m.blocks
              << " fp4=" << m.fp4Blocks
              << " int4=" << m.int4Blocks
              << " bridge=" << m.bridgeBlocks
              << " sp4_subset=" << m.sp4SubsetBlocks
              << " fp16_fallback=" << m.fp16FallbackBlocks
              << " residual_entries=" << m.residualEntries << "\n";
    std::cout << "teacher_fp16_bytes=" << m.teacherWeightBytes
              << " sp4e_bytes=" << m.sp4eWeightBytes
              << " compression=" << compression << "x\n";
    std::cout << "runtime_gate=" << sp4::SP4ERuntimePathName(m.runtimeDecision.path)
              << " enabled=" << (m.runtimeDecision.enabled ? "true" : "false")
              << " reason=" << m.runtimeDecision.reason << "\n";
    return 0;
}
