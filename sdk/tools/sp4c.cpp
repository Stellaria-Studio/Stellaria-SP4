#include "sp4/SP4Compiler.h"

#include <iostream>
#include <string>

namespace {

void PrintHelp() {
    std::cout << "usage: sp4c compile --input model.safetensors --output model.sp4 [--adapter rife] [--target metal-m4pro] [--block-size auto|32|64] [--quality-profile balanced|adaptive-rife|max-accuracy|int8-chase] [--residual-ratio 0.02] [--include-nonruntime-layers] [--conv-only]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "compile") {
        PrintHelp();
        return argc < 2 ? 0 : 2;
    }
    sp4::CompileOptions options;
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
        else if (arg == "--block-size") options.blockSize = next();
        else if (arg == "--quality-profile") options.qualityProfile = next();
        else if (arg == "--residual-ratio") options.residualRatio = std::stof(next());
        else if (arg == "--include-nonruntime-layers") options.includeNonRuntimeLayers = true;
        else if (arg == "--conv-only") options.includeLinearLayers = false;
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
    sp4::CompileResult result = sp4::CompileSafetensorsToSp4(options);
    if (!result.ok) {
        std::cerr << "sp4c failed: " << result.message << "\n";
        return 1;
    }
    const auto& totals = result.manifest;
    const double compression = static_cast<double>(totals.teacherWeightBytes) /
        static_cast<double>(std::max<uint64_t>(1, totals.sp4WeightBytes));
    std::cout << "wrote " << options.outputPath << "\n";
    std::cout << "layers=" << totals.layers.size()
              << " excluded=" << totals.excludedLayers.size()
              << " blocks=" << totals.blocks
              << " residual_entries=" << totals.residualEntries << "\n";
    std::cout << "teacher_fp16_bytes=" << totals.teacherWeightBytes
              << " sp4_bytes=" << totals.sp4WeightBytes
              << " compression=" << compression << "x\n";
    return 0;
}
