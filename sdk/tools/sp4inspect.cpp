#include "sp4/SP4Runtime.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "usage: sp4inspect-cpp model.sp4\n";
        return argc == 1 ? 0 : 2;
    }
    sp4::LoadedAsset asset;
    std::string error;
    if (!sp4::LoadAsset(argv[1], asset, &error)) {
        std::cerr << "sp4inspect-cpp failed: " << error << "\n";
        return 1;
    }
    std::cout << "SP4 asset loaded: " << argv[1] << "\n";
    std::cout << "  qweight_bytes=" << asset.qweight.size()
              << " scale_count=" << asset.scaleFp16.size()
              << " mode_count=" << asset.mode.size()
              << " residual_entries=" << asset.residualIndex.size() << "\n";
    std::cout << "  adapter=" << asset.manifest.adapter
              << " version=" << asset.manifest.version
              << " layers=" << asset.manifest.layers.size() << "\n";
    if (!asset.manifest.layers.empty()) {
        const sp4::LayerDesc& first = asset.manifest.layers.front();
        std::cout << "  first_layer=" << first.name
                  << " blocks=" << first.numBlocks
                  << " block_size=" << first.blockSize
                  << " rmse=" << first.rmse << "\n";
    }
    return 0;
}
