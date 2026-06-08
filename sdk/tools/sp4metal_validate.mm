#include "sp4/SP4Runtime.h"

#include "../src/SP4Common.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace {

struct SP4LayerPlanGPU {
    uint32_t qweightOffsetBytes = 0;
    uint32_t scaleOffset = 0;
    uint32_t modeOffset = 0;
    uint32_t auxOffset = 0;
    uint32_t residualTableOffset = 0;
    uint32_t residualIndexOffset = 0;
    uint32_t residualValueOffset = 0;
    uint32_t blockSize = 0;
    uint32_t blockCount = 0;
    uint32_t inputChannels = 0;
    uint32_t outputChannels = 0;
    uint32_t kernelWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t activationScaleOffset = 0;
    uint32_t flags = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

struct SP4ReconstructDispatch {
    uint32_t layerIndex = 0;
    uint32_t outputOffset = 0;
    uint32_t valueCount = 0;
    uint32_t reserved0 = 0;
};

struct SP4ConvDispatchGPU {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t layerIndex = 0;
    uint32_t activationZeroPoint = 0;
};

struct SP4LinearDispatchGPU {
    uint32_t tokens = 0;
    uint32_t layerIndex = 0;
    uint32_t activationStride = 0;
    uint32_t reserved0 = 0;
};

struct SP4SSMScanDispatchGPU {
    uint32_t tokens = 0;
    uint32_t channels = 0;
    uint32_t stateStride = 0;
    uint32_t reserved0 = 0;
};

struct SP4MlpFusedDispatchGPU {
    uint32_t tokens = 0;
    uint32_t upLayerIndex = 0;
    uint32_t gateLayerIndex = 0;
    uint32_t activationStride = 0;
};

struct ConvBenchResult {
    bool ok = false;
    std::string kernel;
    std::string layer;
    double averageMs = 0.0;
    double peakGops = 0.0;
    double averageGops = 0.0;
    double maxAbs = 0.0;
};

struct FormatBenchResult {
    std::string scheme;
    double averageMs = 0.0;
    double averageGops = 0.0;
    double peakGops = 0.0;
};

struct CompareDispatchGPU {
    uint32_t pixels = 0;
    uint32_t outputChannels = 0;
    uint32_t kernelValues = 0;
    uint32_t blockSize = 0;
};

std::string ReadText(const std::string& path) {
    std::ifstream file(path);
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

std::string StripLocalInclude(std::string text) {
    const std::string include = "#include \"sp4_a1p_common.metal\"";
    const size_t pos = text.find(include);
    if (pos != std::string::npos) {
        text.erase(pos, include.size());
    }
    return text;
}

id<MTLBuffer> Buffer(id<MTLDevice> device, const void* data, size_t bytes) {
    if (bytes == 0) {
        bytes = 4;
    }
    if (data == nullptr) {
        return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    }
    return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
}

template <typename T>
id<MTLBuffer> VectorBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    return Buffer(device, values.empty() ? nullptr : values.data(), values.size() * sizeof(T));
}

uint32_t OriginalValueCount(const sp4::LayerDesc& layer) {
    uint32_t count = 1;
    for (uint32_t dim : layer.shape) {
        count *= dim;
    }
    return count;
}

std::vector<SP4LayerPlanGPU> BuildPlans(const sp4::LoadedAsset& asset) {
    std::vector<SP4LayerPlanGPU> plans;
    plans.reserve(asset.manifest.layers.size());
    for (const sp4::LayerDesc& layer : asset.manifest.layers) {
        SP4LayerPlanGPU plan;
        plan.qweightOffsetBytes = layer.qweightOffsetBytes;
        plan.scaleOffset = layer.scaleOffset;
        plan.modeOffset = layer.modeOffset;
        plan.auxOffset = layer.auxOffset;
        plan.residualTableOffset = layer.residualTableOffset;
        plan.residualIndexOffset = layer.residualIndexOffset;
        plan.residualValueOffset = layer.residualValueOffset;
        plan.blockSize = layer.blockSize;
        plan.blockCount = layer.numBlocks;
        plan.outputChannels = layer.shape.size() > 0 ? layer.shape[0] : 0;
        plan.inputChannels = layer.shape.size() > 1 ? layer.shape[1] : 0;
        plan.kernelHeight = layer.shape.size() > 2 ? layer.shape[2] : 1;
        plan.kernelWidth = layer.shape.size() > 3 ? layer.shape[3] : 1;
        plan.activationScaleOffset = layer.activationScaleOffset;
        plans.push_back(plan);
    }
    return plans;
}

int FindLayerWithBlockSize(const sp4::LoadedAsset& asset, uint32_t blockSize) {
    for (size_t i = 0; i < asset.manifest.layers.size(); ++i) {
        if (asset.manifest.layers[i].blockSize == blockSize && asset.manifest.layers[i].shape.size() == 4) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<int8_t> SyntheticActivation(uint32_t count) {
    std::vector<int8_t> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        values[i] = static_cast<int8_t>((static_cast<int>(i * 13u + 7u) % 127) - 63);
    }
    return values;
}

std::vector<float> SyntheticWeights(uint32_t count) {
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>((i * 17u + 11u) % 257u) - 128;
        values[i] = static_cast<float>(centered) / 512.0f;
    }
    return values;
}

std::vector<uint8_t> PackInt4Values(const std::vector<int>& q) {
    std::vector<uint8_t> packed((q.size() + 1u) / 2u, 0u);
    for (size_t i = 0; i < q.size(); ++i) {
        const uint8_t nibble = static_cast<uint8_t>(q[i]) & 0x0fu;
        if ((i & 1u) == 0u) {
            packed[i / 2u] = nibble;
        } else {
            packed[i / 2u] |= static_cast<uint8_t>(nibble << 4u);
        }
    }
    return packed;
}

struct CompareBuffers {
    std::vector<uint16_t> fp16Weight;
    std::vector<uint16_t> fp16Activation;
    std::vector<int8_t> int8Activation;
    std::vector<int8_t> int8Weight;
    std::vector<uint16_t> int8Scale;
    std::vector<uint8_t> int4PerOutputWeight;
    std::vector<uint16_t> int4PerOutputScale;
    std::vector<uint8_t> nf4Weight;
    std::vector<uint16_t> nf4Scale;
    std::vector<uint8_t> w4a8Weight;
    std::vector<uint16_t> w4a8Scale;
};

CompareBuffers BuildCompareBuffers(uint32_t outputChannels, uint32_t kernelValues, uint32_t blockSize) {
    CompareBuffers buffers;
    const uint32_t valueCount = outputChannels * kernelValues;
    const std::vector<float> weights = SyntheticWeights(valueCount);
    buffers.int8Activation = SyntheticActivation(kernelValues);
    buffers.fp16Weight.reserve(valueCount);
    for (float value : weights) buffers.fp16Weight.push_back(sp4::FloatToHalf(value));
    for (int8_t value : buffers.int8Activation) buffers.fp16Activation.push_back(sp4::FloatToHalf(static_cast<float>(value)));

    buffers.int8Weight.resize(valueCount);
    buffers.int8Scale.resize(outputChannels);
    std::vector<int> int4PerOutput(valueCount);
    buffers.int4PerOutputScale.resize(outputChannels);
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        float maxAbs = 0.0f;
        for (uint32_t k = 0; k < kernelValues; ++k) maxAbs = std::max(maxAbs, std::fabs(weights[oc * kernelValues + k]));
        const float s8 = maxAbs > 1.0e-8f ? maxAbs / 127.0f : 1.0e-8f;
        const float s4 = maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f;
        buffers.int8Scale[oc] = sp4::FloatToHalf(s8);
        buffers.int4PerOutputScale[oc] = sp4::FloatToHalf(s4);
        for (uint32_t k = 0; k < kernelValues; ++k) {
            const uint32_t idx = oc * kernelValues + k;
            buffers.int8Weight[idx] = static_cast<int8_t>(std::max(-128, std::min(127, static_cast<int>(std::lrint(weights[idx] / s8)))));
            int4PerOutput[idx] = std::max(-8, std::min(7, static_cast<int>(std::lrint(weights[idx] / s4))));
        }
    }
    buffers.int4PerOutputWeight = PackInt4Values(int4PerOutput);

    const uint32_t blockCount = (valueCount + blockSize - 1u) / blockSize;
    std::vector<int> w4(valueCount, 0);
    std::vector<int> nf4(valueCount, 0);
    buffers.w4a8Scale.resize(blockCount);
    buffers.nf4Scale.resize(blockCount);
    static constexpr float codebook[16] = {
        -1.000000f, -0.696193f, -0.525073f, -0.394917f,
        -0.284441f, -0.184773f, -0.091050f,  0.000000f,
         0.079580f,  0.160930f,  0.246112f,  0.337915f,
         0.440710f,  0.562617f,  0.722956f,  1.000000f,
    };
    for (uint32_t block = 0; block < blockCount; ++block) {
        const uint32_t base = block * blockSize;
        const uint32_t valid = std::min(blockSize, valueCount - base);
        float maxAbs = 0.0f;
        for (uint32_t i = 0; i < valid; ++i) maxAbs = std::max(maxAbs, std::fabs(weights[base + i]));
        const float s4 = maxAbs > 1.0e-8f ? maxAbs / 7.0f : 1.0e-8f;
        const float snf = maxAbs > 1.0e-8f ? maxAbs : 1.0e-8f;
        buffers.w4a8Scale[block] = sp4::FloatToHalf(s4);
        buffers.nf4Scale[block] = sp4::FloatToHalf(snf);
        for (uint32_t i = 0; i < valid; ++i) {
            const uint32_t idx = base + i;
            w4[idx] = std::max(-8, std::min(7, static_cast<int>(std::lrint(weights[idx] / s4))));
            int best = 0;
            float bestErr = std::fabs(weights[idx] - codebook[0] * snf);
            for (int c = 1; c < 16; ++c) {
                const float err = std::fabs(weights[idx] - codebook[c] * snf);
                if (err < bestErr) {
                    best = c;
                    bestErr = err;
                }
            }
            nf4[idx] = best;
        }
    }
    buffers.w4a8Weight = PackInt4Values(w4);
    buffers.nf4Weight = PackInt4Values(nf4);
    return buffers;
}

std::vector<float> CpuDotOutputs(const sp4::LoadedAsset& asset,
                                 const sp4::LayerDesc& layer,
                                 const std::vector<int8_t>& activation) {
    const std::vector<float> weights = sp4::ReconstructLayer(asset, layer);
    const uint32_t outputChannels = layer.shape[0];
    const uint32_t kernelValues = layer.shape[1] * layer.shape[2] * layer.shape[3];
    std::vector<float> out(outputChannels, 0.0f);
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < kernelValues; ++k) {
            acc += weights[oc * kernelValues + k] * static_cast<float>(activation[k]);
        }
        out[oc] = acc;
    }
    return out;
}

ConvBenchResult RunConvBench(id<MTLDevice> device,
                             id<MTLCommandQueue> queue,
                             id<MTLComputePipelineState> pipeline,
                             const sp4::LoadedAsset& asset,
                             const std::vector<SP4LayerPlanGPU>& plans,
                             uint32_t layerIndex,
                             const char* kernelName) {
    ConvBenchResult result;
    result.kernel = kernelName;
    const sp4::LayerDesc& layer = asset.manifest.layers[layerIndex];
    result.layer = layer.name;
    const uint32_t outputChannels = layer.shape[0];
    const uint32_t kernelValues = layer.shape[1] * layer.shape[2] * layer.shape[3];
    constexpr uint32_t pixels = 4096;
    constexpr uint32_t warmup = 3;
    constexpr uint32_t iterations = 20;
    std::vector<int8_t> activation = SyntheticActivation(kernelValues);
    std::vector<uint16_t> output(static_cast<size_t>(pixels) * outputChannels, 0);

    id<MTLBuffer> qweight = VectorBuffer(device, asset.qweight);
    id<MTLBuffer> scale = VectorBuffer(device, asset.scaleFp16);
    id<MTLBuffer> mode = VectorBuffer(device, asset.mode);
    id<MTLBuffer> aux = VectorBuffer(device, asset.aux);
    id<MTLBuffer> residualIndex = VectorBuffer(device, asset.residualIndex);
    id<MTLBuffer> residualValue = VectorBuffer(device, asset.residualValueFp16);
    id<MTLBuffer> residualTable = VectorBuffer(device, asset.residualTable);
    id<MTLBuffer> layerPlans = VectorBuffer(device, plans);
    id<MTLBuffer> activationBuffer = VectorBuffer(device, activation);
    id<MTLBuffer> outputBuffer = VectorBuffer(device, output);
    SP4ConvDispatchGPU dispatch;
    dispatch.width = pixels;
    dispatch.height = 1;
    dispatch.layerIndex = layerIndex;

    std::vector<double> times;
    times.reserve(iterations);
    for (uint32_t iter = 0; iter < warmup + iterations; ++iter) {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:qweight offset:0 atIndex:0];
        [encoder setBuffer:scale offset:0 atIndex:1];
        [encoder setBuffer:mode offset:0 atIndex:2];
        [encoder setBuffer:aux offset:0 atIndex:3];
        [encoder setBuffer:residualIndex offset:0 atIndex:4];
        [encoder setBuffer:residualValue offset:0 atIndex:5];
        [encoder setBuffer:residualTable offset:0 atIndex:6];
        [encoder setBuffer:layerPlans offset:0 atIndex:7];
        [encoder setBuffer:activationBuffer offset:0 atIndex:8];
        [encoder setBuffer:outputBuffer offset:0 atIndex:9];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:10];
        const MTLSize grid = MTLSizeMake((pixels + 15u) / 16u, 1, outputChannels);
        const NSUInteger tgWidth = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
        const NSUInteger tgHeight = 1;
        const NSUInteger tgDepth = std::max<NSUInteger>(1, std::min<NSUInteger>(4, pipeline.maxTotalThreadsPerThreadgroup / tgWidth));
        const MTLSize group = MTLSizeMake(tgWidth, tgHeight, tgDepth);
        const CFTimeInterval start = CACurrentMediaTime();
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const double ms = (CACurrentMediaTime() - start) * 1000.0;
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return result;
        }
        if (iter >= warmup) {
            times.push_back(ms);
        }
    }

    const std::vector<float> cpu = CpuDotOutputs(asset, layer, activation);
    const uint16_t* gpu = static_cast<const uint16_t*>(outputBuffer.contents);
    double maxAbs = 0.0;
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        const float got = sp4::HalfToFloat(gpu[oc]);
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(got) - static_cast<double>(cpu[oc])));
    }
    const double avgMs = std::accumulate(times.begin(), times.end(), 0.0) / std::max<size_t>(1, times.size());
    const double minMs = *std::min_element(times.begin(), times.end());
    const double ops = 2.0 * static_cast<double>(pixels) * outputChannels * kernelValues;
    result.ok = maxAbs <= 0.25;
    result.averageMs = avgMs;
    result.averageGops = ops / (avgMs * 1.0e6);
    result.peakGops = ops / (minMs * 1.0e6);
    result.maxAbs = maxAbs;
    return result;
}

FormatBenchResult RunFormatBench(id<MTLDevice> device,
                                 id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> pipeline,
                                 const std::string& scheme,
                                 const std::vector<id<MTLBuffer>>& buffers,
                                 const CompareDispatchGPU& dispatch) {
    constexpr uint32_t warmup = 3;
    constexpr uint32_t iterations = 20;
    std::vector<double> times;
    times.reserve(iterations);
    for (uint32_t iter = 0; iter < warmup + iterations; ++iter) {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (NSUInteger i = 0; i < buffers.size(); ++i) {
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        }
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:buffers.size()];
        const MTLSize grid = MTLSizeMake(dispatch.pixels, 1, dispatch.outputChannels);
        const NSUInteger tgWidth = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
        const NSUInteger tgDepth = std::max<NSUInteger>(1, std::min<NSUInteger>(4, pipeline.maxTotalThreadsPerThreadgroup / tgWidth));
        const MTLSize group = MTLSizeMake(tgWidth, 1, tgDepth);
        const CFTimeInterval start = CACurrentMediaTime();
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return FormatBenchResult{scheme};
        }
        if (iter >= warmup) {
            times.push_back((CACurrentMediaTime() - start) * 1000.0);
        }
    }
    const double avgMs = std::accumulate(times.begin(), times.end(), 0.0) / std::max<size_t>(1, times.size());
    const double minMs = *std::min_element(times.begin(), times.end());
    const double ops = 2.0 * static_cast<double>(dispatch.pixels) * dispatch.outputChannels * dispatch.kernelValues;
    return FormatBenchResult{scheme, avgMs, ops / (avgMs * 1.0e6), ops / (minMs * 1.0e6)};
}

std::vector<float> CpuLinearFromPacked(const std::vector<uint8_t>& qweight,
                                       const std::vector<uint16_t>& scaleFp16,
                                       uint32_t outputChannels,
                                       uint32_t inputChannels,
                                       uint32_t blockSize,
                                       const std::vector<int8_t>& activation) {
    std::vector<float> out(outputChannels, 0.0f);
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < inputChannels;) {
            const uint32_t valueIndex = oc * inputChannels + k;
            const uint32_t blockId = valueIndex / blockSize;
            const uint32_t local = valueIndex - blockId * blockSize;
            const uint32_t valid = std::min(inputChannels - k, blockSize - local);
            int isum = 0;
            for (uint32_t i = 0; i < valid; ++i) {
                const uint32_t idx = valueIndex + i;
                const uint8_t packed = qweight[idx >> 1u];
                uint32_t nibble = ((idx & 1u) == 0u ? packed : (packed >> 4u)) & 0x0fu;
                const int q = nibble >= 8u ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
                isum += q * static_cast<int>(activation[k + i]);
            }
            acc += static_cast<float>(isum) * sp4::HalfToFloat(scaleFp16[blockId]);
            k += valid;
        }
        out[oc] = acc;
    }
    return out;
}

ConvBenchResult RunLinearLikeBench(id<MTLDevice> device,
                                   id<MTLCommandQueue> queue,
                                   id<MTLComputePipelineState> pipeline,
                                   const char* kernelName) {
    constexpr uint32_t outputChannels = 96;
    constexpr uint32_t inputChannels = 256;
    constexpr uint32_t blockSize = 64;
    constexpr uint32_t tokens = 4096;
    constexpr uint32_t warmup = 3;
    constexpr uint32_t iterations = 20;

    CompareBuffers compare = BuildCompareBuffers(outputChannels, inputChannels, blockSize);
    const uint32_t blockCount = (outputChannels * inputChannels + blockSize - 1u) / blockSize;
    std::vector<uint8_t> mode(blockCount, 0);
    std::vector<uint8_t> aux(blockCount, 0);
    std::vector<int32_t> residualIndex(1, 0);
    std::vector<uint16_t> residualValue(1, sp4::FloatToHalf(0.0f));
    std::vector<int32_t> residualTable(blockCount * 2u, 0);
    std::vector<SP4LayerPlanGPU> plans(1);
    plans[0].blockSize = blockSize;
    plans[0].blockCount = blockCount;
    plans[0].inputChannels = inputChannels;
    plans[0].outputChannels = outputChannels;
    plans[0].kernelWidth = 1;
    plans[0].kernelHeight = 1;

    std::vector<int8_t> activation = SyntheticActivation(inputChannels);
    std::vector<uint16_t> output(static_cast<size_t>(tokens) * outputChannels, 0);
    SP4LinearDispatchGPU dispatch;
    dispatch.tokens = tokens;
    dispatch.activationStride = inputChannels;

    id<MTLBuffer> qweight = VectorBuffer(device, compare.w4a8Weight);
    id<MTLBuffer> scale = VectorBuffer(device, compare.w4a8Scale);
    id<MTLBuffer> modeBuffer = VectorBuffer(device, mode);
    id<MTLBuffer> auxBuffer = VectorBuffer(device, aux);
    id<MTLBuffer> residualIndexBuffer = VectorBuffer(device, residualIndex);
    id<MTLBuffer> residualValueBuffer = VectorBuffer(device, residualValue);
    id<MTLBuffer> residualTableBuffer = VectorBuffer(device, residualTable);
    id<MTLBuffer> layerPlans = VectorBuffer(device, plans);
    id<MTLBuffer> activationBuffer = VectorBuffer(device, activation);
    id<MTLBuffer> outputBuffer = VectorBuffer(device, output);

    std::vector<double> times;
    times.reserve(iterations);
    for (uint32_t iter = 0; iter < warmup + iterations; ++iter) {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:qweight offset:0 atIndex:0];
        [encoder setBuffer:scale offset:0 atIndex:1];
        [encoder setBuffer:modeBuffer offset:0 atIndex:2];
        [encoder setBuffer:auxBuffer offset:0 atIndex:3];
        [encoder setBuffer:residualIndexBuffer offset:0 atIndex:4];
        [encoder setBuffer:residualValueBuffer offset:0 atIndex:5];
        [encoder setBuffer:residualTableBuffer offset:0 atIndex:6];
        [encoder setBuffer:layerPlans offset:0 atIndex:7];
        [encoder setBuffer:activationBuffer offset:0 atIndex:8];
        [encoder setBuffer:outputBuffer offset:0 atIndex:9];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:10];
        const MTLSize grid = MTLSizeMake((tokens + 15u) / 16u, outputChannels, 1);
        const NSUInteger tgWidth = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
        const NSUInteger tgHeight = std::max<NSUInteger>(1, std::min<NSUInteger>(4, pipeline.maxTotalThreadsPerThreadgroup / tgWidth));
        const MTLSize group = MTLSizeMake(tgWidth, tgHeight, 1);
        const CFTimeInterval start = CACurrentMediaTime();
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return ConvBenchResult{false, kernelName, "synthetic_linear"};
        }
        if (iter >= warmup) {
            times.push_back((CACurrentMediaTime() - start) * 1000.0);
        }
    }

    const std::vector<float> cpu = CpuLinearFromPacked(compare.w4a8Weight, compare.w4a8Scale, outputChannels, inputChannels, blockSize, activation);
    const uint16_t* gpu = static_cast<const uint16_t*>(outputBuffer.contents);
    double maxAbs = 0.0;
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(sp4::HalfToFloat(gpu[oc])) - static_cast<double>(cpu[oc])));
    }
    const double avgMs = std::accumulate(times.begin(), times.end(), 0.0) / std::max<size_t>(1, times.size());
    const double minMs = *std::min_element(times.begin(), times.end());
    const double ops = 2.0 * static_cast<double>(tokens) * outputChannels * inputChannels;
    ConvBenchResult result;
    result.ok = maxAbs <= 0.25;
    result.kernel = kernelName;
    result.layer = "synthetic_linear_96x256_tokens4096";
    result.averageMs = avgMs;
    result.averageGops = ops / (avgMs * 1.0e6);
    result.peakGops = ops / (minMs * 1.0e6);
    result.maxAbs = maxAbs;
    return result;
}

ConvBenchResult RunMlpGateBench(id<MTLDevice> device,
                                id<MTLCommandQueue> queue,
                                id<MTLComputePipelineState> pipeline,
                                const char* kernelName,
                                bool fast4,
                                double tolerance = 0.01) {
    constexpr uint32_t tokens = 4096;
    constexpr uint32_t channels = 256;
    constexpr uint32_t warmup = 3;
    constexpr uint32_t iterations = 20;
    const uint32_t count = tokens * channels;
    std::vector<uint16_t> up(count);
    std::vector<uint16_t> gate(count);
    std::vector<uint16_t> output(count, 0);
    for (uint32_t i = 0; i < count; ++i) {
        up[i] = sp4::FloatToHalf(static_cast<float>((static_cast<int>(i * 7u) % 113) - 56) / 64.0f);
        gate[i] = sp4::FloatToHalf(static_cast<float>((static_cast<int>(i * 11u) % 97) - 48) / 48.0f);
    }
    SP4LinearDispatchGPU dispatch;
    dispatch.tokens = tokens;
    dispatch.activationStride = channels;
    id<MTLBuffer> upBuffer = VectorBuffer(device, up);
    id<MTLBuffer> gateBuffer = VectorBuffer(device, gate);
    id<MTLBuffer> outputBuffer = VectorBuffer(device, output);
    std::vector<double> times;
    for (uint32_t iter = 0; iter < warmup + iterations; ++iter) {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:upBuffer offset:0 atIndex:0];
        [encoder setBuffer:gateBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:3];
        const MTLSize grid = MTLSizeMake(fast4 ? (count + 3u) / 4u : count, 1, 1);
        const MTLSize group = MTLSizeMake(std::max<NSUInteger>(1, pipeline.threadExecutionWidth), 1, 1);
        const CFTimeInterval start = CACurrentMediaTime();
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return ConvBenchResult{false, kernelName, "synthetic_mlp_gate"};
        }
        if (iter >= warmup) times.push_back((CACurrentMediaTime() - start) * 1000.0);
    }
    const uint16_t* gpu = static_cast<const uint16_t*>(outputBuffer.contents);
    double maxAbs = 0.0;
    for (uint32_t i = 0; i < 4096; ++i) {
        const float g = sp4::HalfToFloat(gate[i]);
        const float ref = sp4::HalfToFloat(up[i]) * (g / (1.0f + std::exp(-g)));
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(sp4::HalfToFloat(gpu[i])) - ref));
    }
    const double avgMs = std::accumulate(times.begin(), times.end(), 0.0) / std::max<size_t>(1, times.size());
    const double minMs = *std::min_element(times.begin(), times.end());
    const double ops = 4.0 * static_cast<double>(count);
    ConvBenchResult result;
    result.ok = maxAbs <= tolerance;
    result.kernel = kernelName;
    result.layer = "synthetic_gate_silu";
    result.averageMs = avgMs;
    result.averageGops = ops / (avgMs * 1.0e6);
    result.peakGops = ops / (minMs * 1.0e6);
    result.maxAbs = maxAbs;
    return result;
}

ConvBenchResult RunMlpFusedProjectionGateBench(id<MTLDevice> device,
                                               id<MTLCommandQueue> queue,
                                               id<MTLComputePipelineState> pipeline) {
    constexpr uint32_t outputChannels = 96;
    constexpr uint32_t inputChannels = 256;
    constexpr uint32_t blockSize = 64;
    constexpr uint32_t tokens = 4096;
    constexpr uint32_t warmup = 3;
    constexpr uint32_t iterations = 20;

    CompareBuffers compare = BuildCompareBuffers(outputChannels, inputChannels, blockSize);
    const uint32_t valueCount = outputChannels * inputChannels;
    const uint32_t blockCount = (valueCount + blockSize - 1u) / blockSize;
    std::vector<uint8_t> qweight = compare.w4a8Weight;
    qweight.insert(qweight.end(), compare.w4a8Weight.begin(), compare.w4a8Weight.end());
    std::vector<uint16_t> scales = compare.w4a8Scale;
    scales.insert(scales.end(), compare.w4a8Scale.begin(), compare.w4a8Scale.end());
    std::vector<uint8_t> mode(blockCount * 2u, 0);
    std::vector<uint8_t> aux(blockCount * 2u, 0);
    std::vector<int32_t> residualIndex(1, 0);
    std::vector<uint16_t> residualValue(1, sp4::FloatToHalf(0.0f));
    std::vector<int32_t> residualTable(blockCount * 4u, 0);
    std::vector<SP4LayerPlanGPU> plans(2);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        plans[layer].qweightOffsetBytes = layer == 0 ? 0 : static_cast<uint32_t>(compare.w4a8Weight.size());
        plans[layer].scaleOffset = layer * blockCount;
        plans[layer].modeOffset = layer * blockCount;
        plans[layer].auxOffset = layer * blockCount;
        plans[layer].residualTableOffset = layer * blockCount;
        plans[layer].blockSize = blockSize;
        plans[layer].blockCount = blockCount;
        plans[layer].inputChannels = inputChannels;
        plans[layer].outputChannels = outputChannels;
        plans[layer].kernelWidth = 1;
        plans[layer].kernelHeight = 1;
    }

    std::vector<int8_t> activation = SyntheticActivation(inputChannels);
    std::vector<uint16_t> output(static_cast<size_t>(tokens) * outputChannels, 0);
    SP4MlpFusedDispatchGPU dispatch;
    dispatch.tokens = tokens;
    dispatch.gateLayerIndex = 1;
    dispatch.activationStride = inputChannels;

    id<MTLBuffer> qweightBuffer = VectorBuffer(device, qweight);
    id<MTLBuffer> scaleBuffer = VectorBuffer(device, scales);
    id<MTLBuffer> modeBuffer = VectorBuffer(device, mode);
    id<MTLBuffer> auxBuffer = VectorBuffer(device, aux);
    id<MTLBuffer> residualIndexBuffer = VectorBuffer(device, residualIndex);
    id<MTLBuffer> residualValueBuffer = VectorBuffer(device, residualValue);
    id<MTLBuffer> residualTableBuffer = VectorBuffer(device, residualTable);
    id<MTLBuffer> layerPlans = VectorBuffer(device, plans);
    id<MTLBuffer> activationBuffer = VectorBuffer(device, activation);
    id<MTLBuffer> outputBuffer = VectorBuffer(device, output);

    std::vector<double> times;
    times.reserve(iterations);
    for (uint32_t iter = 0; iter < warmup + iterations; ++iter) {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:qweightBuffer offset:0 atIndex:0];
        [encoder setBuffer:scaleBuffer offset:0 atIndex:1];
        [encoder setBuffer:modeBuffer offset:0 atIndex:2];
        [encoder setBuffer:auxBuffer offset:0 atIndex:3];
        [encoder setBuffer:residualIndexBuffer offset:0 atIndex:4];
        [encoder setBuffer:residualValueBuffer offset:0 atIndex:5];
        [encoder setBuffer:residualTableBuffer offset:0 atIndex:6];
        [encoder setBuffer:layerPlans offset:0 atIndex:7];
        [encoder setBuffer:activationBuffer offset:0 atIndex:8];
        [encoder setBuffer:outputBuffer offset:0 atIndex:9];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:10];
        const MTLSize grid = MTLSizeMake((tokens + 15u) / 16u, outputChannels, 1);
        const NSUInteger tgWidth = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
        const NSUInteger tgHeight = std::max<NSUInteger>(1, std::min<NSUInteger>(4, pipeline.maxTotalThreadsPerThreadgroup / tgWidth));
        const MTLSize group = MTLSizeMake(tgWidth, tgHeight, 1);
        const CFTimeInterval start = CACurrentMediaTime();
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return ConvBenchResult{false, "SP4MLP_FusedProjectionGateSiLU_W4A8", "synthetic_mlp_fused"};
        }
        if (iter >= warmup) {
            times.push_back((CACurrentMediaTime() - start) * 1000.0);
        }
    }

    const std::vector<float> cpu = CpuLinearFromPacked(compare.w4a8Weight, compare.w4a8Scale, outputChannels, inputChannels, blockSize, activation);
    const uint16_t* gpu = static_cast<const uint16_t*>(outputBuffer.contents);
    double maxAbs = 0.0;
    for (uint32_t oc = 0; oc < outputChannels; ++oc) {
        const float gate = cpu[oc];
        const float ref = cpu[oc] * (gate / (1.0f + std::exp(-gate)));
        const float refFp16 = sp4::HalfToFloat(sp4::FloatToHalf(ref));
        maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(sp4::HalfToFloat(gpu[oc])) - static_cast<double>(refFp16)));
    }
    const double avgMs = std::accumulate(times.begin(), times.end(), 0.0) / std::max<size_t>(1, times.size());
    const double minMs = *std::min_element(times.begin(), times.end());
    const double ops = 4.0 * static_cast<double>(tokens) * outputChannels * inputChannels;
    ConvBenchResult result;
    result.ok = maxAbs <= 0.5;
    result.kernel = "SP4MLP_FusedProjectionGateSiLU_W4A8";
    result.layer = "synthetic_mlp_fused_2x96x256_tokens4096";
    result.averageMs = avgMs;
    result.averageGops = ops / (avgMs * 1.0e6);
    result.peakGops = ops / (minMs * 1.0e6);
    result.maxAbs = maxAbs;
    return result;
}

ConvBenchResult RunSsmScanBench(id<MTLDevice> device,
                                id<MTLCommandQueue> queue,
                                id<MTLComputePipelineState> pipeline,
                                const char* kernelName,
                                uint32_t parallelWidth) {
    constexpr uint32_t tokens = 2048;
    constexpr uint32_t channels = 256;
    constexpr uint32_t warmup = 3;
    constexpr uint32_t iterations = 20;
    const uint32_t count = tokens * channels;
    std::vector<uint16_t> input(count);
    std::vector<uint16_t> decay(channels);
    std::vector<uint16_t> initialState(channels);
    std::vector<uint16_t> output(count, 0);
    for (uint32_t i = 0; i < count; ++i) input[i] = sp4::FloatToHalf(static_cast<float>((static_cast<int>(i * 5u) % 31) - 15) / 128.0f);
    for (uint32_t c = 0; c < channels; ++c) {
        decay[c] = sp4::FloatToHalf(0.86f + static_cast<float>(c % 17u) * 0.004f);
        initialState[c] = sp4::FloatToHalf(static_cast<float>(static_cast<int>(c % 9u) - 4) / 64.0f);
    }
    SP4SSMScanDispatchGPU dispatch;
    dispatch.tokens = tokens;
    dispatch.channels = channels;
    dispatch.stateStride = channels;
    id<MTLBuffer> inputBuffer = VectorBuffer(device, input);
    id<MTLBuffer> decayBuffer = VectorBuffer(device, decay);
    id<MTLBuffer> stateBuffer = VectorBuffer(device, initialState);
    id<MTLBuffer> outputBuffer = VectorBuffer(device, output);
    std::vector<double> times;
    for (uint32_t iter = 0; iter < warmup + iterations; ++iter) {
        std::memcpy(stateBuffer.contents, initialState.data(), initialState.size() * sizeof(uint16_t));
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:decayBuffer offset:0 atIndex:1];
        [encoder setBuffer:stateBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:4];
        const MTLSize grid = MTLSizeMake(channels, 1, 1);
        const MTLSize group = MTLSizeMake(parallelWidth ? parallelWidth : std::max<NSUInteger>(1, pipeline.threadExecutionWidth), 1, 1);
        const CFTimeInterval start = CACurrentMediaTime();
        if (parallelWidth) {
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
        } else {
            [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        }
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return ConvBenchResult{false, kernelName, "synthetic_ssm_scan"};
        }
        if (iter >= warmup) times.push_back((CACurrentMediaTime() - start) * 1000.0);
    }
    const uint16_t* gpu = static_cast<const uint16_t*>(outputBuffer.contents);
    double maxAbs = 0.0;
    for (uint32_t c = 0; c < 16; ++c) {
        float state = sp4::HalfToFloat(initialState[c]);
        const float a = sp4::HalfToFloat(decay[c]);
        for (uint32_t t = 0; t < 64; ++t) {
            const uint32_t idx = t * channels + c;
            state = std::fma(state, a, sp4::HalfToFloat(input[idx]));
            maxAbs = std::max(maxAbs, std::fabs(static_cast<double>(sp4::HalfToFloat(gpu[idx])) - state));
        }
    }
    const double avgMs = std::accumulate(times.begin(), times.end(), 0.0) / std::max<size_t>(1, times.size());
    const double minMs = *std::min_element(times.begin(), times.end());
    const double ops = 2.0 * static_cast<double>(count);
    ConvBenchResult result;
    result.ok = maxAbs <= 0.05;
    result.kernel = kernelName;
    result.layer = "synthetic_ssm_scan";
    result.averageMs = avgMs;
    result.averageGops = ops / (avgMs * 1.0e6);
    result.peakGops = ops / (minMs * 1.0e6);
    result.maxAbs = maxAbs;
    return result;
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        const std::string assetPath = argc > 1 ? argv[1] : "build/rife_cpp.sp4";
        sp4::LoadedAsset asset;
        std::string error;
        if (!sp4::LoadAsset(assetPath, asset, &error)) {
            std::cerr << "sp4metal-validate: " << error << "\n";
            return 1;
        }
        if (asset.manifest.layers.empty()) {
            std::cerr << "sp4metal-validate: asset has no parsed layers\n";
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::cerr << "sp4metal-validate: no Metal device\n";
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            std::cerr << "sp4metal-validate: failed to create command queue\n";
            return 1;
        }

        std::string source = ReadText("shaders/metal/sp4_a1p_common.metal") +
                             StripLocalInclude(ReadText("shaders/metal/sp4_reconstruct.metal"));
        NSError* nsError = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                      options:options
                                                        error:&nsError];
        if (library == nil) {
            std::cerr << "sp4metal-validate: Metal compile failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"sp4_reconstruct_a1p"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&nsError];
        if (pipeline == nil) {
            std::cerr << "sp4metal-validate: pipeline failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }

        std::string convSource = ReadText("shaders/metal/sp4_a1p_common.metal") +
                                 StripLocalInclude(ReadText("shaders/metal/sp4_block_w4a8.metal"));
        id<MTLLibrary> convLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:convSource.c_str()]
                                                          options:options
                                                            error:&nsError];
        if (convLibrary == nil) {
            std::cerr << "sp4metal-validate: Metal conv compile failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }
        id<MTLComputePipelineState> conv32Pipeline =
            [device newComputePipelineStateWithFunction:[convLibrary newFunctionWithName:@"sp4_conv2d_block32_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> conv64Pipeline =
            [device newComputePipelineStateWithFunction:[convLibrary newFunctionWithName:@"sp4_conv2d_block64_w4a8"]
                                                  error:&nsError];
        if (conv32Pipeline == nil || conv64Pipeline == nil) {
            std::cerr << "sp4metal-validate: Metal conv pipeline failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }

        std::string compareSource = ReadText("shaders/metal/sp4_compare_kernels.metal");
        id<MTLLibrary> compareLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:compareSource.c_str()]
                                                             options:options
                                                               error:&nsError];
        if (compareLibrary == nil) {
            std::cerr << "sp4metal-validate: Metal compare compile failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }
        auto pipelineFor = [&](NSString* name) -> id<MTLComputePipelineState> {
            return [device newComputePipelineStateWithFunction:[compareLibrary newFunctionWithName:name] error:&nsError];
        };
        id<MTLComputePipelineState> cmpFp16 = pipelineFor(@"cmp_conv_fp16");
        id<MTLComputePipelineState> cmpInt8 = pipelineFor(@"cmp_conv_int8");
        id<MTLComputePipelineState> cmpInt4 = pipelineFor(@"cmp_conv_int4_per_output");
        id<MTLComputePipelineState> cmpNf4 = pipelineFor(@"cmp_conv_nf4_like");
        id<MTLComputePipelineState> cmpW4A8 = pipelineFor(@"cmp_conv_w4a8_block");
        id<MTLComputePipelineState> cmpSp4Int32 = pipelineFor(@"cmp_conv_sp4_int32x8");
        if (cmpFp16 == nil || cmpInt8 == nil || cmpInt4 == nil || cmpNf4 == nil || cmpW4A8 == nil || cmpSp4Int32 == nil) {
            std::cerr << "sp4metal-validate: Metal compare pipeline failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }

        std::string mlpSsmSource = ReadText("shaders/metal/sp4_a1p_common.metal") +
                                   StripLocalInclude(ReadText("shaders/metal/sp4_mlp_ssm.metal"));
        id<MTLLibrary> mlpSsmLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:mlpSsmSource.c_str()]
                                                            options:options
                                                              error:&nsError];
        if (mlpSsmLibrary == nil) {
            std::cerr << "sp4metal-validate: Metal MLP/SSM compile failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }
        id<MTLComputePipelineState> linearPipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_linear_tile16_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> mlpGatePipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_mlp_gated_tile16_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> mlpGateFastPipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_mlp_gated_fast_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> mlpGateFast4Pipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_mlp_gated_fast4_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> mlpGateHardSilu4Pipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_mlp_gated_hardsilu4_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> mlpFusedProjectionGatePipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_mlp_fused_projection_gate_silu_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> ssmProjectionPipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_ssm_projection_w4a8"]
                                                  error:&nsError];
        id<MTLComputePipelineState> ssmScanPipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_ssm_scan_fp16"]
                                                  error:&nsError];
        id<MTLComputePipelineState> ssmScanParallelPipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_ssm_scan_parallel_fp16"]
                                                  error:&nsError];
        id<MTLComputePipelineState> ssmScanParallel512Pipeline =
            [device newComputePipelineStateWithFunction:[mlpSsmLibrary newFunctionWithName:@"sp4_ssm_scan_parallel512_fp16"]
                                                  error:&nsError];
        if (linearPipeline == nil || mlpGatePipeline == nil || mlpGateFastPipeline == nil ||
            mlpGateFast4Pipeline == nil || mlpGateHardSilu4Pipeline == nil || mlpFusedProjectionGatePipeline == nil ||
            ssmProjectionPipeline == nil || ssmScanPipeline == nil ||
            ssmScanParallelPipeline == nil || ssmScanParallel512Pipeline == nil) {
            std::cerr << "sp4metal-validate: Metal MLP/SSM pipeline failed: "
                      << (nsError.localizedDescription.UTF8String ?: "unknown") << "\n";
            return 1;
        }

        std::vector<SP4LayerPlanGPU> plans = BuildPlans(asset);
        uint32_t maxPaddedValues = 0;
        for (const auto& layer : asset.manifest.layers) {
            maxPaddedValues = std::max(maxPaddedValues, layer.paddedValues);
        }
        std::vector<float> zeroOutput(maxPaddedValues, 0.0f);

        id<MTLBuffer> qweight = VectorBuffer(device, asset.qweight);
        id<MTLBuffer> scale = VectorBuffer(device, asset.scaleFp16);
        id<MTLBuffer> mode = VectorBuffer(device, asset.mode);
        id<MTLBuffer> aux = VectorBuffer(device, asset.aux);
        id<MTLBuffer> residualIndex = VectorBuffer(device, asset.residualIndex);
        id<MTLBuffer> residualValue = VectorBuffer(device, asset.residualValueFp16);
        id<MTLBuffer> residualTable = VectorBuffer(device, asset.residualTable);
        id<MTLBuffer> layerPlans = VectorBuffer(device, plans);
        id<MTLBuffer> output = VectorBuffer(device, zeroOutput);

        double totalGpuMs = 0.0;
        double worstAbs = -1.0;
        double worstRmse = 0.0;
        std::string worstLayer;

        for (uint32_t layerIndex = 0; layerIndex < asset.manifest.layers.size(); ++layerIndex) {
            const sp4::LayerDesc& layer = asset.manifest.layers[layerIndex];
            SP4ReconstructDispatch dispatch;
            dispatch.layerIndex = layerIndex;
            dispatch.valueCount = layer.paddedValues;

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:qweight offset:0 atIndex:0];
            [encoder setBuffer:scale offset:0 atIndex:1];
            [encoder setBuffer:mode offset:0 atIndex:2];
            [encoder setBuffer:aux offset:0 atIndex:3];
            [encoder setBuffer:residualIndex offset:0 atIndex:4];
            [encoder setBuffer:residualValue offset:0 atIndex:5];
            [encoder setBuffer:residualTable offset:0 atIndex:6];
            [encoder setBuffer:layerPlans offset:0 atIndex:7];
            [encoder setBuffer:output offset:0 atIndex:8];
            [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:9];
            const NSUInteger threads = 256;
            const MTLSize grid = MTLSizeMake(layer.paddedValues, 1, 1);
            const MTLSize group = MTLSizeMake(threads, 1, 1);
            const CFTimeInterval start = CACurrentMediaTime();
            [encoder dispatchThreads:grid threadsPerThreadgroup:group];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            totalGpuMs += (CACurrentMediaTime() - start) * 1000.0;
            if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
                std::cerr << "sp4metal-validate: command failed on " << layer.name << "\n";
                return 1;
            }

            const float* gpu = static_cast<const float*>(output.contents);
            const std::vector<float> cpu = sp4::ReconstructLayer(asset, layer);
            const uint32_t count = OriginalValueCount(layer);
            double sq = 0.0;
            double maxAbs = 0.0;
            for (uint32_t i = 0; i < count; ++i) {
                const double err = static_cast<double>(gpu[i]) - static_cast<double>(cpu[i]);
                sq += err * err;
                maxAbs = std::max(maxAbs, std::fabs(err));
            }
            const double rmse = std::sqrt(sq / std::max<uint32_t>(1, count));
            if (maxAbs > worstAbs) {
                worstAbs = maxAbs;
                worstRmse = rmse;
                worstLayer = layer.name;
            }
        }

        const bool ok = worstAbs <= 1.0e-5;
        ConvBenchResult conv32;
        ConvBenchResult conv64;
        const int block32Layer = FindLayerWithBlockSize(asset, 32);
        const int block64Layer = FindLayerWithBlockSize(asset, 64);
        if (block32Layer >= 0) {
            conv32 = RunConvBench(device, queue, conv32Pipeline, asset, plans, static_cast<uint32_t>(block32Layer), "SP4Conv2D_W4A8_Block32");
        }
        if (block64Layer >= 0) {
            conv64 = RunConvBench(device, queue, conv64Pipeline, asset, plans, static_cast<uint32_t>(block64Layer), "SP4Conv2D_W4A8_Block64");
        }
        ConvBenchResult linearBench = RunLinearLikeBench(device, queue, linearPipeline, "SP4Linear_Tile16_W4A8");
        ConvBenchResult ssmProjectionBench = RunLinearLikeBench(device, queue, ssmProjectionPipeline, "SP4SSM_Projection_W4A8");
        ConvBenchResult mlpGateBench = RunMlpGateBench(device, queue, mlpGatePipeline, "SP4MLP_Gated_Tile16_W4A8", false);
        ConvBenchResult mlpGateFastBench = RunMlpGateBench(device, queue, mlpGateFastPipeline, "SP4MLP_Gated_Fast_W4A8", false);
        ConvBenchResult mlpGateFast4Bench = RunMlpGateBench(device, queue, mlpGateFast4Pipeline, "SP4MLP_Gated_Fast4_W4A8", true);
        ConvBenchResult mlpGateHardSilu4Bench = RunMlpGateBench(device, queue, mlpGateHardSilu4Pipeline, "SP4MLP_Gated_HardSiLU4_W4A8", true, 0.025);
        ConvBenchResult mlpFusedProjectionGateBench = RunMlpFusedProjectionGateBench(device, queue, mlpFusedProjectionGatePipeline);
        ConvBenchResult ssmScanBench = RunSsmScanBench(device, queue, ssmScanPipeline, "SP4SSM_Scan_FP16", 0);
        ConvBenchResult ssmScanParallelBench = RunSsmScanBench(device, queue, ssmScanParallelPipeline, "SP4SSM_Scan_Parallel256_FP16", 256);
        ConvBenchResult ssmScanParallel512Bench = RunSsmScanBench(device, queue, ssmScanParallel512Pipeline, "SP4SSM_Scan_Parallel512_FP16", 512);
        std::vector<FormatBenchResult> formatBenches;
        if (block64Layer >= 0) {
            const sp4::LayerDesc& benchLayer = asset.manifest.layers[block64Layer];
            CompareDispatchGPU compareDispatch;
            compareDispatch.pixels = 4096;
            compareDispatch.outputChannels = benchLayer.shape[0];
            compareDispatch.kernelValues = benchLayer.shape[1] * benchLayer.shape[2] * benchLayer.shape[3];
            compareDispatch.blockSize = benchLayer.blockSize;
            CompareBuffers compare = BuildCompareBuffers(compareDispatch.outputChannels, compareDispatch.kernelValues, compareDispatch.blockSize);
            std::vector<uint16_t> output(static_cast<size_t>(compareDispatch.pixels) * compareDispatch.outputChannels, 0);
            id<MTLBuffer> outputBuffer = VectorBuffer(device, output);
            formatBenches.push_back(RunFormatBench(device, queue, cmpFp16, "FP16", {
                VectorBuffer(device, compare.fp16Weight), VectorBuffer(device, compare.fp16Activation), outputBuffer
            }, compareDispatch));
            formatBenches.push_back(RunFormatBench(device, queue, cmpInt8, "INT8 per-output", {
                VectorBuffer(device, compare.int8Weight), VectorBuffer(device, compare.int8Scale), VectorBuffer(device, compare.int8Activation), outputBuffer
            }, compareDispatch));
            formatBenches.push_back(RunFormatBench(device, queue, cmpInt4, "INT4 per-output", {
                VectorBuffer(device, compare.int4PerOutputWeight), VectorBuffer(device, compare.int4PerOutputScale), VectorBuffer(device, compare.int8Activation), outputBuffer
            }, compareDispatch));
            formatBenches.push_back(RunFormatBench(device, queue, cmpNf4, "NF4-like", {
                VectorBuffer(device, compare.nf4Weight), VectorBuffer(device, compare.nf4Scale), VectorBuffer(device, compare.int8Activation), outputBuffer
            }, compareDispatch));
            formatBenches.push_back(RunFormatBench(device, queue, cmpW4A8, "W4A8 block", {
                VectorBuffer(device, compare.w4a8Weight), VectorBuffer(device, compare.w4a8Scale), VectorBuffer(device, compare.int8Activation), outputBuffer
            }, compareDispatch));
            formatBenches.push_back(RunFormatBench(device, queue, cmpSp4Int32, "SP4 INT32x8", {
                VectorBuffer(device, compare.w4a8Weight), VectorBuffer(device, compare.w4a8Scale), VectorBuffer(device, compare.int8Activation), outputBuffer
            }, compareDispatch));
        }
        const bool convOk = (!conv32.layer.empty() && conv32.ok) && (!conv64.layer.empty() && conv64.ok);
        std::cout << "SP4 Metal backend validate: " << (ok ? "ok" : "failed") << "\n";
        std::cout << "  device=" << device.name.UTF8String << "\n";
        std::cout << "  layers=" << asset.manifest.layers.size()
                  << " gpu_reconstruct_ms=" << totalGpuMs << "\n";
        std::cout << "  worst_layer=" << worstLayer
                  << " worst_abs=" << worstAbs
                  << " worst_rmse=" << worstRmse << "\n";
        auto printConv = [](const ConvBenchResult& r) {
            std::cout << "  conv_kernel=" << r.kernel
                      << " layer=" << r.layer
                      << " status=" << (r.ok ? "ok" : "failed")
                      << " avg_ms=" << r.averageMs
                      << " avg_gops=" << r.averageGops
                      << " peak_gops=" << r.peakGops
                      << " max_abs=" << r.maxAbs << "\n";
        };
        printConv(conv32);
        printConv(conv64);
        auto printBackendKernel = [](const ConvBenchResult& r) {
            std::cout << "  backend_kernel=" << r.kernel
                      << " layer=" << r.layer
                      << " status=" << (r.ok ? "ok" : "failed")
                      << " avg_ms=" << r.averageMs
                      << " avg_gops=" << r.averageGops
                      << " peak_gops=" << r.peakGops
                      << " max_abs=" << r.maxAbs << "\n";
        };
        printBackendKernel(linearBench);
        printBackendKernel(ssmProjectionBench);
        printBackendKernel(mlpGateBench);
        printBackendKernel(mlpGateFastBench);
        printBackendKernel(mlpGateFast4Bench);
        printBackendKernel(mlpGateHardSilu4Bench);
        printBackendKernel(mlpFusedProjectionGateBench);
        printBackendKernel(ssmScanBench);
        printBackendKernel(ssmScanParallelBench);
        printBackendKernel(ssmScanParallel512Bench);
        for (const FormatBenchResult& r : formatBenches) {
            std::cout << "  throughput_scheme=" << r.scheme
                      << " avg_ms=" << r.averageMs
                      << " avg_gops=" << r.averageGops
                      << " peak_gops=" << r.peakGops << "\n";
        }
        const bool specializedOk = linearBench.ok && ssmProjectionBench.ok && mlpGateBench.ok &&
            mlpGateFastBench.ok && mlpGateFast4Bench.ok && mlpGateHardSilu4Bench.ok && mlpFusedProjectionGateBench.ok && ssmScanBench.ok &&
            ssmScanParallelBench.ok && ssmScanParallel512Bench.ok;
        return (ok && convOk && specializedOk) ? 0 : 1;
    }
}
