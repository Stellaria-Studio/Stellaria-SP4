#include <metal_stdlib>
using namespace metal;

struct SP4LayerPlan {
    uint qweightOffsetBytes;
    uint scaleOffset;
    uint modeOffset;
    uint auxOffset;
    uint residualTableOffset;
    uint residualIndexOffset;
    uint residualValueOffset;
    uint blockSize;
    uint blockCount;
    uint inputChannels;
    uint outputChannels;
    uint kernelWidth;
    uint kernelHeight;
    uint activationScaleOffset;
    uint flags;
    uint reserved0;
    uint reserved1;
};

inline int sp4_sign_extend4(uint v) {
    v &= 0x0fu;
    return v >= 8u ? int(v) - 16 : int(v);
}

inline int sp4_load_i4(device const uchar* qweight, uint valueIndex) {
    const uchar packed = qweight[valueIndex >> 1u];
    return sp4_sign_extend4((valueIndex & 1u) == 0u ? packed : (packed >> 4u));
}

inline uint sp4_load_u4(device const uchar* qweight, uint valueIndex) {
    const uchar packed = qweight[valueIndex >> 1u];
    return ((valueIndex & 1u) == 0u ? packed : (packed >> 4u)) & 0x0fu;
}

inline int sp4_load_i4_from_byte(uchar packed, bool highNibble) {
    return sp4_sign_extend4(highNibble ? (packed >> 4u) : packed);
}

inline float sp4_codebook_value(uint code) {
    constexpr float codebook[16] = {
        -1.000000f, -0.696193f, -0.525073f, -0.394917f,
        -0.284441f, -0.184773f, -0.091050f,  0.000000f,
         0.079580f,  0.160930f,  0.246112f,  0.337915f,
         0.440710f,  0.562617f,  0.722956f,  1.000000f,
    };
    return codebook[code & 0x0fu];
}

inline float sp4_learned_codebook_value(uint code) {
    constexpr float codebook[16] = {
        -1.000000f, -0.770166f, -0.607256f, -0.472373f,
        -0.356107f, -0.252558f, -0.158382f, -0.070901f,
         0.012902f,  0.098159f,  0.189695f,  0.291085f,
         0.407779f,  0.548442f,  0.727477f,  1.000000f,
    };
    return codebook[code & 0x0fu];
}

inline half sp4_sparse_residual(device const int* residualIndex,
                                device const half* residualValue,
                                device const int2* residualTable,
                                uint blockId,
                                uint localIndex) {
    const int2 span = residualTable[blockId];
    for (int i = 0; i < span.y; ++i) {
        const uint entry = uint(span.x + i);
        if (uint(residualIndex[entry]) == localIndex) {
            return residualValue[entry];
        }
    }
    return half(0.0h);
}

inline float sp4_sparse_residual_dot_range(device const int* residualIndex,
                                           device const half* residualValue,
                                           device const int2* residualTable,
                                           device const char* activationInt8,
                                           uint blockId,
                                           uint blockBase,
                                           uint blockLocalStart,
                                           uint blockValid) {
    const int2 span = residualTable[blockId];
    float acc = 0.0f;
    for (int i = 0; i < span.y; ++i) {
        const uint entry = uint(span.x + i);
        const uint local = uint(residualIndex[entry]);
        if (local >= blockLocalStart && local < blockLocalStart + blockValid) {
            acc += float(residualValue[entry]) * float(activationInt8[blockBase + local - blockLocalStart]);
        }
    }
    return acc;
}

inline float sp4_sparse_residual_dot(device const int* residualIndex,
                                     device const half* residualValue,
                                     device const int2* residualTable,
                                     device const char* activationInt8,
                                     uint blockId,
                                     uint blockBase,
                                     uint blockValid) {
    const int2 span = residualTable[blockId];
    float acc = 0.0f;
    for (int i = 0; i < span.y; ++i) {
        const uint entry = uint(span.x + i);
        const uint local = uint(residualIndex[entry]);
        if (local < blockValid) {
            acc += float(residualValue[entry]) * float(activationInt8[blockBase + local]);
        }
    }
    return acc;
}
