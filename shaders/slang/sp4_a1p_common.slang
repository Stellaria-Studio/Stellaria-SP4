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

int sp4SignExtend4(uint v) {
    v &= 0x0f;
    return v >= 8 ? int(v) - 16 : int(v);
}

int sp4LoadI4(StructuredBuffer<uint> qweight32, uint byteOffset, uint valueIndex) {
    uint byteIndex = byteOffset + (valueIndex >> 1);
    uint word = qweight32[byteIndex >> 2];
    uint shift = (byteIndex & 3) * 8;
    uint packed = (word >> shift) & 0xff;
    return sp4SignExtend4(((valueIndex & 1) == 0) ? packed : (packed >> 4));
}

uint sp4LoadU4(StructuredBuffer<uint> qweight32, uint byteOffset, uint valueIndex) {
    uint byteIndex = byteOffset + (valueIndex >> 1);
    uint word = qweight32[byteIndex >> 2];
    uint shift = (byteIndex & 3) * 8;
    uint packed = (word >> shift) & 0xff;
    return (((valueIndex & 1) == 0) ? packed : (packed >> 4)) & 0x0f;
}

float sp4CodebookValue(uint code) {
    static const float codebook[16] = {
        -1.000000, -0.696193, -0.525073, -0.394917,
        -0.284441, -0.184773, -0.091050,  0.000000,
         0.079580,  0.160930,  0.246112,  0.337915,
         0.440710,  0.562617,  0.722956,  1.000000,
    };
    return codebook[code & 0x0f];
}

float sp4LearnedCodebookValue(uint code) {
    static const float codebook[16] = {
        -1.000000, -0.770166, -0.607256, -0.472373,
        -0.356107, -0.252558, -0.158382, -0.070901,
         0.012902,  0.098159,  0.189695,  0.291085,
         0.407779,  0.548442,  0.727477,  1.000000,
    };
    return codebook[code & 0x0f];
}

float sp4SparseResidual(StructuredBuffer<int> residualIndex,
                        StructuredBuffer<float16_t> residualValue,
                        StructuredBuffer<int2> residualTable,
                        uint residualIndexOffset,
                        uint residualValueOffset,
                        uint residualTableOffset,
                        uint blockId,
                        uint localIndex) {
    int2 span = residualTable[residualTableOffset + blockId];
    for (int i = 0; i < span.y; ++i) {
        uint entry = uint(span.x + i);
        if (uint(residualIndex[residualIndexOffset + entry]) == localIndex) {
            return float(residualValue[residualValueOffset + entry]);
        }
    }
    return 0.0;
}
