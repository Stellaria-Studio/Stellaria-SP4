#include "sp4_a1p_common.metal"

struct SP4ConvDispatch {
    uint width;
    uint height;
    uint layerIndex;
    uint activationZeroPoint;
};

kernel void sp4_conv2d_block32_w4a8(device const uchar* qweight [[buffer(0)]],
                                    device const half* scale [[buffer(1)]],
                                    device const uchar* mode [[buffer(2)]],
                                    device const uchar* aux [[buffer(3)]],
                                    device const int* residualIndex [[buffer(4)]],
                                    device const half* residualValue [[buffer(5)]],
                                    device const int2* residualTable [[buffer(6)]],
                                    device const SP4LayerPlan* layers [[buffer(7)]],
                                    device const char* activationInt8 [[buffer(8)]],
                                    device half* outputFp16 [[buffer(9)]],
                                    constant SP4ConvDispatch& dispatch [[buffer(10)]],
                                    uint3 tid [[thread_position_in_grid]]) {
    const SP4LayerPlan plan = layers[dispatch.layerIndex];
    const uint pixelBase = tid.x * 16u;
    if (pixelBase >= dispatch.width || tid.y >= dispatch.height || tid.z >= plan.outputChannels) {
        return;
    }

    const uint oc = tid.z;
    const uint kernelValues = plan.inputChannels * plan.kernelWidth * plan.kernelHeight;
    const uint weightBase = oc * kernelValues;
    float acc = 0.0f;

    for (uint k = 0; k < kernelValues;) {
        const uint valueIndex = weightBase + k;
        const uint blockId = valueIndex / 32u;
        const uint local = valueIndex - blockId * 32u;
        const uint blockValid = min(kernelValues - k, 32u - local);
        const uint modeValue = mode[plan.modeOffset + blockId];
        if (modeValue == 5u || modeValue == 6u) {
            for (uint i = 0; i < blockValid; ++i) {
                const uint code = sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i);
                const float codeValue = modeValue == 6u ? sp4_learned_codebook_value(code) : sp4_codebook_value(code);
                acc += codeValue *
                       float(scale[plan.scaleOffset + blockId]) *
                       float(activationInt8[k + i]);
            }
        } else if (modeValue == 1u) {
            int qsum = 0;
            int asum = 0;
            const int zp = int(aux[plan.auxOffset + blockId] & 0x0fu);
            for (uint i = 0; i < blockValid; ++i) {
                const int a = int(activationInt8[k + i]);
                qsum += int(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i)) * a;
                asum += a;
            }
            acc += float(qsum - zp * asum) * float(scale[plan.scaleOffset + blockId]);
        } else {
            int isum0 = 0;
            int isum1 = 0;
            uint i = 0;
            if (((valueIndex + i) & 1u) != 0u && i < blockValid) {
                isum0 += sp4_load_i4(qweight + plan.qweightOffsetBytes, valueIndex + i) * int(activationInt8[k + i]);
                ++i;
            }
            for (; i + 7u < blockValid; i += 8u) {
                const uint idx = valueIndex + i;
                const uchar b0 = qweight[plan.qweightOffsetBytes + (idx >> 1u)];
                const uchar b1 = qweight[plan.qweightOffsetBytes + ((idx + 2u) >> 1u)];
                const uchar b2 = qweight[plan.qweightOffsetBytes + ((idx + 4u) >> 1u)];
                const uchar b3 = qweight[plan.qweightOffsetBytes + ((idx + 6u) >> 1u)];
                isum0 += sp4_load_i4_from_byte(b0, false) * int(activationInt8[k + i + 0u]);
                isum1 += sp4_load_i4_from_byte(b0, true)  * int(activationInt8[k + i + 1u]);
                isum0 += sp4_load_i4_from_byte(b1, false) * int(activationInt8[k + i + 2u]);
                isum1 += sp4_load_i4_from_byte(b1, true)  * int(activationInt8[k + i + 3u]);
                isum0 += sp4_load_i4_from_byte(b2, false) * int(activationInt8[k + i + 4u]);
                isum1 += sp4_load_i4_from_byte(b2, true)  * int(activationInt8[k + i + 5u]);
                isum0 += sp4_load_i4_from_byte(b3, false) * int(activationInt8[k + i + 6u]);
                isum1 += sp4_load_i4_from_byte(b3, true)  * int(activationInt8[k + i + 7u]);
            }
            for (; i < blockValid; ++i) {
                isum0 += sp4_load_i4(qweight + plan.qweightOffsetBytes, valueIndex + i) * int(activationInt8[k + i]);
            }
            acc += float(isum0 + isum1) * float(scale[plan.scaleOffset + blockId]);
        }
        if (residualTable[plan.residualTableOffset + blockId].y > 0) {
            acc += sp4_sparse_residual_dot_range(residualIndex + plan.residualIndexOffset,
                                                 residualValue + plan.residualValueOffset,
                                                 residualTable + plan.residualTableOffset,
                                                 activationInt8,
                                                 blockId,
                                                 k,
                                                 local,
                                                 blockValid);
        }
        k += blockValid;
    }

    for (uint p = 0; p < 16u && pixelBase + p < dispatch.width; ++p) {
        const uint outIndex = (tid.y * dispatch.width + pixelBase + p) * plan.outputChannels + oc;
        outputFp16[outIndex] = half(acc);
    }
}

kernel void sp4_conv2d_block64_w4a8(device const uchar* qweight [[buffer(0)]],
                                    device const half* scale [[buffer(1)]],
                                    device const uchar* mode [[buffer(2)]],
                                    device const uchar* aux [[buffer(3)]],
                                    device const int* residualIndex [[buffer(4)]],
                                    device const half* residualValue [[buffer(5)]],
                                    device const int2* residualTable [[buffer(6)]],
                                    device const SP4LayerPlan* layers [[buffer(7)]],
                                    device const char* activationInt8 [[buffer(8)]],
                                    device half* outputFp16 [[buffer(9)]],
                                    constant SP4ConvDispatch& dispatch [[buffer(10)]],
                                    uint3 tid [[thread_position_in_grid]]) {
    const SP4LayerPlan plan = layers[dispatch.layerIndex];
    const uint pixelBase = tid.x * 16u;
    if (pixelBase >= dispatch.width || tid.y >= dispatch.height || tid.z >= plan.outputChannels) {
        return;
    }

    const uint oc = tid.z;
    const uint kernelValues = plan.inputChannels * plan.kernelWidth * plan.kernelHeight;
    const uint weightBase = oc * kernelValues;
    float acc = 0.0f;

    for (uint k = 0; k < kernelValues;) {
        const uint valueIndex = weightBase + k;
        const uint blockId = valueIndex / 64u;
        const uint local = valueIndex - blockId * 64u;
        const uint blockValid = min(kernelValues - k, 64u - local);
        const uint modeValue = mode[plan.modeOffset + blockId];
        if (modeValue == 5u || modeValue == 6u) {
            for (uint i = 0; i < blockValid; ++i) {
                const uint code = sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i);
                const float codeValue = modeValue == 6u ? sp4_learned_codebook_value(code) : sp4_codebook_value(code);
                acc += codeValue *
                       float(scale[plan.scaleOffset + blockId]) *
                       float(activationInt8[k + i]);
            }
        } else if (modeValue == 1u) {
            int qsum0 = 0;
            int qsum1 = 0;
            int asum = 0;
            const int zp = int(aux[plan.auxOffset + blockId] & 0x0fu);
            uint i = 0;
            for (; i + 3u < blockValid; i += 4u) {
                const int a0 = int(activationInt8[k + i + 0u]);
                const int a1 = int(activationInt8[k + i + 1u]);
                const int a2 = int(activationInt8[k + i + 2u]);
                const int a3 = int(activationInt8[k + i + 3u]);
                qsum0 += int(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i + 0u)) * a0;
                qsum1 += int(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i + 1u)) * a1;
                qsum0 += int(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i + 2u)) * a2;
                qsum1 += int(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i + 3u)) * a3;
                asum += a0 + a1 + a2 + a3;
            }
            for (; i < blockValid; ++i) {
                const int a = int(activationInt8[k + i]);
                qsum0 += int(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i)) * a;
                asum += a;
            }
            acc += float(qsum0 + qsum1 - zp * asum) * float(scale[plan.scaleOffset + blockId]);
        } else {
            int isum0 = 0;
            int isum1 = 0;
            int isum2 = 0;
            int isum3 = 0;
            uint i = 0;
            if (((valueIndex + i) & 1u) != 0u && i < blockValid) {
                isum0 += sp4_load_i4(qweight + plan.qweightOffsetBytes, valueIndex + i) * int(activationInt8[k + i]);
                ++i;
            }
            for (; i + 7u < blockValid; i += 8u) {
                const uint idx = valueIndex + i;
                const uchar b0 = qweight[plan.qweightOffsetBytes + (idx >> 1u)];
                const uchar b1 = qweight[plan.qweightOffsetBytes + ((idx + 2u) >> 1u)];
                const uchar b2 = qweight[plan.qweightOffsetBytes + ((idx + 4u) >> 1u)];
                const uchar b3 = qweight[plan.qweightOffsetBytes + ((idx + 6u) >> 1u)];
                isum0 += sp4_load_i4_from_byte(b0, false) * int(activationInt8[k + i + 0u]);
                isum1 += sp4_load_i4_from_byte(b0, true)  * int(activationInt8[k + i + 1u]);
                isum2 += sp4_load_i4_from_byte(b1, false) * int(activationInt8[k + i + 2u]);
                isum3 += sp4_load_i4_from_byte(b1, true)  * int(activationInt8[k + i + 3u]);
                isum0 += sp4_load_i4_from_byte(b2, false) * int(activationInt8[k + i + 4u]);
                isum1 += sp4_load_i4_from_byte(b2, true)  * int(activationInt8[k + i + 5u]);
                isum2 += sp4_load_i4_from_byte(b3, false) * int(activationInt8[k + i + 6u]);
                isum3 += sp4_load_i4_from_byte(b3, true)  * int(activationInt8[k + i + 7u]);
            }
            for (; i < blockValid; ++i) {
                isum0 += sp4_load_i4(qweight + plan.qweightOffsetBytes, valueIndex + i) * int(activationInt8[k + i]);
            }
            acc += float(isum0 + isum1 + isum2 + isum3) * float(scale[plan.scaleOffset + blockId]);
        }
        if (residualTable[plan.residualTableOffset + blockId].y > 0) {
            acc += sp4_sparse_residual_dot_range(residualIndex + plan.residualIndexOffset,
                                                 residualValue + plan.residualValueOffset,
                                                 residualTable + plan.residualTableOffset,
                                                 activationInt8,
                                                 blockId,
                                                 k,
                                                 local,
                                                 blockValid);
        }
        k += blockValid;
    }

    for (uint p = 0; p < 16u && pixelBase + p < dispatch.width; ++p) {
        const uint outIndex = (tid.y * dispatch.width + pixelBase + p) * plan.outputChannels + oc;
        outputFp16[outIndex] = half(acc);
    }
}
