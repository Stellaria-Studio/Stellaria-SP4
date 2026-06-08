#include "sp4_a1p_common.metal"

struct SP4LinearDispatch {
    uint tokens;
    uint layerIndex;
    uint activationStride;
    uint reserved0;
};

struct SP4SSMScanDispatch {
    uint tokens;
    uint channels;
    uint stateStride;
    uint reserved0;
};

struct SP4MlpFusedDispatch {
    uint tokens;
    uint upLayerIndex;
    uint gateLayerIndex;
    uint activationStride;
};

inline float sp4_weight_value(device const uchar* qweight,
                              device const half* scale,
                              device const uchar* mode,
                              device const uchar* aux,
                              device const int* residualIndex,
                              device const half* residualValue,
                              device const int2* residualTable,
                              const thread SP4LayerPlan& plan,
                              uint valueIndex,
                              uint blockId,
                              uint localIndex) {
    const uint modeValue = mode[plan.modeOffset + blockId];
    const float blockScale = float(scale[plan.scaleOffset + blockId]);
    float value = 0.0f;
    if (modeValue == 5u) {
        value = sp4_codebook_value(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex)) * blockScale;
    } else if (modeValue == 6u) {
        value = sp4_learned_codebook_value(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex)) * blockScale;
    } else if (modeValue == 1u) {
        const uint zp = uint(aux[plan.auxOffset + blockId] & 0x0fu);
        value = (float(sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex)) - float(zp)) * blockScale;
    } else {
        value = float(sp4_load_i4(qweight + plan.qweightOffsetBytes, valueIndex)) * blockScale;
    }
    if (residualTable[plan.residualTableOffset + blockId].y > 0) {
        value += float(sp4_sparse_residual(residualIndex + plan.residualIndexOffset,
                                           residualValue + plan.residualValueOffset,
                                           residualTable + plan.residualTableOffset,
                                           blockId,
                                           localIndex));
    }
    return value;
}

inline float sp4_dot_int8_w4a8(device const uchar* qweight,
                               device const half* scale,
                               device const uchar* mode,
                               device const uchar* aux,
                               device const int* residualIndex,
                               device const half* residualValue,
                               device const int2* residualTable,
                               device const char* activationInt8,
                               const thread SP4LayerPlan& plan,
                               uint oc) {
    const uint inputChannels = plan.inputChannels;
    const uint weightBase = oc * inputChannels;
    float acc = 0.0f;
    for (uint k = 0; k < inputChannels;) {
        const uint valueIndex = weightBase + k;
        const uint blockId = valueIndex / plan.blockSize;
        const uint local = valueIndex - blockId * plan.blockSize;
        const uint blockValid = min(inputChannels - k, plan.blockSize - local);
        const uint modeValue = mode[plan.modeOffset + blockId];
        if (modeValue == 5u || modeValue == 6u) {
            for (uint i = 0; i < blockValid; ++i) {
                const uint code = sp4_load_u4(qweight + plan.qweightOffsetBytes, valueIndex + i);
                const float codeValue = modeValue == 6u ? sp4_learned_codebook_value(code) : sp4_codebook_value(code);
                acc += codeValue * float(scale[plan.scaleOffset + blockId]) * float(activationInt8[k + i]);
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
    return acc;
}

kernel void sp4_linear_tile16_w4a8(device const uchar* qweight [[buffer(0)]],
                                   device const half* scale [[buffer(1)]],
                                   device const uchar* mode [[buffer(2)]],
                                   device const uchar* aux [[buffer(3)]],
                                   device const int* residualIndex [[buffer(4)]],
                                   device const half* residualValue [[buffer(5)]],
                                   device const int2* residualTable [[buffer(6)]],
                                   device const SP4LayerPlan* layers [[buffer(7)]],
                                   device const char* activationInt8 [[buffer(8)]],
                                   device half* outputFp16 [[buffer(9)]],
                                   constant SP4LinearDispatch& dispatch [[buffer(10)]],
                                   uint3 tid [[thread_position_in_grid]]) {
    const SP4LayerPlan plan = layers[dispatch.layerIndex];
    const uint tokenBase = tid.x * 16u;
    const uint oc = tid.y;
    if (tokenBase >= dispatch.tokens || oc >= plan.outputChannels) return;

    const uint inputChannels = plan.inputChannels;
    const uint weightBase = oc * inputChannels;
    float acc = 0.0f;
    for (uint k = 0; k < inputChannels;) {
        const uint valueIndex = weightBase + k;
        const uint blockId = valueIndex / plan.blockSize;
        const uint local = valueIndex - blockId * plan.blockSize;
        const uint blockValid = min(inputChannels - k, plan.blockSize - local);
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

    for (uint t = 0; t < 16u && tokenBase + t < dispatch.tokens; ++t) {
        outputFp16[(tokenBase + t) * plan.outputChannels + oc] = half(acc);
    }
}

kernel void sp4_mlp_gated_tile16_w4a8(device const half* up [[buffer(0)]],
                                      device const half* gate [[buffer(1)]],
                                      device half* output [[buffer(2)]],
                                      constant SP4LinearDispatch& dispatch [[buffer(3)]],
                                      uint tid [[thread_position_in_grid]]) {
    const uint count = dispatch.tokens * dispatch.activationStride;
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float silu = g / (1.0f + exp(-g));
    output[tid] = half(float(up[tid]) * silu);
}

kernel void sp4_mlp_gated_fast_w4a8(device const half* up [[buffer(0)]],
                                    device const half* gate [[buffer(1)]],
                                    device half* output [[buffer(2)]],
                                    constant SP4LinearDispatch& dispatch [[buffer(3)]],
                                    uint tid [[thread_position_in_grid]]) {
    const uint count = dispatch.tokens * dispatch.activationStride;
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float silu = g / (1.0f + fast::exp(-g));
    output[tid] = half(float(up[tid]) * silu);
}

kernel void sp4_mlp_gated_fast4_w4a8(device const half4* up [[buffer(0)]],
                                     device const half4* gate [[buffer(1)]],
                                     device half4* output [[buffer(2)]],
                                     constant SP4LinearDispatch& dispatch [[buffer(3)]],
                                     uint tid [[thread_position_in_grid]]) {
    const uint count4 = (dispatch.tokens * dispatch.activationStride + 3u) / 4u;
    if (tid >= count4) return;
    const float4 u = float4(up[tid]);
    const float4 g = float4(gate[tid]);
    const float4 sigmoid = 1.0f / (1.0f + fast::exp(-g));
    output[tid] = half4(u * g * sigmoid);
}

kernel void sp4_mlp_gated_hardsilu4_w4a8(device const half4* up [[buffer(0)]],
                                         device const half4* gate [[buffer(1)]],
                                         device half4* output [[buffer(2)]],
                                         constant SP4LinearDispatch& dispatch [[buffer(3)]],
                                         uint tid [[thread_position_in_grid]]) {
    const uint count4 = (dispatch.tokens * dispatch.activationStride + 3u) / 4u;
    if (tid >= count4) return;
    const float4 u = float4(up[tid]);
    const float4 g = float4(gate[tid]);
    const float4 hardSigmoid = clamp(fma(g, 0.25f, 0.5f), 0.0f, 1.0f);
    output[tid] = half4(u * g * hardSigmoid);
}

kernel void sp4_mlp_fused_projection_gate_silu_w4a8(device const uchar* qweight [[buffer(0)]],
                                                    device const half* scale [[buffer(1)]],
                                                    device const uchar* mode [[buffer(2)]],
                                                    device const uchar* aux [[buffer(3)]],
                                                    device const int* residualIndex [[buffer(4)]],
                                                    device const half* residualValue [[buffer(5)]],
                                                    device const int2* residualTable [[buffer(6)]],
                                                    device const SP4LayerPlan* layers [[buffer(7)]],
                                                    device const char* activationInt8 [[buffer(8)]],
                                                    device half* outputFp16 [[buffer(9)]],
                                                    constant SP4MlpFusedDispatch& dispatch [[buffer(10)]],
                                                    uint3 tid [[thread_position_in_grid]]) {
    const SP4LayerPlan upPlan = layers[dispatch.upLayerIndex];
    const SP4LayerPlan gatePlan = layers[dispatch.gateLayerIndex];
    const uint tokenBase = tid.x * 16u;
    const uint oc = tid.y;
    if (tokenBase >= dispatch.tokens || oc >= upPlan.outputChannels || oc >= gatePlan.outputChannels) return;

    const float up = sp4_dot_int8_w4a8(qweight, scale, mode, aux, residualIndex, residualValue, residualTable,
                                       activationInt8, upPlan, oc);
    const float gate = sp4_dot_int8_w4a8(qweight, scale, mode, aux, residualIndex, residualValue, residualTable,
                                         activationInt8, gatePlan, oc);
    const float silu = gate / (1.0f + fast::exp(-gate));
    const half fused = half(up * silu);
    for (uint t = 0; t < 16u && tokenBase + t < dispatch.tokens; ++t) {
        outputFp16[(tokenBase + t) * upPlan.outputChannels + oc] = fused;
    }
}

kernel void sp4_ssm_projection_w4a8(device const uchar* qweight [[buffer(0)]],
                                    device const half* scale [[buffer(1)]],
                                    device const uchar* mode [[buffer(2)]],
                                    device const uchar* aux [[buffer(3)]],
                                    device const int* residualIndex [[buffer(4)]],
                                    device const half* residualValue [[buffer(5)]],
                                    device const int2* residualTable [[buffer(6)]],
                                    device const SP4LayerPlan* layers [[buffer(7)]],
                                    device const char* activationInt8 [[buffer(8)]],
                                    device half* outputFp16 [[buffer(9)]],
                                    constant SP4LinearDispatch& dispatch [[buffer(10)]],
                                    uint3 tid [[thread_position_in_grid]]) {
    const SP4LayerPlan plan = layers[dispatch.layerIndex];
    const uint tokenBase = tid.x * 16u;
    const uint oc = tid.y;
    if (tokenBase >= dispatch.tokens || oc >= plan.outputChannels) return;

    const uint inputChannels = plan.inputChannels;
    const uint weightBase = oc * inputChannels;
    float acc = 0.0f;
    for (uint k = 0; k < inputChannels; ++k) {
        const uint valueIndex = weightBase + k;
        const uint blockId = valueIndex / plan.blockSize;
        const uint local = valueIndex - blockId * plan.blockSize;
        acc += sp4_weight_value(qweight, scale, mode, aux, residualIndex, residualValue, residualTable,
                                plan, valueIndex, blockId, local) * float(activationInt8[k]);
    }
    for (uint t = 0; t < 16u && tokenBase + t < dispatch.tokens; ++t) {
        outputFp16[(tokenBase + t) * plan.outputChannels + oc] = half(acc);
    }
}

kernel void sp4_ssm_scan_fp16(device const half* input [[buffer(0)]],
                              device const half* decay [[buffer(1)]],
                              device half* state [[buffer(2)]],
                              device half* output [[buffer(3)]],
                              constant SP4SSMScanDispatch& dispatch [[buffer(4)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid >= dispatch.channels) return;
    float s = float(state[tid]);
    const float a = float(decay[tid]);
    for (uint t = 0; t < dispatch.tokens; ++t) {
        const uint index = t * dispatch.channels + tid;
        s = fma(s, a, float(input[index]));
        output[index] = half(s);
    }
    state[tid] = half(s);
}

kernel void sp4_ssm_scan_parallel_fp16(device const half* input [[buffer(0)]],
                                       device const half* decay [[buffer(1)]],
                                       device half* state [[buffer(2)]],
                                       device half* output [[buffer(3)]],
                                       constant SP4SSMScanDispatch& dispatch [[buffer(4)]],
                                       uint lane [[thread_position_in_threadgroup]],
                                       uint channel [[threadgroup_position_in_grid]]) {
    constexpr uint kChunk = 256u;
    threadgroup float prefixA[kChunk];
    threadgroup float prefixB[kChunk];
    threadgroup float chunkState;

    if (channel >= dispatch.channels) return;
    if (lane == 0u) {
        chunkState = float(state[channel]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float a = float(decay[channel]);
    for (uint base = 0u; base < dispatch.tokens; base += kChunk) {
        const uint token = base + lane;
        if (token < dispatch.tokens) {
            prefixA[lane] = a;
            prefixB[lane] = float(input[token * dispatch.channels + channel]);
        } else {
            prefixA[lane] = 1.0f;
            prefixB[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint offset = 1u; offset < kChunk; offset <<= 1u) {
            float prevA = 1.0f;
            float prevB = 0.0f;
            if (lane >= offset) {
                prevA = prefixA[lane - offset];
                prevB = prefixB[lane - offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane >= offset) {
                const float curA = prefixA[lane];
                const float curB = prefixB[lane];
                prefixA[lane] = curA * prevA;
                prefixB[lane] = fma(curA, prevB, curB);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (token < dispatch.tokens) {
            output[token * dispatch.channels + channel] = half(fma(prefixA[lane], chunkState, prefixB[lane]));
        }
        if (lane == min(kChunk - 1u, dispatch.tokens - base - 1u)) {
            chunkState = fma(prefixA[lane], chunkState, prefixB[lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        state[channel] = half(chunkState);
    }
}

kernel void sp4_ssm_scan_parallel512_fp16(device const half* input [[buffer(0)]],
                                          device const half* decay [[buffer(1)]],
                                          device half* state [[buffer(2)]],
                                          device half* output [[buffer(3)]],
                                          constant SP4SSMScanDispatch& dispatch [[buffer(4)]],
                                          uint lane [[thread_position_in_threadgroup]],
                                          uint channel [[threadgroup_position_in_grid]]) {
    constexpr uint kChunk = 512u;
    threadgroup float prefixA[kChunk];
    threadgroup float prefixB[kChunk];
    threadgroup float chunkState;

    if (channel >= dispatch.channels) return;
    if (lane == 0u) {
        chunkState = float(state[channel]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float a = float(decay[channel]);
    for (uint base = 0u; base < dispatch.tokens; base += kChunk) {
        const uint token = base + lane;
        if (token < dispatch.tokens) {
            prefixA[lane] = a;
            prefixB[lane] = float(input[token * dispatch.channels + channel]);
        } else {
            prefixA[lane] = 1.0f;
            prefixB[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint offset = 1u; offset < kChunk; offset <<= 1u) {
            float prevA = 1.0f;
            float prevB = 0.0f;
            if (lane >= offset) {
                prevA = prefixA[lane - offset];
                prevB = prefixB[lane - offset];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane >= offset) {
                const float curA = prefixA[lane];
                const float curB = prefixB[lane];
                prefixA[lane] = curA * prevA;
                prefixB[lane] = fma(curA, prevB, curB);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (token < dispatch.tokens) {
            output[token * dispatch.channels + channel] = half(fma(prefixA[lane], chunkState, prefixB[lane]));
        }
        if (lane == min(kChunk - 1u, dispatch.tokens - base - 1u)) {
            chunkState = fma(prefixA[lane], chunkState, prefixB[lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        state[channel] = half(chunkState);
    }
}
