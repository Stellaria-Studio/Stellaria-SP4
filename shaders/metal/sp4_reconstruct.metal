#include "sp4_a1p_common.metal"

struct SP4ReconstructDispatch {
    uint layerIndex;
    uint outputOffset;
    uint valueCount;
    uint reserved0;
};

kernel void sp4_reconstruct_a1p(device const uchar* qweight [[buffer(0)]],
                                device const half* scale [[buffer(1)]],
                                device const uchar* mode [[buffer(2)]],
                                device const uchar* aux [[buffer(3)]],
                                device const int* residualIndex [[buffer(4)]],
                                device const half* residualValue [[buffer(5)]],
                                device const int2* residualTable [[buffer(6)]],
                                device const SP4LayerPlan* layers [[buffer(7)]],
                                device float* output [[buffer(8)]],
                                constant SP4ReconstructDispatch& dispatch [[buffer(9)]],
                                uint tid [[thread_position_in_grid]]) {
    const SP4LayerPlan plan = layers[dispatch.layerIndex];
    if (tid >= dispatch.valueCount) {
        return;
    }

    const uint blockId = tid / plan.blockSize;
    const uint local = tid - blockId * plan.blockSize;
    const uint modeValue = mode[plan.modeOffset + blockId];
    const float blockScale = float(scale[plan.scaleOffset + blockId]);
    float value = 0.0f;
    if (modeValue == 5u) {
        value = sp4_codebook_value(sp4_load_u4(qweight + plan.qweightOffsetBytes, tid)) * blockScale;
    } else if (modeValue == 6u) {
        value = sp4_learned_codebook_value(sp4_load_u4(qweight + plan.qweightOffsetBytes, tid)) * blockScale;
    } else if (modeValue == 1u) {
        const uint zp = uint(aux[plan.auxOffset + blockId] & 0x0fu);
        value = (float(sp4_load_u4(qweight + plan.qweightOffsetBytes, tid)) - float(zp)) * blockScale;
    } else {
        value = float(sp4_load_i4(qweight + plan.qweightOffsetBytes, tid)) * blockScale;
    }
    if (residualTable[plan.residualTableOffset + blockId].y > 0) {
        value += float(sp4_sparse_residual(residualIndex + plan.residualIndexOffset,
                                           residualValue + plan.residualValueOffset,
                                           residualTable + plan.residualTableOffset,
                                           blockId,
                                           local));
    }
    output[dispatch.outputOffset + tid] = value;
}
