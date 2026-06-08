import "sp4_a1p_common.slang";

struct SP4ReconstructDispatch {
    uint layerIndex;
    uint outputOffset;
    uint valueCount;
    uint reserved0;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> g_reconstructQweight32 : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> g_reconstructScale : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<uint> g_reconstructMode32 : register(t2);
[[vk::binding(3, 0)]] StructuredBuffer<int> g_reconstructResidualIndex : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<float16_t> g_reconstructResidualValue : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<int2> g_reconstructResidualTable : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<SP4LayerPlan> g_reconstructLayers : register(t6);
[[vk::binding(7, 0)]] RWStructuredBuffer<float> g_reconstructOutput : register(u7);
[[vk::binding(8, 0)]] ConstantBuffer<SP4ReconstructDispatch> g_reconstructDispatch : register(b8);

uint sp4ReconstructLoadByte(StructuredBuffer<uint> words, uint byteIndex) {
    uint word = words[byteIndex >> 2];
    return (word >> ((byteIndex & 3) * 8)) & 0xff;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void sp4ReconstructA1P(uint3 tid3 : SV_DispatchThreadID) {
    uint tid = tid3.x;
    SP4LayerPlan plan = g_reconstructLayers[g_reconstructDispatch.layerIndex];
    if (tid >= g_reconstructDispatch.valueCount) {
        return;
    }

    uint blockId = tid / plan.blockSize;
    uint local = tid - blockId * plan.blockSize;
    uint mode = sp4ReconstructLoadByte(g_reconstructMode32, plan.modeOffset + blockId);
    float blockScale = float(g_reconstructScale[plan.scaleOffset + blockId]);
    float value = 0.0;
    if (mode == 5) {
        value = sp4CodebookValue(sp4LoadU4(g_reconstructQweight32, plan.qweightOffsetBytes, tid)) * blockScale;
    } else if (mode == 6) {
        value = sp4LearnedCodebookValue(sp4LoadU4(g_reconstructQweight32, plan.qweightOffsetBytes, tid)) * blockScale;
    } else {
        value = float(sp4LoadI4(g_reconstructQweight32, plan.qweightOffsetBytes, tid)) * blockScale;
    }
    if (g_reconstructResidualTable[plan.residualTableOffset + blockId].y > 0) {
        value += sp4SparseResidual(g_reconstructResidualIndex,
                                   g_reconstructResidualValue,
                                   g_reconstructResidualTable,
                                   plan.residualIndexOffset,
                                   plan.residualValueOffset,
                                   plan.residualTableOffset,
                                   blockId,
                                   local);
    }
    g_reconstructOutput[g_reconstructDispatch.outputOffset + tid] = value;
}
