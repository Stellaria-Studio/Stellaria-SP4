import "sp4_a1p_common.slang";

struct SP4ConvDispatch {
    uint width;
    uint height;
    uint layerIndex;
    uint activationZeroPoint;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint> g_qweight32 : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> g_scale : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<uint> g_modeAux32 : register(t2);
[[vk::binding(3, 0)]] StructuredBuffer<int> g_residualIndex : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<float16_t> g_residualValue : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<int2> g_residualTable : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<SP4LayerPlan> g_layers : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<int8_t> g_activationInt8 : register(t7);
[[vk::binding(8, 0)]] RWStructuredBuffer<float16_t> g_outputFp16 : register(u8);
[[vk::binding(9, 0)]] ConstantBuffer<SP4ConvDispatch> g_dispatch : register(b9);

uint sp4LoadByteFromWords(StructuredBuffer<uint> words, uint byteIndex) {
    uint word = words[byteIndex >> 2];
    return (word >> ((byteIndex & 3) * 8)) & 0xff;
}

[numthreads(8, 8, 2)]
[shader("compute")]
void sp4Conv2DBlockW4A8(uint3 tid : SV_DispatchThreadID) {
    SP4LayerPlan plan = g_layers[g_dispatch.layerIndex];
    if (tid.x >= g_dispatch.width || tid.y >= g_dispatch.height || tid.z >= plan.outputChannels) {
        return;
    }

    uint oc = tid.z;
    uint kernelValues = plan.inputChannels * plan.kernelWidth * plan.kernelHeight;
    uint weightBase = oc * kernelValues;
    float acc0 = 0.0;
    float acc1 = 0.0;

    [loop]
    for (uint k = 0; k < kernelValues; k += 2) {
        uint valueIndex0 = weightBase + k;
        uint blockId0 = valueIndex0 / plan.blockSize;
        uint local0 = valueIndex0 - blockId0 * plan.blockSize;
        uint mode0 = sp4LoadByteFromWords(g_modeAux32, plan.modeOffset + blockId0);
        float blockScale0 = float(g_scale[plan.scaleOffset + blockId0]);
        float w0 = mode0 == 5 ? sp4CodebookValue(sp4LoadU4(g_qweight32, plan.qweightOffsetBytes, valueIndex0)) * blockScale0 :
            (mode0 == 6 ? sp4LearnedCodebookValue(sp4LoadU4(g_qweight32, plan.qweightOffsetBytes, valueIndex0)) * blockScale0 :
             float(sp4LoadI4(g_qweight32, plan.qweightOffsetBytes, valueIndex0)) * blockScale0);
        if (mode0 == 3) {
            w0 += sp4SparseResidual(g_residualIndex, g_residualValue, g_residualTable,
                                    plan.residualIndexOffset, plan.residualValueOffset,
                                    plan.residualTableOffset, blockId0, local0);
        }
        acc0 += w0 * float(g_activationInt8[k]);

        if (k + 1 < kernelValues) {
            uint valueIndex1 = valueIndex0 + 1;
            uint blockId1 = valueIndex1 / plan.blockSize;
            uint local1 = valueIndex1 - blockId1 * plan.blockSize;
            uint mode1 = sp4LoadByteFromWords(g_modeAux32, plan.modeOffset + blockId1);
            float blockScale1 = float(g_scale[plan.scaleOffset + blockId1]);
            float w1 = mode1 == 5 ? sp4CodebookValue(sp4LoadU4(g_qweight32, plan.qweightOffsetBytes, valueIndex1)) * blockScale1 :
                (mode1 == 6 ? sp4LearnedCodebookValue(sp4LoadU4(g_qweight32, plan.qweightOffsetBytes, valueIndex1)) * blockScale1 :
                 float(sp4LoadI4(g_qweight32, plan.qweightOffsetBytes, valueIndex1)) * blockScale1);
            if (mode1 == 3) {
                w1 += sp4SparseResidual(g_residualIndex, g_residualValue, g_residualTable,
                                        plan.residualIndexOffset, plan.residualValueOffset,
                                        plan.residualTableOffset, blockId1, local1);
            }
            acc1 += w1 * float(g_activationInt8[k + 1]);
        }
    }

    uint outIndex = (tid.y * g_dispatch.width + tid.x) * plan.outputChannels + oc;
    g_outputFp16[outIndex] = float16_t(acc0 + acc1);
}
