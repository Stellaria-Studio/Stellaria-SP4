import "sp4_a1p_common.slang";

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

[[vk::binding(0, 0)]] StructuredBuffer<uint> g_qweight32 : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<float16_t> g_scale : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<uint> g_mode32 : register(t2);
[[vk::binding(3, 0)]] StructuredBuffer<uint> g_aux32 : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<int> g_residualIndex : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<float16_t> g_residualValue : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<int2> g_residualTable : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<SP4LayerPlan> g_layers : register(t7);
[[vk::binding(8, 0)]] StructuredBuffer<int8_t> g_activationInt8 : register(t8);
[[vk::binding(9, 0)]] RWStructuredBuffer<float16_t> g_linearOutputFp16 : register(u9);
[[vk::binding(10, 0)]] ConstantBuffer<SP4LinearDispatch> g_linearDispatch : register(b10);

[[vk::binding(11, 0)]] StructuredBuffer<float16_t> g_mlpUp : register(t11);
[[vk::binding(12, 0)]] StructuredBuffer<float16_t> g_mlpGate : register(t12);
[[vk::binding(13, 0)]] RWStructuredBuffer<float16_t> g_mlpOutput : register(u13);

[[vk::binding(14, 0)]] StructuredBuffer<float16_t> g_ssmInput : register(t14);
[[vk::binding(15, 0)]] StructuredBuffer<float16_t> g_ssmDecay : register(t15);
[[vk::binding(16, 0)]] RWStructuredBuffer<float16_t> g_ssmState : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<float16_t> g_ssmOutput : register(u17);
[[vk::binding(18, 0)]] ConstantBuffer<SP4SSMScanDispatch> g_ssmDispatch : register(b18);

uint sp4LoadByteFromWords(StructuredBuffer<uint> words, uint byteIndex) {
    uint word = words[byteIndex >> 2];
    return (word >> ((byteIndex & 3) * 8)) & 0xff;
}

float sp4WeightValuePortable(SP4LayerPlan plan, uint valueIndex, uint blockId, uint localIndex) {
    uint modeValue = sp4LoadByteFromWords(g_mode32, plan.modeOffset + blockId);
    float blockScale = float(g_scale[plan.scaleOffset + blockId]);
    float value = 0.0;
    if (modeValue == 5) {
        value = sp4CodebookValue(sp4LoadU4(g_qweight32, plan.qweightOffsetBytes, valueIndex)) * blockScale;
    } else if (modeValue == 6) {
        value = sp4LearnedCodebookValue(sp4LoadU4(g_qweight32, plan.qweightOffsetBytes, valueIndex)) * blockScale;
    } else if (modeValue == 1) {
        uint zp = sp4LoadByteFromWords(g_aux32, plan.auxOffset + blockId) & 0x0f;
        value = (float(sp4LoadU4(g_qweight32, plan.qweightOffsetBytes, valueIndex)) - float(zp)) * blockScale;
    } else {
        value = float(sp4LoadI4(g_qweight32, plan.qweightOffsetBytes, valueIndex)) * blockScale;
    }
    int2 span = g_residualTable[plan.residualTableOffset + blockId];
    if (span.y > 0) {
        value += sp4SparseResidual(g_residualIndex, g_residualValue, g_residualTable,
                                   plan.residualIndexOffset, plan.residualValueOffset,
                                   plan.residualTableOffset, blockId, localIndex);
    }
    return value;
}

[numthreads(16, 4, 1)]
[shader("compute")]
void sp4LinearTile16W4A8(uint3 tid : SV_DispatchThreadID) {
    SP4LayerPlan plan = g_layers[g_linearDispatch.layerIndex];
    uint token = tid.x;
    uint oc = tid.y;
    if (token >= g_linearDispatch.tokens || oc >= plan.outputChannels) {
        return;
    }

    uint inputChannels = plan.inputChannels;
    uint weightBase = oc * inputChannels;
    float acc = 0.0;
    [loop]
    for (uint k = 0; k < inputChannels; ++k) {
        uint valueIndex = weightBase + k;
        uint blockId = valueIndex / plan.blockSize;
        uint local = valueIndex - blockId * plan.blockSize;
        acc += sp4WeightValuePortable(plan, valueIndex, blockId, local) * float(g_activationInt8[k]);
    }
    g_linearOutputFp16[token * plan.outputChannels + oc] = float16_t(acc);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void sp4MlpGatedTile16W4A8(uint tid : SV_DispatchThreadID) {
    uint count = g_linearDispatch.tokens * g_linearDispatch.activationStride;
    if (tid >= count) {
        return;
    }
    float gate = float(g_mlpGate[tid]);
    float silu = gate / (1.0 + exp(-gate));
    g_mlpOutput[tid] = float16_t(float(g_mlpUp[tid]) * silu);
}

[numthreads(16, 4, 1)]
[shader("compute")]
void sp4SsmProjectionW4A8(uint3 tid : SV_DispatchThreadID) {
    sp4LinearTile16W4A8(tid);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void sp4SsmScanFp16(uint tid : SV_DispatchThreadID) {
    if (tid >= g_ssmDispatch.channels) {
        return;
    }
    float state = float(g_ssmState[tid]);
    float decay = float(g_ssmDecay[tid]);
    [loop]
    for (uint t = 0; t < g_ssmDispatch.tokens; ++t) {
        uint index = t * g_ssmDispatch.channels + tid;
        state = state * decay + float(g_ssmInput[index]);
        g_ssmOutput[index] = float16_t(state);
    }
    g_ssmState[tid] = float16_t(state);
}
