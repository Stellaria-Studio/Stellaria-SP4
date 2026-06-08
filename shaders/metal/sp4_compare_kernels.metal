#include <metal_stdlib>
using namespace metal;

struct CompareDispatch {
    uint pixels;
    uint outputChannels;
    uint kernelValues;
    uint blockSize;
};

inline int cmp_sign_extend4(uint v) {
    v &= 0x0fu;
    return v >= 8u ? int(v) - 16 : int(v);
}

inline int cmp_load_i4(device const uchar* qweight, uint valueIndex) {
    const uchar packed = qweight[valueIndex >> 1u];
    return cmp_sign_extend4((valueIndex & 1u) == 0u ? packed : (packed >> 4u));
}

inline int cmp_load_i4_from_byte(uchar packed, bool highNibble) {
    return cmp_sign_extend4(highNibble ? (packed >> 4u) : packed);
}

kernel void cmp_conv_fp16(device const half* weight [[buffer(0)]],
                          device const half* activation [[buffer(1)]],
                          device half* output [[buffer(2)]],
                          constant CompareDispatch& d [[buffer(3)]],
                          uint3 tid [[thread_position_in_grid]]) {
    if (tid.x >= d.pixels || tid.z >= d.outputChannels) return;
    const uint oc = tid.z;
    float acc = 0.0f;
    for (uint k = 0; k < d.kernelValues; ++k) {
        acc += float(weight[oc * d.kernelValues + k]) * float(activation[k]);
    }
    output[tid.x * d.outputChannels + oc] = half(acc);
}

kernel void cmp_conv_int8(device const char* qweight [[buffer(0)]],
                          device const half* scale [[buffer(1)]],
                          device const char* activation [[buffer(2)]],
                          device half* output [[buffer(3)]],
                          constant CompareDispatch& d [[buffer(4)]],
                          uint3 tid [[thread_position_in_grid]]) {
    if (tid.x >= d.pixels || tid.z >= d.outputChannels) return;
    const uint oc = tid.z;
    float acc = 0.0f;
    const float s = float(scale[oc]);
    for (uint k = 0; k < d.kernelValues; ++k) {
        acc += float(qweight[oc * d.kernelValues + k]) * s * float(activation[k]);
    }
    output[tid.x * d.outputChannels + oc] = half(acc);
}

kernel void cmp_conv_int4_per_output(device const uchar* qweight [[buffer(0)]],
                                     device const half* scale [[buffer(1)]],
                                     device const char* activation [[buffer(2)]],
                                     device half* output [[buffer(3)]],
                                     constant CompareDispatch& d [[buffer(4)]],
                                     uint3 tid [[thread_position_in_grid]]) {
    if (tid.x >= d.pixels || tid.z >= d.outputChannels) return;
    const uint oc = tid.z;
    const uint base = oc * d.kernelValues;
    float acc = 0.0f;
    const float s = float(scale[oc]);
    for (uint k = 0; k < d.kernelValues; ++k) {
        acc += float(cmp_load_i4(qweight, base + k)) * s * float(activation[k]);
    }
    output[tid.x * d.outputChannels + oc] = half(acc);
}

kernel void cmp_conv_nf4_like(device const uchar* qweight [[buffer(0)]],
                              device const half* scale [[buffer(1)]],
                              device const char* activation [[buffer(2)]],
                              device half* output [[buffer(3)]],
                              constant CompareDispatch& d [[buffer(4)]],
                              uint3 tid [[thread_position_in_grid]]) {
    constexpr float codebook[16] = {
        -1.000000f, -0.696193f, -0.525073f, -0.394917f,
        -0.284441f, -0.184773f, -0.091050f,  0.000000f,
         0.079580f,  0.160930f,  0.246112f,  0.337915f,
         0.440710f,  0.562617f,  0.722956f,  1.000000f,
    };
    if (tid.x >= d.pixels || tid.z >= d.outputChannels) return;
    const uint oc = tid.z;
    const uint base = oc * d.kernelValues;
    float acc = 0.0f;
    for (uint k = 0; k < d.kernelValues; ++k) {
        const uint v = uint(cmp_load_i4(qweight, base + k)) & 0x0fu;
        const uint blockId = (base + k) / d.blockSize;
        acc += codebook[v] * float(scale[blockId]) * float(activation[k]);
    }
    output[tid.x * d.outputChannels + oc] = half(acc);
}

kernel void cmp_conv_w4a8_block(device const uchar* qweight [[buffer(0)]],
                                device const half* scale [[buffer(1)]],
                                device const char* activation [[buffer(2)]],
                                device half* output [[buffer(3)]],
                                constant CompareDispatch& d [[buffer(4)]],
                                uint3 tid [[thread_position_in_grid]]) {
    if (tid.x >= d.pixels || tid.z >= d.outputChannels) return;
    const uint oc = tid.z;
    const uint base = oc * d.kernelValues;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint k = 0; k < d.kernelValues; k += 2u) {
        const uint idx0 = base + k;
        acc0 += float(cmp_load_i4(qweight, idx0)) *
                float(scale[idx0 / d.blockSize]) *
                float(activation[k]);
        if (k + 1u < d.kernelValues) {
            const uint idx1 = idx0 + 1u;
            acc1 += float(cmp_load_i4(qweight, idx1)) *
                    float(scale[idx1 / d.blockSize]) *
                    float(activation[k + 1u]);
        }
    }
    output[tid.x * d.outputChannels + oc] = half(acc0 + acc1);
}

kernel void cmp_conv_sp4_int32x8(device const uchar* qweight [[buffer(0)]],
                                 device const half* scale [[buffer(1)]],
                                 device const char* activation [[buffer(2)]],
                                 device half* output [[buffer(3)]],
                                 constant CompareDispatch& d [[buffer(4)]],
                                 uint3 tid [[thread_position_in_grid]]) {
    if (tid.x >= d.pixels || tid.z >= d.outputChannels) return;
    const uint oc = tid.z;
    const uint base = oc * d.kernelValues;
    float acc = 0.0f;
    for (uint k = 0; k < d.kernelValues;) {
        const uint idx = base + k;
        const uint blockId = idx / d.blockSize;
        const uint local = idx - blockId * d.blockSize;
        const uint blockValid = min(d.kernelValues - k, d.blockSize - local);
        int isum0 = 0;
        int isum1 = 0;
        int isum2 = 0;
        int isum3 = 0;
        uint i = 0;
        if (((idx + i) & 1u) != 0u && i < blockValid) {
            isum0 += cmp_load_i4(qweight, idx + i) * int(activation[k + i]);
            ++i;
        }
        for (; i + 7u < blockValid; i += 8u) {
            const uint vi = idx + i;
            const uchar b0 = qweight[vi >> 1u];
            const uchar b1 = qweight[(vi + 2u) >> 1u];
            const uchar b2 = qweight[(vi + 4u) >> 1u];
            const uchar b3 = qweight[(vi + 6u) >> 1u];
            isum0 += cmp_load_i4_from_byte(b0, false) * int(activation[k + i + 0u]);
            isum1 += cmp_load_i4_from_byte(b0, true)  * int(activation[k + i + 1u]);
            isum2 += cmp_load_i4_from_byte(b1, false) * int(activation[k + i + 2u]);
            isum3 += cmp_load_i4_from_byte(b1, true)  * int(activation[k + i + 3u]);
            isum0 += cmp_load_i4_from_byte(b2, false) * int(activation[k + i + 4u]);
            isum1 += cmp_load_i4_from_byte(b2, true)  * int(activation[k + i + 5u]);
            isum2 += cmp_load_i4_from_byte(b3, false) * int(activation[k + i + 6u]);
            isum3 += cmp_load_i4_from_byte(b3, true)  * int(activation[k + i + 7u]);
        }
        for (; i < blockValid; ++i) {
            isum0 += cmp_load_i4(qweight, idx + i) * int(activation[k + i]);
        }
        acc += float(isum0 + isum1 + isum2 + isum3) * float(scale[blockId]);
        k += blockValid;
    }
    output[tid.x * d.outputChannels + oc] = half(acc);
}
