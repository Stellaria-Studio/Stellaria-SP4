#pragma once

#include "sp4/SP4Format.h"

#include <cstdint>

extern "C" {

typedef struct SP4BackendCaps {
    uint32_t supportsBlock32;
    uint32_t supportsBlock64;
    uint32_t supportsW4A8;
    uint32_t supportsSparseResidual;
    uint32_t supportsSensitiveFp16;
    uint32_t supportsConv2D;
    uint32_t supportsDepthwiseConv2D;
    uint32_t supportsPointwiseConv2D;
    uint32_t supportsLinear;
    uint32_t supportsElementwise;
    uint32_t supportsActivation;
    uint32_t supportsRequantize;
    uint32_t supportsFp16WarpGridSample;
    uint32_t supportsFp16Blend;
    uint32_t supportsExternalTexture;
    uint32_t supportsExternalBuffer;
    uint32_t supportsAsyncDispatch;
    uint32_t supportsFusedConv;
    uint32_t supportsMatrixAccel;
    uint32_t supportsSlangSpirv;
    uint32_t supportsSlangMetal;
    uint32_t supportsMixedPrecisionPlan;
    uint32_t supportsCustomKernelRegistry;
    uint32_t supportsTile16Reuse;
    uint32_t supportsAsyncFpCorrection;
    uint32_t supportsSmallKernelBatch;
    uint32_t supportsOutOfOrderDispatch;
    uint32_t supportsAutoFusion;
    uint32_t reserved[1];
} SP4BackendCaps;

typedef struct SP4DeviceDesc {
    uint32_t backendKind;
    void* nativeDevice;
    void* nativeQueue;
} SP4DeviceDesc;

typedef struct SP4Context SP4Context;
typedef struct SP4Model SP4Model;
typedef struct SP4Tensor {
    void* nativeHandle;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t format;
} SP4Tensor;

typedef struct SP4DispatchDesc {
    void* nativeCommandBuffer;
    uint32_t allowFallback;
    uint32_t reserved0;
    uint32_t reserved1;
} SP4DispatchDesc;

typedef enum SP4SchedulePolicy {
    SP4_SCHEDULE_IN_ORDER = 0,
    SP4_SCHEDULE_BATCH_IDENTICAL = 1,
    SP4_SCHEDULE_OUT_OF_ORDER = 2,
    SP4_SCHEDULE_FUSE_WHEN_SAFE = 3
} SP4SchedulePolicy;

typedef enum SP4FusionKind {
    SP4_FUSION_NONE = 0,
    SP4_FUSION_CONV_ACT_REQUANT = 1,
    SP4_FUSION_LINEAR_ACT_REQUANT = 2,
    SP4_FUSION_LINEAR_GATE_SILU = 3,
    SP4_FUSION_SSM_PROJECTION_SCAN = 4,
    SP4_FUSION_POINTWISE_BLEND = 5
} SP4FusionKind;

typedef struct SP4DispatchItem {
    uint32_t opKind;
    uint32_t precisionKind;
    uint32_t kernelFamily;
    uint32_t layerIndex;
    uint32_t batchKey;
    uint32_t dependencyMask;
    uint32_t fusionKind;
    uint32_t estimatedWorkItems;
    const void* opParams;
    uint32_t opParamsBytes;
} SP4DispatchItem;

typedef struct SP4KernelBatchDesc {
    void* nativeCommandBuffer;
    const SP4DispatchItem* items;
    uint32_t itemCount;
    uint32_t schedulePolicy;
    uint32_t maxOutOfOrderDepth;
    uint32_t allowFallback;
} SP4KernelBatchDesc;

typedef struct SP4KernelDesc {
    uint32_t opKind;
    uint32_t precisionKind;
    uint32_t kernelFamily;
    uint32_t abiVersion;
    const char* name;
    const char* semanticTag;
    void* nativePipeline;
} SP4KernelDesc;

typedef struct SP4Backend {
    const char* name;
    uint32_t abiVersion;
    SP4BackendCaps (*caps)(void* backendUserData);
    int (*supports)(void* backendUserData, const SP4Model* model);
    int (*createPipeline)(SP4Context* context, SP4Model* model);
    int (*registerKernel)(SP4Context* context, const SP4KernelDesc* desc);
    int (*bindResources)(SP4Context* context, SP4Model* model);
    int (*dispatch)(SP4Context* context, SP4Model* model, const SP4DispatchDesc* desc);
    int (*dispatchBatch)(SP4Context* context, SP4Model* model, const SP4KernelBatchDesc* desc);
    int (*flush)(SP4Context* context, SP4Model* model, void* nativeCommandBuffer);
    void (*destroyModel)(SP4Context* context, SP4Model* model);
    void* userData;
} SP4Backend;

} // extern "C"
