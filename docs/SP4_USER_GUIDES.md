# SP4 SDK User Guides

SP4 has three user groups. They should not all touch the same layer.

## 1. Ordinary Users

Ordinary users want to compile and run a model.

```bash
sp4c compile \
  --input Models/RIFE-safetensors/flownet.safetensors \
  --output build/rife.sp4 \
  --adapter rife \
  --target metal-m4pro \
  --quality-profile adaptive-rife
```

Runtime usage should look like this:

```cpp
SP4Context* ctx = sp4CreateContextMetal(device);
SP4Model* model = sp4LoadModel(ctx, "rife.sp4");

sp4SetInputTexture(model, 0, frame0);
sp4SetInputTexture(model, 1, frame1);
sp4SetOutputTexture(model, 0, output);

sp4Dispatch(model, commandBuffer);
```

They should not need to know how INT4 is unpacked, how `mode` branches are handled, or how residual buffers are indexed.

## 2. Engine Integrators

Engine integrators need stable asset and resource binding APIs.

They care about:

```text
C ABI stability
external GPU texture/buffer wrapping
backend capability checks
async command-buffer integration
fallback behavior
validation before shipping
```

Typical flow:

```cpp
SP4DeviceDesc desc{};
desc.backendKind = 1; // Metal
desc.nativeDevice = metalDevice;
desc.nativeQueue = metalQueue;

SP4Context* ctx = sp4CreateContext(&desc);
SP4BackendCaps caps = sp4GetBackendCaps(ctx);

SP4Model* model = sp4LoadModel(ctx, "model.sp4");
if (!sp4BackendSupportsModel(ctx, model)) {
    // reject or route to CPU/reference fallback
}
```

The SDK must allow them to bind native resources:

```text
Metal texture / buffer
IOSurface
DX12 resource
Vulkan image / buffer
engine-owned command buffers
```

## 3. Backend Developers

Backend developers implement `SP4Backend` from [SP4Backend.h](/Users/minsawa/Documents/Stellaria%20SP4/sdk/include/sp4/SP4Backend.h).

They must implement:

```text
caps
supports
createPipeline
bindResources
dispatch
destroyModel
```

They must follow:

```text
SP4 Format Spec
SP4 Kernel Contract
SoA buffer layout
mode / aux semantics
sparse residual table semantics
backend plan compatibility
version compatibility
```

This makes the design reusable:

```text
same .sp4 asset
different backend plugin
fixed runtime plan
no shader editing for ordinary users
```
