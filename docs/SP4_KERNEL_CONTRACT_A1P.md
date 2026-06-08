# SP4 Kernel Contract A1P

The compiler writes a fixed backend plan into the `.sp4` manifest. Runtime execution should not rediscover the graph every frame.

## SoA Sections

The A1P SDK stores high-frequency data in separate arrays:

```text
qweight
scale
bias
mode
aux
residual_index
residual_value
residual_table
activation_scale
```

This SoA layout lets kernels coalesce the main qweight and scale reads while touching residual buffers only for blocks whose residual table span is non-empty.

## Block Addressing

For a flattened weight value:

```text
block_id    = value_index / block_size
local_index = value_index - block_id * block_size
packed_byte = qweight[value_index / 2]
q4          = low nibble for even values, high nibble for odd values
```

Signed INT4 is sign-extended from two's-complement nibble range `[-8, 7]`.

For `SP4_MODE_ADAPTIVE_PROJECT`, the same nibble is interpreted as an unsigned codebook index in `[0, 15]`, then multiplied by the block FP16 scale.

## Sparse Residual

At rest, C++ A1P assets store compact residual metadata:

```text
residual_index: uint8 local index within block
residual_table: int32 triples { local_block_id, start, count } for non-empty blocks
```

Loaders expand it before backend dispatch into the dense table used by the current kernels:

```text
dense_residual_table[block_id] = { start, count }
```

For residual entry `start + i` after expansion:

```text
local_index = residual_index[start + i]
correction  = residual_value[start + i]
```

Any block with `count > 0` needs this path. The current compiler can attach sparse residual correction to either linear W4A8 blocks or AdaptiveProject codebook blocks.

## Performance Rules

Backends should preserve these rules:

```text
ordinary blocks do not read residual buffers
qweight and scale are read as separate streams
block metadata is byte-addressable
kernel variants are selected from the asset plan
INT32 grouped accumulation is preferred for linear W4A8 blocks
FP16/FP32 residual correction should happen at the block boundary, not per nibble
6-8 output pixels per thread can reuse packed weights and hide correction latency
the current Metal backend uses 16 output pixels per thread for benchmarked Conv2D reuse
threadgroup shape should keep occupancy high before maximizing per-thread work
```

The Metal source in [sp4_block_w4a8.metal](/Users/minsawa/Documents/Stellaria%20SP4/shaders/metal/sp4_block_w4a8.metal) is the first native contract implementation. The Slang source in [sp4_block_w4a8.slang](/Users/minsawa/Documents/Stellaria%20SP4/shaders/slang/sp4_block_w4a8.slang) mirrors the same addressing model for portable backends.
