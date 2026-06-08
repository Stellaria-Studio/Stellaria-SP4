# SP4 A1P Format Spec

This repository implements the first practical preview ABI for `SP4block32(A1P)` and `SP4block64(A1P)`.

## Precision Contract

- Weight main path: signed INT4, symmetric, packed two values per byte.
- Zero-point main path: unsigned INT4 with per-block zero point in `aux & 0x0F`, selected for skewed blocks in high-accuracy profiles.
- Input tensor dtypes accepted by the C++ compiler: F32, F16, and BF16 safetensors.
- Adaptive project path: unsigned 4-bit codebook index, packed in the same `qweight` stream when mode is `SP4_MODE_ADAPTIVE_PROJECT`.
- Activation contract: INT8 activation scale table is stored as FP16 metadata. The CPU reference backend reconstructs weights and validates with synthetic INT8 activation samples.
- Scale: FP16 per block.
- Bias: FP16 per output channel when a matching bias tensor exists.
- Accumulation: INT32 in backend contracts, restored to FP16/FP32.
- Residual: sparse FP16 residual entries attached to selected blocks.

## Binary Container

```text
magic:          8 bytes, "STSP4A1P"
container_ver:  uint32 little endian, currently 1
manifest_len:   uint32 little endian
manifest_json:  UTF-8 JSON
payload:        concatenated binary sections
```

The manifest stores section offsets and SHA-256 hashes relative to the payload. The whole payload also has a SHA-256 checksum.

## Sections

```text
qweight          uint8 packed signed int4
scale            float16 per block
bias             float16 per output channel
mode             uint8 per block
aux              uint8 per block
residual_index   uint8 sparse residual index within the block in C++ A1P assets; legacy int32 is still accepted by loaders
residual_value   float16 sparse residual value
residual_table   int32 compact triples: block, start, count for non-empty residual blocks
activation_scale float16 per layer
```

## Modes

```text
0 = SP4_MODE_LINEAR_W4A8
1 = SP4_MODE_LINEAR_W4A8_ZP
2 = SP4_MODE_RESIDUAL
3 = SP4_MODE_SPARSE_RESIDUAL
4 = SP4_MODE_SENSITIVE_FP16
5 = SP4_MODE_ADAPTIVE_PROJECT
6 = SP4_MODE_LEARNED_PROJECT
7 = SP4_MODE_RESERVED_SP4E
```

The current C++ compiler emits:

```text
0 for optimized linear W4A8 blocks
1 for asymmetric zero-point W4A8 blocks in max-accuracy / int8-chase profiles
3 for linear W4A8 blocks with sparse residual correction
5 for AdaptiveProject codebook blocks, optionally with sparse residual correction
6 for LearnedProject codebook blocks, optionally with sparse residual correction
```

Mode `1` keeps the same packed nibble stream but reconstructs `value = (u4 - zero_point) * scale`; Metal dot kernels compute this as `sum(u4 * int8) - zero_point * sum(int8)` before the FP scale restore, so the main path still uses INT32 ALU grouping. Mode `5` uses the shared SP4/NF4-like 16-entry codebook. Mode `6` uses a learned 16-entry codebook fit from normalized visual/RIFE-style weight blocks. Codebook modes are selected only when offline validation beats the optimized linear or zero-point scale for that block. Residual correction is determined by the compact residual table span, so mode `1`, mode `5`, and mode `6` blocks can still have sparse FP16 residual entries.

## Compact Residual Metadata

Current C++ A1P assets use:

```text
residual_index_encoding: u8_local_index_v1
residual_table_encoding: sparse_triplet_v1
```

`sparse_triplet_v1` stores only blocks with residual entries:

```text
triplet = { local_block_id, residual_start, residual_count }
```

Loaders expand this into the dense table expected by existing CPU and GPU kernels, so backend shader code can stay simple while the file avoids paying 8 bytes for every empty block.

## Aux Metadata

The base ABI treats `aux` as project-owned metadata. The included RIFE adapter uses:

```text
bit 0: flow branch
bit 1: mask branch
bit 2: early motion-boundary-sensitive stage
bit 3: deconvolution / upsample path
bits 4-7: block stage id, clamped to 0..15
```

When `mode == SP4_MODE_LINEAR_W4A8_ZP`, the low nibble is reserved for the zero point and the high nibble can still carry a coarse project stage id. Project adapters that need all 8 aux bits should avoid mode `1` or move their adapter metadata into a side table in a future ABI.

## Compatibility

`A1P` is a preview ABI. The loader rejects incompatible magic, unsupported container versions, missing sections, checksum mismatches, invalid modes, and invalid residual table spans.
