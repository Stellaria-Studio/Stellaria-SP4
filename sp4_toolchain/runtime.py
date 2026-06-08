from __future__ import annotations

from pathlib import Path

import numpy as np

from .format import SP4Loader
from .modes import SP4_MODE_ADAPTIVE_PROJECT, SP4_MODE_LEARNED_PROJECT, SP4_MODE_LINEAR_W4A8_ZP
from .quantization import unpack_int4, unpack_uint4


SP4_CODEBOOK = np.asarray(
    [
        -1.000000, -0.696193, -0.525073, -0.394917,
        -0.284441, -0.184773, -0.091050, 0.000000,
        0.079580, 0.160930, 0.246112, 0.337915,
        0.440710, 0.562617, 0.722956, 1.000000,
    ],
    dtype=np.float32,
)

SP4_LEARNED_CODEBOOK = np.asarray(
    [
        -1.000000, -0.770166, -0.607256, -0.472373,
        -0.356107, -0.252558, -0.158382, -0.070901,
         0.012902,  0.098159,  0.189695,  0.291085,
         0.407779,  0.548442,  0.727477,  1.000000,
    ],
    dtype=np.float32,
)


class CPUReferenceBackend:
    def __init__(self, asset: SP4Loader):
        self.asset = asset
        self.qweight = asset.section_array("qweight", np.uint8)
        self.scale = asset.section_array("scale", np.float16)
        self.bias = asset.section_array("bias", np.float16)
        self.residual_value = asset.section_array("residual_value", np.float16)
        residual_index_raw = asset.sections.get("residual_index", b"")
        if len(residual_index_raw) == self.residual_value.size:
            self.residual_index = np.frombuffer(residual_index_raw, dtype=np.uint8).astype(np.int32)
        else:
            self.residual_index = asset.section_array("residual_index", np.int32)
        self.residual_table = asset.section_array("residual_table", np.int32)
        self._total_blocks = sum(int(layer["num_blocks"]) for layer in asset.manifest.get("layers", []))

    @classmethod
    def load(cls, path: Path) -> "CPUReferenceBackend":
        return cls(SP4Loader(path))

    def reconstruct_layer(self, layer: dict) -> np.ndarray:
        block_size = int(layer["block_size"])
        num_blocks = int(layer["num_blocks"])
        padded_values = int(layer["padded_values"])
        q_offset = int(layer["qweight_offset_bytes"])
        q_len = int(layer["qweight_length_bytes"])
        packed = self.qweight[q_offset:q_offset + q_len]
        q = unpack_int4(packed, padded_values).reshape(num_blocks, block_size).astype(np.float32)
        uq = unpack_uint4(packed, padded_values).reshape(num_blocks, block_size)
        scales = self.scale[int(layer["scale_offset"]):int(layer["scale_offset"]) + num_blocks].astype(np.float32)
        values = q * scales[:, None]
        mode_offset = int(layer["mode_offset"])
        modes = self.asset.section_array("mode", np.uint8)[mode_offset:mode_offset + num_blocks]
        codebook_blocks = modes == SP4_MODE_ADAPTIVE_PROJECT
        if np.any(codebook_blocks):
            values[codebook_blocks] = SP4_CODEBOOK[uq[codebook_blocks]] * scales[codebook_blocks, None]
        learned_blocks = modes == SP4_MODE_LEARNED_PROJECT
        if np.any(learned_blocks):
            values[learned_blocks] = SP4_LEARNED_CODEBOOK[uq[learned_blocks]] * scales[learned_blocks, None]
        zp_blocks = modes == SP4_MODE_LINEAR_W4A8_ZP
        if np.any(zp_blocks):
            aux_offset = int(layer["aux_offset"])
            aux = self.asset.section_array("aux", np.uint8)[aux_offset:aux_offset + num_blocks]
            zp = (aux[zp_blocks] & 0x0F).astype(np.float32)
            values[zp_blocks] = (uq[zp_blocks].astype(np.float32) - zp[:, None]) * scales[zp_blocks, None]
        table = self._residual_table_for_layer(layer, num_blocks)
        index_base = int(layer["residual_index_offset"])
        value_base = int(layer["residual_value_offset"])
        for block_id, (start, count) in enumerate(table):
            if count <= 0:
                continue
            indices = self.residual_index[index_base + start:index_base + start + count]
            residuals = self.residual_value[value_base + start:value_base + start + count].astype(np.float32)
            values[block_id, indices] += residuals
        flat_count = int(np.prod(layer["shape"]))
        return values.reshape(-1)[:flat_count].reshape(tuple(layer["shape"]))

    def _residual_table_for_layer(self, layer: dict, num_blocks: int) -> np.ndarray:
        if self.residual_table.size == self._total_blocks * 2:
            table_start = int(layer["residual_table_offset"])
            return self.residual_table[table_start * 2:(table_start + num_blocks) * 2].reshape(num_blocks, 2)

        layers = self.asset.manifest.get("layers", [])
        layer_index = layers.index(layer)
        compact_start = int(layer["residual_table_offset"])
        compact_end = int(layers[layer_index + 1]["residual_table_offset"]) if layer_index + 1 < len(layers) else self.residual_table.size // 3
        table = np.zeros((num_blocks, 2), dtype=np.int32)
        compact = self.residual_table[compact_start * 3:compact_end * 3].reshape(-1, 3)
        for block, start, count in compact:
            if 0 <= block < num_blocks:
                table[int(block)] = (start, count)
        return table

    def bias_for_layer(self, layer: dict) -> np.ndarray:
        offset = int(layer["bias_offset"])
        count = int(layer["bias_count"])
        return self.bias[offset:offset + count].astype(np.float32)


def conv2d_nchw(input_tensor: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None, stride: int = 1, padding: int = 1) -> np.ndarray:
    if input_tensor.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d_nchw expects input [N,C,H,W] and weight [O,I,KH,KW]")
    n, in_channels, height, width = input_tensor.shape
    out_channels, weight_channels, kernel_h, kernel_w = weight.shape
    if in_channels != weight_channels:
        raise ValueError(f"input channels {in_channels} do not match weight channels {weight_channels}")
    padded = np.pad(input_tensor, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    out_h = (height + 2 * padding - kernel_h) // stride + 1
    out_w = (width + 2 * padding - kernel_w) // stride + 1
    output = np.zeros((n, out_channels, out_h, out_w), dtype=np.float32)
    for oy in range(out_h):
        y0 = oy * stride
        for ox in range(out_w):
            x0 = ox * stride
            patch = padded[:, :, y0:y0 + kernel_h, x0:x0 + kernel_w]
            output[:, :, oy, ox] = np.tensordot(patch, weight, axes=([1, 2, 3], [1, 2, 3]))
    if bias is not None and bias.size:
        output += bias.reshape(1, -1, 1, 1)
    return output
