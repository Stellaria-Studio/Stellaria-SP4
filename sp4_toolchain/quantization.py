from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modes import SP4_MODE_LINEAR_W4A8, SP4_MODE_SPARSE_RESIDUAL


@dataclass
class QuantizedTensor:
    qweight: np.ndarray
    scale: np.ndarray
    mode: np.ndarray
    aux: np.ndarray
    residual_index: np.ndarray
    residual_value: np.ndarray
    residual_table: np.ndarray
    reconstruction: np.ndarray
    num_blocks: int
    padded_values: int


def quantize_tensor(values: np.ndarray, block_size: int, aux_value: int, residual_ratio: float) -> QuantizedTensor:
    flat = values.astype(np.float32, copy=False).reshape(-1)
    original_count = flat.size
    num_blocks = int((original_count + block_size - 1) // block_size)
    padded_values = num_blocks * block_size
    padded = np.zeros(padded_values, dtype=np.float32)
    padded[:original_count] = flat
    blocks = padded.reshape(num_blocks, block_size)

    scales = np.maximum(np.max(np.abs(blocks), axis=1) / 7.0, np.float32(1.0e-8)).astype(np.float16)
    q = np.rint(blocks / scales.astype(np.float32)[:, None]).clip(-8, 7).astype(np.int8)
    dequant = q.astype(np.float32) * scales.astype(np.float32)[:, None]
    error = blocks - dequant

    modes = np.full(num_blocks, SP4_MODE_LINEAR_W4A8, dtype=np.uint8)
    aux = np.full(num_blocks, aux_value & 0xFF, dtype=np.uint8)
    residual_indices: list[int] = []
    residual_values: list[float] = []
    residual_table = np.zeros((num_blocks, 2), dtype=np.int32)

    ratio = max(0.0, float(residual_ratio))
    if ratio > 0.0:
        total_budget = max(1, int(round(original_count * ratio)))
        entries_per_selected_block = max(1, int(np.ceil(block_size * min(0.125, max(ratio * 4.0, ratio)))))
        valid_counts = np.minimum(block_size, np.maximum(0, original_count - np.arange(num_blocks) * block_size))
        block_scores = np.sqrt(np.mean(np.square(error), axis=1))
        candidates = [int(i) for i in np.argsort(block_scores)[::-1] if valid_counts[int(i)] > 0 and block_scores[int(i)] > 0]
    else:
        total_budget = 0
        entries_per_selected_block = 0
        valid_counts = np.zeros(num_blocks, dtype=np.int32)
        candidates = []

    used_budget = 0
    for block_id in candidates:
        valid_count = min(block_size, max(0, original_count - block_id * block_size))
        if valid_count <= 0 or used_budget >= total_budget:
            continue
        count = min(valid_count, entries_per_selected_block, total_budget - used_budget)
        local_error = np.abs(error[block_id, :valid_count])
        if not np.any(local_error > 0):
            continue
        threshold_indices = np.argpartition(local_error, -count)[-count:]
        threshold_indices = threshold_indices[np.argsort(threshold_indices)]
        start = len(residual_indices)
        for local_index in threshold_indices:
            residual_indices.append(int(local_index))
            residual_values.append(float(error[block_id, local_index]))
            dequant[block_id, local_index] += error[block_id, local_index]
        residual_table[block_id] = (start, len(threshold_indices))
        modes[block_id] = SP4_MODE_SPARSE_RESIDUAL
        used_budget += len(threshold_indices)

    return QuantizedTensor(
        qweight=pack_int4(q.reshape(-1)),
        scale=scales,
        mode=modes,
        aux=aux,
        residual_index=np.asarray(residual_indices, dtype=np.int32),
        residual_value=np.asarray(residual_values, dtype=np.float16),
        residual_table=residual_table,
        reconstruction=dequant.reshape(-1)[:original_count].reshape(values.shape),
        num_blocks=num_blocks,
        padded_values=padded_values,
    )


def pack_int4(qvalues: np.ndarray) -> np.ndarray:
    q = qvalues.astype(np.int8, copy=False).reshape(-1)
    if q.size % 2:
        q = np.concatenate([q, np.zeros(1, dtype=np.int8)])
    low = q[0::2] & 0x0F
    high = (q[1::2] & 0x0F) << 4
    return (low | high).astype(np.uint8)


def unpack_int4(packed: np.ndarray, count: int) -> np.ndarray:
    raw = packed.astype(np.uint8, copy=False).reshape(-1)
    out = np.empty(raw.size * 2, dtype=np.int8)
    out[0::2] = raw & 0x0F
    out[1::2] = (raw >> 4) & 0x0F
    out[out >= 8] -= 16
    return out[:count]


def unpack_uint4(packed: np.ndarray, count: int) -> np.ndarray:
    raw = packed.astype(np.uint8, copy=False).reshape(-1)
    out = np.empty(raw.size * 2, dtype=np.uint8)
    out[0::2] = raw & 0x0F
    out[1::2] = (raw >> 4) & 0x0F
    return out[:count]
