from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SafeTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


class SafeTensorFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        if len(raw) < 8:
            raise ValueError(f"{self.path}: safetensors file is too small")
        header_len = struct.unpack("<Q", raw[:8])[0]
        header_end = 8 + header_len
        if header_end > len(raw):
            raise ValueError(f"{self.path}: safetensors header length is invalid")
        self.header = json.loads(raw[8:header_end])
        self.payload = memoryview(raw)[header_end:]
        self.tensors = {
            name: SafeTensor(
                name=name,
                dtype=entry["dtype"],
                shape=tuple(int(v) for v in entry["shape"]),
                data_offsets=(int(entry["data_offsets"][0]), int(entry["data_offsets"][1])),
            )
            for name, entry in self.header.items()
            if name != "__metadata__"
        }

    def names(self) -> list[str]:
        return sorted(self.tensors)

    def tensor(self, name: str) -> np.ndarray:
        entry = self.tensors[name]
        start, end = entry.data_offsets
        raw = self.payload[start:end]
        dtype = _numpy_dtype(entry.dtype)
        expected = int(np.prod(entry.shape)) * np.dtype(dtype).itemsize
        if len(raw) != expected:
            raise ValueError(f"{name}: expected {expected} bytes, got {len(raw)}")
        return np.frombuffer(raw, dtype=dtype).reshape(entry.shape).copy()

    def conv_weight_names(self) -> list[str]:
        first = [f"block{i}.conv0.0.0.weight" for i in range(3)]
        weights = [
            name
            for name, entry in self.tensors.items()
            if entry.dtype in {"F32", "F16", "BF16"} and len(entry.shape) == 4 and name.endswith(".weight")
        ]
        ordered = [name for name in first if name in weights]
        ordered.extend(name for name in sorted(weights) if name not in ordered)
        return ordered


def _numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "F32":
        return np.dtype("<f4")
    if dtype == "F16":
        return np.dtype("<f2")
    if dtype == "I64":
        return np.dtype("<i8")
    if dtype == "I32":
        return np.dtype("<i4")
    if dtype == "U8":
        return np.dtype("u1")
    raise ValueError(f"unsupported safetensors dtype {dtype}")
