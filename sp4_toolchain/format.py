from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"STSP4A1P"
CONTAINER_VERSION = 1


@dataclass
class SP4Asset:
    manifest: dict
    sections: dict[str, bytes]

    def write(self, path: Path) -> None:
        path = Path(path)
        payload = bytearray()
        section_meta = {}
        for name, data in self.sections.items():
            offset = len(payload)
            payload.extend(data)
            section_meta[name] = {
                "offset": offset,
                "length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        manifest = dict(self.manifest)
        manifest["sections"] = section_meta
        manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(MAGIC + struct.pack("<II", CONTAINER_VERSION, len(manifest_bytes)) + manifest_bytes + payload)


class SP4Loader:
    def __init__(self, path: Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        if len(raw) < 16:
            raise ValueError(f"{self.path}: SP4 asset is too small")
        if raw[:8] != MAGIC:
            raise ValueError(f"{self.path}: invalid SP4 magic")
        version, manifest_len = struct.unpack("<II", raw[8:16])
        if version != CONTAINER_VERSION:
            raise ValueError(f"{self.path}: unsupported container version {version}")
        manifest_start = 16
        manifest_end = manifest_start + manifest_len
        if manifest_end > len(raw):
            raise ValueError(f"{self.path}: manifest length is invalid")
        self.manifest = json.loads(raw[manifest_start:manifest_end])
        self.payload = raw[manifest_end:]
        expected_payload_hash = self.manifest.get("payload_sha256")
        if expected_payload_hash and hashlib.sha256(self.payload).hexdigest() != expected_payload_hash:
            raise ValueError(f"{self.path}: payload checksum mismatch")
        self.sections = {}
        for name, meta in self.manifest.get("sections", {}).items():
            offset = int(meta["offset"])
            length = int(meta["length"])
            end = offset + length
            if offset < 0 or end > len(self.payload):
                raise ValueError(f"{self.path}: section {name} range is invalid")
            data = self.payload[offset:end]
            expected_hash = meta.get("sha256")
            if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError(f"{self.path}: section {name} checksum mismatch")
            self.sections[name] = data

    def section_array(self, name: str, dtype: str | np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
        if name not in self.sections:
            raise KeyError(f"missing SP4 section {name}")
        arr = np.frombuffer(self.sections[name], dtype=np.dtype(dtype))
        if shape is not None:
            return arr.reshape(shape)
        return arr
