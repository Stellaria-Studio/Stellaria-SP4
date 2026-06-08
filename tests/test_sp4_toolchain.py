from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from sp4_toolchain.compiler import compile_safetensors
from sp4_toolchain.format import SP4Loader
from sp4_toolchain.runtime import CPUReferenceBackend
from sp4_toolchain.validator import validate_asset


def test_compile_load_validate_synthetic(tmp_path: Path) -> None:
    model = tmp_path / "tiny.safetensors"
    rng = np.random.default_rng(42)
    tensors = {
        "block0.conv0.0.0.weight": rng.normal(0.0, 0.08, size=(4, 3, 3, 3)).astype(np.float32),
        "block0.conv0.0.0.bias": rng.normal(0.0, 0.02, size=(4,)).astype(np.float32),
        "block0.conv1.0.weight": rng.normal(0.0, 0.06, size=(3, 4, 3, 3)).astype(np.float32),
        "block0.conv1.0.bias": rng.normal(0.0, 0.02, size=(3,)).astype(np.float32),
    }
    save_file(tensors, str(model))

    asset = tmp_path / "tiny.sp4"
    manifest = compile_safetensors(model, asset, "rife", "32", 0.05)
    assert asset.exists()
    assert manifest["totals"]["blocks"] > 0

    loaded = SP4Loader(asset)
    backend = CPUReferenceBackend(loaded)
    first = loaded.manifest["layers"][0]
    reconstructed = backend.reconstruct_layer(first)
    assert reconstructed.shape == tensors[first["name"]].shape

    report = validate_asset(asset, model, max_rmse=0.05)
    assert report["ok"], report
