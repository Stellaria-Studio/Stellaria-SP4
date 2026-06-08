from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .format import SP4Loader
from .modes import MODE_NAMES, SUPPORTED_A1P_MODES
from .runtime import CPUReferenceBackend, conv2d_nchw
from .safetensors_io import SafeTensorFile


def validate_asset(asset_path: Path, teacher_path: Path | None = None, max_rmse: float = 0.04) -> dict:
    asset = SP4Loader(asset_path)
    report: dict = {"ok": True, "format_errors": [], "numeric": [], "conv_probe": None}
    _validate_manifest(asset, report)
    if teacher_path is not None:
        _validate_numeric(asset, teacher_path, report, max_rmse)
    report["ok"] = not report["format_errors"] and all(item["rmse"] <= max_rmse for item in report["numeric"])
    return report


def _validate_manifest(asset: SP4Loader, report: dict) -> None:
    required_sections = {
        "qweight",
        "scale",
        "bias",
        "mode",
        "aux",
        "residual_index",
        "residual_value",
        "residual_table",
        "activation_scale",
    }
    missing = sorted(required_sections - set(asset.sections))
    if missing:
        report["format_errors"].append(f"missing sections: {', '.join(missing)}")
    if asset.manifest.get("version") != "A1P":
        report["format_errors"].append(f"unsupported SP4 version: {asset.manifest.get('version')}")
    modes = asset.section_array("mode", np.uint8) if "mode" in asset.sections else np.asarray([], dtype=np.uint8)
    invalid = sorted({int(v) for v in modes if int(v) not in MODE_NAMES})
    unsupported = sorted({int(v) for v in modes if int(v) not in SUPPORTED_A1P_MODES})
    if invalid:
        report["format_errors"].append(f"invalid mode values: {invalid}")
    if unsupported:
        report["format_errors"].append(f"unsupported A1P mode values: {unsupported}")
    table = asset.section_array("residual_table", np.int32) if "residual_table" in asset.sections else np.asarray([], dtype=np.int32)
    if "residual_index" in asset.sections:
        residual_index_bytes = asset.sections["residual_index"]
        residual_value_count = asset.section_array("residual_value", np.float16).size if "residual_value" in asset.sections else 0
        residual_count = len(residual_index_bytes) if len(residual_index_bytes) == residual_value_count else asset.section_array("residual_index", np.int32).size
    else:
        residual_count = 0
    layers = asset.manifest.get("layers", [])
    total_blocks = sum(int(layer["num_blocks"]) for layer in layers)
    if table.size == total_blocks * 2:
        for layer in layers:
            start = int(layer["residual_table_offset"]) * 2
            end = start + int(layer["num_blocks"]) * 2
            if end > table.size:
                report["format_errors"].append(f"{layer['name']}: residual table out of range")
                continue
            for local_start, count in table[start:end].reshape(-1, 2):
                if local_start < 0 or count < 0 or local_start + count > residual_count:
                    report["format_errors"].append(f"{layer['name']}: residual span out of range")
                    break
    elif table.size % 3 == 0:
        for layer_index, layer in enumerate(layers):
            start = int(layer["residual_table_offset"])
            end = int(layers[layer_index + 1]["residual_table_offset"]) if layer_index + 1 < len(layers) else table.size // 3
            if start < 0 or end < start or end * 3 > table.size:
                report["format_errors"].append(f"{layer['name']}: compact residual table out of range")
                continue
            for block, local_start, count in table[start * 3:end * 3].reshape(-1, 3):
                if block < 0 or block >= int(layer["num_blocks"]):
                    report["format_errors"].append(f"{layer['name']}: compact residual block out of range")
                    break
                if local_start < 0 or count < 0 or local_start + count > residual_count:
                    report["format_errors"].append(f"{layer['name']}: compact residual span out of range")
                    break
    elif table.size:
        report["format_errors"].append("residual_table section is neither dense int2 nor compact triplet encoded")


def _validate_numeric(asset: SP4Loader, teacher_path: Path, report: dict, max_rmse: float) -> None:
    teacher = SafeTensorFile(teacher_path)
    backend = CPUReferenceBackend(asset)
    worst = None
    for layer in asset.manifest.get("layers", []):
        name = layer["name"]
        if name not in teacher.tensors:
            report["format_errors"].append(f"{name}: teacher tensor missing")
            continue
        expected = teacher.tensor(name).astype(np.float32)
        actual = backend.reconstruct_layer(layer)
        err = expected - actual
        rmse = float(np.sqrt(np.mean(np.square(err)))) if err.size else 0.0
        mae = float(np.mean(np.abs(err))) if err.size else 0.0
        max_abs = float(np.max(np.abs(err))) if err.size else 0.0
        relative_l2 = float(np.linalg.norm(err.reshape(-1)) / max(np.linalg.norm(expected.reshape(-1)), 1.0e-12))
        item = {"name": name, "rmse": rmse, "mae": mae, "max_abs": max_abs, "relative_l2": relative_l2, "ok": rmse <= max_rmse}
        report["numeric"].append(item)
        if worst is None or rmse > worst["rmse"]:
            worst = item
    if asset.manifest.get("layers"):
        first = asset.manifest["layers"][0]
        weight = teacher.tensor(first["name"]).astype(np.float32)
        sp4_weight = backend.reconstruct_layer(first)
        bias = backend.bias_for_layer(first)
        rng = np.random.default_rng(20260608)
        sample = rng.normal(0.0, 0.25, size=(1, weight.shape[1], 8, 8)).astype(np.float32)
        teacher_out = conv2d_nchw(sample, weight, bias=bias, stride=1, padding=1)
        sp4_out = conv2d_nchw(sample, sp4_weight, bias=bias, stride=1, padding=1)
        err = teacher_out - sp4_out
        report["conv_probe"] = {
            "layer": first["name"],
            "rmse": float(np.sqrt(np.mean(np.square(err)))),
            "mae": float(np.mean(np.abs(err))),
            "max_abs": float(np.max(np.abs(err))),
        }
    report["worst_layer"] = worst


def add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate", help="validate an SP4 asset")
    parser.add_argument("asset", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--max-rmse", type=float, default=0.04)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=_main)


def _main(args: argparse.Namespace) -> int:
    report = validate_asset(args.asset, args.teacher, args.max_rmse)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "ok" if report["ok"] else "failed"
        print(f"SP4 validate: {status}")
        if report["format_errors"]:
            for error in report["format_errors"]:
                print(f"  error: {error}")
        if report.get("worst_layer"):
            worst = report["worst_layer"]
            print(f"  worst: {worst['name']} rmse={worst['rmse']:.6f} rel_l2={worst['relative_l2']:.6f}")
        if report.get("conv_probe"):
            probe = report["conv_probe"]
            print(f"  conv_probe: {probe['layer']} rmse={probe['rmse']:.6f} mae={probe['mae']:.6f}")
    return 0 if report["ok"] else 1
