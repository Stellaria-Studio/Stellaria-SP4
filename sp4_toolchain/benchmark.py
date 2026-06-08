from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .format import SP4Loader
from .runtime import CPUReferenceBackend
from .validator import validate_asset


def benchmark_asset(asset_path: Path, teacher_path: Path | None = None, iterations: int = 5) -> dict:
    load_times = []
    reconstruct_times = []
    layer_count = 0
    total_values = 0
    for _ in range(iterations):
        start = time.perf_counter()
        asset = SP4Loader(asset_path)
        backend = CPUReferenceBackend(asset)
        load_times.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        for layer in asset.manifest.get("layers", []):
            reconstructed = backend.reconstruct_layer(layer)
            total_values += int(reconstructed.size)
            layer_count += 1
        reconstruct_times.append((time.perf_counter() - start) * 1000.0)
    validation = validate_asset(asset_path, teacher_path) if teacher_path is not None else None
    totals = SP4Loader(asset_path).manifest.get("totals", {})
    return {
        "asset": str(asset_path),
        "iterations": iterations,
        "load_ms_avg": float(np.mean(load_times)),
        "load_ms_min": float(np.min(load_times)),
        "reconstruct_ms_avg": float(np.mean(reconstruct_times)),
        "reconstruct_ms_min": float(np.min(reconstruct_times)),
        "layers_per_iteration": layer_count // max(iterations, 1),
        "values_per_iteration": total_values // max(iterations, 1),
        "compression": totals.get("teacher_weight_bytes", 0) / max(1, totals.get("sp4_weight_bytes", 1)),
        "validation": validation,
    }


def add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("benchmark", help="benchmark SP4 load/reconstruct and optional quality validation")
    parser.add_argument("asset", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=_main)


def _main(args: argparse.Namespace) -> int:
    report = benchmark_asset(args.asset, args.teacher, args.iterations)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"SP4 benchmark: {report['asset']}")
        print(f"  load avg/min: {report['load_ms_avg']:.3f} / {report['load_ms_min']:.3f} ms")
        print(f"  reconstruct avg/min: {report['reconstruct_ms_avg']:.3f} / {report['reconstruct_ms_min']:.3f} ms")
        print(f"  layers={report['layers_per_iteration']} values={report['values_per_iteration']} compression={report['compression']:.2f}x")
        validation = report.get("validation")
        if validation and validation.get("worst_layer"):
            worst = validation["worst_layer"]
            print(f"  quality worst rmse={worst['rmse']:.6f} layer={worst['name']}")
    return 0
