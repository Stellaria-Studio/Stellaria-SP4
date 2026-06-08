from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .format import SP4Loader
from .modes import MODE_NAMES


def inspect_asset(asset_path: Path) -> dict:
    asset = SP4Loader(asset_path)
    modes = asset.section_array("mode", np.uint8)
    mode_counts = Counter(int(v) for v in modes)
    totals = asset.manifest.get("totals", {})
    layers = []
    for layer in asset.manifest.get("layers", []):
        layers.append(
            {
                "name": layer["name"],
                "shape": layer["shape"],
                "block_abi": layer["block_abi"],
                "blocks": layer["num_blocks"],
                "residual_entries": layer["residual_entries"],
                "aux_policy": layer["aux_policy"],
                "rmse": layer["metrics"]["rmse"],
            }
        )
    return {
        "path": str(asset_path),
        "version": asset.manifest.get("version"),
        "adapter": asset.manifest.get("adapter"),
        "layers": len(layers),
        "excluded_layers": len(asset.manifest.get("excluded_layers", [])),
        "graph_tensor_count": asset.manifest.get("graph_tensor_count"),
        "totals": totals,
        "compression": totals.get("teacher_weight_bytes", 0) / max(1, totals.get("sp4_weight_bytes", 1)),
        "mode_counts": {MODE_NAMES.get(k, str(k)): v for k, v in sorted(mode_counts.items())},
        "backend_plan": asset.manifest.get("backend_plan", {}),
        "layer_preview": layers[:12],
        "sections": asset.manifest.get("sections", {}),
    }


def add_inspect_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("inspect", help="inspect an SP4 asset")
    parser.add_argument("asset", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=_main)


def _main(args: argparse.Namespace) -> int:
    report = inspect_asset(args.asset)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"SP4 asset: {report['path']}")
        print(f"  version={report['version']} adapter={report['adapter']} layers={report['layers']} excluded={report['excluded_layers']} graph_tensors={report['graph_tensor_count']}")
        print(f"  compression={report['compression']:.2f}x blocks={report['totals'].get('blocks')} residual_entries={report['totals'].get('residual_entries')}")
        print("  modes:")
        for name, count in report["mode_counts"].items():
            print(f"    {name}: {count}")
        print("  layers:")
        for layer in report["layer_preview"]:
            print(f"    {layer['name']} {layer['shape']} {layer['block_abi']} blocks={layer['blocks']} residual={layer['residual_entries']} rmse={layer['rmse']:.6f}")
    return 0
