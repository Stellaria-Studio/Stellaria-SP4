from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from .adapters import adapter_by_name
from .format import SP4Asset
from .quantization import quantize_tensor
from .safetensors_io import SafeTensorFile


def compile_safetensors(
    input_path: Path,
    output_path: Path,
    adapter_name: str,
    block_size: str,
    residual_ratio: float,
    include_nonruntime_layers: bool = False,
) -> dict:
    started = time.perf_counter()
    source = SafeTensorFile(input_path)
    adapter = adapter_by_name(adapter_name)
    sections = {
        "qweight": bytearray(),
        "scale": bytearray(),
        "bias": bytearray(),
        "mode": bytearray(),
        "aux": bytearray(),
        "residual_index": bytearray(),
        "residual_value": bytearray(),
        "residual_table": bytearray(),
        "activation_scale": bytearray(),
    }
    layers = []
    totals = {
        "teacher_weight_bytes": 0,
        "sp4_weight_bytes": 0,
        "blocks": 0,
        "residual_entries": 0,
    }

    excluded_layers = []
    for name in source.conv_weight_names():
        if not include_nonruntime_layers and not adapter.include_layer(name):
            excluded_layers.append({"name": name, "reason": f"excluded_by_{adapter.name}_runtime_policy"})
            continue
        tensor = source.tensor(name).astype(np.float32)
        policy = adapter.policy_for_layer(name, tuple(tensor.shape), block_size, residual_ratio)
        quantized = quantize_tensor(tensor, policy.block_size, policy.aux, policy.residual_ratio)

        qweight_offset = len(sections["qweight"])
        scale_offset = len(sections["scale"]) // np.dtype(np.float16).itemsize
        mode_offset = len(sections["mode"])
        aux_offset = len(sections["aux"])
        residual_index_offset = len(sections["residual_index"]) // np.dtype(np.int32).itemsize
        residual_value_offset = len(sections["residual_value"]) // np.dtype(np.float16).itemsize
        residual_table_offset = len(sections["residual_table"]) // (np.dtype(np.int32).itemsize * 2)
        activation_scale_offset = len(sections["activation_scale"]) // np.dtype(np.float16).itemsize

        bias_name = name[: -len(".weight")] + ".bias"
        if bias_name in source.tensors:
            bias = source.tensor(bias_name).astype(np.float16).reshape(-1)
        else:
            bias = np.zeros((tensor.shape[0],), dtype=np.float16)
        activation_scale = np.asarray([1.0 / 127.0], dtype=np.float16)

        sections["qweight"].extend(quantized.qweight.tobytes())
        sections["scale"].extend(quantized.scale.tobytes())
        sections["bias"].extend(bias.tobytes())
        sections["mode"].extend(quantized.mode.tobytes())
        sections["aux"].extend(quantized.aux.tobytes())
        sections["residual_index"].extend(quantized.residual_index.tobytes())
        sections["residual_value"].extend(quantized.residual_value.tobytes())
        sections["residual_table"].extend(quantized.residual_table.tobytes())
        sections["activation_scale"].extend(activation_scale.tobytes())

        teacher_bytes = tensor.size * np.dtype(np.float16).itemsize
        layer_sp4_bytes = (
            quantized.qweight.nbytes
            + quantized.scale.nbytes
            + bias.nbytes
            + quantized.mode.nbytes
            + quantized.aux.nbytes
            + quantized.residual_index.nbytes
            + quantized.residual_value.nbytes
            + quantized.residual_table.nbytes
            + activation_scale.nbytes
        )
        totals["teacher_weight_bytes"] += teacher_bytes
        totals["sp4_weight_bytes"] += layer_sp4_bytes
        totals["blocks"] += quantized.num_blocks
        totals["residual_entries"] += int(quantized.residual_index.size)

        err = tensor - quantized.reconstruction
        rmse = float(np.sqrt(np.mean(np.square(err)))) if err.size else 0.0
        mae = float(np.mean(np.abs(err))) if err.size else 0.0
        max_abs = float(np.max(np.abs(err))) if err.size else 0.0
        layers.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "block_abi": f"SP4block{policy.block_size}(A1P)",
                "block_size": policy.block_size,
                "num_blocks": quantized.num_blocks,
                "padded_values": quantized.padded_values,
                "qweight_offset_bytes": qweight_offset,
                "qweight_length_bytes": int(quantized.qweight.nbytes),
                "scale_offset": scale_offset,
                "bias_offset": len(sections["bias"]) // np.dtype(np.float16).itemsize - int(bias.size),
                "bias_count": int(bias.size),
                "mode_offset": mode_offset,
                "aux_offset": aux_offset,
                "residual_index_offset": residual_index_offset,
                "residual_value_offset": residual_value_offset,
                "residual_table_offset": residual_table_offset,
                "residual_entries": int(quantized.residual_index.size),
                "activation_scale_offset": activation_scale_offset,
                "aux_policy": policy.aux,
                "sensitive": policy.sensitive,
                "fallback_fp16": policy.fallback_fp16,
                "metrics": {"rmse": rmse, "mae": mae, "max_abs": max_abs},
            }
        )

    manifest = {
        "format": "stellaria.sp4",
        "architecture": "SP4block32/64(A1P)",
        "version": "A1P",
        "adapter": adapter.name,
        "source": str(input_path),
        "created_unix": int(time.time()),
        "compiler": {
            "name": "sp4c",
            "version": "0.1.0",
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "precision": {
            "weight": "int4",
            "activation": "int8",
            "scale": "fp16",
            "bias": "fp16",
            "accumulate": "int32",
            "restore": "fp16/fp32",
            "residual": "sparse_fp16",
        },
        "layers": layers,
        "excluded_layers": excluded_layers,
        "graph_tensor_count": len(source.tensors),
        "backend_plan": adapter.backend_plan(),
        "totals": totals,
        "compile_ms": (time.perf_counter() - started) * 1000.0,
    }
    SP4Asset(manifest=manifest, sections={name: bytes(data) for name, data in sections.items()}).write(output_path)
    return manifest


def add_compile_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("compile", help="compile a safetensors model into an SP4 A1P asset")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", default="generic")
    parser.add_argument("--block-size", default="auto", choices=["auto", "32", "64", "block32", "block64"])
    parser.add_argument("--residual-ratio", default=0.02, type=float)
    parser.add_argument("--include-nonruntime-layers", action="store_true")
    parser.set_defaults(func=_main)


def _main(args: argparse.Namespace) -> int:
    manifest = compile_safetensors(
        args.input,
        args.output,
        args.adapter,
        args.block_size,
        args.residual_ratio,
        args.include_nonruntime_layers,
    )
    totals = manifest["totals"]
    ratio = totals["teacher_weight_bytes"] / max(1, totals["sp4_weight_bytes"])
    print(f"wrote {args.output}")
    print(f"layers={len(manifest['layers'])} excluded={len(manifest.get('excluded_layers', []))} blocks={totals['blocks']} residual_entries={totals['residual_entries']}")
    print(f"teacher_fp16_bytes={totals['teacher_weight_bytes']} sp4_bytes={totals['sp4_weight_bytes']} compression={ratio:.2f}x")
    return 0
