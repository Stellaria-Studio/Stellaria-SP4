from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerPolicy:
    block_size: int
    aux: int
    residual_ratio: float
    sensitive: bool
    fallback_fp16: bool = False


class ProjectAdapter:
    name = "generic"

    def include_layer(self, layer_name: str) -> bool:
        return True

    def policy_for_layer(self, layer_name: str, shape: tuple[int, ...], requested_block_size: str, residual_ratio: float) -> LayerPolicy:
        block_size = _resolve_block_size(requested_block_size, shape)
        return LayerPolicy(block_size=block_size, aux=0, residual_ratio=residual_ratio, sensitive=False)

    def backend_plan(self) -> dict:
        return {
            "preferred": "cpu_reference",
            "available": ["cpu_reference", "metal_future", "dx12_stub", "vulkan_stub", "matrix_accel_future"],
        }


class RIFEAdapter(ProjectAdapter):
    name = "rife"

    def include_layer(self, layer_name: str) -> bool:
        # RIFE safetensors can include block_tea training/teacher tensors that are
        # not part of the runtime interpolation path.
        return not layer_name.startswith("block_tea.")

    def policy_for_layer(self, layer_name: str, shape: tuple[int, ...], requested_block_size: str, residual_ratio: float) -> LayerPolicy:
        block_size = _resolve_block_size(requested_block_size, shape)
        stage = _stage_id(layer_name)
        aux = (stage & 0x0F) << 4
        sensitive = False
        if ".conv1." in layer_name:
            aux |= 0b0001 | 0b1000
        if ".conv2." in layer_name:
            aux |= 0b0010 | 0b1000
        if ".conv0." in layer_name or stage == 0:
            aux |= 0b0100
            sensitive = True
        if "convblock" in layer_name:
            sensitive = stage <= 1
        layer_ratio = residual_ratio
        if sensitive:
            layer_ratio = max(layer_ratio, residual_ratio * 1.5)
        if ".conv1." in layer_name or ".conv2." in layer_name:
            layer_ratio = max(layer_ratio, residual_ratio * 1.25)
        return LayerPolicy(block_size=block_size, aux=aux, residual_ratio=layer_ratio, sensitive=sensitive)

    def backend_plan(self) -> dict:
        return {
            "preferred": "metal_alu_w4a8",
            "fallback": "cpu_reference",
            "available": ["cpu_reference", "metal_alu_w4a8", "matrix_accel_future"],
            "notes": "RIFE SP4 A1P: W4A8 main path plus sparse FP16 residuals for motion-boundary and flow/mask branches.",
        }


def adapter_by_name(name: str) -> ProjectAdapter:
    normalized = name.lower().strip()
    if normalized == "rife":
        return RIFEAdapter()
    if normalized == "generic":
        return ProjectAdapter()
    raise ValueError(f"unknown SP4 project adapter: {name}")


def _resolve_block_size(requested: str, shape: tuple[int, ...]) -> int:
    if requested in {"32", "block32"}:
        return 32
    if requested in {"64", "block64"}:
        return 64
    if requested != "auto":
        raise ValueError("--block-size must be auto, 32, or 64")
    values_per_output = int(shape[1] * shape[2] * shape[3]) if len(shape) == 4 else int(shape[-1])
    return 32 if values_per_output <= 128 else 64


def _stage_id(layer_name: str) -> int:
    if layer_name.startswith("block"):
        end = layer_name.find(".")
        try:
            return int(layer_name[len("block"):end])
        except ValueError:
            return 0
    return 0
