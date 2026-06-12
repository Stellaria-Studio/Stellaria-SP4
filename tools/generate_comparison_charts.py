#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "Models/RIFE-safetensors/flownet.safetensors"
OUT_DIR = ROOT / "docs/assets"


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return result.stdout


def parse_quantbench(text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) != 8 or parts[0] in {"scheme", ""}:
            continue
        try:
            rows[parts[0]] = {
                "bytes": float(parts[1]),
                "compression": float(parts[2].rstrip("x")),
                "rmse": float(parts[3]),
                "mae": float(parts[4]),
                "max_abs": float(parts[5]),
                "task_acc": float(parts[6].rstrip("%")),
                "cpu_ms": float(parts[7]),
            }
        except ValueError:
            continue
    return rows


def parse_metal(text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if "conv_kernel=" in line:
            kernel = re.search(r"conv_kernel=([^ ]+)", line)
            avg = re.search(r"avg_gops=([0-9.eE+-]+)", line)
            peak = re.search(r"peak_gops=([0-9.eE+-]+)", line)
            if kernel and avg and peak:
                rows[kernel.group(1)] = {"avg_gops": float(avg.group(1)), "peak_gops": float(peak.group(1))}
        if "backend_kernel=" in line:
            kernel = re.search(r"backend_kernel=([^ ]+)", line)
            avg = re.search(r"avg_gops=([0-9.eE+-]+)", line)
            peak = re.search(r"peak_gops=([0-9.eE+-]+)", line)
            if kernel and avg and peak:
                rows[kernel.group(1)] = {"avg_gops": float(avg.group(1)), "peak_gops": float(peak.group(1))}
        if "throughput_scheme=" in line:
            scheme = re.search(r"throughput_scheme=(.*?) avg_ms=", line)
            avg = re.search(r"avg_gops=([0-9.eE+-]+)", line)
            peak = re.search(r"peak_gops=([0-9.eE+-]+)", line)
            if scheme and avg and peak:
                rows[scheme.group(1)] = {"avg_gops": float(avg.group(1)), "peak_gops": float(peak.group(1))}
    return rows


def parse_scheduler(text: str) -> dict[str, float | int | str]:
    for line in text.splitlines():
        if "scheduler_bench=" not in line:
            continue
        name = re.search(r"scheduler_bench=([^ ]+)", line)
        small_kernels = re.search(r"small_kernels=([0-9]+)", line)
        naive_ms = re.search(r"naive_ms=([0-9.eE+-]+)", line)
        batched_ms = re.search(r"batched_ms=([0-9.eE+-]+)", line)
        speedup = re.search(r"speedup=([0-9.eE+-]+)", line)
        naive_gops = re.search(r"naive_gops=([0-9.eE+-]+)", line)
        batched_gops = re.search(r"batched_gops=([0-9.eE+-]+)", line)
        if name and small_kernels and naive_ms and batched_ms and speedup and naive_gops and batched_gops:
            return {
                "name": name.group(1),
                "small_kernels": int(small_kernels.group(1)),
                "naive_ms": float(naive_ms.group(1)),
                "batched_ms": float(batched_ms.group(1)),
                "speedup": float(speedup.group(1)),
                "naive_gops": float(naive_gops.group(1)),
                "batched_gops": float(batched_gops.group(1)),
            }
    return {}


def svg_polyline(points: list[tuple[float, float]], color: str, width: float = 2.4) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def make_svg_chart(path: Path,
                   title: str,
                   x_label: str,
                   y_label: str,
                   series: list[dict],
                   scatter: list[dict] | None = None,
                   width: int = 980,
                   height: int = 620) -> None:
    scatter = scatter or []
    left, right, top, bottom = 82, 32, 58, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [p["x"] for s in series for p in s["points"]] + [p["x"] for p in scatter]
    ys = [p["y"] for s in series for p in s["points"]] + [p["y"] for p in scatter]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax += 1
    if ymin == ymax:
        ymax += 1
    xpad = (xmax - xmin) * 0.08
    ypad = (ymax - ymin) * 0.10
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfc"/>',
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Inter,Arial" font-size="22" font-weight="700" fill="#1f2937">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#d1d5db"/>',
    ]
    for i in range(6):
        x = left + plot_w * i / 5
        y = top + plot_h * i / 5
        xv = xmin + (xmax - xmin) * i / 5
        yv = ymax - (ymax - ymin) * i / 5
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{xv:.2f}</text>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{yv:.2f}</text>')
    parts.append(f'<text x="{left+plot_w/2:.0f}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15" fill="#111827">{x_label}</text>')
    parts.append(f'<text transform="translate(24 {top+plot_h/2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15" fill="#111827">{y_label}</text>')

    legend_y = top + 18
    for s in series:
        pts = [(sx(p["x"]), sy(p["y"])) for p in s["points"]]
        parts.append(svg_polyline(pts, s["color"]))
        for p, (px, py) in zip(s["points"], pts):
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.5" fill="{s["color"]}"/>')
            if "label" in p:
                parts.append(f'<text x="{px+7:.2f}" y="{py-7:.2f}" font-family="Arial" font-size="12" fill="{s["color"]}">{p["label"]}</text>')
        parts.append(f'<line x1="{width-260}" y1="{legend_y}" x2="{width-232}" y2="{legend_y}" stroke="{s["color"]}" stroke-width="3"/>')
        parts.append(f'<text x="{width-224}" y="{legend_y+4}" font-family="Arial" font-size="13" fill="#111827">{s["name"]}</text>')
        legend_y += 20
    for p in scatter:
        px, py = sx(p["x"]), sy(p["y"])
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" fill="{p["color"]}"/>')
        parts.append(f'<text x="{px+8:.2f}" y="{py+4:.2f}" font-family="Arial" font-size="12" fill="#111827">{p["name"]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_throughput_chart(path: Path, metal: dict[str, dict[str, float]]) -> None:
    preferred = [
        "FP16",
        "INT8 per-output",
        "INT4 per-output",
        "NF4-like",
        "W4A8 block",
        "SP4 INT32x8",
        "SP4Conv2D_W4A8_Block64",
        "SP4Linear_Tile16_W4A8",
        "SP4SSM_Projection_W4A8",
        "SP4MLP_FusedProjectionGateSiLU_W4A8",
    ]
    rows = [(name, metal[name]) for name in preferred if name in metal]
    if not rows:
        return
    display_name = {
        "INT8 per-output": "INT8",
        "INT4 per-output": "INT4",
        "W4A8 block": "W4A8",
        "SP4Conv2D_W4A8_Block64": "SP4 Conv2D",
        "SP4Linear_Tile16_W4A8": "SP4 Linear",
        "SP4SSM_Projection_W4A8": "SP4 SSM proj",
        "SP4MLP_FusedProjectionGateSiLU_W4A8": "SP4 MLP fused",
    }
    width, height = 1180, 660
    left, right, top, bottom = 96, 32, 58, 188
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(max(row["avg_gops"], row["peak_gops"]) for _, row in rows) * 1.18
    group_w = plot_w / len(rows)
    bar_w = min(44.0, group_w * 0.28)

    def sy(v: float) -> float:
        return top + (ymax - v) / ymax * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfc"/>',
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Inter,Arial" font-size="22" font-weight="700" fill="#1f2937">Metal throughput: quantization schemes and SP4 backend kernels</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#d1d5db"/>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        v = ymax - ymax * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{v:.0f}</text>')
    parts.append(f'<text transform="translate(24 {top+plot_h/2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15" fill="#111827">GOPS</text>')
    parts.append(f'<rect x="{width-230}" y="{top+8}" width="14" height="14" fill="#2563eb"/><text x="{width-208}" y="{top+20}" font-family="Arial" font-size="13" fill="#111827">average</text>')
    parts.append(f'<rect x="{width-230}" y="{top+30}" width="14" height="14" fill="#f97316"/><text x="{width-208}" y="{top+42}" font-family="Arial" font-size="13" fill="#111827">peak</text>')
    for i, (name, row) in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        avg_h = top + plot_h - sy(row["avg_gops"])
        peak_h = top + plot_h - sy(row["peak_gops"])
        parts.append(f'<rect x="{cx-bar_w-3:.2f}" y="{sy(row["avg_gops"]):.2f}" width="{bar_w:.2f}" height="{avg_h:.2f}" fill="#2563eb"/>')
        parts.append(f'<rect x="{cx+3:.2f}" y="{sy(row["peak_gops"]):.2f}" width="{bar_w:.2f}" height="{peak_h:.2f}" fill="#f97316"/>')
        label = display_name.get(name, name)
        parts.append(f'<text transform="translate({cx:.2f} {top+plot_h+28}) rotate(35)" text-anchor="start" font-family="Arial" font-size="12" fill="#111827">{label}</text>')
        parts.append(f'<text x="{cx-bar_w/2-3:.2f}" y="{sy(row["avg_gops"])-6:.2f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#1d4ed8">{row["avg_gops"]:.0f}</text>')
        parts.append(f'<text x="{cx+bar_w/2+3:.2f}" y="{sy(row["peak_gops"])-6:.2f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#c2410c">{row["peak_gops"]:.0f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_scheduler_chart(path: Path, scheduler: dict[str, float | int | str]) -> None:
    if not scheduler:
        return
    rows = [
        ("one command buffer per small kernel", float(scheduler["naive_ms"]), float(scheduler["naive_gops"]), "#dc2626"),
        ("SP4 command-buffer batch", float(scheduler["batched_ms"]), float(scheduler["batched_gops"]), "#16a34a"),
    ]
    width, height = 900, 480
    left, right, top, bottom = 96, 32, 60, 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(row[2] for row in rows) * 1.20

    def sy(v: float) -> float:
        return top + (ymax - v) / ymax * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfc"/>',
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Inter,Arial" font-size="22" font-weight="700" fill="#1f2937">SP4 small-kernel scheduling benefit</text>',
        f'<text x="{width/2:.0f}" y="52" text-anchor="middle" font-family="Arial" font-size="13" fill="#4b5563">{int(scheduler["small_kernels"])} identical small kernels, speedup {float(scheduler["speedup"]):.2f}x</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#d1d5db"/>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        v = ymax - ymax * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{v:.1f}</text>')
    parts.append(f'<text transform="translate(24 {top+plot_h/2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15" fill="#111827">Effective GOPS</text>')
    group_w = plot_w / len(rows)
    bar_w = 112.0
    for i, (name, ms, gops, color) in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        y = sy(gops)
        h = top + plot_h - y
        parts.append(f'<rect x="{cx-bar_w/2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{cx:.2f}" y="{y-8:.2f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{gops:.1f} GOPS</text>')
        parts.append(f'<text x="{cx:.2f}" y="{top+plot_h+28}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{name}</text>')
        parts.append(f'<text x="{cx:.2f}" y="{top+plot_h+48}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{ms:.3f} ms</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_backend_gate_chart(path: Path, sp4e_gate: dict) -> None:
    decision = sp4e_gate.get("runtime_decision", {})
    enabled = bool(decision.get("enabled"))
    path_name = decision.get("path", "SP4_FALLBACK")
    reason = decision.get("reason", "no_profitable_fp4_or_lowq_acceleration")
    rows = [
        ("native FP4 tensor", 1.0 if sp4e_gate.get("native_fp4_tensor_support") else 0.0, "#2563eb"),
        ("native INT4 tensor", 1.0 if sp4e_gate.get("native_int4_tensor_support") else 0.0, "#0f766e"),
        ("lowQ ALU speedup", float(sp4e_gate.get("low_precision_alu_speedup", 1.0)), "#f97316"),
        ("SP4E/SP4 profiled", float(sp4e_gate.get("sp4e_vs_sp4_speedup", 0.0)), "#9333ea"),
    ]
    width, height = 920, 460
    left, right, top, bottom = 132, 42, 86, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    xmax = max(2.2, max(v for _, v, _ in rows) * 1.2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfc"/>',
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Inter,Arial" font-size="22" font-weight="700" fill="#1f2937">SP4E local backend gate</text>',
        f'<text x="{width/2:.0f}" y="56" text-anchor="middle" font-family="Arial" font-size="13" fill="#4b5563">decision: {path_name}, enabled={str(enabled).lower()}, reason={reason}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fff" stroke="#d1d5db"/>',
    ]
    for i in range(6):
        x = left + plot_w * i / 5
        v = xmax * i / 5
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{v:.1f}</text>')
    bar_h = 34
    gap = 24
    for i, (name, value, color) in enumerate(rows):
        y = top + 28 + i * (bar_h + gap)
        w = plot_w * min(value, xmax) / xmax
        parts.append(f'<text x="{left-12}" y="{y+22}" text-anchor="end" font-family="Arial" font-size="13" fill="#111827">{name}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.2f}" height="{bar_h}" fill="{color}"/>')
        parts.append(f'<text x="{left+w+8:.2f}" y="{y+22}" font-family="Arial" font-size="13" fill="#111827">{value:.2f}</text>')
    parts.append(f'<line x1="{left + plot_w * 2.0 / xmax:.2f}" y1="{top}" x2="{left + plot_w * 2.0 / xmax:.2f}" y2="{top+plot_h}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{left + plot_w * 2.0 / xmax + 6:.2f}" y="{top+16}" font-family="Arial" font-size="12" fill="#dc2626">2.0x lowQ threshold</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(["cmake", "-S", ".", "-B", "build-sdk"])
    run(["cmake", "--build", "build-sdk"])

    ratios = [0.0, 0.005, 0.01, 0.02, 0.025, 0.03, 0.04]
    sp4_curve = []
    sp4e_curve = []
    sp4e_gate = {}
    baseline_rows = None
    for ratio in ratios:
        asset = ROOT / f"build/rife_curve_r{int(ratio * 1000):04d}.sp4"
        sp4e_asset = ROOT / f"build/rife_curve_r{int(ratio * 1000):04d}.sp4e"
        run([
            "./build-sdk/sp4c",
            "compile",
            "--input",
            str(MODEL),
            "--output",
            str(asset),
            "--adapter",
            "rife",
            "--target",
            "metal-m4pro",
            "--block-size",
            "auto",
            "--quality-profile",
            "adaptive-rife",
            "--residual-ratio",
            str(ratio),
        ])
        sp4e_compile = run([
            "./build-sdk/sp4ec",
            "compile",
            "--input",
            str(MODEL),
            "--output",
            str(sp4e_asset),
            "--adapter",
            "rife",
            "--target",
            "cpu-reference",
            "--activation-precision",
            "fp16",
            "--quality-profile",
            "balanced",
            "--residual-ratio",
            str(ratio),
        ])
        rows = parse_quantbench(run(["./build-sdk/sp4quantbench", str(MODEL), str(asset), "--sp4e", str(sp4e_asset)]))
        sp4 = rows["SP4 A1P sparse residual"]
        sp4["residual_ratio"] = ratio
        sp4_curve.append(sp4)
        sp4e = rows["SP4E A1P CPU reference"]
        sp4e["residual_ratio"] = ratio
        sp4e_curve.append(sp4e)
        if ratio == 0.02:
            gate_match = re.search(r"runtime_gate=([^ ]+) enabled=([^ ]+) reason=([^\n]+)", sp4e_compile)
            sp4e_gate = {
                "native_fp4_tensor_support": False,
                "native_int4_tensor_support": False,
                "low_precision_alu_speedup": 1.0,
                "sp4e_vs_sp4_speedup": 0.0,
                "runtime_decision": {
                    "path": gate_match.group(1) if gate_match else "SP4_FALLBACK",
                    "enabled": gate_match.group(2) == "true" if gate_match else False,
                    "reason": gate_match.group(3) if gate_match else "no_profitable_fp4_or_lowq_acceleration",
                },
            }
        if ratio == 0.02:
            baseline_rows = rows

    int8_chase_asset = ROOT / "build/rife_int8_chase.sp4"
    run([
        "./build-sdk/sp4c",
        "compile",
        "--input",
        str(MODEL),
        "--output",
        str(int8_chase_asset),
        "--adapter",
        "rife",
        "--target",
        "metal-m4pro",
        "--block-size",
        "auto",
        "--quality-profile",
        "int8-chase",
        "--residual-ratio",
        "0.02",
    ])
    assert baseline_rows is not None
    int8_rows = parse_quantbench(run(["./build-sdk/sp4quantbench", str(MODEL), str(int8_chase_asset)]))
    int8_sp4 = dict(int8_rows["SP4 A1P sparse residual"])
    int8_sp4["residual_ratio"] = 0.02
    int8_sp4["quality_profile"] = "int8-chase"
    baseline_rows["SP4 int8-chase ZP+learned"] = int8_sp4

    sp4e_profiles = []
    for profile_name in ["int8-chase", "max-accuracy"]:
        sp4e_profile_asset = ROOT / f"build/rife_sp4e_{profile_name}.sp4e"
        run([
            "./build-sdk/sp4ec",
            "compile",
            "--input",
            str(MODEL),
            "--output",
            str(sp4e_profile_asset),
            "--adapter",
            "rife",
            "--target",
            "cpu-reference",
            "--activation-precision",
            "fp16",
            "--quality-profile",
            profile_name,
            "--residual-ratio",
            "0.01" if profile_name == "int8-chase" else "0.02",
            "--fp16-fallback-ratio",
            "auto",
        ])
        profile_rows = parse_quantbench(run(["./build-sdk/sp4quantbench", str(MODEL), str(ROOT / "build/rife_cpp.sp4"), "--sp4e", str(sp4e_profile_asset)]))
        item = dict(profile_rows["SP4E A1P CPU reference"])
        item["quality_profile"] = profile_name
        item["residual_ratio"] = 0.01 if profile_name == "int8-chase" else 1.0
        baseline_rows[f"SP4E {profile_name}"] = item
        sp4e_profiles.append(item)

    if baseline_rows and sp4e_curve:
        best_sp4e = max((p for p in sp4e_curve if p["compression"] >= 2.0), key=lambda p: p["task_acc"])
        baseline_rows["SP4E A1P CPU reference"] = dict(best_sp4e)

    metal_output = run(["./build-sdk/sp4metal-validate", "build/rife_cpp.sp4"])
    metal = parse_metal(metal_output)
    scheduler = parse_scheduler(metal_output)

    palette = ["#2563eb", "#16a34a", "#f97316", "#9333ea", "#dc2626"]
    scatter = []
    for idx, (name, row) in enumerate(baseline_rows.items()):
        if name == "SP4 A1P sparse residual":
            continue
        scatter.append({"name": name, "x": row["compression"], "y": row["task_acc"], "color": palette[idx % len(palette)]})

    chart1 = OUT_DIR / "sp4_accuracy_compression_curve.svg"
    make_svg_chart(
        chart1,
        "RIFE task-proxy accuracy vs compression",
        "Compression ratio vs FP16",
        "Task-proxy accuracy (%)",
        [{
            "name": "SP4 residual sweep",
            "color": "#0f766e",
            "points": [{"x": p["compression"], "y": p["task_acc"], "label": f"{p['residual_ratio']:.0%}"} for p in sp4_curve],
        }],
        scatter,
    )

    chart2 = OUT_DIR / "sp4_residual_tradeoff_curve.svg"
    make_svg_chart(
        chart2,
        "SP4 residual budget tradeoff",
        "Sparse residual budget (%)",
        "Compression / scaled error",
        [
            {
                "name": "compression",
                "color": "#2563eb",
                "points": [{"x": p["residual_ratio"] * 100.0, "y": p["compression"], "label": f"{p['residual_ratio']:.0%}"} for p in sp4_curve],
            },
            {
                "name": "RMSE x100",
                "color": "#dc2626",
                "points": [{"x": p["residual_ratio"] * 100.0, "y": p["rmse"] * 100.0} for p in sp4_curve],
            },
            {
                "name": "max_abs x10",
                "color": "#9333ea",
                "points": [{"x": p["residual_ratio"] * 100.0, "y": p["max_abs"] * 10.0} for p in sp4_curve],
            },
        ],
    )

    chart_sp4e = OUT_DIR / "sp4e_accuracy_comparison.svg"
    make_svg_chart(
        chart_sp4e,
        "SP4 vs SP4E CPU-reference accuracy sweep",
        "Sparse residual budget (%)",
        "Task-proxy accuracy (%)",
        [
            {
                "name": "SP4 A1P",
                "color": "#2563eb",
                "points": [{"x": p["residual_ratio"] * 100.0, "y": p["task_acc"], "label": f"{p['residual_ratio']:.0%}"} for p in sp4_curve],
            },
            {
                "name": "SP4E A1P CPU reference",
                "color": "#0f766e",
                "points": [{"x": p["residual_ratio"] * 100.0, "y": p["task_acc"], "label": f"{p['residual_ratio']:.0%}"} for p in sp4e_curve],
            },
        ],
        [
            {"name": f"SP4E {p['quality_profile']}", "x": p["residual_ratio"] * 100.0, "y": p["task_acc"], "color": "#dc2626" if p["quality_profile"] == "int8-chase" else "#9333ea"}
            for p in sp4e_profiles
        ],
    )
    chart_gate = OUT_DIR / "sp4e_backend_gate.svg"
    make_backend_gate_chart(chart_gate, sp4e_gate)

    chart3 = OUT_DIR / "sp4_throughput_comparison.svg"
    make_throughput_chart(chart3, metal)
    chart4 = OUT_DIR / "sp4_scheduler_speedup.svg"
    make_scheduler_chart(chart4, scheduler)

    report = {
        "sp4_curve": sp4_curve,
        "sp4e_curve": sp4e_curve,
        "sp4e_gate": sp4e_gate,
        "baseline_rows": baseline_rows,
        "metal_throughput": metal,
        "scheduler_bench": scheduler,
        "charts": [
            str(chart1.relative_to(ROOT)),
            str(chart2.relative_to(ROOT)),
            str(chart3.relative_to(ROOT)),
            str(chart4.relative_to(ROOT)),
            str(chart_sp4e.relative_to(ROOT)),
            str(chart_gate.relative_to(ROOT)),
        ],
    }
    (OUT_DIR / "sp4_comparison_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
