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
        if "throughput_scheme=" in line:
            scheme = re.search(r"throughput_scheme=(.*?) avg_ms=", line)
            avg = re.search(r"avg_gops=([0-9.eE+-]+)", line)
            peak = re.search(r"peak_gops=([0-9.eE+-]+)", line)
            if scheme and avg and peak:
                rows[scheme.group(1)] = {"avg_gops": float(avg.group(1)), "peak_gops": float(peak.group(1))}
    return rows


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
    ]
    rows = [(name, metal[name]) for name in preferred if name in metal]
    if not rows:
        return
    width, height = 980, 560
    left, right, top, bottom = 96, 32, 56, 128
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
        f'<text x="{width/2:.0f}" y="30" text-anchor="middle" font-family="Inter,Arial" font-size="22" font-weight="700" fill="#1f2937">Metal throughput by quantization scheme</text>',
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
        parts.append(f'<text x="{cx:.2f}" y="{top+plot_h+18}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{name}</text>')
        parts.append(f'<text x="{cx-bar_w/2-3:.2f}" y="{sy(row["avg_gops"])-6:.2f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#1d4ed8">{row["avg_gops"]:.0f}</text>')
        parts.append(f'<text x="{cx+bar_w/2+3:.2f}" y="{sy(row["peak_gops"])-6:.2f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#c2410c">{row["peak_gops"]:.0f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(["cmake", "-S", ".", "-B", "build-sdk"])
    run(["cmake", "--build", "build-sdk"])

    ratios = [0.0, 0.005, 0.01, 0.02, 0.04]
    sp4_curve = []
    baseline_rows = None
    for ratio in ratios:
        asset = ROOT / f"build/rife_curve_r{int(ratio * 1000):04d}.sp4"
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
        rows = parse_quantbench(run(["./build-sdk/sp4quantbench", str(MODEL), str(asset)]))
        sp4 = rows["SP4 A1P sparse residual"]
        sp4["residual_ratio"] = ratio
        sp4_curve.append(sp4)
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

    metal = parse_metal(run(["./build-sdk/sp4metal-validate", "build/rife_cpp.sp4"]))

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

    chart3 = OUT_DIR / "sp4_throughput_comparison.svg"
    make_throughput_chart(chart3, metal)

    report = {
        "sp4_curve": sp4_curve,
        "baseline_rows": baseline_rows,
        "metal_throughput": metal,
        "charts": [str(chart1.relative_to(ROOT)), str(chart2.relative_to(ROOT)), str(chart3.relative_to(ROOT))],
    }
    (OUT_DIR / "sp4_comparison_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
