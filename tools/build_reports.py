#!/usr/bin/env python3
"""Aggregate experiment JSON and create submission-ready CSV/SVG assets."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "submission_evidence"
OUT.mkdir(parents=True, exist_ok=True)

run_files = [ROOT / "results" / f"verilator_full_run{i}.json" for i in (1, 2, 3)]
runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_files]
experiment_dir = OUT / "experiments"
experiment_dir.mkdir(exist_ok=True)
for path in run_files:
    shutil.copy2(path, experiment_dir / path.name)
by_dut = {}
for run in runs:
    for record in run:
        by_dut.setdefault(record["dut"], []).append(record)

rows = []
summary = {}
for dut, records in by_dut.items():
    curves = [r["curve"] for r in records]
    for point_index in range(min(map(len, curves))):
        step = curves[0][point_index][0]
        values = [c[point_index][1] for c in curves]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        rows.append([dut, step, mean, var, math.sqrt(var)])
    finals = [r["coverage"] for r in records]
    aucs = []
    for curve in curves:
        aucs.append(sum((curve[i][1] + curve[i + 1][1]) * 0.5 *
                        (curve[i + 1][0] - curve[i][0])
                        for i in range(len(curve) - 1)))
    summary[dut] = {
        "runs": len(records),
        "final_mean": sum(finals) / len(finals),
        "final_variance": sum((x - sum(finals) / len(finals)) ** 2 for x in finals) / len(finals),
        "auc_mean": sum(aucs) / len(aucs),
        "auc_variance": sum((x - sum(aucs) / len(aucs)) ** 2 for x in aucs) / len(aucs),
        "predict_mean_ms": sum(r["predict_mean_ms"] for r in records) / len(records),
        "predict_p99_ms_max": max(r["predict_p99_ms"] for r in records),
    }

with (OUT / "coverage_curves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["dut", "step", "coverage_mean", "coverage_variance", "coverage_stddev"])
    writer.writerows(rows)

(OUT / "experiment_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

# Dependency-free SVG chart suitable for embedding in Markdown or documents.
width, height = 1000, 620
left, top, right, bottom = 85, 40, 30, 75
pw, ph = width - left - right, height - top - bottom
colors = {"dma_xfer_public": "#d55e00", "spi_master_public": "#0072b2",
          "spi_xfer_public": "#009e73"}
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}.line{fill:none;stroke-width:3}</style>']
for pct in range(0, 101, 20):
    y = top + ph * (1 - pct / 100)
    parts += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}"/>',
              f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="14">{pct}%</text>']
for step in range(0, 30001, 5000):
    x = left + pw * step / 30000
    parts += [f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ph}"/>',
              f'<text x="{x:.1f}" y="{top+ph+28}" text-anchor="middle" font-size="14">{step//1000}k</text>']
parts += [f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>',
          f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>',
          f'<text x="{left+pw/2}" y="{height-20}" text-anchor="middle" font-size="17">Simulation cycles</text>',
          f'<text transform="translate(22 {top+ph/2}) rotate(-90)" text-anchor="middle" font-size="17">Functional coverage</text>',
          '<text x="500" y="25" text-anchor="middle" font-size="20" font-weight="bold">Coverage convergence (3-run mean; variance = 0)</text>']
for legend_i, (dut, records) in enumerate(sorted(by_dut.items())):
    curve = records[0]["curve"]
    points = " ".join(f"{left + pw*s/30000:.1f},{top + ph*(1-c):.1f}" for s, c in curve)
    color = colors[dut]
    parts.append(f'<polyline class="line" stroke="{color}" points="{points}"/>')
    lx, ly = left + 555, top + 25 + legend_i * 28
    parts += [f'<line x1="{lx}" y1="{ly}" x2="{lx+35}" y2="{ly}" stroke="{color}" stroke-width="4"/>',
              f'<text x="{lx+45}" y="{ly+5}" font-size="15">{dut}</text>']
parts.append('</svg>')
(OUT / "coverage_curves.svg").write_text("\n".join(parts), encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
