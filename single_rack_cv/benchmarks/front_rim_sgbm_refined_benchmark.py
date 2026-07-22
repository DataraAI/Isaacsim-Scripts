#!/usr/bin/env python3
"""Run the frozen benchmark with refined strict local SGBM geometry."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math

import numpy as np

from benchmarks import front_rim_sgbm_benchmark as base
from front_rim_sgbm_refined import estimate_front_plane_sgbm_refined


STRICT_PLANE_RESIDUAL_P95_MM = 0.50


def _finite_percentile(values: list[float], q: float) -> float:
    finite = np.asarray(
        [value for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.percentile(finite, q)) if finite.size else math.nan


def _rewrite_report(summary: dict[str, object]) -> None:
    keys = (
        "pair_success_rate",
        "track_switch_count",
        "radial_jitter_mm",
        "ray_gap_p95_mm",
        "plane_residual_p95_mm",
        "plane_error_median_mm",
        "plane_error_p95_mm",
        "sgbm_valid_count_median",
        "sgbm_consistent_count_median",
        "sgbm_ring_count_median",
        "sgbm_triangulated_count_median",
        "sgbm_cluster_count_median",
        "sgbm_median_disparity_px",
        "QUALIFIED",
    )
    lines = ["FRONT-RIM REFINED LOCAL SGBM BENCHMARK SUMMARY"]
    lines.extend(f"{key}={summary.get(key)}" for key in keys)
    lines.extend(
        (
            "",
            "Strict qualification:",
            "  pair_success_rate>=0.95",
            "  track_switch_count=0",
            "  radial_jitter_mm<=0.5",
            "  ray_gap_p95_mm<=0.5",
            "  plane_residual_p95_mm<=0.5",
            "  plane_error_median_mm<=0.5",
            "  plane_error_p95_mm<=1.0",
            "",
            "Rejection counts:",
        )
    )
    for reason, count in summary.get("rejection_counts", {}).items():
        lines.append(f"  {int(count):3d}  {reason}")
    text = "\n".join(lines) + "\n"
    (base.OUTPUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> int:
    base.estimate_front_plane_sgbm = estimate_front_plane_sgbm_refined

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        base_status = int(base.main())

    summary_path = base.OUTPUT_DIR / "summary.json"
    details_path = base.OUTPUT_DIR / "details.csv"
    if not summary_path.is_file() or not details_path.is_file():
        print(captured.getvalue(), end="", flush=True)
        raise RuntimeError(
            "Refined SGBM benchmark did not write summary/details output."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with details_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    residuals: list[float] = []
    for row in rows:
        if str(row.get("pair_success", "")).lower() != "true":
            continue
        try:
            residuals.append(float(row["plane_residual_mm"]))
        except (KeyError, TypeError, ValueError):
            continue

    plane_residual_p95_mm = _finite_percentile(residuals, 95.0)
    base_qualified = bool(summary.get("QUALIFIED", False))
    strict_qualified = bool(
        base_qualified
        and math.isfinite(plane_residual_p95_mm)
        and plane_residual_p95_mm <= STRICT_PLANE_RESIDUAL_P95_MM
    )

    summary.update(
        {
            "schema_version": 5,
            "mode": "local_sgbm_refined_v7",
            "base_status_before_postprocess": base_status,
            "plane_residual_p95_mm": plane_residual_p95_mm,
            "strict_plane_residual_p95_gate_mm": (
                STRICT_PLANE_RESIDUAL_P95_MM
            ),
            "QUALIFIED": strict_qualified,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _rewrite_report(summary)
    return 0 if strict_qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
