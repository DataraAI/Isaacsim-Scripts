#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "prompt_benchmark_core.py"


def load_core():
    if not MODULE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("prompt_benchmark_core", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptBenchmarkCoreTests(unittest.TestCase):
    def require_core(self):
        core = load_core()
        self.assertIsNotNone(core, "benchmarks/prompt_benchmark_core.py has not been implemented")
        return core

    def test_rms_radial_jitter_uses_median_center(self):
        core = self.require_core()
        value = core.rms_radial_jitter([[0.0, 0.0], [2.0, 0.0]])
        self.assertAlmostEqual(value, 1.0)

    def test_track_switch_counts_frame_when_either_eye_jumps(self):
        core = self.require_core()
        records = [
            {"pair_success": True, "left_center_u": 10.0, "left_center_v": 20.0, "right_center_u": 30.0, "right_center_v": 20.0},
            {"pair_success": True, "left_center_u": 11.0, "left_center_v": 20.0, "right_center_u": 31.0, "right_center_v": 20.0},
            {"pair_success": True, "left_center_u": 90.0, "left_center_v": 20.0, "right_center_u": 110.0, "right_center_v": 20.0},
        ]
        self.assertEqual(core.count_track_switches(records, 45.0), 1)

    def test_summary_rates_and_quality_gates(self):
        core = self.require_core()
        records = []
        for index in range(20):
            records.append({
                "frame_index": index + 1,
                "left_success": True,
                "right_success": True,
                "pair_success": index != 19,
                "inference_ms": 20.0 + index * 0.01,
                "left_center_u": 100.0 + index * 0.01,
                "left_center_v": 200.0,
                "right_center_u": 80.0 + index * 0.01,
                "right_center_v": 200.0,
                "center_world_x": 1.0 + index * 1.0e-6,
                "center_world_y": 2.0,
                "center_world_z": 3.0,
                "ray_gap_mm": 0.1,
                "center_error_px": 1.0,
            })
        summary = core.summarize_records("A_five_scale", records, 20, 45.0)
        self.assertAlmostEqual(summary["left_success_rate"], 1.0)
        self.assertAlmostEqual(summary["pair_success_rate"], 0.95)
        self.assertEqual(summary["track_switch_count"], 0)
        self.assertTrue(summary["base_quality_pass"])

    def test_manifest_requires_exact_frame_count_and_unique_indices(self):
        core = self.require_core()
        manifest = {
            "schema_version": 1,
            "frame_count": 2,
            "frames": [
                {"frame_index": 1, "left_image": "a.png", "right_image": "b.png"},
                {"frame_index": 2, "left_image": "c.png", "right_image": "d.png"},
            ],
        }
        core.validate_manifest(manifest, 2)
        manifest["frames"][1]["frame_index"] = 1
        with self.assertRaises(ValueError):
            core.validate_manifest(manifest, 2)

    def test_relative_speed_gate_and_winner(self):
        core = self.require_core()
        summaries = [
            {"strategy": "A_five_scale", "base_quality_pass": True, "pair_success_rate": 0.98, "track_switch_count": 0, "center_3d_jitter_mm": 0.20, "inference_median_ms": 100.0},
            {"strategy": "B_single_runtime_scale", "base_quality_pass": True, "pair_success_rate": 0.98, "track_switch_count": 0, "center_3d_jitter_mm": 0.15, "inference_median_ms": 110.0},
        ]
        gated = core.apply_relative_speed_gate(summaries, 1.25)
        self.assertTrue(all(item["qualified"] for item in gated))
        self.assertEqual(core.choose_winner(gated), "B_single_runtime_scale")

    def test_no_winner_when_neither_strategy_qualifies(self):
        core = self.require_core()
        summaries = [
            {"strategy": "A", "qualified": False, "pair_success_rate": 0.90, "track_switch_count": 0, "center_3d_jitter_mm": 0.1, "inference_median_ms": 10.0},
            {"strategy": "B", "qualified": False, "pair_success_rate": 0.99, "track_switch_count": 2, "center_3d_jitter_mm": 0.1, "inference_median_ms": 9.0},
        ]
        self.assertIsNone(core.choose_winner(summaries))


if __name__ == "__main__":
    unittest.main()
