from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

FORBIDDEN_MODULE_MARKERS = (
    "from front_rim ",
    "import front_rim",
    "front_rim_match",
    "front_rim_stereo",
    "front_rim_sgbm",
    "front_rim_sgbm_refined",
    "live_front_plane",
    "highres_config",
    "prompt_benchmark_core",
)

PRODUCTION_FILES = (
    ROOT / "main.py",
    ROOT / "config.py",
    ROOT / "front_plane.py",
    ROOT / "stereo_geometry.py",
    ROOT / "live_control.py",
    ROOT / "benchmarks" / "front_plane_benchmark.py",
    ROOT / "benchmarks" / "capture_dataset.py",
    ROOT / "tools" / "run_benchmark_isaac.py",
    ROOT / "tools" / "run_benchmark.sh",
    ROOT / "tools" / "generate_ground_truth.py",
)

FORBIDDEN_PATHS = (
    "front_rim.py",
    "front_rim_match.py",
    "front_rim_stereo.py",
    "front_rim_sgbm.py",
    "front_rim_sgbm_refined.py",
    "live_front_plane.py",
    "main_highres.py",
    "highres_config.py",
    "benchmarks/front_rim_benchmark.py",
    "benchmarks/front_rim_benchmark_epipolar.py",
    "benchmarks/front_rim_sgbm_diagnostic.py",
    "benchmarks/front_rim_sgbm_benchmark.py",
    "benchmarks/front_rim_sgbm_refined_benchmark.py",
    "benchmarks/front_rim_sgbm_highres_benchmark.py",
    "benchmarks/prompt_benchmark_capture.py",
    "benchmarks/prompt_benchmark_capture_highres.py",
    "benchmarks/prompt_benchmark_core.py",
    "benchmarks/prompt_benchmark_evaluate.py",
    "benchmarks/prompt_benchmark_evaluate_isaac_bootstrap.py",
    "benchmarks/run_prompt_ab_benchmark.py",
    "tools/run_front_rim_benchmark_isaac.py",
    "tools/run_front_rim_benchmark.sh",
    "tools/run_front_rim_ground_truth.sh",
    "tools/extract_front_rim_ground_truth_bootstrap.py",
    "tools/run_center_diagnostic_isaac.py",
    "tools/diagnose_stereo_centers.py",
    "front_mouth_projective_center.py",
    "lower_mouth_projective_center.py",
    "stereo_center_projective.py",
    "stereo_front_rim_plane.py",
    "stereo_center.py",
    "tests/test_front_mouth_outer_edges.py",
    "tests/test_lower_mouth_projective_center.py",
    "tests/test_projective_front_rim_center.py",
    "tests/test_stereo_front_rim_plane.py",
    "tests/test_stereo_center.py",
)


class RepositoryCleanlinessTests(unittest.TestCase):
    def test_production_imports_no_legacy_modules(self):
        for path in PRODUCTION_FILES:
            self.assertTrue(path.is_file(), path)
            source = path.read_text(encoding="utf-8")
            for marker in FORBIDDEN_MODULE_MARKERS:
                self.assertNotIn(marker, source, f"{marker!r} found in {path}")

    def test_legacy_paths_are_deleted(self):
        for relative in FORBIDDEN_PATHS:
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_generated_outputs_and_worktrees_are_ignored(self):
        source = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (
            ".worktrees/",
            "single_rack_cv/camera_output/",
            "single_rack_cv/assets/models/*.pt",
            "single_rack_cv/benchmarks/front_plane_ground_truth.json",
        ):
            self.assertIn(entry, source)


if __name__ == "__main__":
    unittest.main()
