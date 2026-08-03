#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "stereo_handoff_runtime.py"
SETTLED_RUNTIME_PATH = ROOT / "settled_stereo_handoff_runtime.py"
MAIN_PATH = ROOT / "main.py"
CONFIG_PATH = ROOT / "config.py"
DEBUG_PATH = ROOT / "debug.py"


class StereoHandoffWiringTests(unittest.TestCase):
    def test_main_selects_handoff_runtime(self):
        source = MAIN_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "AngledHandStereoHandoffRuntime "
            "as CableMountedSimulationRuntime",
            source,
        )

    def test_runtime_qualifies_stationary_port_before_any_motion(self):
        source = RUNTIME_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn("qualify_stationary_port_goal", source)
        self.assertIn("observation.center_world_xyz_m", source)
        self.assertIn("actual_position_before + correction_world_m", source)
        self.assertIn("_frozen_port_point_world_m", source)
        self.assertIn("STATIONARY PORT POSE QUALIFIED", source)
        self.assertIn("state.acquired", source)

        observe_start = source.index("def observe_visual_servo(")
        failure_start = source.index("def note_perception_failure(")
        observe_source = source[observe_start:failure_start]
        self.assertNotIn(
            "super().observe_visual_servo(",
            observe_source,
            "Stationary qualification must not issue the old continuous "
            "visual-servo step before freezing the port pose.",
        )

    def test_runtime_stops_camera_after_qualified_goal(self):
        source = RUNTIME_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn("if self._handoff_active", source)
        self.assertIn("self._advance_handoff_if_settled()", source)
        self.assertIn("return False", source)
        self.assertIn("bounded_step_to_goal", source)
        self.assertIn("_MAXIMUM_HANDOFF_STEP_M = 0.005", source)
        self.assertIn("destination: frozen physical port pose", source)

    def test_runtime_preserves_stationary_evidence_across_rejected_frames(self):
        source = RUNTIME_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn("accepted stationary 3D evidence", source)
        self.assertIn("state.left_reference = None", source)
        self.assertIn("state.right_reference = None", source)
        self.assertNotIn("state.acquisition_features.clear()", source)

    def test_frozen_port_marker_is_wired(self):
        debug_source = DEBUG_PATH.read_text(encoding="utf-8")
        main_source = MAIN_PATH.read_text(encoding="utf-8")

        self.assertIn("update_frozen_port_point", debug_source)
        self.assertIn('"/World/FrozenPortPoint"', debug_source)
        self.assertIn("runtime.frozen_port_point_world_m", main_source)

    def test_camera_remains_on_last_visible_mount(self):
        source = CONFIG_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "local_y_rotation_deg: float = 186.0248",
            source,
        )
        self.assertIn(
            "        0.04,\n"
            "        -0.020,\n"
            "        0.025,",
            source,
        )
        self.assertIn(
            "        0.04,\n"
            "        +0.020,\n"
            "        0.025,",
            source,
        )

    def test_live_runtime_uses_tight_quiet_fine_settling(self):
        source = SETTLED_RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn("fine_settle_tolerance_m=0.0001", source)
        self.assertIn("fine_required_settled_frames=10", source)
        self.assertIn("fine_max_motion_per_frame_m=0.00003", source)
        self.assertIn("FINE INSERTION SETTLING", source)


if __name__ == "__main__":
    unittest.main()
