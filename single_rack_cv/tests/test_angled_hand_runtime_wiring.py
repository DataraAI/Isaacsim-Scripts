from __future__ import annotations

from pathlib import Path
import unittest
from robot.angled_hand_config import ANGLED_HAND_CONFIG

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "runtime" / "angled_hand_runtime.py"
HANDOFF_RUNTIME_PATH = ROOT / "runtime" / "stereo_handoff_runtime.py"
MAIN_PATH = ROOT / "main.py"


class AngledHandRuntimeWiringTests(unittest.TestCase):
    def test_shared_geometry_defaults_match_requested_pose(self):
        self.assertEqual(
            ANGLED_HAND_CONFIG.hand_downward_pitch_deg,
            30.0,
        )
        self.assertEqual(
            ANGLED_HAND_CONFIG.palm_side_tolerance_deg,
            1.0,
        )
        self.assertEqual(
            ANGLED_HAND_CONFIG.plug_body_length_m,
            0.036152,
        )

    def test_main_selects_the_angled_stereo_handoff_runtime(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        handoff_source = HANDOFF_RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "AngledHandStereoHandoffRuntime as "
            "CableMountedSimulationRuntime",
            main_source,
        )
        self.assertIn("runtime = CableMountedSimulationRuntime(", main_source)
        self.assertIn(
            "from runtime.angled_hand_runtime import AngledHandCableRuntime",
            handoff_source,
        )
        self.assertIn(
            "class AngledHandStereoHandoffRuntime(\n"
            "    AngledHandCableRuntime\n"
            "):",
            handoff_source,
        )

    def test_runtime_solves_hand_pose_and_tool_transform_together(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        solve_index = source.index(
            "compute_angled_hand_pose_preserving_tool("
        )
        super_index = source.index(
            "super().__init__(\n            simulation_app=simulation_app,"
        )
        self.assertLess(solve_index, super_index)
        self.assertIn(
            "initial_position=tuple(",
            source,
        )
        self.assertIn(
            "initial_orientation_wxyz=angled_hand_orientation",
            source,
        )
        self.assertIn(
            "tool_center_local_position_m=tuple(",
            source,
        )
        self.assertIn(
            "tool_center_local_orientation_wxyz=(",
            source,
        )
        self.assertIn("ExplicitInsertionAxisAdapter", source)

    def test_runtime_recenters_rear_and_preserves_camera_to_plug_extrinsics(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "recenter_horizontal_plug_rear_in_pitched_hand(",
            source,
        )
        self.assertIn("camera=replace(", source)
        self.assertIn("left_local_position=left_camera_position", source)
        self.assertIn("right_local_position=right_camera_position", source)
        self.assertIn("virtual_local_position=virtual_camera_position", source)
        self.assertIn("camera-to-plug calibration: preserved exactly", source)
        self.assertIn("RJ45 rear centered between fingers", source)

    def test_runtime_preserves_plug_target_instead_of_local_roll_guess(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("preserved plug-tip target", source)
        self.assertIn("solved hand target", source)
        self.assertNotIn("palm_roll_deg", source)
        self.assertNotIn("compute_pitched_hand_from_tool_rotation", source)

    def test_runtime_rejects_wrong_palm_side(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("palm_roll_error_deg", source)
        self.assertIn("palm_side_tolerance_deg", source)
        self.assertIn(
            "palm side does not match the previous working pose",
            source,
        )

    def test_runtime_logs_post_settle_geometry_once(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("ANGLED HAND GEOMETRY VALIDATED", source)
        self.assertIn("self._geometry_success_logged", source)

    def test_runtime_separates_live_plug_axis_from_tool_orientation(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("def _live_plug_tip_and_axis", source)
        self.assertIn(
            "self._insertion_axis_adapter.set_axis_world",
            source,
        )
        self.assertIn("def _sample_mount_validation_live", source)
        self.assertIn("plug_horizontal_error_deg", source)

    def test_runtime_does_not_override_camera_poses(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("left_camera_sensor.set", source)
        self.assertNotIn("right_camera_sensor.set", source)
        self.assertNotIn("camera_prim.Set", source)


if __name__ == "__main__":
    unittest.main()
