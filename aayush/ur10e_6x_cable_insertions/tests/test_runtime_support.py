import importlib
import unittest
import numpy as np


def load_support():
    try:
        return importlib.import_module("ur10e_6x_cable_insertions.runtime_support")
    except ModuleNotFoundError as exc:
        raise AssertionError("runtime_support module is missing") from exc


def physical_grasp_is_valid(**kwargs):
    callback = getattr(load_support(), "physical_grasp_is_valid", None)
    if callback is None:
        raise AssertionError("physical_grasp_is_valid is missing")
    return callback(**kwargs)


class StageUnitTests(unittest.TestCase):
    def test_centimeter_stage_round_trip(self):
        support = load_support()
        metres = np.array([0.5783, -1.35, 3.1366])
        stage = support.meters_to_stage(metres, 0.01)
        np.testing.assert_allclose(stage, [57.83, -135.0, 313.66])
        np.testing.assert_allclose(support.stage_to_meters(stage, 0.01), metres)

    def test_meter_stage_is_identity(self):
        support = load_support()
        point = np.array([1.0, -2.0, 3.0])
        np.testing.assert_allclose(support.meters_to_stage(point, 1.0), point)
        np.testing.assert_allclose(support.stage_to_meters(point, 1.0), point)

    def test_invalid_stage_units_are_rejected(self):
        support = load_support()
        for value in (0.0, -0.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                support.validate_meters_per_unit(value)


class PhysicalGraspTests(unittest.TestCase):
    def test_stationary_cable_fails_even_with_closed_fingers(self):
        self.assertFalse(physical_grasp_is_valid(
            initial_part=np.array([0.7, -1.35, 3.06]),
            current_part=np.array([0.7, -1.35, 3.06]),
            expected_tip=np.array([0.7, -1.35, 3.06]),
            fingers=np.array([0.82]),
            min_lift_m=0.04,
            max_tip_error_m=0.06,
            contact_rad=0.21,
        ))

    def test_lifted_cable_near_tool_passes(self):
        self.assertTrue(physical_grasp_is_valid(
            initial_part=np.array([0.7, -1.35, 3.06]),
            current_part=np.array([0.7, -1.35, 3.12]),
            expected_tip=np.array([0.7, -1.35, 3.12]),
            fingers=np.array([0.82]),
            min_lift_m=0.04,
            max_tip_error_m=0.06,
            contact_rad=0.21,
        ))

    def test_nonzero_x_offset_requires_grasp_tip_not_part_center(self):
        support = load_support()
        grasp_tip_from_part = getattr(support, "grasp_tip_from_part", None)
        if grasp_tip_from_part is None:
            raise AssertionError("grasp_tip_from_part is missing")

        part_center = np.array([0.7, -1.35, 3.12])
        x_offset_m = 0.05
        grasp_tip = grasp_tip_from_part(part_center, x_offset_m)
        np.testing.assert_allclose(grasp_tip, [0.75, -1.35, 3.12])

        initial = np.array([0.7, -1.35, 3.06])
        expected_tip = grasp_tip.copy()
        fingers = np.array([0.82])
        kwargs = dict(
            initial_part=initial,
            expected_tip=expected_tip,
            fingers=fingers,
            min_lift_m=0.04,
            max_tip_error_m=0.03,
            contact_rad=0.21,
        )

        self.assertTrue(physical_grasp_is_valid(current_part=grasp_tip, **kwargs))
        self.assertFalse(physical_grasp_is_valid(current_part=part_center, **kwargs))
