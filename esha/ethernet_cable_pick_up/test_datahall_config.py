import unittest

from config import SceneConfig, datahall_world_position


class DataHallWorldPositionTests(unittest.TestCase):
    def test_default_offset_is_plus_x_from_cable_spawn(self) -> None:
        scene = SceneConfig()
        x, y, z = datahall_world_position(scene)
        self.assertAlmostEqual(x, scene.cable_spawn_xy[0] + 1.5)
        self.assertAlmostEqual(y, scene.cable_spawn_xy[1] + 0.0)
        self.assertAlmostEqual(z, 0.0)

    def test_custom_offset_and_spawn(self) -> None:
        scene = SceneConfig(
            cable_spawn_xy=(0.74, 0.10),
            datahall_offset_from_cable_xy=(2.0, -0.5),
        )
        self.assertEqual(datahall_world_position(scene), (2.74, -0.4, 0.0))


if __name__ == "__main__":
    unittest.main()
