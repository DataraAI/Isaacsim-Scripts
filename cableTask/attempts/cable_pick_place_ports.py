"""
Franka sequential pick-and-place through datacenter QSFP port prims (Isaac Sim 4.5.0).

Uses PickPlaceController per:
https://docs.isaacsim.omniverse.nvidia.com/4.5.0/core_api_tutorials/tutorial_core_adding_manipulator.html#using-the-pickandplace-controller

By default thin, long DynamicCylinder cables are used: cable1 → first port, cable2 → second port, etc.
A new cylinder is spawned at the spawn pose after each place. Set CABLE_URDF_PATH to use a single URDF
cable instead. Port goals are taken from PORT_PRIM_PATH_LIST (each element is a connector prim path).
"""

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.core.api.objects import DynamicCylinder
import omni.kit.commands
import omni.usd

from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd
import numpy as np
from typing import List, Optional, Tuple

import carb

# --- Scene / task configuration (tune for your machine and asset paths) ---

USD_PATH = r"/home/advaith/Downloads/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"

# When set, cable is loaded from URDF instead of placeholder cylinders (no per-port spawn).
CABLE_URDF_PATH = ""

# If the cable URDF exposes a specific link for grasping, set its prim path; otherwise the URDF import
# root prim path returned by URDFImportRobot is used for pick position sampling.
CABLE_PICK_PRIM_PATH = ""  # e.g. "/World/.../cable_link"

# Offset added to each port world position for the place goal (e.g. approach along normal / height).
PLACE_POSITION_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Franka: same distance_scale as baseline_franka_hello_world URDF import (import_config.distance_scale).
# For the extension Franka prim, match that via uniform local scale (see baseline_follow_target_datacenter_ik).
FRANKA_DISTANCE_SCALE = 57.0

# Franka world pose in the datacenter (same ballpark as baseline_franka_hello_world).
FRANKA_WORLD_POSITION = np.array([30.0, -90.0, 150.0], dtype=np.float64)
FRANKA_WORLD_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

# Placeholder cable: thin (x,y) and long (z) cylinder when CABLE_URDF_PATH is empty (tune for your stage).
CABLE_WORLD_POSITION = np.array([28.0, -92.0, 141.85], dtype=np.float64)
CABLE_CYLINDER_SCALE = np.array([0.012, 0.012, 0.45], dtype=np.float64)

num_quads = 4
num_pairs = 4
num_conn_a = 2

PORT_BASE_PRIM_PATH = (
    "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base"
    "/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01"
)

PORT_PRIM_PATH_LIST: List[str] = []
for q in range(1, num_quads + 1):
    for p in range(1, num_pairs + 1):
        for a in range(1, num_conn_a + 1):
            suffix = f"/Connector_Quad_{q:02d}/Connector_Pair_{p:02d}/QSFP_DD_Connector_A_{a:02d}"
            PORT_PRIM_PATH_LIST.append(PORT_BASE_PRIM_PATH + suffix)


def get_world_transform_xform(prim: Usd.Prim) -> Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """World translation, rotation, and scale for a prim (same helper as baseline)."""
    world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale


def _import_cable_urdf(urdf_path: str) -> str:
    """Import cable URDF; returns prim path for the pick point (CABLE_PICK_PRIM_PATH or import root)."""
    import_config = _urdf.ImportConfig()
    import_config.convex_decomp = False
    import_config.fix_base = False
    import_config.make_default_prim = False
    import_config.self_collision = False
    import_config.distance_scale = 1.0
    import_config.density = 0.0

    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=urdf_path,
        import_config=import_config,
    )
    if not result:
        carb.log_warn(f"URDFParseFile failed for {urdf_path}")
    result, prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )
    carb.log_info(f"Cable URDF imported at prim_path={prim_path}")
    return CABLE_PICK_PRIM_PATH if CABLE_PICK_PRIM_PATH else prim_path


def _delete_prims_at_paths(prim_paths: List[str]) -> None:
    """Remove prims from the active stage (used to clear cable1, cable2, ... on reset)."""
    for prim_path in prim_paths:
        omni.kit.commands.execute("DeletePrims", paths=[Sdf.Path(prim_path)])


def _port_place_position(port_prim_path: str) -> np.ndarray:
    port_prim = omni.usd.get_context().get_stage().GetPrimAtPath(port_prim_path)
    if not port_prim.IsValid():
        carb.log_error(f"Port prim missing: {port_prim_path}")
        return np.zeros(3, dtype=np.float64)
    pos = get_world_transform_xform(port_prim)[0]
    return np.array(pos, dtype=np.float64) + PLACE_POSITION_OFFSET


class CablePickPlacePorts(BaseSample):
    """Pick cable (or surrogate), place at PORT_PRIM_PATH_LIST[i], repeat until all ports are done."""

    def __init__(self) -> None:
        super().__init__()
        self._cable_pick_prim_path: Optional[str] = None
        self._cable_object = None  # DynamicCylinder or None when using URDF prim-only cable
        self._spawned_cable_paths: List[str] = []
        self._port_index = 0

    def _spawn_cylinder_cable(self, world, index: int):
        """Spawn numbered cylinder: prim /World/cable{n}, scene name cable{n} (n is 1-based)."""
        prim_path = f"/World/cable{index}"
        name = f"cable{index}"
        cyl = world.scene.add(
            DynamicCylinder(
                prim_path=prim_path,
                name=name,
                position=CABLE_WORLD_POSITION,
                scale=CABLE_CYLINDER_SCALE,
                color=np.array([0.2, 0.6, 0.9]),
            )
        )
        self._spawned_cable_paths.append(prim_path)
        return cyl

    def _apply_franka_distance_scale_and_pose(self):
        s = float(FRANKA_DISTANCE_SCALE)
        self._franka.set_local_scale(np.array([s, s, s], dtype=np.float64))
        self._franka.set_world_pose(
            position=FRANKA_WORLD_POSITION,
            orientation=FRANKA_WORLD_ORIENTATION,
        )

    def setup_scene(self):
        world = self.get_world()
        add_reference_to_stage(USD_PATH, "/World/Datacenter")

        world.scene.add(
            Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"),
        )

        if CABLE_URDF_PATH:
            self._cable_pick_prim_path = _import_cable_urdf(CABLE_URDF_PATH)
            self._cable_object = None
            self._spawned_cable_paths = []
        else:
            self._spawned_cable_paths = []
            self._cable_object = self._spawn_cylinder_cable(world, 1)
            self._cable_pick_prim_path = None

        self._port_index = 0
        return

    def _cable_pick_position(self) -> np.ndarray:
        if self._cable_object is not None:
            pos, _ = self._cable_object.get_world_pose()
            return np.array(pos, dtype=np.float64)
        prim_path = self._cable_pick_prim_path
        assert prim_path is not None
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        pos = get_world_transform_xform(prim)[0]
        return np.array(pos, dtype=np.float64)

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._apply_franka_distance_scale_and_pose()

        self._controller = PickPlaceController(
            name="cable_pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        self._controller.reset()
        self._port_index = 0
        self._apply_franka_distance_scale_and_pose()
        if not CABLE_URDF_PATH:
            if self._spawned_cable_paths:
                _delete_prims_at_paths(list(self._spawned_cable_paths))
                self._spawned_cable_paths.clear()
            self._cable_object = self._spawn_cylinder_cable(self._world, 1)
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        if self._port_index >= len(PORT_PRIM_PATH_LIST):
            return

        picking_position = self._cable_pick_position()
        placing_position = _port_place_position(PORT_PRIM_PATH_LIST[self._port_index])
        current_joint_positions = self._franka.get_joint_positions()

        actions = self._controller.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=current_joint_positions,
        )
        self._franka.apply_action(actions)

        if self._controller.is_done():
            self._port_index += 1
            self._controller.reset()
            self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
            if self._port_index >= len(PORT_PRIM_PATH_LIST):
                self._world.pause()
            elif not CABLE_URDF_PATH:
                # Previous cable stays at the port; spawn the next numbered cylinder for the next port.
                self._cable_object = self._spawn_cylinder_cable(self._world, self._port_index + 1)
        return

    def world_cleanup(self):
        return
