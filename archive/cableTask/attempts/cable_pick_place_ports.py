"""
Franka sequential pick-and-place through datacenter QSFP port prims (Isaac Sim 4.5.0).

PickPlaceController + task observations follow:
https://docs.isaacsim.omniverse.nvidia.com/4.5.0/core_api_tutorials/tutorial_core_adding_manipulator.html#use-the-pick-and-place-task

CablePortPickPlaceTask mirrors the built-in PickPlace task pattern (get_params / get_observations with
position + target_position) so physics_step can use world.get_observations() like the tutorial cube example.
"""

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.api.tasks import BaseTask
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.core.api.objects import DynamicCylinder
import omni.kit.commands
import omni.usd

from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd, PhysxSchema
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import carb

# --- Scene / task configuration (tune for your machine and asset paths) ---

USD_PATH = r"/home/advaith/Downloads/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"

CABLE_PORT_TASK_NAME = "cable_port_task"

# When set, cable is loaded from URDF instead of placeholder cylinders (no per-port spawn).
CABLE_URDF_PATH = ""

# If the cable URDF exposes a specific link for grasping, set its prim path; otherwise the URDF import
# root prim path returned by URDFImportRobot is used for pick position sampling.
CABLE_PICK_PRIM_PATH = ""  # e.g. "/World/.../cable_link"

# Offset added to each port world position for the place goal (e.g. approach along normal / height).
PLACE_POSITION_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# Franka: same distance_scale as baseline_franka_hello_world URDF import (import_config.distance_scale).
FRANKA_DISTANCE_SCALE = 57.0

# Franka world pose in the datacenter (same ballpark as baseline_franka_hello_world).
FRANKA_WORLD_POSITION = np.array([30.0, -90.0, 150.0], dtype=np.float64)
FRANKA_WORLD_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

# Cable geometry (stage units): diameter / length define the cylinder; CABLE_CYLINDER_SCALE scales each axis.
CABLE_DIAMETER = 0.1
CABLE_LENGTH = 1.0
CABLE_CYLINDER_SCALE = np.array([1.0, 1.0, 1.0], dtype=np.float64)

# World position where each spawned cylinder cable is placed (same units as the stage).
CABLE_WORLD_POSITION = np.array([10.0, -90.0, 170.0], dtype=np.float64)

# Observation key for pick target (same as PickPlace task tutorial uses for the cube).
PICK_TARGET_OBSERVATION_KEY = "fancy_cube"

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


def _apply_franka_scale_pose(franka) -> None:
    s = float(FRANKA_DISTANCE_SCALE)
    franka.set_local_scale(np.array([s, s, s], dtype=np.float64))
    franka.set_world_pose(
        position=FRANKA_WORLD_POSITION,
        orientation=FRANKA_WORLD_ORIENTATION,
    )


def _cable_dynamic_cylinder_scale_vector() -> np.ndarray:
    r = CABLE_DIAMETER * 0.5
    h = CABLE_LENGTH
    s = CABLE_CYLINDER_SCALE
    return np.array([r * s[0], r * s[1], h * s[2]], dtype=np.float64)


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
    for prim_path in prim_paths:
        omni.kit.commands.execute("DeletePrims", paths=[Sdf.Path(prim_path)])


def _fix_placed_cable_kinematic(prim_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return
    stack = [root]
    touched = False
    while stack:
        prim = stack.pop()
        stack.extend(prim.GetChildren())
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            rb = PhysxSchema.PhysxRigidBodyAPI(prim)
            kin = rb.GetKinematicEnabledAttr()
            if kin:
                kin.Set(True)
            else:
                rb.CreateKinematicEnabledAttr(True)
            touched = True
    if not touched:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rb.CreateKinematicEnabledAttr(True)


def _port_place_position(port_prim_path: str) -> np.ndarray:
    port_prim = omni.usd.get_context().get_stage().GetPrimAtPath(port_prim_path)
    if not port_prim.IsValid():
        carb.log_error(f"Port prim missing: {port_prim_path}")
        return np.zeros(3, dtype=np.float64)
    pos = get_world_transform_xform(port_prim)[0]
    return np.array(pos, dtype=np.float64) + PLACE_POSITION_OFFSET


class CablePortPickPlaceTask(BaseTask):
    """
    Task-scene + observations aligned with PickPlace (cube) tutorial, but for sequential cylinder cables
    and datacenter ports. Exposes get_params robot_name / cube_name and observations fancy_franka + fancy_cube.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name, offset=None)
        self._scene = None
        self._franka = None
        self._cable_object = None
        self._spawned_cable_paths: List[str] = []
        self._port_index = 0
        self._cable_pick_prim_path: Optional[str] = None

    def set_up_scene(self, scene) -> None:
        super().set_up_scene(scene)
        self._scene = scene
        add_reference_to_stage(USD_PATH, "/World/Datacenter")
        scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))
        if CABLE_URDF_PATH:
            self._cable_pick_prim_path = _import_cable_urdf(CABLE_URDF_PATH)
            self._cable_object = None
            self._spawned_cable_paths = []
        else:
            self._cable_pick_prim_path = None
            self._spawned_cable_paths = []
            self._cable_object = None
        self._port_index = 0

    def get_params(self) -> Dict[str, Any]:
        """Same keys as isaacsim PickPlace task (tutorial: Use the Pick and Place Task)."""
        return {
            "robot_name": {"value": "fancy_franka", "modifiable": True},
            "cube_name": {"value": PICK_TARGET_OBSERVATION_KEY, "modifiable": True},
        }

    def _spawn_cylinder_cable(self, scene, index: int, position: np.ndarray):
        prim_path = f"/World/cable{index}"
        name = f"cable{index}"
        cyl = scene.add(
            DynamicCylinder(
                prim_path=prim_path,
                name=name,
                position=position,
                scale=_cable_dynamic_cylinder_scale_vector(),
                color=np.array([0.2, 0.6, 0.9]),
            )
        )
        self._spawned_cable_paths.append(prim_path)
        return cyl

    def _current_pick_position(self) -> np.ndarray:
        if self._cable_object is not None:
            pos, _ = self._cable_object.get_world_pose()
            return np.array(pos, dtype=np.float64)
        assert self._cable_pick_prim_path is not None
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(self._cable_pick_prim_path)
        pos = get_world_transform_xform(prim)[0]
        return np.array(pos, dtype=np.float64)

    def get_observations(self) -> Dict[str, Any]:
        """Match PickPlace-style observations: cube (here: cable) has position + target_position."""
        if self._franka is None:
            self._franka = self._scene.get_object("fancy_franka")
        joints = self._franka.get_joint_positions()
        if self._port_index >= len(PORT_PRIM_PATH_LIST):
            pick_pos = self._current_pick_position()
            target = pick_pos
        else:
            pick_pos = self._current_pick_position()
            target = _port_place_position(PORT_PRIM_PATH_LIST[self._port_index])
        return {
            "fancy_franka": {"joint_positions": joints},
            PICK_TARGET_OBSERVATION_KEY: {
                "position": pick_pos,
                "target_position": target,
            },
        }

    def post_reset(self) -> None:
        super().post_reset()
        self._franka = self._scene.get_object("fancy_franka")
        _apply_franka_scale_pose(self._franka)
        self._port_index = 0
        if not CABLE_URDF_PATH:
            if self._spawned_cable_paths:
                _delete_prims_at_paths(list(self._spawned_cable_paths))
                self._spawned_cable_paths.clear()
            self._cable_object = self._spawn_cylinder_cable(self._scene, 1, CABLE_WORLD_POSITION)
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)

    def has_more_pick_segments(self) -> bool:
        return self._port_index < len(PORT_PRIM_PATH_LIST)

    def on_pick_place_done(self) -> bool:
        """Advance port, fix placed cylinder, spawn next. Returns True if all ports finished."""
        self._port_index += 1
        if not CABLE_URDF_PATH and self._port_index > 0:
            _fix_placed_cable_kinematic(f"/World/cable{self._port_index}")
        if self._port_index >= len(PORT_PRIM_PATH_LIST):
            return True
        if not CABLE_URDF_PATH:
            self._cable_object = self._spawn_cylinder_cable(
                self._scene, self._port_index + 1, CABLE_WORLD_POSITION
            )
        return False


class HelloWorld(BaseSample):
    """Uses CablePortPickPlaceTask + PickPlaceController like the PickPlace task tutorial."""

    def __init__(self) -> None:
        super().__init__()
        self._controller = None
        self._franka = None
        self._cube_name = PICK_TARGET_OBSERVATION_KEY
        self._pick_task: Optional[CablePortPickPlaceTask] = None

    def setup_scene(self):
        world = self.get_world()
        world.add_task(CablePortPickPlaceTask(name=CABLE_PORT_TASK_NAME))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._pick_task = self._world.get_task(CABLE_PORT_TASK_NAME)
        task_params = self._pick_task.get_params()
        self._franka = self._world.scene.get_object(task_params["robot_name"]["value"])
        self._cube_name = task_params["cube_name"]["value"]
        _apply_franka_scale_pose(self._franka)

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
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        if not self._pick_task.has_more_pick_segments():
            return

        current_observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=current_observations[self._cube_name]["position"],
            placing_position=current_observations[self._cube_name]["target_position"],
            current_joint_positions=current_observations[self._franka.name]["joint_positions"],
        )
        self._franka.apply_action(actions)

        if self._controller.is_done():
            pause_sim = self._pick_task.on_pick_place_done()
            self._controller.reset()
            self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
            if pause_sim:
                self._world.pause()
        return

    def world_cleanup(self):
        return
