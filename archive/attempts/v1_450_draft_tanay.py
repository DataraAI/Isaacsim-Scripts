from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget

import omni.kit.commands
import omni.usd

from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Gf, Sdf, Usd, UsdPhysics
from isaacsim.core.prims import Articulation

import numpy as np
import carb
import os
import typing

# Surface gripper
# If this exact import differs on your 4.5.0 build, only this line should need changing.
import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
from omni.isaac.dynamic_control import _dynamic_control as dc


USD_PATH = r"/home/advaith/Downloads/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"

PORT_BASE_PRIM_PATH = (
    "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/"
    "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01"
)

num_quads = 4
num_pairs = 4
num_conn_a = 2

PORT_PRIM_PATH_LIST = []
for q in range(1, num_quads + 1):
    for p in range(1, num_pairs + 1):
        for a in range(1, num_conn_a + 1):
            suffix = f"/Connector_Quad_{q:02d}/Connector_Pair_{p:02d}/QSFP_DD_Connector_A_{a:02d}"
            PORT_PRIM_PATH_LIST.append(PORT_BASE_PRIM_PATH + suffix)

TARGET_CUBE_PATH = "/World/TargetCube"
WIRE_ROOT = "/World/Datacenter/WireBank"

# --- Tuning knobs ---
ROBOT_WORLD_POS = np.array([30.0, -90.0, 150.0])   # same as your baseline
ROBOT_WORLD_ORI = np.array([1.0, 0.0, 0.0, 0.0])

WIRE_BANK_OFFSET = np.array([2.5, -4.0, -1.0])     # relative to robot base
WIRE_COL_SPACING = 0.9
WIRE_ROW_SPACING = 0.7
WIRE_LENGTH = 0.75
WIRE_RADIUS = 0.04
WIRE_HEAD_SIZE = 0.14

PICK_APPROACH_OFFSET = np.array([0.0, 0.0, 0.45])
PICK_CONTACT_OFFSET = np.array([0.0, 0.0, 0.10])
LIFT_OFFSET = np.array([0.0, 0.0, 0.80])

PORT_APPROACH_DISTANCE = 2.0
PORT_TOUCH_DISTANCE = 0.20

PHASE_DURATIONS = {
    "to_pick_approach": 1.5,
    "to_pick_contact": 1.2,
    "wait_after_close": 0.5,
    "to_lift": 1.5,
    "to_port_approach": 2.0,
    "to_port_touch": 1.2,
    "wait_after_open": 0.3,
}

DEBUG_SHOW_TARGET_CUBE = True


def set_prim_visibility_attribute(prim_path: str, value: str):
    prop_path = f"{prim_path}.visibility"
    omni.kit.commands.execute(
        "ChangeProperty", prop_path=Sdf.Path(prop_path), value=value, prev=None
    )


def hide_prim(prim_path: str):
    set_prim_visibility_attribute(prim_path, "invisible")


def show_prim(prim_path: str):
    set_prim_visibility_attribute(prim_path, "inherited")


def get_world_transform_xform(prim: Usd.Prim) -> typing.Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale


def get_world_pos(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    return np.array(get_world_transform_xform(prim)[0], dtype=np.float64)


def ensure_translate_op(prim: Usd.Prim):
    xformable = UsdGeom.Xformable(prim)
    attr = prim.GetAttribute("xformOp:translate")
    if not attr.IsValid():
        xformable.AddTranslateOp()
        attr = prim.GetAttribute("xformOp:translate")
    return attr


def ensure_orient_op(prim: Usd.Prim):
    xformable = UsdGeom.Xformable(prim)
    attr = prim.GetAttribute("xformOp:orient")
    if not attr.IsValid():
        xformable.AddOrientOp()
        attr = prim.GetAttribute("xformOp:orient")
    return attr


def set_world_xform_simple(prim: Usd.Prim, pos, quat_wxyz=None):
    ensure_translate_op(prim).Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    if quat_wxyz is not None:
        q = Gf.Quatf(float(quat_wxyz[0]), Gf.Vec3f(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])))
        ensure_orient_op(prim).Set(q)


def lerp(a, b, t):
    return a + (b - a) * t


def find_descendant_prim_path(stage, root_path, leaf_name):
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(root_path + "/") and p.endswith("/" + leaf_name):
            return p
    raise RuntimeError(f"Could not find descendant '{leaf_name}' under {root_path}")


def create_wire(stage, root_path, world_pos, color=(0.05, 0.05, 0.05)):
    """
    Creates one rigid placeholder 'wire':
      - rigid Xform root
      - connector head cube at local origin
      - cable tail cylinder extending in -X
    """
    root = stage.DefinePrim(root_path, "Xform")
    set_world_xform_simple(root, world_pos, [1, 0, 0, 0])

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.05)

    # connector head
    head_path = root_path + "/head"
    head_geom = UsdGeom.Cube.Define(stage, head_path)
    head_geom.CreateSizeAttr(WIRE_HEAD_SIZE)
    head_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    set_world_xform_simple(head_geom.GetPrim(), [0, 0, 0], [1, 0, 0, 0])
    UsdPhysics.CollisionAPI.Apply(head_geom.GetPrim())

    # cable tail
    tail_path = root_path + "/tail"
    tail_geom = UsdGeom.Cylinder.Define(stage, tail_path)
    tail_geom.CreateRadiusAttr(WIRE_RADIUS)
    tail_geom.CreateHeightAttr(WIRE_LENGTH)
    tail_geom.CreateAxisAttr(UsdGeom.Tokens.x)
    tail_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    set_world_xform_simple(tail_geom.GetPrim(), [-(WIRE_LENGTH * 0.5 + WIRE_HEAD_SIZE * 0.5), 0, 0], [1, 0, 0, 0])
    UsdPhysics.CollisionAPI.Apply(tail_geom.GetPrim())

    return {
        "root": root_path,
        "head": head_path,
        "tail": tail_path,
    }


def build_wire_bank(stage, robot_world_pos, port_paths):
    """
    One wire per port, laid out near the robot in a simple grid.
    """
    stage.DefinePrim(WIRE_ROOT, "Xform")

    bank_anchor = np.array(robot_world_pos, dtype=np.float64) + WIRE_BANK_OFFSET
    wire_infos = []

    cols = 4
    for i, port_path in enumerate(port_paths):
        row = i // cols
        col = i % cols

        spawn = bank_anchor + np.array([
            0.0,
            col * WIRE_COL_SPACING,
            -row * WIRE_ROW_SPACING
        ], dtype=np.float64)

        root_path = f"{WIRE_ROOT}/wire_{i:02d}"
        info = create_wire(stage, root_path, spawn)
        info["port"] = port_path
        wire_infos.append(info)

    return wire_infos


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._curr_wire = 0
        self._wire_infos = []
        self._phase = None
        self._phase_elapsed = 0.0
        self._phase_duration = 0.0
        self._phase_start = None
        self._phase_end = None
        self._dc = None
        self._surface_gripper = None
        self._sgp = None
        self._robot_root_prim_path = None
        self._hand_path = None
        self._target_cube_prim = None
        return

    def setup_scene(self):
        world = self.get_world()

        add_reference_to_stage(USD_PATH, "/World/Datacenter")

        urdf_interface = _urdf.acquire_urdf_interface()

        import_config = _urdf.ImportConfig()
        import_config.convex_decomp = False
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False
        import_config.distance_scale = 57.0
        import_config.density = 0.0

        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = extension_path + "/data/urdf/robots/franka_description/robots"
        file_name = "panda_arm_hand.urdf"

        result, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config
        )

        for joint in robot_model.joints:
            robot_model.joints[joint].drive.strength = 1047.19751
            robot_model.joints[joint].drive.damping = 52.35988

        result, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=import_config,
        )

        self._robot_root_prim_path = prim_path

        first_port_prim = omni.usd.get_context().get_stage().GetPrimAtPath(PORT_PRIM_PATH_LIST[0])
        first_port_world_position = get_world_transform_xform(first_port_prim)[0]

        my_task = FollowTarget(
            name="follow_target_task",
            franka_prim_path=prim_path,
            franka_robot_name="fancy_franka",
            target_name="target",
            target_position=np.array(first_port_world_position) + np.array([0, 0, 2]),
        )
        world.add_task(my_task)
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._stage = omni.usd.get_context().get_stage()

        self._target_cube_prim = self._stage.GetPrimAtPath(TARGET_CUBE_PATH)
        if not DEBUG_SHOW_TARGET_CUBE:
            hide_prim(TARGET_CUBE_PATH)
        else:
            show_prim(TARGET_CUBE_PATH)

        # Same robot placement as your baseline
        Articulation(self._robot_root_prim_path).set_world_poses(
            positions=ROBOT_WORLD_POS,
            orientations=ROBOT_WORLD_ORI
        )

        self._controller = RMPFlowController(
            name="target_follower_controller",
            robot_articulation=self._franka
        )

        # Build wire bank after robot placement
        self._wire_infos = build_wire_bank(self._stage, ROBOT_WORLD_POS, PORT_PRIM_PATH_LIST)

        # Resolve panda hand path for surface gripper parent
        self._hand_path = find_descendant_prim_path(self._stage, self._robot_root_prim_path, "panda_hand")
        carb.log_info(f"Using surface gripper parent path: {self._hand_path}")

        # Surface gripper setup
        self._dc = dc.acquire_dynamic_control_interface()
        self._sgp = surface_gripper.Surface_Gripper_Properties()
        self._sgp.d6JointPath = ""
        self._sgp.parentPath = self._hand_path
        self._sgp.offset = dc.Transform()

        # IMPORTANT: Surface gripper closes in the local +X direction of this offset pose,
        # and the grip point should be outside the parent collision shape.
        # These are the only numbers you may need to tune on your build.
        self._sgp.offset.p.x = 0.12
        self._sgp.offset.p.y = 0.0
        self._sgp.offset.p.z = 0.0
        self._sgp.offset.r = [1.0, 0.0, 0.0, 0.0]

        self._sgp.gripThreshold = 0.08
        self._sgp.forceLimit = 1.0e4
        self._sgp.torqueLimit = 1.0e4
        self._sgp.bendAngle = np.pi / 6
        self._sgp.stiffness = 1.0e4
        self._sgp.damping = 1.0e3
        self._sgp.retryClose = True
        self._sgp.disableGravity = True

        self._surface_gripper = surface_gripper.Surface_Gripper(self._dc)
        ok = self._surface_gripper.initialize(self._sgp)
        if not ok:
            raise RuntimeError("Surface gripper failed to initialize")

        self._curr_wire = 0
        self._start_sequence_for_current_wire()

        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        self._controller.reset()
        self._curr_wire = 0
        self._start_sequence_for_current_wire()
        await self._world.play_async()
        return

    def _get_target_cube_pos(self):
        return get_world_pos(self._stage, TARGET_CUBE_PATH)

    def _set_target_cube_pos(self, pos):
        set_world_xform_simple(self._target_cube_prim, pos, [1, 0, 0, 0])

    def _compute_port_offsets(self, idx, port_world_pos):
        sign = 1.0 if (idx % 2 == 0) else -1.0
        port_approach = np.array(port_world_pos, dtype=np.float64) + np.array([0.0, 0.0, PORT_APPROACH_DISTANCE * sign])
        port_touch = np.array(port_world_pos, dtype=np.float64) + np.array([0.0, 0.0, PORT_TOUCH_DISTANCE * sign])
        return port_approach, port_touch

    def _wire_pick_points(self, wire_info):
        head_world_pos = get_world_pos(self._stage, wire_info["head"])
        pick_approach = head_world_pos + PICK_APPROACH_OFFSET
        pick_contact = head_world_pos + PICK_CONTACT_OFFSET
        lift = head_world_pos + LIFT_OFFSET
        return pick_approach, pick_contact, lift

    def _start_motion_phase(self, name, end_pos, duration):
        self._phase = name
        self._phase_elapsed = 0.0
        self._phase_duration = duration
        self._phase_start = self._get_target_cube_pos()
        self._phase_end = np.array(end_pos, dtype=np.float64)

        carb.log_info(f"Phase -> {name} | end={self._phase_end} | dur={duration}")

    def _start_sequence_for_current_wire(self):
        if self._curr_wire >= len(self._wire_infos):
            carb.log_info("All wires processed. Pausing world.")
            self._world.pause()
            return

        wire_info = self._wire_infos[self._curr_wire]
        pick_approach, pick_contact, lift = self._wire_pick_points(wire_info)
        self._cached_pick_approach = pick_approach
        self._cached_pick_contact = pick_contact
        self._cached_lift = lift

        port_world_pos = get_world_pos(self._stage, wire_info["port"])
        self._cached_port_approach, self._cached_port_touch = self._compute_port_offsets(
            self._curr_wire, port_world_pos
        )

        self._start_motion_phase("to_pick_approach", self._cached_pick_approach, PHASE_DURATIONS["to_pick_approach"])

    def _advance_phase(self):
        if self._phase == "to_pick_approach":
            self._start_motion_phase("to_pick_contact", self._cached_pick_contact, PHASE_DURATIONS["to_pick_contact"])

        elif self._phase == "to_pick_contact":
            closed = self._surface_gripper.close()
            carb.log_info(f"Surface gripper close() returned {closed}")
            self._start_motion_phase("wait_after_close", self._cached_pick_contact, PHASE_DURATIONS["wait_after_close"])

        elif self._phase == "wait_after_close":
            self._start_motion_phase("to_lift", self._cached_lift, PHASE_DURATIONS["to_lift"])

        elif self._phase == "to_lift":
            self._start_motion_phase("to_port_approach", self._cached_port_approach, PHASE_DURATIONS["to_port_approach"])

        elif self._phase == "to_port_approach":
            self._start_motion_phase("to_port_touch", self._cached_port_touch, PHASE_DURATIONS["to_port_touch"])

        elif self._phase == "to_port_touch":
            opened = self._surface_gripper.open()
            carb.log_info(f"Surface gripper open() returned {opened}")
            self._start_motion_phase("wait_after_open", self._cached_port_touch, PHASE_DURATIONS["wait_after_open"])

        elif self._phase == "wait_after_open":
            self._curr_wire += 1
            self._start_sequence_for_current_wire()

    def _update_target_motion(self, step_size):
        self._phase_elapsed += step_size
        alpha = min(self._phase_elapsed / max(self._phase_duration, 1e-6), 1.0)
        current = lerp(self._phase_start, self._phase_end, alpha)
        self._set_target_cube_pos(current)

        if self._phase_elapsed >= self._phase_duration:
            self._advance_phase()

    def physics_step(self, step_size):
        observations = self._world.get_observations()

        # Mandatory every step while using the surface gripper
        self._surface_gripper.update()

        # Move hidden/visible target cube according to state machine
        self._update_target_motion(step_size)

        actions = self._controller.forward(
            target_end_effector_position=observations["target"]["position"],
            target_end_effector_orientation=observations["target"]["orientation"],
        )

        self._franka.apply_action(actions)

        # Optional debug
        if self._curr_wire < len(self._wire_infos):
            wire_info = self._wire_infos[self._curr_wire]
            cube_world_position = self._get_target_cube_pos()
            carb.log_info(
                f"[wire={self._curr_wire:02d}] phase={self._phase} "
                f"cube={cube_world_position} "
                f"closed={self._surface_gripper.is_closed()}"
            )
        return

    def world_cleanup(self):
        return