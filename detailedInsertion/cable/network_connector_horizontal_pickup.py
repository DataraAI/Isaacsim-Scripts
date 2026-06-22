from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import typing
from pathlib import Path

import numpy as np
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

sys.path.append(str(Path(__file__).resolve().parent))
from franka_motion_controller import FrankaMotionController


# =============================================================================
# 1. PATHS / PRIMS
# =============================================================================

NETWORK_CABLE_USD_PATH = "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd"
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head1_45"

FRANKA_PATH = "/World/Franka"
BLOCK_PATH = "/World/PickupBlock"
POST_LEFT_PATH = "/World/PickupPostLeft"
POST_RIGHT_PATH = "/World/PickupPostRight"


# =============================================================================
# 2. SCENE SETTINGS
# =============================================================================

# Support block/cable spawn position.
# You asked for this around 0.35 height.
BLOCK_CENTER = np.array([0.65, 0.00, 0.1], dtype=np.float64)
BLOCK_SIZE = np.array([0.04, 0.005, 0.025], dtype=np.float64)

# Two small red posts make the cable slot visible.
# These are restored from your previous setup.
POST_SIZE = np.array([0.001, 0.001, 0.005], dtype=np.float64)
POST_GAP_Y = 0.008
POST_X_OFFSET = -0.020

# Cable plug rests this far above the top of the block.
PLUG_BLOCK_CLEARANCE = 0.004

PHYSICS_DT = 1.0 / 120.0
RENDERING_DT = 1.0 / 60.0

# Used by FrankaMotionController when you add cartesian waypoints.
TOOL_OFFSET = 0.111

TOP_DOWN_ORI = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
DIAGONAL_DOWN_ORI = np.array([0.5, 0.0, 0.86602540, 0.0], dtype=np.float64)
DIAGONAL_INSERT_ORI = np.array([0.0, -0.86602540, 0.0, 0.5], dtype=np.float64)
HORIZONTAL_ORI = np.array([0.70710678, 0, 0.70710678, 0.0], dtype=np.float64)

# Grip/contact tuning. These change physics, not the motion path.
# The original gripper delta was 0.02m per frame, which snaps shut in only a few
# frames and can squeeze/eject the connector before contact settles.
GRIPPER_ACTION_DELTA = 0.0025
GRIP_CLOSE_WAIT_FRAMES = 240

# High-friction physics material for the finger pads and connector plug.
# If the plug still slips, raise both friction values together. If the plug sticks
# unnaturally to everything, lower them together.
GRIP_STATIC_FRICTION = 8.0
GRIP_DYNAMIC_FRICTION = 8.0
GRIP_RESTITUTION = 0.0
GRIP_MATERIAL_PATH = "/World/Looks/HighGripPhysicsMaterial"

# Small contact offsets help tiny mesh contacts resolve before visible penetration.
GRIP_CONTACT_OFFSET = 0.003
GRIP_REST_OFFSET = 0.0


# =============================================================================
# 3. EDIT YOUR WAYPOINTS HERE
# =============================================================================

def queue_user_waypoints(controller: FrankaMotionController) -> None:
    """
    Safe controller test waypoint.

    This version is intentionally boring so it does not throw IK/Lula errors.
    Your previous failing test asked for a horizontal wrist pose before the robot
    was in a reachable configuration. That poisoned the next linear command.

    Add your own waypoints below once this file runs cleanly.
    """

    controller.clear_queue()

    # SAFE DEFAULT: top-down, reachable, joint-space setup move.
    # This verifies that the robot/controller/IK stack works without using the
    # fragile horizontal wrist pose.
    #
    # Keep this until your base scene runs without errors. Then replace it with
    # ONE of your own waypoints at a time.

    controller.add_cartesian_waypoint(
        position=np.array([0.642, 0.00, 0.1125], dtype=np.float64),
        orientation=DIAGONAL_DOWN_ORI,
        max_frames=600,
        pos_tolerance=0.001,
        linear=True,
        linear_step=0.001,
        label="diagonal_pickup_down",
    )
    controller.add_gripper_command(action="close", wait_frames=GRIP_CLOSE_WAIT_FRAMES)

    controller.add_cartesian_waypoint(
        position=np.array([0.642, 0.00, 0.2], dtype=np.float64),
        orientation=DIAGONAL_DOWN_ORI,
        max_frames=600,
        pos_tolerance=0.001,
        linear=True,
        linear_step=0.001,
        hold_gripper=True,
        label="diagonal_lift_after_grasp",
    )

    controller.add_cartesian_waypoint(
        position=np.array([-0.642, 0.00, 0.2], dtype=np.float64),
        orientation=DIAGONAL_INSERT_ORI,
        max_frames=600,
        pos_tolerance=0.001,
        joint_interp=True,
        joint_steps=240,
        hold_gripper=True,
        label="diagonal_insert_no_wrist_flip",
    )

    # controller.add_cartesian_waypoint(
    #     position=np.array([-0.55, 0.00, 0.5175], dtype=np.float64),
    #     orientation=np.array([0.7071, 0.0, -0.7071, 0.0], dtype=np.float64),
    #     max_frames=240,
    #     pos_tolerance=0.006,
    #     linear=True,
    #     label="safe_topdown_controller_test",
    # )

    # controller.add_cartesian_waypoint(
    #     position=np.array([-0.6, 0.00, 0.5175], dtype=np.float64),
    #     orientation=np.array([0.7071, 0.0, -0.7071, 0.0], dtype=np.float64),
    #     max_frames=240,
    #     pos_tolerance=0.001,
    #     linear=True,
    #     linear_step=0.001,
    # )
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    # -------------------------------------------------------------------------
    # Add ONE waypoint at a time. If one fails, fix that one before adding more.
    #
    # Example:
    # controller.add_cartesian_waypoint(
    #     position=np.array([0.60, 0.00, 0.40], dtype=np.float64),
    #     orientation=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    #     max_frames=240,
    #     pos_tolerance=0.006,
    #     joint_interp=True,
    #     joint_steps=180,
    #     label="my_waypoint",
    # )
    # -------------------------------------------------------------------------


# =============================================================================
# 4. SMALL USD HELPERS
# =============================================================================

def remove_prim_if_exists(prim_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def get_bbox(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = np.array(box.GetMin(), dtype=np.float64)
    mx = np.array(box.GetMax(), dtype=np.float64)
    center = 0.5 * (mn + mx)
    dims = mx - mn
    return mn, mx, center, dims


# =============================================================================
# DEBUG HELPERS
# =============================================================================

def fmt_vec(value: np.ndarray, decimals: int = 5) -> str:
    return np.array2string(
        np.round(np.asarray(value, dtype=np.float64), decimals),
        precision=decimals,
        suppress_small=False,
    )


def get_prim_world_pose(prim_path: str):
    """Return world-space translation and orientation quaternion [w, x, y, z]."""

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None, None

    mat = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    pos = np.array(mat.ExtractTranslation(), dtype=np.float64)

    quat_gf = mat.ExtractRotationQuat()
    imag = quat_gf.GetImaginary()
    quat_wxyz = np.array(
        [quat_gf.GetReal(), imag[0], imag[1], imag[2]],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat_wxyz))
    if norm > 1e-12:
        quat_wxyz /= norm
    return pos, quat_wxyz


def log_cable_pose(tag: str) -> None:
    """Print the cable/root pose plus the tracked plug pose/bbox.

    Root xform can stay mostly constant for deformable assets, so the tracked
    plug bbox center is the more useful number for pickup/insertion debugging.
    """

    print("-" * 88)
    print(f"[CABLE POSE DEBUG] {tag}")

    root_pos, root_quat = get_prim_world_pose(NETWORK_CABLE_ROOT_PATH)
    if root_pos is None:
        print(f"  cable_root missing: {NETWORK_CABLE_ROOT_PATH}")
    else:
        print(f"  root_path:       {NETWORK_CABLE_ROOT_PATH}")
        print(f"  root_pos:        {fmt_vec(root_pos)}")
        print(f"  root_ori_wxyz:   {fmt_vec(root_quat)}")

    plug_pos, plug_quat = get_prim_world_pose(TRACKED_PLUG_PRIM_PATH)
    if plug_pos is None:
        print(f"  tracked_plug missing: {TRACKED_PLUG_PRIM_PATH}")
    else:
        print(f"  plug_path:       {TRACKED_PLUG_PRIM_PATH}")
        print(f"  plug_xform_pos:  {fmt_vec(plug_pos)}")
        print(f"  plug_ori_wxyz:   {fmt_vec(plug_quat)}")

        plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)
        print(f"  plug_bbox_min:   {fmt_vec(plug_min)}")
        print(f"  plug_bbox_max:   {fmt_vec(plug_max)}")
        print(f"  plug_bbox_ctr:   {fmt_vec(plug_center)}")
        print(f"  plug_dims_mm:    {fmt_vec(plug_dims * 1000.0, 3)}")

    print("-" * 88)


def get_current_command_info(controller: FrankaMotionController):
    """Read the active command from the controller for debug printing."""

    if controller.is_done():
        return None

    index = int(getattr(controller, "_current_command_index", -1))
    queue = getattr(controller, "_command_queue", [])
    if index < 0 or index >= len(queue):
        return None

    cmd = queue[index]
    cmd_type = str(cmd.get("type", "unknown"))
    label = str(cmd.get("label", ""))
    if not label:
        if cmd_type == "gripper":
            label = f"gripper_{cmd.get('action', 'command')}"
        else:
            label = f"command_{index}"

    return {
        "index": index,
        "type": cmd_type,
        "label": label,
        "cmd": cmd,
    }


def print_queued_commands(controller: FrankaMotionController) -> None:
    queue = getattr(controller, "_command_queue", [])
    print("=" * 88)
    print(f"[QUEUED CONTROLLER COMMANDS] count={len(queue)}")
    for i, cmd in enumerate(queue):
        cmd_type = str(cmd.get("type", "unknown"))
        label = str(cmd.get("label", "")) or f"gripper_{cmd.get('action', 'command')}"
        print(f"  [{i}] type={cmd_type} label={label}")
        if cmd_type == "cartesian":
            print(f"      target_position={fmt_vec(cmd.get('pos'))}")
            print(f"      target_ori_wxyz={fmt_vec(cmd.get('ori'))}")
            print(
                "      mode="
                f"linear={cmd.get('linear')} "
                f"joint_interp={cmd.get('joint_interp')} "
                f"hold_gripper={cmd.get('hold_gripper')} "
                f"target_is_hand={cmd.get('target_is_hand')}"
            )
        elif cmd_type == "gripper":
            print(f"      action={cmd.get('action')} wait_frames={cmd.get('max_frames')}")
    print("=" * 88)


def log_command_boundary(boundary: str, info) -> None:
    if info is None:
        return

    cmd = info["cmd"]
    print("=" * 88)
    print(
        f"[COMMAND {boundary}] index={info['index']} "
        f"type={info['type']} label={info['label']}"
    )
    if info["type"] == "cartesian":
        print(f"  commanded_position={fmt_vec(cmd.get('pos'))}")
        print(f"  commanded_ori_wxyz={fmt_vec(cmd.get('ori'))}")
        print(
            "  command_mode="
            f"linear={cmd.get('linear')} "
            f"joint_interp={cmd.get('joint_interp')} "
            f"hold_gripper={cmd.get('hold_gripper')} "
            f"target_is_hand={cmd.get('target_is_hand')}"
        )
    elif info["type"] == "gripper":
        print(f"  gripper_action={cmd.get('action')} wait_frames={cmd.get('max_frames')}")
    print("=" * 88)
    log_cable_pose(f"{boundary} command {info['index']} / {info['label']}")


def set_world_translate(prim_path: str, translation: np.ndarray) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    value = Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2]))

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(value)
            return

    xform.AddTranslateOp().Set(value)


def make_static_box(prim_path: str, center: np.ndarray, size: np.ndarray, color=(0.2, 0.35, 1.0)) -> None:
    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(prim_path)

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _set_schema_attr(api, create_attr_name: str, value) -> bool:
    """Best-effort setter for generated USD/PhysX schema attributes."""

    create_attr = getattr(api, create_attr_name, None)
    if create_attr is None:
        return False

    try:
        attr = create_attr()
    except TypeError:
        attr = create_attr(value)

    attr.Set(value)
    return True


def create_high_grip_physics_material() -> UsdShade.Material:
    """Create one reusable physics material for plug/finger contact."""

    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material.Define(stage, Sdf.Path(GRIP_MATERIAL_PATH))
    prim = material.GetPrim()

    usd_physics_mat = UsdPhysics.MaterialAPI.Apply(prim)
    _set_schema_attr(usd_physics_mat, "CreateStaticFrictionAttr", GRIP_STATIC_FRICTION)
    _set_schema_attr(usd_physics_mat, "CreateDynamicFrictionAttr", GRIP_DYNAMIC_FRICTION)
    _set_schema_attr(usd_physics_mat, "CreateRestitutionAttr", GRIP_RESTITUTION)

    # Force high friction to win even if the other contacting surface has a lower
    # material. These attributes exist in PhysX-backed Isaac builds; the guards
    # keep the script usable if a schema version omits them.
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    _set_schema_attr(physx_mat, "CreateFrictionCombineModeAttr", "max")
    _set_schema_attr(physx_mat, "CreateRestitutionCombineModeAttr", "min")

    return material


def _bind_material_to_prim(prim: Usd.Prim, material: UsdShade.Material) -> None:
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    try:
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    except TypeError:
        try:
            binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
        except TypeError:
            binding_api.Bind(material)


def _tune_collision_contact(prim: Usd.Prim) -> bool:
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return False

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    touched = False
    touched |= _set_schema_attr(physx_collision, "CreateContactOffsetAttr", GRIP_CONTACT_OFFSET)
    touched |= _set_schema_attr(physx_collision, "CreateRestOffsetAttr", GRIP_REST_OFFSET)
    return touched


def bind_high_grip_material_recursive(root_path: str, material: UsdShade.Material) -> typing.Tuple[int, int]:
    """Bind high-friction material to all geometry/collider prims below root_path."""

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[HIGH GRIP WARNING] Missing prim: {root_path}")
        return 0, 0

    bound_count = 0
    contact_tuned_count = 0

    for prim in Usd.PrimRange(root):
        is_geom = prim.IsA(UsdGeom.Gprim)
        is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        if not (is_geom or is_collider):
            continue

        _bind_material_to_prim(prim, material)
        bound_count += 1
        if _tune_collision_contact(prim):
            contact_tuned_count += 1

    # Binding the root gives descendants a fallback even if the referenced asset
    # has collision prims that are not regular Gprims.
    if bound_count == 0:
        _bind_material_to_prim(root, material)
        bound_count = 1

    return bound_count, contact_tuned_count


def apply_high_grip_setup() -> None:
    """Make the physical grasp grippier without attaching the cable to the robot."""

    material = create_high_grip_physics_material()

    targets = [
        TRACKED_PLUG_PRIM_PATH,
        f"{FRANKA_PATH}/panda_leftfinger",
        f"{FRANKA_PATH}/panda_rightfinger",
    ]

    print("=" * 88)
    print("[APPLYING HIGH GRIP SETUP]")
    print("  No robot-to-cable attachment is being created.")
    print("  This only changes contact friction, restitution, contact offsets, and clamp speed.")
    print(f"  gripper_action_delta={GRIPPER_ACTION_DELTA:.4f}m/frame")
    print(f"  close_wait_frames={GRIP_CLOSE_WAIT_FRAMES}")
    print(f"  static_friction={GRIP_STATIC_FRICTION:.2f}")
    print(f"  dynamic_friction={GRIP_DYNAMIC_FRICTION:.2f}")
    print(f"  contact_offset={GRIP_CONTACT_OFFSET * 1000.0:.1f}mm")

    total_bound = 0
    total_contact_tuned = 0
    for path in targets:
        bound, contact_tuned = bind_high_grip_material_recursive(path, material)
        total_bound += bound
        total_contact_tuned += contact_tuned
        print(f"  target={path}")
        print(f"    bound_prims={bound}")
        print(f"    contact_tuned_prims={contact_tuned}")

    print(f"  total_bound_prims={total_bound}")
    print(f"  total_contact_tuned_prims={total_contact_tuned}")
    print("=" * 88)


def enable_gpu_dynamics() -> None:
    stage = omni.usd.get_context().get_stage()
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()]

    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True).Set(True)
        api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
        api.CreateSolverTypeAttr("TGS").Set("TGS")


# =============================================================================
# 5. SPAWN ROBOT / BLOCK / CABLE
# =============================================================================

def spawn_robot(world: World) -> SingleManipulator:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not find Isaac Sim assets folder")

    robot_prim = add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        prim_path=FRANKA_PATH,
    )

    robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    physics_variant = robot_prim.GetVariantSet("Physics")
    physics_variants = list(physics_variant.GetVariantNames())
    if physics_variants:
        physics_variant.SetVariantSelection(
            next((v for v in physics_variants if v.lower() == "physx"), physics_variants[0])
        )

    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PATH}/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.001, 0.001]),
        action_deltas=np.array([GRIPPER_ACTION_DELTA, GRIPPER_ACTION_DELTA]),
    )

    return world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PATH,
            name="my_franka",
            end_effector_prim_path=f"{FRANKA_PATH}/panda_rightfinger",
            gripper=gripper,
        )
    )


def spawn_block_and_posts() -> float:
    """Spawn the blue support block plus the two red slot posts."""

    remove_prim_if_exists(BLOCK_PATH)
    remove_prim_if_exists(POST_LEFT_PATH)
    remove_prim_if_exists(POST_RIGHT_PATH)

    block_top_z = BLOCK_CENTER[2] + 0.5 * BLOCK_SIZE[2]

    make_static_box(
        BLOCK_PATH,
        BLOCK_CENTER,
        BLOCK_SIZE,
        color=(0.2, 0.35, 1.0),
    )

    inner_y = 0.5 * POST_GAP_Y + 0.5 * POST_SIZE[1]
    post_z = block_top_z + 0.5 * POST_SIZE[2]
    post_x = BLOCK_CENTER[0] + POST_X_OFFSET

    left_post_center = np.array(
        [post_x, BLOCK_CENTER[1] + inner_y, post_z],
        dtype=np.float64,
    )
    right_post_center = np.array(
        [post_x, BLOCK_CENTER[1] - inner_y, post_z],
        dtype=np.float64,
    )

    make_static_box(
        POST_LEFT_PATH,
        left_post_center,
        POST_SIZE,
        color=(1.0, 0.0, 0.0),
    )
    make_static_box(
        POST_RIGHT_PATH,
        right_post_center,
        POST_SIZE,
        color=(1.0, 0.0, 0.0),
    )

    print("[SUPPORT SPAWNED]")
    print(f"  block_center={np.round(BLOCK_CENTER, 5)}")
    print(f"  block_size_mm={np.round(BLOCK_SIZE * 1000.0, 2)}")
    print(f"  block_top_z={block_top_z:.5f}")
    print(f"  left_post_center={np.round(left_post_center, 5)}")
    print(f"  right_post_center={np.round(right_post_center, 5)}")
    print(f"  post_size_mm={np.round(POST_SIZE * 1000.0, 2)}")
    print(f"  post_gap_y_mm={POST_GAP_Y * 1000.0:.2f}")
    print(f"  post_x_offset_mm={POST_X_OFFSET * 1000.0:.2f}")

    return block_top_z


def spawn_cable_on_block(block_top_z: float) -> None:
    remove_prim_if_exists(NETWORK_CABLE_ROOT_PATH)

    add_reference_to_stage(
        usd_path=NETWORK_CABLE_USD_PATH,
        prim_path=NETWORK_CABLE_ROOT_PATH,
    )

    stage = omni.usd.get_context().get_stage()

    plug_min, _, plug_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    root_prim = stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH)
    root_tf = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim)
    root_t = np.array(root_tf.ExtractTranslation(), dtype=np.float64)

    desired_plug_center_xy = BLOCK_CENTER[:2]
    desired_plug_min_z = block_top_z + PLUG_BLOCK_CLEARANCE

    delta = np.array(
        [
            desired_plug_center_xy[0] - plug_center[0],
            desired_plug_center_xy[1] - plug_center[1],
            desired_plug_min_z - plug_min[2],
        ],
        dtype=np.float64,
    )

    set_world_translate(NETWORK_CABLE_ROOT_PATH, root_t + delta)

    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)

    print("[CABLE SPAWNED]")
    print(f"  cable_root={NETWORK_CABLE_ROOT_PATH}")
    print(f"  tracked_plug={TRACKED_PLUG_PRIM_PATH}")
    print(f"  plug_center={np.round(plug_center, 5)}")
    print(f"  plug_dims_mm={np.round(plug_dims * 1000.0, 3)}")
    print(f"  plug_min_z={plug_min[2]:.5f}")
    print(f"  plug_max_z={plug_max[2]:.5f}")
    log_cable_pose("AFTER cable spawn/place")


# =============================================================================
# 6. CONTROLLER / IK SETUP
# =============================================================================

def build_controller(franka: SingleManipulator):
    kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")

    kinematics_solver = LulaKinematicsSolver(**kinematics_config)
    task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**kinematics_config)
    art_kinematics = ArticulationKinematicsSolver(franka, kinematics_solver, "panda_hand")

    base_position, base_orientation = franka.get_world_pose()
    kinematics_solver.set_robot_base_pose(base_position, base_orientation)

    controller = FrankaMotionController(
        name="network_connector_controller_base_v10",
        robot_articulation=franka,
        task_traj_gen=task_traj_gen,
        art_kinematics=art_kinematics,
        gripper=franka.gripper,
        tool_offset=TOOL_OFFSET,
        debug=True,
    )

    return controller, kinematics_solver, art_kinematics


def build_scene_and_controller():
    world = World(stage_units_in_meters=1.0)
    world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)

    stage = omni.usd.get_context().get_stage()

    light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    light.CreateIntensityAttr(500.0)
    light.CreateColorAttr((1.0, 1.0, 1.0))

    world.scene.add_default_ground_plane()

    franka = spawn_robot(world)
    block_top_z = spawn_block_and_posts()
    spawn_cable_on_block(block_top_z)
    apply_high_grip_setup()
    enable_gpu_dynamics()

    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    world.reset()

    controller, kinematics_solver, art_kinematics = build_controller(franka)

    return world, franka, controller, kinematics_solver, art_kinematics


# =============================================================================
# 7. OPTIONAL PER-FRAME HOOK
# =============================================================================

def user_robot_step(
    world: World,
    franka: SingleManipulator,
    controller: FrankaMotionController,
    art_kinematics: ArticulationKinematicsSolver,
) -> None:
    """
    Optional hook that runs every frame while Play is active.

    You probably do not need this at first.
    Add printouts, live measurements, or custom checks here later.
    """
    pass


# =============================================================================
# 8. MAIN LOOP
# =============================================================================

world, franka, controller, kinematics_solver, art_kinematics = build_scene_and_controller()
waypoints_queued = False
active_command_info = None
all_done_logged = False

print("[READY - CONTROLLER BASE V11 SAFE]")
print("  Spawned: Franka robot, pickup block, two slot posts, and network cable.")
print("  Controller/IK/Lula setup is complete. Safe default waypoint is active.")
print("  Replace the safe default waypoint inside queue_user_waypoints() when ready.")
print("  Press Play.")

while simulation_app.is_running():
    world.step(render=True)

    if not world.is_playing():
        continue

    if not waypoints_queued:
        queue_user_waypoints(controller)
        print_queued_commands(controller)
        log_cable_pose("AFTER queue_user_waypoints / BEFORE first command")
        waypoints_queued = True

    user_robot_step(world, franka, controller, art_kinematics)

    if controller.is_done():
        if not all_done_logged:
            if active_command_info is not None:
                log_command_boundary("AFTER", active_command_info)
                active_command_info = None
            log_cable_pose("ALL CONTROLLER COMMANDS DONE")
            all_done_logged = True
        continue

    joint_pos = franka.get_joint_positions()
    if joint_pos is None:
        continue

    current_info = get_current_command_info(controller)

    if current_info is not None:
        if active_command_info is None:
            active_command_info = current_info
            log_command_boundary("BEFORE", active_command_info)
        elif current_info["index"] != active_command_info["index"]:
            # The previous command completed between frames. Print its after-pose,
            # then immediately print the before-pose for the new command.
            log_command_boundary("AFTER", active_command_info)
            active_command_info = current_info
            log_command_boundary("BEFORE", active_command_info)

    previous_info = active_command_info
    action = controller.forward(joint_pos)
    franka.get_articulation_controller().apply_action(action)

    # controller.forward() can complete a command immediately once tolerance /
    # wait-frame criteria are met. Detect that transition right away so every
    # command gets an AFTER pose print.
    next_info = get_current_command_info(controller)
    if previous_info is not None:
        if next_info is None or next_info["index"] != previous_info["index"]:
            log_command_boundary("AFTER", previous_info)
            active_command_info = None

simulation_app.close()
