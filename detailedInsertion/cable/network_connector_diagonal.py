from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
import typing
from pathlib import Path

import numpy as np
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
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
GRIP_STATIC_FRICTION = 10.0
GRIP_DYNAMIC_FRICTION = 10.0
GRIP_RESTITUTION = 0.0
GRIP_MATERIAL_PATH = "/World/Looks/HighGripPhysicsMaterial"

# Small contact offsets help tiny mesh contacts resolve before visible penetration.

GRIP_CONTACT_OFFSET = 0.003
GRIP_REST_OFFSET = 0.0

# Closed-loop plug pose servo. This is the cable version of the block insertion
# feedback loop: it reads /World/NetworkCable/E_crystal_head1_45 every frame and
# corrects the robot until the plug position AND orientation are inside tolerance.
# Position uses the tracked plug XFORM TRANSLATE because that is the value shown
# in the Isaac Sim Transform panel. Bbox center is still printed as secondary
# debug, but it is not the controlled target in this version.
ENABLE_PLUG_POSE_SERVO = True

# =============================================================================
# USER-SELECTED CABLE / PORT TARGET
# =============================================================================
# Change ONLY this position when you want the cable/plug to end somewhere else.
# This is the desired world-space TRANSLATE value shown in the Transform panel
# for /World/NetworkCable/E_crystal_head1_45. It is not the robot hand target
# and it is not the bbox center.
USER_SELECTED_CABLE_TARGET_POSITION = np.array([-0.55, 0.00, 0.275], dtype=np.float64)

# The plug's long dimension is local +X in the USD. This target is a 180-degree
# yaw around world Z, so local +X points along world -X and the connector stays
# horizontal for later insertion.
USER_SELECTED_CABLE_TARGET_ORI_WXYZ = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

# Internal names used by the servo. Do not edit these; edit USER_SELECTED_* above.
PLUG_TARGET_POSITION = USER_SELECTED_CABLE_TARGET_POSITION.copy()
PLUG_TARGET_ORI_WXYZ = USER_SELECTED_CABLE_TARGET_ORI_WXYZ.copy()
PLUG_INSERT_AXIS_WORLD = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

# Coarse routing is still hand/tool motion, but the last coarse waypoint is now
# derived from the cable target above. The final measured position is corrected
# by the plug feedback servo, not by trusting the hand waypoint.
COARSE_TRANSFER_LANE_Y = -0.50
COARSE_TRANSFER_Z = max(0.325, float(PLUG_TARGET_POSITION[2]) + 0.125)
COARSE_TRANSFER_POSITION = np.array([0.0, COARSE_TRANSFER_LANE_Y, COARSE_TRANSFER_Z], dtype=np.float64)
COARSE_NEAR_TARGET_POSITION = PLUG_TARGET_POSITION.copy()

# Visual marker for the desired plug target. It has no collision and will not
# affect physics. Disable if it annoys you.
SHOW_USER_TARGET_MARKER = True
USER_TARGET_MARKER_PATH = "/World/UserSelectedCableTarget"
USER_TARGET_MARKER_SIZE = 0.008

# Final plug-position alignment tolerances.
# For now this is NOT insertion. The goal is to put the tracked plug transform
# translate at the user-selected port/alignment point with <= 0.5 mm total Y/Z
# offset, matching the Transform panel value.
PLUG_POSITION_TOL = 0.0010          # legacy total XYZ print limit
PLUG_X_TOL = 0.0010                 # 1.0 mm X tolerance for staging/alignment
PLUG_YZ_TOTAL_TOL = 0.00050         # 0.5 mm radial Y/Z tolerance — main metric
PLUG_ORIENTATION_TOL_DEG = 1.00     # full quaternion angular error
PLUG_HOLD_FRAMES = 4                # faster settle; still avoids one-frame false positives
PLUG_SERVO_MAX_FRAMES = 500
PLUG_SERVO_DEBUG_EVERY = 15

# Per-frame correction limits. These are intentionally small so the correction
# behaves like a servo, not a teleporting IK target.
PLUG_SERVO_POS_KP = 0.65           # legacy fallback value; adaptive gains below are used
# Fast/fine position servo inspired by the block insert: move aggressively while
# far from target, then slow down only near the 0.5 mm Y/Z gate.
PLUG_FAST_MODE_DISTANCE = 0.0060     # >6 mm error: fast catch-up
PLUG_MID_MODE_DISTANCE = 0.0020      # 2-6 mm error: medium catch-up
PLUG_FAST_POS_KP = 1.00
PLUG_MID_POS_KP = 0.85
PLUG_FINE_POS_KP = 0.95
PLUG_FAST_MAX_POS_STEP = 0.0040      # 4.0 mm/frame while far
PLUG_MID_MAX_POS_STEP = 0.0025       # 2.5 mm/frame while medium
PLUG_FINE_MAX_POS_STEP = 0.0020      # 2.0 mm/frame while settling
PLUG_SERVO_MAX_POS_STEP = PLUG_FAST_MAX_POS_STEP
PLUG_FAST_ORI_STEP_DEG = 1.20
PLUG_FINE_ORI_STEP_DEG = 0.65
PLUG_SERVO_MAX_ORI_STEP_DEG = PLUG_FAST_ORI_STEP_DEG
PLUG_SERVO_IK_POS_TOL = 0.0002
PLUG_SERVO_IK_ORI_TOL = 0.020       # radians-ish tolerance used by Lula IK
PLUG_LOCAL_DRIFT_WARN_POS = 0.0040  # if exceeded, plug is moving in fingers
PLUG_LOCAL_DRIFT_WARN_ORI_DEG = 5.0

# The uploaded baseline consistently plateaued around +1.2 mm Z error at the
# final pose. This Y/Z overdrive is the missing block-servo idea: if the measured
# plug sits low/right, command the hand slightly high/left until the measured
# plug itself is on the target line. This only runs near the target so it does
# not yank the cable during the long approach.
PLUG_YZ_OVERDRIVE_ENABLE_RADIUS = 0.012  # activate earlier; final bias was previously starving
PLUG_YZ_OVERDRIVE_X_ENABLE = 0.0100      # allow final Y/Z bias as soon as X is reasonably close
# Gentler than v3. The old 4 mm overdrive was accurate but could oscillate for
# hundreds of frames. This is closer to the block servo idea: enough bias to beat
# steady-state cable sag, not enough to throw the plug past the target line.
PLUG_YZ_OVERDRIVE_KP = 1.65
PLUG_YZ_OVERDRIVE_KI = 0.0025
PLUG_YZ_INTEGRAL_LEAK = 0.995
PLUG_YZ_INTEGRAL_LIMIT = 0.40
PLUG_YZ_MAX_OVERDRIVE = 0.0030      # max 3.0 mm Y/Z target bias; enough to beat cable sag quickly


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
        position=np.array([0.642, 0.00, 0.325], dtype=np.float64),
        orientation=DIAGONAL_DOWN_ORI,
        max_frames=600,
        pos_tolerance=0.025,
        linear=True,
        linear_step=0.025,
        hold_gripper=True,
        label="diagonal_lift_after_grasp",
    )

    controller.add_cartesian_waypoint(
        position=COARSE_TRANSFER_POSITION.copy(),
        orientation=DIAGONAL_INSERT_ORI,
        max_frames=600,
        pos_tolerance=0.025,
        # Keep this as the same normal Cartesian/IK style that worked in your
        # baseline. Do not make this an insertion stroke yet.
        hold_gripper=True,
        label="coarse_transfer_lane_from_cable_target",
    )

    controller.add_cartesian_waypoint(
        # This is a coarse robot waypoint derived from the desired cable target.
        # It is NOT trusted as the final cable position. The closed-loop servo
        # below measures E_crystal_head1_45 and finishes the plug position.
        position=COARSE_NEAR_TARGET_POSITION.copy(),
        orientation=DIAGONAL_INSERT_ORI,
        max_frames=800,
        pos_tolerance=0.001,
        hold_gripper=True,
        label="coarse_move_near_user_cable_target",
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




def spawn_user_target_marker() -> None:
    """Spawn a small no-collision visual marker at the selected plug target."""

    if not SHOW_USER_TARGET_MARKER:
        return

    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(USER_TARGET_MARKER_PATH)

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(USER_TARGET_MARKER_PATH))
    cube.CreateSizeAttr(float(USER_TARGET_MARKER_SIZE))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.0)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    target = PLUG_TARGET_POSITION
    xform.AddTranslateOp().Set(Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])))

    print("[USER CABLE TARGET MARKER]")
    print(f"  marker_path={USER_TARGET_MARKER_PATH}")
    print(f"  target_plug_transform_translate={fmt_vec(PLUG_TARGET_POSITION)}")
    print("  collision=disabled")


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
        name="network_connector_controller_fast_v5",
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
    spawn_user_target_marker()
    apply_high_grip_setup()
    enable_gpu_dynamics()

    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    world.reset()

    controller, kinematics_solver, art_kinematics = build_controller(franka)

    return world, franka, controller, kinematics_solver, art_kinematics



# =============================================================================
# 7. CLOSED-LOOP PLUG POSE SERVO
# =============================================================================

PHASE_COARSE_WAYPOINTS = 0
PHASE_PLUG_POSE_SERVO = 1
PHASE_DONE = 2

plug_pose_servo: dict = {}
plug_pose_samples: typing.List[dict] = []


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).flatten()
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_to_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    return quats_to_rot_matrices(normalize_quat_wxyz(quat).reshape(1, 4))[0]


def rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat = rot_matrices_to_quats(np.asarray(rot, dtype=np.float64).reshape(1, 3, 3))
    if quat.ndim > 1:
        quat = quat[0]
    return normalize_quat_wxyz(quat)


def quat_angle_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    a = normalize_quat_wxyz(actual)
    b = normalize_quat_wxyz(target)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def quat_slerp_shortest(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)
    t = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_wxyz(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return normalize_quat_wxyz(s0 * q0 + s1 * q1)


def rotation_angle_error_deg(actual_rot: np.ndarray, target_rot: np.ndarray) -> float:
    rel = np.asarray(target_rot, dtype=np.float64).T @ np.asarray(actual_rot, dtype=np.float64)
    trace_value = float(np.trace(rel))
    cos_angle = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def get_hand_pose_matrix() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_pos, hand_rot = art_kinematics.compute_end_effector_pose()
    hand_pos = np.asarray(hand_pos, dtype=np.float64).flatten()
    hand_rot = np.asarray(hand_rot, dtype=np.float64)
    if hand_rot.ndim == 3:
        hand_rot = hand_rot[0]
    hand_quat = rot_to_quat_wxyz(hand_rot)
    return hand_pos, hand_rot, hand_quat


def get_tracked_plug_pose_matrix() -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return controlled plug position, rotation matrix, quaternion, and bbox center.

    Controlled position is the xform/world translate of E_crystal_head1_45 — the
    same value shown in the Isaac Sim Transform panel. The bbox center is returned
    only for debug so we can see how far visual geometry differs from the xform.
    """

    xform_pos, quat = get_prim_world_pose(TRACKED_PLUG_PRIM_PATH)
    if xform_pos is None or quat is None:
        raise RuntimeError(f"Could not read tracked plug pose: {TRACKED_PLUG_PRIM_PATH}")
    _, _, bbox_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    quat = normalize_quat_wxyz(quat)
    rot = quat_to_rot_wxyz(quat)
    return np.asarray(xform_pos, dtype=np.float64), rot, quat, np.asarray(bbox_center, dtype=np.float64)


def plug_long_axis_world(plug_rot: np.ndarray) -> np.ndarray:
    axis = np.asarray(plug_rot, dtype=np.float64) @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-12 else axis


def plug_axis_angle_to_insert_deg(plug_rot: np.ndarray) -> float:
    axis = plug_long_axis_world(plug_rot)
    target_axis = PLUG_INSERT_AXIS_WORLD / np.linalg.norm(PLUG_INSERT_AXIS_WORLD)
    dot = float(np.clip(np.dot(axis, target_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def plug_tilt_out_of_horizontal_deg(plug_rot: np.ndarray) -> float:
    axis = plug_long_axis_world(plug_rot)
    return float(np.degrees(np.arcsin(np.clip(abs(axis[2]), 0.0, 1.0))))


def compute_plug_hand_offsets() -> typing.Tuple[np.ndarray, np.ndarray]:
    plug_pos, plug_rot, _, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, _ = get_hand_pose_matrix()
    plug_offset_local = hand_rot.T @ (plug_pos - hand_pos)
    plug_rot_local = hand_rot.T @ plug_rot
    return plug_offset_local, plug_rot_local


def hand_pose_for_plug_pose(
    target_plug_pos: np.ndarray,
    target_plug_quat: np.ndarray,
    plug_offset_local: np.ndarray,
    plug_rot_local: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_plug_rot = quat_to_rot_wxyz(target_plug_quat)
    target_hand_rot = target_plug_rot @ np.asarray(plug_rot_local, dtype=np.float64).T
    target_hand_pos = np.asarray(target_plug_pos, dtype=np.float64) - target_hand_rot @ np.asarray(plug_offset_local, dtype=np.float64)
    target_hand_quat = rot_to_quat_wxyz(target_hand_rot)
    return target_hand_pos, target_hand_rot, target_hand_quat


def closed_gripper_hold_action(n_dof: int) -> ArticulationAction:
    return controller._with_closed_gripper(controller._hold_action(n_dof), n_dof)


def print_plug_pose_error(tag: str) -> typing.Tuple[float, float]:
    plug_pos, plug_rot, plug_quat, plug_bbox_center = get_tracked_plug_pose_matrix()
    target_quat = normalize_quat_wxyz(PLUG_TARGET_ORI_WXYZ)
    pos_err = plug_pos - PLUG_TARGET_POSITION
    pos_norm = float(np.linalg.norm(pos_err))
    yz_total = float(np.linalg.norm(pos_err[1:3]))
    ori_err = quat_angle_error_deg(plug_quat, target_quat)
    long_axis = plug_long_axis_world(plug_rot)
    axis_angle = plug_axis_angle_to_insert_deg(plug_rot)
    tilt = plug_tilt_out_of_horizontal_deg(plug_rot)

    print("=" * 88)
    print(f"[PLUG POSE CHECK] {tag}")
    bbox_minus_xform = plug_bbox_center - plug_pos
    print(f"  target_transform_pos:   {fmt_vec(PLUG_TARGET_POSITION)}")
    print(f"  actual_xform_translate: {fmt_vec(plug_pos)}  ← controlled value / Transform panel")
    print(f"  actual_bbox_center:     {fmt_vec(plug_bbox_center)}  secondary debug")
    print(f"  bbox_minus_xform_mm:    {fmt_vec(bbox_minus_xform * 1000.0, 3)}")
    print(f"  position_error_xyz_mm:  {fmt_vec(pos_err * 1000.0, 3)}")
    print(f"  position_error_norm_mm: {pos_norm * 1000.0:.3f}  legacy_limit={PLUG_POSITION_TOL * 1000.0:.3f}")
    print(f"  yz_total_error_mm:      {yz_total * 1000.0:.3f}  main_limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
    print(f"  x_error_mm:             {pos_err[0] * 1000.0:.3f}  limit={PLUG_X_TOL * 1000.0:.3f}")
    print(f"  target_ori_wxyz:        {fmt_vec(target_quat)}")
    print(f"  actual_ori_wxyz:        {fmt_vec(plug_quat)}")
    print(f"  orientation_error_deg:  {ori_err:.3f}  limit={PLUG_ORIENTATION_TOL_DEG:.3f}")
    print(f"  plug_long_axis_world:   {fmt_vec(long_axis)}")
    print(f"  axis_angle_to_-X_deg:   {axis_angle:.3f}")
    print(f"  tilt_out_horizontal_deg:{tilt:.3f}")
    print("=" * 88)
    return pos_norm, ori_err


def print_plug_hand_offset_debug(tag: str) -> None:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, hand_quat = get_hand_pose_matrix()
    offset_local = hand_rot.T @ (plug_pos - hand_pos)
    rot_local = hand_rot.T @ plug_rot

    ref_offset = plug_pose_servo.get("plug_offset_local")
    ref_rot = plug_pose_servo.get("plug_rot_local")
    offset_drift = float(np.linalg.norm(offset_local - ref_offset)) if ref_offset is not None else 0.0
    rot_drift = rotation_angle_error_deg(rot_local, ref_rot) if ref_rot is not None else 0.0

    print("-" * 88)
    print(f"[PLUG/HAND OFFSET DEBUG] {tag}")
    print(f"  hand_pos:              {fmt_vec(hand_pos)}")
    print(f"  hand_ori_wxyz:         {fmt_vec(hand_quat)}")
    _, _, bbox_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    print(f"  plug_xform_translate:  {fmt_vec(plug_pos)}")
    print(f"  plug_bbox_center:      {fmt_vec(bbox_center)}")
    print(f"  plug_ori_wxyz:         {fmt_vec(plug_quat)}")
    print(f"  plug_offset_local:     {fmt_vec(offset_local)}")
    print(f"  plug_offset_drift_mm:  {offset_drift * 1000.0:.3f}")
    print(f"  plug_rot_local_drift:  {rot_drift:.3f} deg")
    if offset_drift > PLUG_LOCAL_DRIFT_WARN_POS or rot_drift > PLUG_LOCAL_DRIFT_WARN_ORI_DEG:
        print("  WARNING: plug is moving relative to the hand; this is grip/slip, not IK error.")
    print("-" * 88)


def init_plug_pose_servo() -> None:
    plug_pose_servo.clear()
    plug_pose_samples.clear()

    plug_offset_local, plug_rot_local = compute_plug_hand_offsets()
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, hand_quat = get_hand_pose_matrix()
    pos_err_norm, ori_err_deg = print_plug_pose_error("BEFORE closed-loop plug pose servo")

    target_hand_pos, _, target_hand_quat = hand_pose_for_plug_pose(
        PLUG_TARGET_POSITION,
        PLUG_TARGET_ORI_WXYZ,
        plug_offset_local,
        plug_rot_local,
    )

    plug_pose_servo.update({
        "frames": 0,
        "stable_frames": 0,
        "ik_fail_count": 0,
        "warned_ik": False,
        "plug_offset_local": plug_offset_local,
        "plug_rot_local": plug_rot_local,
        "initial_plug_pos": plug_pos.copy(),
        "initial_plug_quat": plug_quat.copy(),
        "initial_hand_pos": hand_pos.copy(),
        "initial_hand_quat": hand_quat.copy(),
        "target_hand_pos": target_hand_pos.copy(),
        "target_hand_quat": target_hand_quat.copy(),
        "max_position_error": pos_err_norm,
        "max_orientation_error_deg": ori_err_deg,
        "max_offset_drift": 0.0,
        "max_rot_local_drift_deg": 0.0,
        "yz_integral": np.zeros(2, dtype=np.float64),
        "last_yz_overdrive": np.zeros(2, dtype=np.float64),
        "last_command_plug_pos": plug_pos.copy(),
        "last_target_hand_pos": target_hand_pos.copy(),
        "last_servo_mode": "init",
        "last_pos_step_mm": 0.0,
    })

    print("=" * 88)
    print("[PLUG POSE SERVO INIT]")
    print("  control_object:          /World/NetworkCable/E_crystal_head1_45")
    print("  position_source:         tracked plug xform translate / Transform panel value")
    print("  orientation_source:      tracked plug xform quaternion")
    print(f"  plug_offset_local:       {fmt_vec(plug_offset_local)}")
    print(f"  target_transform_pos:    {fmt_vec(PLUG_TARGET_POSITION)}")
    print(f"  target_plug_ori_wxyz:    {fmt_vec(normalize_quat_wxyz(PLUG_TARGET_ORI_WXYZ))}")
    print(f"  target_hand_pos:         {fmt_vec(target_hand_pos)}")
    print(f"  target_hand_ori_wxyz:    {fmt_vec(target_hand_quat)}")
    print(f"  x_tolerance:             {PLUG_X_TOL * 1000.0:.3f} mm")
    print(f"  yz_total_tolerance:      {PLUG_YZ_TOTAL_TOL * 1000.0:.3f} mm  ← main port-position metric")
    print(f"  orientation_tolerance:   {PLUG_ORIENTATION_TOL_DEG:.3f} deg")
    print(f"  stable_hold_frames:      {PLUG_HOLD_FRAMES}")
    print(f"  fast/mid/fine pos step:  {PLUG_FAST_MAX_POS_STEP * 1000.0:.1f} / {PLUG_MID_MAX_POS_STEP * 1000.0:.1f} / {PLUG_FINE_MAX_POS_STEP * 1000.0:.1f} mm/frame")
    print(f"  fast/fine ori step:      {PLUG_FAST_ORI_STEP_DEG:.2f} / {PLUG_FINE_ORI_STEP_DEG:.2f} deg/frame")
    print(f"  yz_overdrive_kp/ki:      {PLUG_YZ_OVERDRIVE_KP:.3f} / {PLUG_YZ_OVERDRIVE_KI:.4f}")
    print(f"  yz_overdrive_limit:      {PLUG_YZ_MAX_OVERDRIVE * 1000.0:.3f} mm")
    print("=" * 88)
    print_plug_hand_offset_debug("SERVO INIT local plug-to-hand relationship")


def sample_plug_pose_servo_path(pos_err_norm: float, ori_err_deg: float) -> None:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    hand_pos, hand_rot, _ = get_hand_pose_matrix()
    ref_offset = plug_pose_servo.get("plug_offset_local")
    ref_rot = plug_pose_servo.get("plug_rot_local")
    current_offset = hand_rot.T @ (plug_pos - hand_pos)
    current_rot_local = hand_rot.T @ plug_rot
    offset_drift = float(np.linalg.norm(current_offset - ref_offset)) if ref_offset is not None else 0.0
    rot_drift = rotation_angle_error_deg(current_rot_local, ref_rot) if ref_rot is not None else 0.0

    plug_pose_servo["max_position_error"] = max(float(plug_pose_servo.get("max_position_error", 0.0)), float(pos_err_norm))
    plug_pose_servo["max_orientation_error_deg"] = max(float(plug_pose_servo.get("max_orientation_error_deg", 0.0)), float(ori_err_deg))
    plug_pose_servo["max_offset_drift"] = max(float(plug_pose_servo.get("max_offset_drift", 0.0)), offset_drift)
    plug_pose_servo["max_rot_local_drift_deg"] = max(float(plug_pose_servo.get("max_rot_local_drift_deg", 0.0)), rot_drift)

    plug_pose_samples.append({
        "pos": plug_pos.copy(),
        "quat": plug_quat.copy(),
        "long_axis": plug_long_axis_world(plug_rot).copy(),
        "pos_err_norm": float(pos_err_norm),
        "ori_err_deg": float(ori_err_deg),
        "axis_angle_deg": plug_axis_angle_to_insert_deg(plug_rot),
        "tilt_deg": plug_tilt_out_of_horizontal_deg(plug_rot),
        "offset_drift": offset_drift,
        "rot_drift_deg": rot_drift,
    })


def update_plug_pose_servo_state() -> bool:
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    pos_err = PLUG_TARGET_POSITION - plug_pos
    pos_err_norm = float(np.linalg.norm(pos_err))
    x_err = float(pos_err[0])
    yz_err = np.asarray(pos_err[1:3], dtype=np.float64)
    yz_total = float(np.linalg.norm(yz_err))
    ori_err_deg = quat_angle_error_deg(plug_quat, PLUG_TARGET_ORI_WXYZ)
    sample_plug_pose_servo_path(pos_err_norm, ori_err_deg)

    plug_pose_servo["frames"] += 1
    inside_x = abs(x_err) <= PLUG_X_TOL
    inside_yz = yz_total <= PLUG_YZ_TOTAL_TOL
    inside_ori = ori_err_deg <= PLUG_ORIENTATION_TOL_DEG
    inside = inside_x and inside_yz and inside_ori
    if inside:
        plug_pose_servo["stable_frames"] += 1
    else:
        plug_pose_servo["stable_frames"] = 0

    if plug_pose_servo["frames"] == 1 or plug_pose_servo["frames"] % PLUG_SERVO_DEBUG_EVERY == 0 or inside:
        axis_angle = plug_axis_angle_to_insert_deg(plug_rot)
        tilt = plug_tilt_out_of_horizontal_deg(plug_rot)
        yz_overdrive = np.asarray(plug_pose_servo.get("last_yz_overdrive", np.zeros(2)), dtype=np.float64)
        command_plug_pos = np.asarray(plug_pose_servo.get("last_command_plug_pos", plug_pos), dtype=np.float64)
        servo_mode = str(plug_pose_servo.get("last_servo_mode", "?"))
        pos_step_mm = float(plug_pose_servo.get("last_pos_step_mm", 0.0))
        print(
            f"[PLUG PORT ALIGN SERVO] frame={plug_pose_servo['frames']} "
            f"pos_err={pos_err_norm * 1000.0:.3f}mm "
            f"x_err={x_err * 1000.0:.3f}mm "
            f"y_err={yz_err[0] * 1000.0:.3f}mm "
            f"z_err={yz_err[1] * 1000.0:.3f}mm "
            f"yz_total={yz_total * 1000.0:.3f}mm "
            f"ori_err={ori_err_deg:.3f}deg "
            f"axis_to_-X={axis_angle:.3f}deg "
            f"tilt={tilt:.3f}deg "
            f"mode={servo_mode} "
            f"pos_step={pos_step_mm:.3f}mm "
            f"yz_overdrive_mm={np.round(yz_overdrive * 1000.0, 3)} "
            f"command_plug={fmt_vec(command_plug_pos)} "
            f"inside_x={inside_x} inside_yz={inside_yz} inside_ori={inside_ori} "
            f"stable={plug_pose_servo['stable_frames']}/{PLUG_HOLD_FRAMES}"
        )

    if plug_pose_servo["stable_frames"] >= PLUG_HOLD_FRAMES:
        print("\n" + "=" * 88)
        print("[PLUG PORT ALIGN SERVO] target port/alignment pose held inside tolerance")
        print(f"  frames:               {plug_pose_servo['frames']}")
        print(f"  final_x_error_mm:     {x_err * 1000.0:.3f}")
        print(f"  final_y_error_mm:     {yz_err[0] * 1000.0:.3f}")
        print(f"  final_z_error_mm:     {yz_err[1] * 1000.0:.3f}")
        print(f"  final_yz_total_mm:    {yz_total * 1000.0:.3f}  limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_ori_error_deg:  {ori_err_deg:.3f}")
        print("=" * 88)
        return True

    if plug_pose_servo["frames"] >= PLUG_SERVO_MAX_FRAMES:
        print("\n" + "=" * 88)
        print("[PLUG PORT ALIGN SERVO] FAILED: timed out before Y/Z port tolerance was met")
        print(f"  final_x_error_mm:     {x_err * 1000.0:.3f}")
        print(f"  final_y_error_mm:     {yz_err[0] * 1000.0:.3f}")
        print(f"  final_z_error_mm:     {yz_err[1] * 1000.0:.3f}")
        print(f"  final_yz_total_mm:    {yz_total * 1000.0:.3f}  limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_ori_error_deg:  {ori_err_deg:.3f}")
        print(f"  stable_frames:        {plug_pose_servo['stable_frames']}/{PLUG_HOLD_FRAMES}")
        print("=" * 88)
        return True

    return False

def plug_pose_servo_action(joint_pos: np.ndarray) -> ArticulationAction:
    n_dof = int(joint_pos.shape[0])
    plug_pos, plug_rot, plug_quat, _ = get_tracked_plug_pose_matrix()
    target_quat = normalize_quat_wxyz(PLUG_TARGET_ORI_WXYZ)

    # Measured error from actual tracked plug xform translate to the user-selected
    # Transform-panel target.
    measured_err = PLUG_TARGET_POSITION - plug_pos

    pos_err_norm = float(np.linalg.norm(measured_err))
    yz_total = float(np.linalg.norm(measured_err[1:3]))

    # Fast/fine mode split, copied from the lesson of the block insert: get close
    # quickly first, then use a smaller correction only for the final sub-mm settle.
    if pos_err_norm > PLUG_FAST_MODE_DISTANCE:
        servo_mode = "fast"
        pos_kp = PLUG_FAST_POS_KP
        max_pos_step = PLUG_FAST_MAX_POS_STEP
    elif pos_err_norm > PLUG_MID_MODE_DISTANCE:
        servo_mode = "mid"
        pos_kp = PLUG_MID_POS_KP
        max_pos_step = PLUG_MID_MAX_POS_STEP
    else:
        servo_mode = "fine"
        pos_kp = PLUG_FINE_POS_KP
        max_pos_step = PLUG_FINE_MAX_POS_STEP

    # Y/Z overdrive only in fine/mid close range. v3 enabled a 4 mm overdrive too
    # early, so it often overshot and spent hundreds of frames unwinding.
    overdrive_allowed = (
        abs(float(measured_err[0])) <= PLUG_YZ_OVERDRIVE_X_ENABLE
        and yz_total <= PLUG_YZ_OVERDRIVE_ENABLE_RADIUS
    )
    if overdrive_allowed:
        yz_integral = np.asarray(plug_pose_servo.get("yz_integral", np.zeros(2)), dtype=np.float64)
        yz_integral = PLUG_YZ_INTEGRAL_LEAK * yz_integral + measured_err[1:3]
        yz_integral = np.clip(yz_integral, -PLUG_YZ_INTEGRAL_LIMIT, PLUG_YZ_INTEGRAL_LIMIT)
        plug_pose_servo["yz_integral"] = yz_integral
        yz_overdrive = PLUG_YZ_OVERDRIVE_KP * measured_err[1:3] + PLUG_YZ_OVERDRIVE_KI * yz_integral

        # Deadband kick: the previous fast version parked forever at ~0.56 mm
        # Y/Z error because the bias was barely smaller than the cable/contact
        # restoring force. When we are just outside the pass band, add a small
        # push in the measured-error direction so it crosses 0.5 mm quickly.
        if yz_total > PLUG_YZ_TOTAL_TOL:
            yz_dir = measured_err[1:3] / max(yz_total, 1e-9)
            kick_mag = min(0.0010, max(0.00020, yz_total - PLUG_YZ_TOTAL_TOL + 0.00030))
            yz_overdrive = yz_overdrive + yz_dir * kick_mag

        yz_overdrive = np.clip(yz_overdrive, -PLUG_YZ_MAX_OVERDRIVE, PLUG_YZ_MAX_OVERDRIVE)
    else:
        plug_pose_servo["yz_integral"] = np.zeros(2, dtype=np.float64)
        yz_overdrive = np.zeros(2, dtype=np.float64)

    servo_goal = PLUG_TARGET_POSITION.copy()
    servo_goal[1:3] += yz_overdrive
    plug_pose_servo["last_yz_overdrive"] = yz_overdrive.copy()
    plug_pose_servo["last_servo_mode"] = servo_mode

    err = servo_goal - plug_pos
    raw_step = pos_kp * err
    raw_step_norm = float(np.linalg.norm(raw_step))
    if raw_step_norm > max_pos_step:
        raw_step *= max_pos_step / raw_step_norm
    command_plug_pos = plug_pos + raw_step
    plug_pose_servo["last_command_plug_pos"] = command_plug_pos.copy()
    plug_pose_servo["last_pos_step_mm"] = float(np.linalg.norm(raw_step) * 1000.0)

    # Orientation also uses a fast/fine split. Big rotations are allowed to close
    # quickly, then we slow down near the target to avoid twisting the plug.
    ori_err_deg = quat_angle_error_deg(plug_quat, target_quat)
    max_ori_step = PLUG_FAST_ORI_STEP_DEG if ori_err_deg > 2.0 else PLUG_FINE_ORI_STEP_DEG
    if ori_err_deg <= 1e-9:
        command_plug_quat = target_quat
    else:
        frac = min(1.0, max_ori_step / ori_err_deg)
        command_plug_quat = quat_slerp_shortest(plug_quat, target_quat, frac)

    target_hand_pos, _, target_hand_quat = hand_pose_for_plug_pose(
        command_plug_pos,
        command_plug_quat,
        plug_pose_servo["plug_offset_local"],
        plug_pose_servo["plug_rot_local"],
    )
    plug_pose_servo["last_target_hand_pos"] = target_hand_pos.copy()

    action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_hand_pos,
        target_orientation=target_hand_quat,
        position_tolerance=PLUG_SERVO_IK_POS_TOL,
        orientation_tolerance=PLUG_SERVO_IK_ORI_TOL,
    )

    if not success:
        plug_pose_servo["ik_fail_count"] = int(plug_pose_servo.get("ik_fail_count", 0)) + 1
        if not plug_pose_servo.get("warned_ik", False) or plug_pose_servo["ik_fail_count"] % 120 == 0:
            print("[PLUG PORT ALIGN SERVO] IK failed; holding closed gripper")
            print(f"  measured_err_mm:      {fmt_vec(measured_err * 1000.0, 3)}")
            print(f"  yz_overdrive_mm:      {fmt_vec(yz_overdrive * 1000.0, 3)}")
            print(f"  command_plug_pos:     {fmt_vec(command_plug_pos)}")
            print(f"  command_plug_quat:    {fmt_vec(command_plug_quat)}")
            print(f"  target_hand_pos:      {fmt_vec(target_hand_pos)}")
            print(f"  target_hand_quat:     {fmt_vec(target_hand_quat)}")
            print(f"  ik_fail_count:        {plug_pose_servo['ik_fail_count']}")
            plug_pose_servo["warned_ik"] = True
        return closed_gripper_hold_action(n_dof)

    return controller._with_closed_gripper(action, n_dof)

def measure_plug_pose_servo_result() -> bool:
    pos_norm, ori_err = print_plug_pose_error("FINAL closed-loop plug pose result")
    print_plug_hand_offset_debug("FINAL local plug-to-hand relationship")

    plug_pos, _, _, _ = get_tracked_plug_pose_matrix()
    final_err = PLUG_TARGET_POSITION - plug_pos
    final_x_abs = abs(float(final_err[0]))
    final_yz_total = float(np.linalg.norm(final_err[1:3]))

    if len(plug_pose_samples) >= 2:
        positions = np.asarray([s["pos"] for s in plug_pose_samples], dtype=np.float64)
        pos_errs = np.asarray([s["pos_err_norm"] for s in plug_pose_samples], dtype=np.float64)
        ori_errs = np.asarray([s["ori_err_deg"] for s in plug_pose_samples], dtype=np.float64)
        axis_angles = np.asarray([s["axis_angle_deg"] for s in plug_pose_samples], dtype=np.float64)
        tilts = np.asarray([s["tilt_deg"] for s in plug_pose_samples], dtype=np.float64)
        offset_drifts = np.asarray([s["offset_drift"] for s in plug_pose_samples], dtype=np.float64)
        rot_drifts = np.asarray([s["rot_drift_deg"] for s in plug_pose_samples], dtype=np.float64)
        yz_errs = np.linalg.norm(PLUG_TARGET_POSITION[1:3].reshape(1, 2) - positions[:, 1:3], axis=1)

        dx = np.diff(positions[:, 0])
        max_positive_x_backtrack = float(np.max(np.maximum(0.0, dx))) if len(dx) else 0.0

        print("=" * 88)
        print("[PLUG TRANSFORM TARGET ALIGN PATH SUMMARY]")
        print(f"  samples:                    {len(plug_pose_samples)}")
        print(f"  start_plug_pos:             {fmt_vec(positions[0])}")
        print(f"  end_plug_pos:               {fmt_vec(positions[-1])}")
        print(f"  max_position_error_mm:      {float(np.max(pos_errs)) * 1000.0:.3f}")
        print(f"  final_position_error_mm:    {pos_norm * 1000.0:.3f}")
        print(f"  max_yz_total_error_mm:      {float(np.max(yz_errs)) * 1000.0:.3f}")
        print(f"  final_yz_total_error_mm:    {final_yz_total * 1000.0:.3f}  limit={PLUG_YZ_TOTAL_TOL * 1000.0:.3f}")
        print(f"  final_x_abs_error_mm:       {final_x_abs * 1000.0:.3f}  limit={PLUG_X_TOL * 1000.0:.3f}")
        print(f"  max_orientation_error_deg:  {float(np.max(ori_errs)):.3f}")
        print(f"  final_orientation_error_deg:{ori_err:.3f}")
        print(f"  max_axis_angle_to_-X_deg:   {float(np.max(axis_angles)):.3f}")
        print(f"  final_axis_angle_to_-X_deg: {float(axis_angles[-1]):.3f}")
        print(f"  max_tilt_horizontal_deg:    {float(np.max(tilts)):.3f}")
        print(f"  final_tilt_horizontal_deg:  {float(tilts[-1]):.3f}")
        print(f"  max_plug_offset_drift_mm:   {float(np.max(offset_drifts)) * 1000.0:.3f}")
        print(f"  max_plug_rot_drift_deg:     {float(np.max(rot_drifts)):.3f}")
        print(f"  max_positive_x_backtrack:   {max_positive_x_backtrack * 1000.0:.3f} mm")
        print(f"  ik_fail_count:              {int(plug_pose_servo.get('ik_fail_count', 0))}")
        print("=" * 88)

    ok = final_x_abs <= PLUG_X_TOL and final_yz_total <= PLUG_YZ_TOTAL_TOL and ori_err <= PLUG_ORIENTATION_TOL_DEG
    print("[PLUG TRANSFORM TARGET ALIGN RESULT] ✓ plug xform translate is within requested port-position tolerance" if ok else "[PLUG TRANSFORM TARGET ALIGN RESULT] ✗ plug xform translate is outside requested port-position tolerance")
    return ok


# =============================================================================
# 8. OPTIONAL PER-FRAME HOOK
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
# 9. MAIN LOOP
# =============================================================================

world, franka, controller, kinematics_solver, art_kinematics = build_scene_and_controller()
waypoints_queued = False
active_command_info = None
phase = PHASE_COARSE_WAYPOINTS
final_result_logged = False

print("[READY - USER-SELECTED CABLE TRANSFORM TARGET ALIGNMENT FAST V5]")
print("  Spawned: Franka robot, pickup block, two slot posts, and network cable.")
print("  You choose the final cable/plug target; the robot hand is only used as a carrier.")
print(f"  tracked plug: {TRACKED_PLUG_PRIM_PATH}")
print(f"  selected plug target xform translate: {fmt_vec(PLUG_TARGET_POSITION)}")
print(f"  selected plug target orientation: {fmt_vec(PLUG_TARGET_ORI_WXYZ)}")
print(f"  coarse transfer waypoint derived from target: {fmt_vec(COARSE_TRANSFER_POSITION)}")
print(f"  coarse near-target waypoint derived from target: {fmt_vec(COARSE_NEAR_TARGET_POSITION)}")
print("  Final pass metric: tracked plug Y/Z radial offset <= 0.5 mm.")
print("  Press Play.")

while simulation_app.is_running():
    world.step(render=True)

    if not world.is_playing():
        continue

    if not waypoints_queued:
        queue_user_waypoints(controller)
        print_queued_commands(controller)
        print_plug_pose_error("AFTER queue_user_waypoints / BEFORE first command")
        log_cable_pose("AFTER queue_user_waypoints / BEFORE first command")
        waypoints_queued = True

    user_robot_step(world, franka, controller, art_kinematics)

    joint_pos = franka.get_joint_positions()
    if joint_pos is None:
        continue

    if phase == PHASE_PLUG_POSE_SERVO:
        if update_plug_pose_servo_state():
            print("\n[PHASE] CLOSED-LOOP PLUG POSE SERVO complete")
            measure_plug_pose_servo_result()
            phase = PHASE_DONE
            print("\n[PHASE] DONE — gripper remains closed for inspection. Press Stop to reset/rerun.")
        else:
            franka.get_articulation_controller().apply_action(plug_pose_servo_action(joint_pos))
        continue

    if phase == PHASE_DONE:
        if not final_result_logged:
            log_cable_pose("FINAL HOLD / ALL PHASES DONE")
            final_result_logged = True
        continue

    # PHASE_COARSE_WAYPOINTS: execute your existing waypoint sequence with the
    # same before/after command debug you already had.
    if controller.is_done():
        if active_command_info is not None:
            log_command_boundary("AFTER", active_command_info)
            active_command_info = None

        print("\n[PHASE] COARSE WAYPOINTS complete")
        print_plug_pose_error("AFTER coarse waypoints / BEFORE closed-loop correction")
        log_cable_pose("AFTER coarse waypoints / BEFORE closed-loop correction")

        if ENABLE_PLUG_POSE_SERVO:
            print("[PHASE] → CLOSED-LOOP PLUG POSE SERVO")
            controller.clear_queue()
            init_plug_pose_servo()
            phase = PHASE_PLUG_POSE_SERVO
        else:
            print("[PHASE] Plug pose servo disabled. Holding final coarse pose.")
            phase = PHASE_DONE
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
            print_plug_pose_error(f"AFTER command {active_command_info['index']} / {active_command_info['label']}")
            active_command_info = current_info
            log_command_boundary("BEFORE", active_command_info)
            print_plug_pose_error(f"BEFORE command {active_command_info['index']} / {active_command_info['label']}")

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
            print_plug_pose_error(f"AFTER command {previous_info['index']} / {previous_info['label']}")
            active_command_info = None

simulation_app.close()
