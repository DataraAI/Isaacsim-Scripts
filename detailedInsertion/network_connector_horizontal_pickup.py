from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
from pathlib import Path

import numpy as np
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

sys.path.append(str(Path(__file__).resolve().parent))
from franka_motion_controller import FrankaMotionController


# ------------------------- Paths -------------------------

NETWORK_CABLE_USD_PATH = "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd"
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head1_45"

BLOCK_PATH = "/World/PickupBlock"


# ------------------------- Spawn config -------------------------

# Moved 50 mm farther away from the robot than the previous 0.50 m spawn.
# If this is too far, try 0.53. If still too close, try 0.57.
BLOCK_CENTER = np.array([0.55, 0.00, 0.175], dtype=np.float64)

# Thin support block selected by user.
BLOCK_SIZE = np.array([0.04, 0.005, 0.025], dtype=np.float64)

# Two small top posts create a shallow slot so the cable/strain relief can sit between them.
POST_SIZE = np.array([0.001, 0.001, 0.005], dtype=np.float64)
POST_GAP_Y = 0.008         # clear gap between the inner faces of the two posts
POST_X_OFFSET = -0.02      # keep posts centered on the support block for visibility

PLUG_BLOCK_CLEARANCE = 0.004
SETTLE_FRAMES = 120


# ------------------------- Horizontal pickup config -------------------------

# -1.0 means the hand starts on the robot/near side and moves toward +X.
# Flip back to +1.0 only if it approaches from the far side.
APPROACH_SIGN = -1.0

APPROACH_BACKOFF = 0.03      # meters away from pickup pose before moving in
RETRACT_DISTANCE = 0.120      # meters backward after grasp

# Stop before the mathematically exact grasp pose so the hand/fingers do not ram the block.
# Larger = stops farther from the block. Smaller = moves deeper toward the connector.
# Start here. If it still touches the block, try 0.040. If it misses the connector, try 0.020.
PICKUP_X_STANDOFF = 0.030     # meters

TOOL_OFFSET = 0.050           # panda_hand to grasp point estimate
LINEAR_STEP = 0.001           # 1 mm per sim step during horizontal approach/retract

# Grasp slightly behind the connector head, same idea as the old pickup script.
GRASP_LOCAL_OFFSET = np.array([-0.006, 0.0, 0.0005], dtype=np.float64)

# Small Z trim if the fingers scrape the block or miss too high.
# Try +0.002 if the fingers hit the block.
# Try -0.002 if the fingers close above the connector.
HAND_Z_BIAS = 0.000

# Hand orientation for horizontal pickup.
# For APPROACH_SIGN = +1:
#   hand local +Z points toward world -X, so the fingers approach the connector from +X.
# For APPROACH_SIGN = -1:
#   hand local +Z points toward world +X.
if APPROACH_SIGN > 0:
    PICKUP_ORI = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=np.float64)
else:
    PICKUP_ORI = np.array([0.70710678, 0.0, 0.70710678, 0.0], dtype=np.float64)


# ------------------------- USD helpers -------------------------

def get_bbox(prim_path):
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


def set_world_translate(prim_path, translation):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    value = Gf.Vec3d(
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(value)
            return

    xform.AddTranslateOp().Set(value)


def remove_prim_if_exists(prim_path):
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def enable_gpu_dynamics():
    stage = omni.usd.get_context().get_stage()
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene")).GetPrim()]

    for prim in scenes:
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        api.CreateEnableGPUDynamicsAttr(True).Set(True)
        api.CreateBroadphaseTypeAttr("GPU").Set("GPU")
        api.CreateSolverTypeAttr("TGS").Set("TGS")


def make_static_box(prim_path, center, size, color=(0.2, 0.35, 1.0)):
    stage = omni.usd.get_context().get_stage()
    remove_prim_if_exists(prim_path)

    cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    xform.AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def create_static_block():
    remove_prim_if_exists(BLOCK_PATH)
    remove_prim_if_exists("/World/PickupPostLeft")
    remove_prim_if_exists("/World/PickupPostRight")

    block_top_z = BLOCK_CENTER[2] + 0.5 * BLOCK_SIZE[2]

    # Main blue support block.
    make_static_box(
        BLOCK_PATH,
        BLOCK_CENTER,
        BLOCK_SIZE,
        color=(0.2, 0.35, 1.0),
    )

    # Two big red visible posts on top of the block.
    # The cable/connector should sit in the gap between these posts.
    inner_face_offset = 0.5 * POST_GAP_Y + 0.5 * POST_SIZE[1]
    post_center_z = block_top_z + 0.5 * POST_SIZE[2]
    post_center_x = BLOCK_CENTER[0] + POST_X_OFFSET

    left_post_center = np.array(
        [post_center_x, BLOCK_CENTER[1] + inner_face_offset, post_center_z],
        dtype=np.float64,
    )
    right_post_center = np.array(
        [post_center_x, BLOCK_CENTER[1] - inner_face_offset, post_center_z],
        dtype=np.float64,
    )

    make_static_box(
        "/World/PickupPostLeft",
        left_post_center,
        POST_SIZE,
        color=(1.0, 0.0, 0.0),
    )
    make_static_box(
        "/World/PickupPostRight",
        right_post_center,
        POST_SIZE,
        color=(1.0, 0.0, 0.0),
    )

    print("[BLOCK]")
    print(f"  center={np.round(BLOCK_CENTER, 5)}")
    print(f"  size_mm={np.round(BLOCK_SIZE * 1000.0, 2)}")
    print(f"  top_z={block_top_z:.5f}")
    print("[POSTS]")
    print(f"  left_post_center={np.round(left_post_center, 5)}")
    print(f"  right_post_center={np.round(right_post_center, 5)}")
    print(f"  post_size_mm={np.round(POST_SIZE * 1000.0, 2)}")
    print(f"  post_gap_y_mm={POST_GAP_Y * 1000.0:.2f}")
    print(f"  post_x_offset_mm={POST_X_OFFSET * 1000.0:.2f}")

    return block_top_z

def load_cable():
    remove_prim_if_exists(NETWORK_CABLE_ROOT_PATH)
    add_reference_to_stage(
        usd_path=NETWORK_CABLE_USD_PATH,
        prim_path=NETWORK_CABLE_ROOT_PATH,
    )


def place_tracked_plug_on_block(block_top_z):
    """
    Move the whole cable root so E_crystal_head1_45 sits on top of the block.

    Do NOT move the connector child by itself.
    Do NOT rotate the connector child.
    """

    stage = omni.usd.get_context().get_stage()

    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)

    root_prim = stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH)
    root_pose = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim)
    root_t = np.array(root_pose.ExtractTranslation(), dtype=np.float64)

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

    root_min, root_max, _, _ = get_bbox(NETWORK_CABLE_ROOT_PATH)
    plug_min, plug_max, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)

    print("[CABLE PLACED]")
    print(f"  plug_center={np.round(plug_center, 5)}")
    print(f"  plug_min_z={plug_min[2]:.5f}")
    print(f"  plug_dims_mm={np.round(plug_dims * 1000.0, 3)}")
    print(f"  root_min={np.round(root_min, 5)}")
    print(f"  root_max={np.round(root_max, 5)}")


def reload_cable_and_block():
    remove_prim_if_exists(BLOCK_PATH)
    remove_prim_if_exists("/World/PickupPostLeft")
    remove_prim_if_exists("/World/PickupPostRight")
    remove_prim_if_exists(NETWORK_CABLE_ROOT_PATH)

    block_top_z = create_static_block()
    load_cable()
    place_tracked_plug_on_block(block_top_z)
    enable_gpu_dynamics()


def get_grasp_target():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(TRACKED_PLUG_PRIM_PATH)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing connector head: {TRACKED_PLUG_PRIM_PATH}")

    xform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    target = np.array(xform.Transform(Gf.Vec3d(*GRASP_LOCAL_OFFSET)), dtype=np.float64)
    target[2] += HAND_Z_BIAS
    return target


def hand_from_grasp_point(grasp_point, orientation_wxyz):
    """
    Convert desired connector grasp point into panda_hand world position.

    The panda_hand frame is TOOL_OFFSET behind the actual grasp/contact point
    along the hand local +Z direction.
    """

    rot = quats_to_rot_matrices(np.asarray(orientation_wxyz, dtype=np.float64).reshape(1, 4))[0]
    tool_vec_world = rot @ np.array([0.0, 0.0, TOOL_OFFSET], dtype=np.float64)
    return np.asarray(grasp_point, dtype=np.float64) - tool_vec_world


# ------------------------- Scene setup -------------------------

assets_root = get_assets_root_path()
if assets_root is None:
    raise RuntimeError("Could not find Isaac Sim assets folder")

world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)

stage = omni.usd.get_context().get_stage()

light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
light.CreateIntensityAttr(500.0)
light.CreateColorAttr((1.0, 1.0, 1.0))

world.scene.add_default_ground_plane()

robot_prim = add_reference_to_stage(
    usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path="/World/Franka",
)

robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
physics_variant = robot_prim.GetVariantSet("Physics")
physics_variants = list(physics_variant.GetVariantNames())
if physics_variants:
    physics_variant.SetVariantSelection(
        next((v for v in physics_variants if v.lower() == "physx"), physics_variants[0])
    )

gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.001, 0.001]),
    action_deltas=np.array([0.02, 0.02]),
)

franka = world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="my_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
    )
)

reload_cable_and_block()
franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
world.reset()

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
kinematics_solver = LulaKinematicsSolver(**kinematics_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**kinematics_config)
art_kinematics = ArticulationKinematicsSolver(franka, kinematics_solver, "panda_hand")

base_position, base_orientation = franka.get_world_pose()
kinematics_solver.set_robot_base_pose(base_position, base_orientation)

controller = FrankaMotionController(
    name="horizontal_pickup_controller",
    robot_articulation=franka,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=franka.gripper,
    tool_offset=TOOL_OFFSET,
    debug=False,
)


# ------------------------- Horizontal pickup sequence -------------------------

def queue_horizontal_pickup():
    grasp_point = get_grasp_target()

    # This is the exact pose that would put the tool point on the connector target.
    # It was too aggressive and caused block contact, so we stop short by PICKUP_X_STANDOFF.
    hand_exact_grasp = hand_from_grasp_point(grasp_point, PICKUP_ORI)

    approach_axis = np.array([APPROACH_SIGN, 0.0, 0.0], dtype=np.float64)

    # Final close pose: same orientation and Z, but pulled back on the robot-side approach axis.
    # This keeps the palm/fingers from driving into the support block.
    hand_pick = hand_exact_grasp + approach_axis * PICKUP_X_STANDOFF

    hand_pre = hand_pick + approach_axis * APPROACH_BACKOFF
    hand_pre_above = hand_pre + np.array([0.0, 0.0, 0.010], dtype=np.float64)
    hand_retract = hand_pick + approach_axis * RETRACT_DISTANCE

    _, _, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)

    print("[HORIZONTAL PICKUP]")
    print(f"  pickup_ori={np.round(PICKUP_ORI, 5)}")
    print(f"  approach_sign={APPROACH_SIGN:+.1f}")
    print(f"  pickup_x_standoff_m={PICKUP_X_STANDOFF:.3f}")
    print(f"  plug_center={np.round(plug_center, 5)}")
    print(f"  plug_dims_mm={np.round(plug_dims * 1000.0, 3)}")
    print(f"  grasp_point={np.round(grasp_point, 5)}")
    print(f"  hand_exact_grasp={np.round(hand_exact_grasp, 5)}  # too deep; not commanded")
    print(f"  hand_pre_above={np.round(hand_pre_above, 5)}")
    print(f"  hand_pre={np.round(hand_pre, 5)}")
    print(f"  hand_pick={np.round(hand_pick, 5)}")
    print(f"  hand_retract={np.round(hand_retract, 5)}")
    print("  Motion: pre-pickup -> straight X approach to offset pose -> close -> straight X retract")

    controller.clear_queue()

    # First move to the same pickup orientation, but 1 cm above the current pre-pickup pose.
    # This gives the arm a safer staging point before dropping to the real horizontal approach height.
    controller.add_cartesian_waypoint(
        position=hand_pre_above,
        orientation=PICKUP_ORI,
        max_frames=240,
        pos_tolerance=0.003,
        target_is_hand=True,
        label="pre_horizontal_pickup_above",
    )

    # Then move down 1 cm to the original pre-pickup pose.
    controller.add_cartesian_waypoint(
        position=hand_pre,
        orientation=PICKUP_ORI,
        max_frames=120,
        pos_tolerance=0.003,
        target_is_hand=True,
        label="pre_horizontal_pickup",
    )

    # Move only along X, but stop short of the block.
    controller.add_cartesian_waypoint(
        position=hand_pick,
        orientation=PICKUP_ORI,
        max_frames=240,
        pos_tolerance=0.0015,
        linear=True,
        linear_step=LINEAR_STEP,
        target_is_hand=True,
        label="horizontal_x_approach_offset",
    )

    controller.add_gripper_command(action="close", wait_frames=45)

    # Move backward along X while holding the connector.
    # controller.add_cartesian_waypoint(
    #     position=hand_retract,
    #     orientation=PICKUP_ORI,
    #     max_frames=240,
    #     pos_tolerance=0.002,
    #     linear=True,
    #     linear_step=LINEAR_STEP,
    #     hold_gripper=True,
    #     target_is_hand=True,
    #     label="horizontal_x_retract",
    # )


def print_done():
    fingers = np.asarray(franka.gripper.get_joint_positions(), dtype=np.float64).flatten()
    _, _, plug_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)

    print("[DONE]")
    print(f"  fingers_mm={np.round(fingers * 1000.0, 3)}")
    print(f"  total_gap_mm={np.sum(fingers) * 1000.0:.3f}")
    print(f"  plug_center={np.round(plug_center, 5)}")
    print("  If the hand came from the wrong side, flip APPROACH_SIGN.")


def reset_task_for_next_play():
    global phase, settle_count, run_count

    run_count += 1
    print(f"\n[RESET FOR RUN {run_count}]")

    controller.clear_queue()
    reload_cable_and_block()
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    world.reset()

    base_position, base_orientation = franka.get_world_pose()
    kinematics_solver.set_robot_base_pose(base_position, base_orientation)

    phase = "settle"
    settle_count = 0


phase = "settle"
settle_count = 0
run_count = 1
was_playing = False
reset_needed_on_next_play = False

print("[READY - VISIBLE SLOT BLOCK VERSION]")
print("  Press Play.")
print("  The cable spawns on the support block with two top posts forming a shallow slot.")
print("  The hand will move to PICKUP_ORI, approach horizontally in X, stop 30 mm short of the exact grasp pose, close, then retract.")
print("  If it approaches from the wrong X side, set APPROACH_SIGN = -1.0.")


while simulation_app.is_running():
    world.step(render=True)
    playing = world.is_playing()

    if not playing:
        if was_playing:
            reset_needed_on_next_play = True
            phase = "waiting_for_play"
        was_playing = False
        continue

    if playing and not was_playing:
        if reset_needed_on_next_play:
            reset_task_for_next_play()
            reset_needed_on_next_play = False
        was_playing = True

    if phase == "done" or phase == "waiting_for_play":
        continue

    try:
        joint_pos = franka.get_joint_positions()
    except Exception:
        continue

    if joint_pos is None:
        continue

    if phase == "settle":
        settle_count += 1
        if settle_count >= SETTLE_FRAMES:
            queue_horizontal_pickup()
            phase = "pickup"
        continue

    if phase == "pickup":
        if controller.is_done():
            print_done()
            phase = "done"
            continue

        action = controller.forward(joint_pos)
        franka.get_articulation_controller().apply_action(action)

simulation_app.close()
