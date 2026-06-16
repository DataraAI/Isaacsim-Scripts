from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema

sys.path.append(str(Path(__file__).resolve().parent))
from franka_motion_controller import FrankaMotionController


NETWORK_CABLE_USD_PATH = "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd"
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head1_45"

PLUG_SPAWN_XY = np.array([0.5, 0.0], dtype=np.float64)
GROUND_CLEARANCE = 0.002
SETTLE_FRAMES = 15

GRASP_LOCAL_OFFSET = np.array([-0.006, 0.0, 0.0005], dtype=np.float64)
GRASP_Z_OFFSET = 0.010
ABOVE_Z = 0.080
CLOSE_WAIT_FRAMES = 30
# Top-down pickup, but yawed 90 degrees so the connector is aligned with the hand/gripper frame.
# If this is the wrong side, flip the sign on the 0.70710678 in the Y component.
PICKUP_ORI = np.array([0.0, 0.70710678, 0.70710678, 0.0], dtype=np.float64)


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
    return mn, mx, 0.5 * (mn + mx), mx - mn


def set_world_translate(prim_path, translation):
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


def place_cable_on_ground():
    stage = omni.usd.get_context().get_stage()
    root_min, _, _, _ = get_bbox(NETWORK_CABLE_ROOT_PATH)
    _, _, plug_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)

    root_prim = stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH)
    root_pose = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim)
    root_t = np.array(root_pose.ExtractTranslation(), dtype=np.float64)

    delta = np.array([
        PLUG_SPAWN_XY[0] - plug_center[0],
        PLUG_SPAWN_XY[1] - plug_center[1],
        GROUND_CLEARANCE - root_min[2],
    ], dtype=np.float64)
    set_world_translate(NETWORK_CABLE_ROOT_PATH, root_t + delta)

    _, _, plug_center, plug_dims = get_bbox(TRACKED_PLUG_PRIM_PATH)
    print(f"[SPAWN] plug_center={np.round(plug_center, 5)} dims_mm={np.round(plug_dims * 1000.0, 3)}")


def reload_cable():
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(NETWORK_CABLE_ROOT_PATH))

    add_reference_to_stage(usd_path=NETWORK_CABLE_USD_PATH, prim_path=NETWORK_CABLE_ROOT_PATH)
    place_cable_on_ground()


def get_grasp_target():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(TRACKED_PLUG_PRIM_PATH)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing connector head: {TRACKED_PLUG_PRIM_PATH}")

    xform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return np.array(xform.Transform(Gf.Vec3d(*GRASP_LOCAL_OFFSET)), dtype=np.float64)


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

franka = world.scene.add(SingleManipulator(
    prim_path="/World/Franka",
    name="my_franka",
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    gripper=gripper,
))

reload_cable()
enable_gpu_dynamics()
franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
world.reset()

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
kinematics_solver = LulaKinematicsSolver(**kinematics_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**kinematics_config)
art_kinematics = ArticulationKinematicsSolver(franka, kinematics_solver, "panda_hand")
base_position, base_orientation = franka.get_world_pose()
kinematics_solver.set_robot_base_pose(base_position, base_orientation)

controller = FrankaMotionController(
    name="minimal_pickup_controller",
    robot_articulation=franka,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=franka.gripper,
    tool_offset=0.05,
    debug=False,
)


# ------------------------- Pickup sequence -------------------------

def queue_pickup():
    target = get_grasp_target()
    at_connector = np.array([target[0], target[1], target[2] + GRASP_Z_OFFSET], dtype=np.float64)
    above = np.array([at_connector[0], at_connector[1], ABOVE_Z], dtype=np.float64)

    print("[PICKUP - HAND PARALLEL TO CONNECTOR]")
    print(f"  target={np.round(target, 5)}")
    print(f"  above={np.round(above, 5)}")
    print(f"  at_connector={np.round(at_connector, 5)}")
    print(f"  pickup_ori={np.round(PICKUP_ORI, 5)}")

    controller.clear_queue()
    controller.add_cartesian_waypoint(above, PICKUP_ORI, max_frames=30, pos_tolerance=0.001, label="above")
    controller.add_cartesian_waypoint(at_connector, PICKUP_ORI, max_frames=30, pos_tolerance=0.001, label="at_connector")
    controller.add_gripper_command(action="close", wait_frames=CLOSE_WAIT_FRAMES)
    controller.add_cartesian_waypoint(above, PICKUP_ORI, max_frames=30, pos_tolerance=0.001, hold_gripper=True, label="lift")


def reset_task_for_next_play():
    global phase, settle_count, run_count

    run_count += 1
    print(f"\n[RESET FOR RUN {run_count}]")

    controller.clear_queue()
    reload_cable()                       # clean deformable state every run
    enable_gpu_dynamics()
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    world.reset()                        # rebuilds Physics Simulation View

    base_position, base_orientation = franka.get_world_pose()
    kinematics_solver.set_robot_base_pose(base_position, base_orientation)

    phase = "settle"
    settle_count = 0


def print_done():
    fingers = np.asarray(franka.gripper.get_joint_positions(), dtype=np.float64).flatten()
    _, _, plug_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    print("[DONE]")
    print(f"  fingers_mm={np.round(fingers * 1000.0, 3)} total_mm={np.sum(fingers) * 1000.0:.3f}")
    print(f"  plug_center={np.round(plug_center, 5)}")
    print("  Press Stop, then Play to run again in the same Python process.")


phase = "settle"
settle_count = 0
run_count = 1
was_playing = False
reset_needed_on_next_play = False
print("[READY] Press Play. After a run finishes, press Stop then Play to rerun.")

while simulation_app.is_running():
    world.step(render=True)
    playing = world.is_playing()

    if not playing:
        if was_playing:
            # The user pressed Stop. Do not touch robot articulation here; the Physics Simulation View is gone.
            reset_needed_on_next_play = True
            phase = "waiting_for_play"
        was_playing = False
        continue

    if playing and not was_playing:
        # The user pressed Play. If this is not the first play, rebuild the task cleanly.
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
            queue_pickup()
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
