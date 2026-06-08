# Core Isaac Sim App Initialization
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys

import carb
import numpy as np
import omni.timeline
import omni.ui as ui
import omni.usd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.xforms import get_world_pose
try:
    from isaacsim.core.utils.viewports import set_camera_view
except Exception:
    set_camera_view = None
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    LulaTaskSpaceTrajectoryGenerator,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path

from collision_setup import (
    apply_datahall_scale,
    enable_articulation_collisions,
    enable_static_collisions,
)
from franka_lula_controller import FrankaLulaController
from port_frame import PortFrame
from qsfp_insert_job import (
    APPROACH_STANDOFF,
    INSERT_STOP_DEPTH_M,
    MIN_SEATED_TIP_AXIAL_M,
    SEAT_LATERAL_TOL_M,
    SEAT_SETTLE_FRAMES,
    add_qsfp_insert_job,
)
from qsfp_module import (
    QSFP_LENGTH_M,
    create_qsfp_module,
    grasp_tool_offset,
    gripper_closed_positions,
)

DEBUG_TRAJ_LOG = True
VIEWPORT_ONLY_LAYOUT = False
# Only these exact viewport titles stay visible.
VIEWPORT_KEEP_TITLES = ("Viewport", "Viewport 1")
# Extra windows Isaac may spawn that should always be hidden.
VIEWPORT_HIDE_TOKENS = ("Sensors Output",)
AUTO_CAMERA_ON_OPEN = True
# Set both to fixed [x, y, z] lists to use a custom view on open.
# Leave as None to auto-frame the robot, pick tray, and ports.
STARTUP_CAMERA_EYE = [1.3438452186512029, 1.2295915975598826, 1.6191613759601924]
STARTUP_CAMERA_TARGET = [-0.20123143762967688, -0.1274897566344806, 0.22558066095187224]
# Set to a positive step count to print the current viewport camera once
# (frame the view manually first, then copy the printed values above).
LOG_VIEWPORT_CAMERA_AT_STEP = 900

DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)
DATAHALL_SCALE = 0.5
INSERT_PORT_PRIM_PATHS = [
    (
        "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
        "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
        "pcb003636_idf_01/Connector_Quad_04/Connector_Pair_04/"
        "QSFP_DD_Connector_A_02/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
    ),    
    (
        "/World/DataHall/Network_Switches/SN4600C_CS2FC_01/msn4600_cs2fc_01/"
        "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
        "pcb003636_idf_01/Connector_Quad_02/Connector_Pair_01/"
        "QSFP_DD_Connector_A_01/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
    ),
    (
        "/World/DataHall/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/"
        "SN4600C_A_01/msn4600_cs2fc_base/SM4600_CS2FC_01/NetworkConnectors/"
        "pcb003636_idf_01/Connector_Quad_04/Connector_Pair_04/"
        "QSFP_DD_Connector_A_01/QSFP_DD_Connector_01/con002228_13_15/con002228_13"
    )
]
MAX_PORTS = len(INSERT_PORT_PRIM_PATHS)
PICK_TRAY_XY = np.array([0.30, 0.30])
PICK_SPACING = 0.08
INSERT_CARTESIAN_STEP = 0.002
# con002228_* connector prims use local +Z as the rack-facing insert normal.
INSERT_LOCAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
INSERT_LATERAL_Z_BY_CONNECTOR = {
    "QSFP_DD_Connector_A_01": -0.00075,
    "QSFP_DD_Connector_A_02": -0.00375,
}
ROBOT_BASE_POS = np.array([0.45, -0.15, 0.0])
MODULE_SPAWN_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
POST_RESET_WARMUP_FRAMES = 60
POST_RESET_SETTLE_STEPS = 10


def insert_lateral_offset_for_path(prim_path: str) -> np.ndarray:
    for connector_name, z_offset in INSERT_LATERAL_Z_BY_CONNECTOR.items():
        if connector_name in prim_path:
            return np.array([0.0, 0.0, z_offset], dtype=np.float64)
    carb.log_warn(
        f"No connector-specific lateral Z offset for {prim_path}; using 0."
    )
    return np.zeros(3, dtype=np.float64)


def _should_keep_viewport_window(title: str) -> bool:
    if any(token in title for token in VIEWPORT_HIDE_TOKENS):
        return False
    return title in VIEWPORT_KEEP_TITLES


def configure_viewport_only_layout() -> None:
    """Hide dock windows while keeping the live viewport window alive."""
    if not VIEWPORT_ONLY_LAYOUT:
        return

    try:
        for window in ui.Workspace.get_windows():
            title = getattr(window, "title", str(window))
            ui.Workspace.show_window(title, _should_keep_viewport_window(title))

        for viewport_title in VIEWPORT_KEEP_TITLES:
            viewport = ui.Workspace.get_window(viewport_title)
            if viewport is not None:
                ui.Workspace.show_window(viewport_title, True)
    except Exception as exc:
        carb.log_warn(f"Could not apply viewport-only layout: {exc}")


def work_area_camera_target() -> np.ndarray:
    points = [
        ROBOT_BASE_POS + np.array([0.0, 0.0, 0.25], dtype=np.float64),
        np.array([PICK_TRAY_XY[0], PICK_TRAY_XY[1], QSFP_LENGTH_M / 2.0], dtype=np.float64),
    ]
    points.extend(port.insert_origin for port in ports)
    target = np.mean(np.asarray(points, dtype=np.float64), axis=0)
    target[2] += 0.1
    return target


def startup_camera_pose() -> tuple[np.ndarray, np.ndarray]:
    if STARTUP_CAMERA_EYE is not None and STARTUP_CAMERA_TARGET is not None:
        return (
            np.asarray(STARTUP_CAMERA_EYE, dtype=np.float64),
            np.asarray(STARTUP_CAMERA_TARGET, dtype=np.float64),
        )
    target = work_area_camera_target()
    eye = target + np.array([1.2, -0.8, 0.7], dtype=np.float64)
    return eye, target


def apply_startup_camera_view() -> None:
    if not AUTO_CAMERA_ON_OPEN or set_camera_view is None:
        return
    eye, target = startup_camera_pose()
    set_camera_view(eye=eye.tolist(), target=target.tolist())
    carb.log_info(f"Startup camera eye={eye.tolist()} target={target.tolist()}")


def read_viewport_camera_pose() -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        viewport_api = get_active_viewport()
        if viewport_api is None:
            return None, None
        camera_path = viewport_api.get_active_camera()
        state = ViewportCameraState(camera_path, viewport_api)
        return (
            np.asarray(state.position_world, dtype=np.float64),
            np.asarray(state.target_world, dtype=np.float64),
        )
    except Exception as exc:
        carb.log_warn(f"Could not read viewport camera pose: {exc}")
        return None, None


def log_viewport_camera_for_config() -> None:
    eye, target = read_viewport_camera_pose()
    if eye is None or target is None:
        return
    msg = (
        "Copy these into insert_at_prim_lula.py:\n"
        f"STARTUP_CAMERA_EYE = {eye.tolist()}\n"
        f"STARTUP_CAMERA_TARGET = {target.tolist()}"
    )
    carb.log_info(msg)
    print(msg)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)
my_world.get_physics_context().enable_ccd(True)

add_reference_to_stage(usd_path=DATAHALL_USD, prim_path="/World/DataHall")
apply_datahall_scale("/World/DataHall", DATAHALL_SCALE)
switch_collider_count = enable_static_collisions(
    "/World/DataHall/Network_Switches", "none"
)
carb.log_info(
    f"Runtime colliders enabled on {switch_collider_count} Network_Switches prims"
)

stage = omni.usd.get_context().get_stage()

add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
    prim_path="/World/ground",
)

robot = add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path="/World/Franka",
)
robot.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
robot.GetVariantSet("Mesh").SetVariantSelection("Quality")
enable_articulation_collisions("/World/Franka")

closed_grip = gripper_closed_positions()
gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=closed_grip,
    action_deltas=np.array([0.02, 0.02]),
)

port_paths = INSERT_PORT_PRIM_PATHS[:MAX_PORTS]
if not port_paths:
    carb.log_error("No insert port prim paths configured.")
    simulation_app.close()
    sys.exit()


def build_ports() -> list:
    built = []
    for path in port_paths:
        if not stage.GetPrimAtPath(path).IsValid():
            carb.log_error(f"Insert port prim not found: {path}")
            continue
        lateral_offset = insert_lateral_offset_for_path(path)
        port = PortFrame.from_prim_path(
            path,
            local_insert_axis=INSERT_LOCAL_AXIS,
            lateral_offset=lateral_offset,
            robot_position=ROBOT_BASE_POS,
        )
        if port is None:
            carb.log_warn(f"Skipping invalid port: {path}")
            continue
        built.append(port)
        prim_position, prim_orientation = get_world_pose(path)
        approach_goal = port.approach_position(APPROACH_STANDOFF)
        insert_goal = port.point_along_axis(INSERT_STOP_DEPTH_M)
        carb.log_info(
            f"Port frame: {path} prim_pos={prim_position} prim_ori={prim_orientation} "
            f"lateral_offset={lateral_offset}"
        )
        carb.log_info(
            f"Port insert: origin={port.insert_origin} axis={port.insert_axis} "
            f"rot={port.insert_rot}"
        )
        carb.log_info(
            f"Port goals: approach={approach_goal} insert={insert_goal} "
            f"approach_yz_delta={approach_goal[1:3] - port.insert_origin[1:3]} "
            f"insert_yz_delta={insert_goal[1:3] - port.insert_origin[1:3]}"
        )
    return built


ports: list = []

PICK_SURFACE_Z = QSFP_LENGTH_M / 2.0

my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="my_franka",
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        gripper=gripper,
        position=ROBOT_BASE_POS,
    )
)

modules = []
for i in range(len(port_paths)):
    pick_xy = PICK_TRAY_XY + np.array([i * PICK_SPACING, 0.0])
    mod = create_qsfp_module(
        my_world,
        prim_path=f"/World/QSFP_Module_{i}",
        name=f"qsfp_module_{i}",
        position=np.array([pick_xy[0], pick_xy[1], PICK_SURFACE_Z]),
    )
    modules.append((pick_xy, mod))
    carb.log_info(f"Module {i} at pick_xy={pick_xy} -> port {port_paths[i]}")

my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)

PHYSICS_DT = 1.0 / 60.0

my_world.reset()
ROBOT_HOME_POSITION, ROBOT_HOME_ORIENTATION = my_franka.get_world_pose()
_home_joints = my_franka.get_joint_positions()
if _home_joints is not None:
    if hasattr(_home_joints, "cpu"):
        _home_joints = _home_joints.cpu().numpy()
    ROBOT_HOME_JOINT_POSITIONS = np.asarray(_home_joints, dtype=np.float64)
else:
    ROBOT_HOME_JOINT_POSITIONS = None
if hasattr(my_franka, "post_reset"):
    my_franka.post_reset()
ports = build_ports()
if not ports:
    carb.log_error("No valid port frames.")
    simulation_app.close()
    sys.exit()
simulation_app.update()
configure_viewport_only_layout()
apply_startup_camera_view()
simulation_app.update()

lula_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
lula_kinematics = LulaKinematicsSolver(**lula_config)
task_traj_gen = LulaTaskSpaceTrajectoryGenerator(**lula_config)
art_kinematics = ArticulationKinematicsSolver(my_franka, lula_kinematics, "panda_hand")

base_pos, base_ori = my_franka.get_world_pose()
lula_kinematics.set_robot_base_pose(base_pos, base_ori)
articulation_controller = my_franka.get_articulation_controller()

franka_controller = FrankaLulaController(
    name="franka_controller",
    robot_articulation=my_franka,
    lula_kinematics=lula_kinematics,
    task_traj_gen=task_traj_gen,
    art_kinematics=art_kinematics,
    gripper=my_franka.gripper,
    tool_offset=grasp_tool_offset(),
    physics_dt=PHYSICS_DT,
    position_tolerance=0.005,
    orientation_tolerance=0.02,
    cartesian_step=INSERT_CARTESIAN_STEP,
    settle_frames=10,
    insert_orientation_tolerance=0.005,
)

job_start_indices: list[int] = []
port_frames_by_job: list = []


def queue_all_insert_jobs() -> None:
    global job_start_indices, port_frames_by_job
    if len(ports) != len(modules):
        carb.log_error(
            f"Port/module count mismatch: {len(ports)} ports, {len(modules)} modules."
        )
        return
    franka_controller._command_queue.clear()
    job_start_indices = []
    for i, (port, (pick_xy, _mod)) in enumerate(zip(ports, modules)):
        job_start_indices.append(len(franka_controller._command_queue))
        franka_controller.reset_grasp_calibration()
        add_qsfp_insert_job(
            franka_controller,
            port,
            pick_xy=pick_xy,
            pick_z=PICK_SURFACE_Z,
        )
        carb.log_info(
            f"Queued insert job {i} starting at command index {job_start_indices[-1]}"
        )
    port_frames_by_job = list(ports)
    carb.log_info(f"Running {len(ports)} insert jobs ({len(modules)} modules)")


def _settle_physics_steps() -> None:
    for i in range(POST_RESET_SETTLE_STEPS):
        my_world.step(render=(i >= POST_RESET_SETTLE_STEPS - 3))


def reset_modules_to_pick() -> None:
    spawn_ori = np.asarray(MODULE_SPAWN_ORIENTATION, dtype=np.float64).reshape(4)
    for pick_xy, mod in modules:
        pos = np.array([pick_xy[0], pick_xy[1], PICK_SURFACE_Z], dtype=np.float64)
        try:
            mod.set_world_pose(position=pos, orientation=spawn_ori)
        except ValueError as exc:
            carb.log_warn(
                f"set_world_pose failed for {getattr(mod, 'name', mod)} ({exc}); "
                "will retry after physics settles."
            )
            continue
        if hasattr(mod, "set_linear_velocity"):
            mod.set_linear_velocity(np.zeros(3, dtype=np.float64))
        if hasattr(mod, "set_angular_velocity"):
            mod.set_angular_velocity(np.zeros(3, dtype=np.float64))


def reset_robot_to_home() -> None:
    my_franka.set_world_pose(
        position=np.asarray(ROBOT_HOME_POSITION, dtype=np.float64),
        orientation=np.asarray(ROBOT_HOME_ORIENTATION, dtype=np.float64),
    )
    if ROBOT_HOME_JOINT_POSITIONS is not None:
        joints = np.asarray(ROBOT_HOME_JOINT_POSITIONS, dtype=np.float64)
        my_franka.set_joint_positions(joints)
        if hasattr(my_franka, "set_joint_velocities"):
            my_franka.set_joint_velocities(np.zeros_like(joints))

    opened = np.asarray(my_franka.gripper.joint_opened_positions, dtype=np.float64)
    my_franka.gripper.set_joint_positions(opened)
    my_franka.gripper.set_default_state(opened)


def restart_job_at_pick(job_idx: int) -> None:
    """Rewind the controller to the start of one insert job."""
    franka_controller._current_command_index = job_start_indices[job_idx]
    franka_controller._clear_segment_playback()
    franka_controller.reset_grasp_calibration()
    opened = np.asarray(my_franka.gripper.joint_opened_positions, dtype=np.float64)
    my_franka.gripper.set_joint_positions(opened)
    carb.log_info(f"Restarting job {job_idx} at command {job_start_indices[job_idx]}")


def log_reset_state(label: str) -> None:
    for i, (pick_xy, mod) in enumerate(modules):
        pos, _ = mod.get_world_pose()
        carb.log_info(
            f"{label} module {i} pick_xy={pick_xy.tolist()} actual_pos={pos.tolist()}"
        )
    for i, port in enumerate(ports):
        carb.log_info(f"{label} port {i} insert_origin={port.insert_origin.tolist()}")
    carb.log_info(
        f"{label} job_start_indices={job_start_indices} "
        f"queued_commands={len(franka_controller._command_queue)}"
    )


def on_simulation_reset() -> None:
    global ports, _post_reset_warmup_frames
    msg = "Simulation restart: rebuilding ports, jobs, and scene state."
    carb.log_info(msg)
    print(msg)
    my_world.reset()
    if hasattr(my_franka, "post_reset"):
        my_franka.post_reset()
    for _, mod in modules:
        if hasattr(mod, "post_reset"):
            mod.post_reset()
    _settle_physics_steps()
    reset_robot_to_home()
    reset_modules_to_pick()
    ports = build_ports()
    if not ports:
        carb.log_error("No valid port frames after simulation reset.")
        return
    queue_all_insert_jobs()
    _settle_physics_steps()
    reset_modules_to_pick()
    reset_robot_to_home()
    franka_controller.reset()
    register_robot_base_pose()
    log_reset_state("After simulation reset:")
    done_msg = (
        f"Simulation restart complete: {len(ports)} ports, "
        f"{len(franka_controller._command_queue)} queued commands, "
        f"warmup={POST_RESET_WARMUP_FRAMES} frames."
    )
    carb.log_info(done_msg)
    print(done_msg)
    _post_reset_warmup_frames = POST_RESET_WARMUP_FRAMES
    simulation_app.update()
    configure_viewport_only_layout()
    apply_startup_camera_view()


queue_all_insert_jobs()


def register_robot_base_pose():
    base_pos, base_ori = my_franka.get_world_pose()
    lula_kinematics.set_robot_base_pose(base_pos, base_ori)


def active_job_index(cmd_index: int) -> int:
    idx = 0
    for i, start in enumerate(job_start_indices):
        if cmd_index >= start:
            idx = i
    return idx


def active_module(cmd_index: int):
    return modules[active_job_index(cmd_index)][1]


def check_pick_after_lift(completed_cmd_idx: int) -> None:
    """Verify the module left the tray after the lift waypoint; retry grasp once."""
    if completed_cmd_idx < 0 or completed_cmd_idx >= len(franka_controller._command_queue):
        return
    prev_cmd = franka_controller._command_queue[completed_cmd_idx]
    if prev_cmd.get("type") != "cartesian" or not prev_cmd.get("verify_pick_lift"):
        return

    job_idx = active_job_index(completed_cmd_idx)
    _, mod = modules[job_idx]
    pos, _ = mod.get_world_pose()
    min_lift_z = prev_cmd.get("verify_pick_min_z")
    if min_lift_z is None:
        from qsfp_insert_job import HOVER_CLEARANCE

        min_lift_z = PICK_SURFACE_Z + HOVER_CLEARANCE - 0.08
    if float(pos[2]) >= float(min_lift_z):
        return

    msg = (
        f"Pick failed for job {job_idx} — module did not lift "
        f"(z={pos[2]:.4f}, expected >= {float(min_lift_z):.4f})"
    )
    carb.log_warn(msg)
    print(msg)

    if _pick_retry_by_job.get(job_idx, False):
        if not _full_job_retry_by_job.get(job_idx, False):
            _full_job_retry_by_job[job_idx] = True
            _pick_retry_by_job[job_idx] = False
            restart_job_at_pick(job_idx)
            retry_msg = f"Restarting full job {job_idx} after repeated pick failure."
            carb.log_warn(retry_msg)
            print(retry_msg)
        return

    _pick_retry_by_job[job_idx] = True
    grasp_idx = job_start_indices[job_idx] + 1
    franka_controller._current_command_index = grasp_idx
    franka_controller._clear_segment_playback()
    opened = np.asarray(my_franka.gripper.joint_opened_positions, dtype=np.float64)
    my_franka.gripper.set_joint_positions(opened)
    carb.log_info(f"Retrying pick for job {job_idx} from command {grasp_idx}")


reset_needed = False
task_completed = False
_last_cmd_index = -1
_prev_controller_cmd_idx = 0
_job_results = []
_seat_settle_count = 0
_viewport_camera_log_step = 0
_viewport_camera_logged = False
_viewport_layout_frames = 0
_was_playing = False
_has_seen_play = False
_timeline = omni.timeline.get_timeline_interface()
_last_timeline_time = 0.0
_pick_retry_by_job: dict[int, bool] = {}
_full_job_retry_by_job: dict[int, bool] = {}
_post_reset_warmup_frames = 0

while simulation_app.is_running():
    playing = my_world.is_playing()
    timeline_time = float(_timeline.get_current_time())

    if _was_playing and not playing:
        reset_needed = True
        task_completed = False
        msg = "Simulation stopped — will fully reset on next play."
        carb.log_info(msg)
        print(msg)

    if (
        not playing
        and _has_seen_play
        and timeline_time + 1e-6 < _last_timeline_time
    ):
        reset_needed = True
        task_completed = False
        msg = "Timeline reset — will fully reset on next play."
        carb.log_info(msg)
        print(msg)

    if (
        playing
        and task_completed
        and franka_controller.is_done()
        and timeline_time + 1e-6 < _last_timeline_time
    ):
        reset_needed = True
        task_completed = False
        msg = "Run complete + timeline rewind — will fully reset."
        carb.log_info(msg)
        print(msg)

    _last_timeline_time = timeline_time

    if playing:
        if _has_seen_play and reset_needed:
            on_simulation_reset()
            reset_needed = False
            task_completed = False
            _last_cmd_index = -1
            _prev_controller_cmd_idx = 0
            _job_results = []
            _seat_settle_count = 0
            _viewport_layout_frames = 0
            _pick_retry_by_job = {}
            _full_job_retry_by_job = {}

        _has_seen_play = True

        my_world.step(render=True)

        if not my_world.is_playing():
            reset_needed = True
            task_completed = False
            msg = "Simulation stopped — will fully reset on next play."
            carb.log_info(msg)
            print(msg)
        elif _post_reset_warmup_frames > 0:
            _post_reset_warmup_frames -= 1
            if _post_reset_warmup_frames == 0:
                reset_robot_to_home()
                register_robot_base_pose()
                carb.log_info("Post-reset warmup complete — starting control.")
                print("Post-reset warmup complete — starting control.")
        else:
            if VIEWPORT_ONLY_LAYOUT and _viewport_layout_frames < 120:
                _viewport_layout_frames += 1
                configure_viewport_only_layout()

            if LOG_VIEWPORT_CAMERA_AT_STEP > 0 and not _viewport_camera_logged:
                _viewport_camera_log_step += 1
                if _viewport_camera_log_step >= LOG_VIEWPORT_CAMERA_AT_STEP:
                    log_viewport_camera_for_config()
                    _viewport_camera_logged = True

            current_joint_pos = my_franka.get_joint_positions()
            if current_joint_pos is None:
                carb.log_warn("Skipping control update: joint positions unavailable.")
            else:
                if hasattr(current_joint_pos, "cpu"):
                    current_joint_pos = current_joint_pos.cpu().numpy()
                current_joint_pos = np.asarray(current_joint_pos, dtype=np.float64)

                cmd_idx = franka_controller._current_command_index
                module = active_module(cmd_idx)

                tracked_pos = None
                if not franka_controller.is_done():
                    current_cmd = franka_controller._command_queue[cmd_idx]
                    if (
                        current_cmd.get("type") == "cartesian"
                        and current_cmd.get("track_block")
                    ):
                        tracked_pos, _ = module.get_world_pose()

                actions = franka_controller.forward(
                    current_joint_positions=current_joint_pos,
                    current_tracked_position=tracked_pos,
                )
                articulation_controller.apply_action(actions)

                new_cmd_idx = franka_controller._current_command_index
                if new_cmd_idx != _prev_controller_cmd_idx:
                    check_pick_after_lift(_prev_controller_cmd_idx)
                    _prev_controller_cmd_idx = new_cmd_idx

                if DEBUG_TRAJ_LOG:
                    dbg = franka_controller.get_traj_debug_state()
                    if dbg is not None and dbg.get("cmd_index") != _last_cmd_index:
                        _last_cmd_index = dbg["cmd_index"]
                        err_parts = []
                        for key in (
                            "x_err",
                            "y_err",
                            "z_err",
                            "yz_err",
                            "y_drift",
                            "z_drift",
                            "yz_drift",
                        ):
                            if key in dbg:
                                err_parts.append(f"{key}={dbg[key]:.4f}m")
                        err_msg = " " + " ".join(err_parts) if err_parts else ""
                        carb.log_info(
                            f"lula wp={dbg['cmd_index']} "
                            f"job={active_job_index(dbg['cmd_index'])} "
                            f"mode={dbg.get('mode')}{err_msg}"
                        )

                if franka_controller.is_done() and not task_completed:
                    if _seat_settle_count < SEAT_SETTLE_FRAMES:
                        _seat_settle_count += 1
                    else:
                        for i, port in enumerate(port_frames_by_job):
                            _, mod = modules[i]
                            pos, ori = mod.get_world_pose()
                            passed, metrics = port.evaluate_seat(
                                pos,
                                seat_depth=MIN_SEATED_TIP_AXIAL_M,
                                module_orientation=ori,
                                module_half_length=QSFP_LENGTH_M * 0.5,
                                lateral_tol=SEAT_LATERAL_TOL_M,
                                depth_fraction=1.0,
                            )
                            _job_results.append((port.prim_path, passed, metrics))
                            msg = (
                                f"Job {i} seat check path={port.prim_path} passed={passed} "
                                f"lateral={metrics['lateral_error_m']:.4f}m "
                                f"center_axial={metrics['axial_depth_m']:.4f}m "
                                f"tip_axial={metrics['tip_axial_m']:.4f}m "
                                f"lateral_ok={metrics['lateral_ok']} "
                                f"depth_ok={metrics['depth_ok']}"
                            )
                            carb.log_info(msg)
                            print(msg)
                        all_pass = all(r[1] for r in _job_results)
                        print(
                            f"Wire insert run complete. "
                            f"{sum(r[1] for r in _job_results)}/{len(_job_results)} seated."
                        )
                        if not all_pass:
                            carb.log_warn("One or more inserts did not meet seat criteria.")
                        task_completed = True

    else:
        simulation_app.update()

    _was_playing = playing

simulation_app.close()
