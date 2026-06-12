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
from isaacsim.core.api.objects import FixedCuboid
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
from port_collision_proxy import (
    PORT_SLEEVE_DEPTH_M,
    build_port_insert_colliders,
    disable_all_port_insert_colliders,
    enable_port_insert_colliders_for_port,
)
from port_frame import PortFrame
from qsfp_insert_job import (
    ALIGN_STANDOFF,
    JOB_CMD_CLEAR,
    JOB_CMD_INSERT,
    MIN_SEATED_TIP_AXIAL_M,
    SEAT_LATERAL_TOL_M,
    SEAT_SETTLE_FRAMES,
    add_qsfp_insert_job,
)
from qsfp_module import (
    QSFP_GRASP_OFFSET_TO_TOP_M,
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
STARTUP_CAMERA_EYE = [1.8809555265110907, 1.5749548281281234, 2.8130942736157993]
STARTUP_CAMERA_TARGET = [-0.40583929622548287, -0.43359704675040334, 0.7505214224942818]
# Set to a positive step count to print the current viewport camera once
# (frame the view manually first, then copy the printed values above).
LOG_VIEWPORT_CAMERA_AT_STEP = 1000

DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)
DATAHALL_SCALE = 1.0
TABLE_HEIGHT = 1.0
TABLE_THICKNESS = 0.05
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
    )
]
MAX_PORTS = len(INSERT_PORT_PRIM_PATHS)
PICK_TRAY_XY = np.array([0.30, 0.30])
PICK_SPACING = 0.08
INSERT_CARTESIAN_STEP = 0.002
# con002228_* connector prims use local +Z as the rack-facing insert normal.
INSERT_LOCAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
INSERT_LATERAL_Z_BY_CONNECTOR = {
    "QSFP_DD_Connector_A_01": -0.00085,
    "QSFP_DD_Connector_A_02": -0.00425,
}
# Per-port world-frame lateral offset overrides (meters). Index matches
# INSERT_PORT_PRIM_PATHS / job order. Use these when two ports share the same
# connector name but need different tuning because they come from different
# switch instances or connector pairs.
INSERT_LATERAL_OFFSET_BY_PORT_INDEX = {
    0: np.array([0.0, 0.0, -0.0105], dtype=np.float64),
    1: np.array([0.0, 0.0, -0.00675], dtype=np.float64),
}
# Default leading-tip depth (m) past the port opening along +insert_axis.
# Override at runtime with the "QSFP Insert" UI panel (stop → play to apply).
INSERT_TIP_DEPTH_M_DEFAULT = 0.048
INSERT_TIP_DEPTH_M_MIN = 0.020
# Stay inside the port sleeve back wall (60 mm) to avoid switch / arm collisions.
INSERT_TIP_DEPTH_M_MAX = PORT_SLEEVE_DEPTH_M - 0.005
# Vertical pick: shift IK block-center toward module +Z (top). Higher = more body below fingers.
PICK_GRASP_OFFSET_TO_TOP_M_DEFAULT = QSFP_GRASP_OFFSET_TO_TOP_M
PICK_GRASP_OFFSET_TO_TOP_M_MIN = 0.0
PICK_GRASP_OFFSET_TO_TOP_M_MAX = QSFP_LENGTH_M * 0.5 * 0.95
ROBOT_BASE_POS = np.array([0.45, -0.15, TABLE_HEIGHT])
PICK_SURFACE_Z = TABLE_HEIGHT + QSFP_LENGTH_M / 2.0
MODULE_SPAWN_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
POST_RESET_WARMUP_FRAMES = 20
POST_RESET_SETTLE_STEPS = 15

_insert_depth_model = ui.SimpleFloatModel(INSERT_TIP_DEPTH_M_DEFAULT)
_pick_grasp_offset_model = ui.SimpleFloatModel(PICK_GRASP_OFFSET_TO_TOP_M_DEFAULT)
_tune_status = ui.SimpleStringModel(
    "Stop, then play to apply slider changes"
)


def build_qsfp_tune_panel() -> None:
    window = ui.Window("QSFP Tuning", width=360, height=175)
    with window.frame:
        with ui.VStack(spacing=6, height=0):
            ui.Label("Pick grasp height — offset toward module top (m)")
            ui.Label(
                "Higher = grip nearer the top, more length hangs below the fingers",
                word_wrap=True,
            )
            ui.FloatDrag(
                _pick_grasp_offset_model,
                min=PICK_GRASP_OFFSET_TO_TOP_M_MIN,
                max=PICK_GRASP_OFFSET_TO_TOP_M_MAX,
                step=0.002,
                format="%.3f",
            )
            ui.Label("Insert tip depth past port face (m)")
            ui.FloatDrag(
                _insert_depth_model,
                min=INSERT_TIP_DEPTH_M_MIN,
                max=INSERT_TIP_DEPTH_M_MAX,
                step=0.005,
                format="%.3f",
            )
            ui.StringField(
                _tune_status,
                read_only=True,
                style={"background_color": 0x00000000},
            )


def get_insert_tip_depth_m() -> float:
    depth = float(_insert_depth_model.get_value_as_float())
    return float(
        np.clip(depth, INSERT_TIP_DEPTH_M_MIN, INSERT_TIP_DEPTH_M_MAX)
    )


def get_pick_grasp_offset_to_top_m() -> float:
    offset = float(_pick_grasp_offset_model.get_value_as_float())
    return float(
        np.clip(
            offset,
            PICK_GRASP_OFFSET_TO_TOP_M_MIN,
            PICK_GRASP_OFFSET_TO_TOP_M_MAX,
        )
    )


build_qsfp_tune_panel()


def insert_lateral_offset_for_path(prim_path: str, port_index: int | None = None) -> np.ndarray:
    if (
        port_index is not None
        and port_index in INSERT_LATERAL_OFFSET_BY_PORT_INDEX
    ):
        return INSERT_LATERAL_OFFSET_BY_PORT_INDEX[port_index].copy()
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
        np.array([PICK_TRAY_XY[0], PICK_TRAY_XY[1], PICK_SURFACE_Z], dtype=np.float64),
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


def select_robot_physics_variant(robot_prim) -> None:
    physics_variant = robot_prim.GetVariantSet("Physics")
    names = list(physics_variant.GetVariantNames())
    if not names:
        carb.log_warn("Franka asset has no Physics variant set.")
        return
    selection = next((name for name in names if name.lower() == "physx"), names[0])
    physics_variant.SetVariantSelection(selection)
    carb.log_info(f"Selected Franka Physics variant: {selection}")

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
select_robot_physics_variant(robot)
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


def build_work_table(world, module_count: int) -> None:
    """Spawn a physics-enabled slab under the robot and module tray."""
    _TABLE_MARGIN_M = 0.20
    _CUBOID_BASE_SIZE = 1.0

    robot_xy = ROBOT_BASE_POS[:2]
    center_xy = (robot_xy + PICK_TRAY_XY) / 2.0

    coverage_x = [robot_xy[0]] + [
        PICK_TRAY_XY[0] + i * PICK_SPACING for i in range(module_count)
    ]
    coverage_y = [robot_xy[1], PICK_TRAY_XY[1]]

    half_x = max(abs(x - center_xy[0]) for x in coverage_x) + _TABLE_MARGIN_M
    half_y = max(abs(y - center_xy[1]) for y in coverage_y) + _TABLE_MARGIN_M
    size_xy = np.array([2.0 * half_x, 2.0 * half_y], dtype=np.float64)

    center_z = TABLE_HEIGHT - TABLE_THICKNESS / 2.0
    position = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
    scale = np.array([size_xy[0], size_xy[1], TABLE_THICKNESS], dtype=np.float64)
    orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    color = np.array([0.55, 0.35, 0.18], dtype=np.float64)

    world.scene.add(
        FixedCuboid(
            name="work_table",
            prim_path="/World/WorkTable",
            position=position,
            orientation=orientation,
            scale=scale,
            size=_CUBOID_BASE_SIZE,
            color=color,
            visible=True,
        )
    )
    carb.log_info(
        f"Work table: center={position.tolist()} size_xy={size_xy.tolist()} "
        f"thickness={TABLE_THICKNESS} top_surface_z={TABLE_HEIGHT}"
    )


def build_ports() -> list:
    built = []
    for port_index, path in enumerate(port_paths):
        if not stage.GetPrimAtPath(path).IsValid():
            carb.log_error(f"Insert port prim not found: {path}")
            continue
        lateral_offset = insert_lateral_offset_for_path(path, port_index)
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
        approach_goal = port.approach_position(ALIGN_STANDOFF)
        insert_goal = port.center_goal_for_tip_depth(
            get_insert_tip_depth_m(),
            QSFP_LENGTH_M / 2.0,
        )
        carb.log_info(
            f"Port frame {port_index}: {path} prim_pos={prim_position} "
            f"prim_ori={prim_orientation} lateral_offset={lateral_offset}"
        )
        carb.log_info(
            f"Port insert: origin={port.insert_origin} axis={port.insert_axis} "
            f"rot={port.insert_rot}"
        )
        carb.log_info(
            f"Port goals: approach={approach_goal} insert_tip_depth_m={get_insert_tip_depth_m()} "
            f"insert_center={insert_goal} "
            f"approach_yz_delta={approach_goal[1:3] - port.insert_origin[1:3]} "
            f"insert_yz_delta={insert_goal[1:3] - port.insert_origin[1:3]}"
        )
    if built:
        build_port_insert_colliders(my_world, stage, built)
    return built


ports: list = []
_cached_ports: list = []

build_work_table(my_world, len(port_paths))

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
        port_index=i,
    )
    modules.append((pick_xy, mod))
    carb.log_info(f"Module {i} at pick_xy={pick_xy} -> port {port_paths[i]}")

my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)

# Build port frames and invisible sleeve colliders before the first physics reset
# so PhysX registers the static shapes. They stay collision-disabled until each
# insert completes, so they do not fight the gripper during the crawl.
_cached_ports = build_ports()
ports = list(_cached_ports)

PHYSICS_DT = 1.0 / 60.0

def _settle_physics_steps() -> None:
    for i in range(POST_RESET_SETTLE_STEPS):
        my_world.step(render=(i >= POST_RESET_SETTLE_STEPS - 3))


def update_view_if_available(obj) -> None:
    update = getattr(obj, "update", None)
    if callable(update):
        try:
            update()
        except TypeError:
            carb.log_warn(f"Could not update runtime view for {getattr(obj, 'name', obj)}.")


def flush_runtime_views() -> None:
    update_view_if_available(my_franka)
    for _, mod in modules:
        update_view_if_available(mod)


def get_ports(rebuild: bool = False) -> list:
    global _cached_ports
    if rebuild or not _cached_ports:
        _cached_ports = build_ports()
    return list(_cached_ports)


my_world.reset()
_settle_physics_steps()
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
    insert_depth_m = get_insert_tip_depth_m()
    pick_grasp_offset_m = get_pick_grasp_offset_to_top_m()
    _tune_status.set_value(
        f"Queued {len(modules)} jobs: pick_offset={pick_grasp_offset_m:.3f} m "
        f"insert_depth={insert_depth_m:.3f} m"
    )
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
            insert_tip_depth_m=insert_depth_m,
            pick_grasp_offset_to_top_m=pick_grasp_offset_m,
        )
        tip_goal = port.center_goal_for_tip_depth(
            insert_depth_m,
            QSFP_LENGTH_M / 2.0,
        )
        msg = (
            f"Queued insert job {i} pick_grasp_offset={pick_grasp_offset_m:.3f} m "
            f"insert_tip_depth={insert_depth_m:.3f} m "
            f"center_goal_axial={port.axial_coordinate(tip_goal):.4f} m "
            f"cmd_index={job_start_indices[-1]}"
        )
        carb.log_info(msg)
        print(msg)
    port_frames_by_job = list(ports)
    carb.log_info(f"Running {len(ports)} insert jobs ({len(modules)} modules)")


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
    flush_runtime_views()


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
    flush_runtime_views()
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
    global _post_reset_warmup_frames, _run_ready
    msg = "Simulation restart: staging scene state before warmup."
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
    franka_controller.reset()
    register_robot_base_pose()
    done_msg = (
        f"Simulation restart staged: warmup={POST_RESET_WARMUP_FRAMES} frames."
    )
    carb.log_info(done_msg)
    print(done_msg)
    _run_ready = False
    _post_reset_warmup_frames = POST_RESET_WARMUP_FRAMES
    simulation_app.update()
    if VIEWPORT_ONLY_LAYOUT:
        configure_viewport_only_layout()
    apply_startup_camera_view()


def prepare_run_after_warmup() -> bool:
    global ports, _run_ready
    flush_runtime_views()
    reset_modules_to_pick()
    reset_robot_to_home()
    _settle_physics_steps()
    flush_runtime_views()
    ports = get_ports(rebuild=True)
    if not ports:
        carb.log_error("No valid port frames.")
        return False
    disable_all_port_insert_colliders(stage)
    queue_all_insert_jobs()
    franka_controller.reset()
    register_robot_base_pose()
    log_reset_state("After warmup:")
    msg = (
        f"Run ready: {len(ports)} ports, "
        f"{len(franka_controller._command_queue)} queued commands."
    )
    carb.log_info(msg)
    print(msg)
    _run_ready = True
    return True


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
    """Log-only pick check; never rewind or open the gripper (that caused drops)."""
    return


def enable_port_colliders_after_insert(completed_cmd_idx: int) -> None:
    """Enable the matching sleeve after insertion, before opening the gripper."""
    for job_idx, start_idx in enumerate(job_start_indices):
        if completed_cmd_idx != start_idx + JOB_CMD_INSERT:
            continue
        enable_port_insert_colliders_for_port(stage, job_idx)
        print(f"Enabled port sleeve collisions for job {job_idx}.")
        return


def log_job_completion_seat(completed_cmd_idx: int) -> None:
    for job_idx, start_idx in enumerate(job_start_indices):
        # Check once right after insert and again after the final rack-clear retreat.
        if completed_cmd_idx == start_idx + JOB_CMD_INSERT:
            label = "post-insert"
        elif completed_cmd_idx == start_idx + JOB_CMD_CLEAR:
            label = "post-retreat"
        else:
            continue
        port = port_frames_by_job[job_idx]
        _, mod = modules[job_idx]
        flush_runtime_views()
        pos, ori = mod.get_world_pose()
        passed, metrics = port.evaluate_seat(
            pos,
            seat_depth=MIN_SEATED_TIP_AXIAL_M,
            module_orientation=ori,
            module_half_length=QSFP_LENGTH_M * 0.5,
            lateral_tol=SEAT_LATERAL_TOL_M,
            depth_fraction=1.0,
        )
        msg = (
            f"Job {job_idx} {label} seat check passed={passed} "
            f"lateral={metrics['lateral_error_m']:.4f}m "
            f"tip_axial={metrics['tip_axial_m']:.4f}m"
        )
        carb.log_info(msg)
        print(msg)
        return


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
_post_reset_warmup_frames = POST_RESET_WARMUP_FRAMES
_run_ready = False

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

        if _post_reset_warmup_frames > 0:
            my_world.step(render=True)
            if not my_world.is_playing():
                reset_needed = True
                task_completed = False
                msg = "Simulation stopped — will fully reset on next play."
                carb.log_info(msg)
                print(msg)
                _was_playing = playing
                continue
            _post_reset_warmup_frames -= 1
            if _post_reset_warmup_frames == 0:
                if prepare_run_after_warmup():
                    carb.log_info("Warmup complete — starting control.")
                    print("Warmup complete — starting control.")
        else:
            if not _run_ready and not prepare_run_after_warmup():
                my_world.step(render=True)
                _was_playing = playing
                continue

            if VIEWPORT_ONLY_LAYOUT and _viewport_layout_frames < 120:
                _viewport_layout_frames += 1
                configure_viewport_only_layout()

            if LOG_VIEWPORT_CAMERA_AT_STEP > 0 and not _viewport_camera_logged:
                _viewport_camera_log_step += 1
                if _viewport_camera_log_step >= LOG_VIEWPORT_CAMERA_AT_STEP:
                    log_viewport_camera_for_config()
                    _viewport_camera_logged = True

            flush_runtime_views()
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
                tracked_ori = None
                if not franka_controller.is_done():
                    current_cmd = franka_controller._command_queue[cmd_idx]
                    if (
                        current_cmd.get("type") == "cartesian"
                        and current_cmd.get("track_block")
                    ):
                        flush_runtime_views()
                        tracked_pos, tracked_ori = module.get_world_pose()
                        if tracked_ori is not None and hasattr(tracked_ori, "cpu"):
                            tracked_ori = tracked_ori.cpu().numpy()
                        if tracked_ori is not None:
                            tracked_ori = np.asarray(tracked_ori, dtype=np.float64)

                actions = franka_controller.forward(
                    current_joint_positions=current_joint_pos,
                    current_tracked_position=tracked_pos,
                    current_tracked_orientation=tracked_ori,
                )
                articulation_controller.apply_action(actions)

            my_world.step(render=True)

            if not my_world.is_playing():
                reset_needed = True
                task_completed = False
                msg = "Simulation stopped — will fully reset on next play."
                carb.log_info(msg)
                print(msg)
                _was_playing = playing
                continue

            flush_runtime_views()
            new_cmd_idx = franka_controller._current_command_index
            if new_cmd_idx != _prev_controller_cmd_idx:
                check_pick_after_lift(_prev_controller_cmd_idx)
                enable_port_colliders_after_insert(_prev_controller_cmd_idx)
                log_job_completion_seat(_prev_controller_cmd_idx)
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
                    flush_runtime_views()
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
