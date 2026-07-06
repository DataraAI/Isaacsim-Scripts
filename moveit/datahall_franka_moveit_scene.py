from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})

import os
import sys

import carb
import numpy as np
import omni
import omni.graph.core as og
import omni.timeline

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path


DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)

# Keep /panda. The official isaac_moveit launch/config expects this robot prim.
ROBOT_PRIM_PATH = "/panda"
END_EFFECTOR_PRIM_PATH = "/panda/panda_rightfinger"

JOINT_STATES_TOPIC = "/isaac_joint_states"
JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"

# Same general "ready" posture used by common Franka/Panda MoveIt configs.
# Starting here keeps RViz's <current> start state out of self-collision.
PANDA_READY_JOINT_POSITIONS = np.array(
    [
        0.0,
        -0.7853981633974483,
        0.0,
        -2.356194490192345,
        0.0,
        1.5707963267948966,
        0.7853981633974483,
        0.05,
        0.05,
    ]
)


def log(message: str) -> None:
    print(f"[DATAHALL_MOVEIT_V3] {message}", flush=True)
    carb.log_info(f"[DATAHALL_MOVEIT_V3] {message}")


def update_frames(count: int, reason: str) -> None:
    log(f"Updating {count} frames: {reason}")
    for _ in range(count):
        simulation_app.update()


def enable_ros2_extensions() -> None:
    log("Enabling ROS2 extensions")
    manager = omni.kit.app.get_app().get_extension_manager()
    for ext in ("isaacsim.ros2.bridge", "isaacsim.ros2.nodes"):
        if not manager.is_extension_enabled(ext):
            manager.set_extension_enabled_immediate(ext, True)
            log(f"Enabled extension: {ext}")
        else:
            log(f"Extension already enabled: {ext}")
    update_frames(20, "after enabling ROS2 extensions")


def create_moveit_joint_graph() -> None:
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/ActionGraph").IsValid():
        log("/ActionGraph already exists; reusing it")
        return

    log("Creating ROS2 MoveIt joint command/state ActionGraph")
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", ROBOT_PRIM_PATH),
                ("PublishJointState.inputs:targetPrim", ROBOT_PRIM_PATH),
                ("PublishJointState.inputs:topicName", JOINT_STATES_TOPIC),
                ("SubscribeJointState.inputs:topicName", JOINT_COMMANDS_TOPIC),
                ("PublishClock.inputs:topicName", "clock"),
            ],
        },
    )
    log(f"Graph topics: publish={JOINT_STATES_TOPIC}, subscribe={JOINT_COMMANDS_TOPIC}")


def main() -> None:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit(1)

    if not os.path.exists(DATAHALL_USD):
        carb.log_error(f"DATAHALL_USD does not exist: {DATAHALL_USD}")
        simulation_app.close()
        sys.exit(1)

    log("Starting DataHall Franka MoveIt scene v3")
    enable_ros2_extensions()

    my_world = World(stage_units_in_meters=1.0)

    log(f"Loading DataHall via add_reference_to_stage: {DATAHALL_USD}")
    add_reference_to_stage(usd_path=DATAHALL_USD, prim_path="/World/DataHall")
    update_frames(120, "after loading DataHall")

    franka_usd = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    log(f"Loading Franka via add_reference_to_stage: {franka_usd}")
    robot_prim = add_reference_to_stage(usd_path=franka_usd, prim_path=ROBOT_PRIM_PATH)
    robot_prim.GetVariantSet("Gripper").SetVariantSelection("AlternateFinger")
    robot_prim.GetVariantSet("Mesh").SetVariantSelection("Quality")

    gripper = ParallelGripper(
        end_effector_prim_path=END_EFFECTOR_PRIM_PATH,
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05]),
        joint_closed_positions=np.array([0.005, 0.005]),
        action_deltas=np.array([0.02, 0.02]),
    )

    my_franka = my_world.scene.add(
        SingleManipulator(
            prim_path=ROBOT_PRIM_PATH,
            name="panda",
            end_effector_prim_path=END_EFFECTOR_PRIM_PATH,
            gripper=gripper,
            position=np.array([0.45, -0.25, 0.0]),
        )
    )
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    my_franka.set_joints_default_state(positions=PANDA_READY_JOINT_POSITIONS)

    log("Resetting Isaac World")
    my_world.reset()
    my_franka.set_joint_positions(PANDA_READY_JOINT_POSITIONS)
    update_frames(60, "after World.reset")

    create_moveit_joint_graph()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    log("=" * 80)
    log("Scene is running. Now launch RViz in another terminal:")
    log("ros2 launch isaac_moveit isaac_moveit.launch.py")
    log("=" * 80)

    while simulation_app.is_running():
        my_world.step(render=True)

    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
