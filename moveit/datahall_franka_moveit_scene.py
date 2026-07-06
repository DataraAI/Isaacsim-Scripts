from isaacsim import SimulationApp


simulation_app = SimulationApp({"headless": False})

import os
import sys
import time

import carb
import numpy as np
import omni
import omni.graph.core as og
import omni.timeline

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysxSchema, UsdPhysics

try:
    import rclpy
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
    from sensor_msgs.msg import JointState
    from shape_msgs.msg import SolidPrimitive
except Exception as exc:
    rclpy = None
    Pose = None
    AttachedCollisionObject = None
    CollisionObject = None
    JointState = None
    SolidPrimitive = None
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None


DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)

# Keep /panda. The official isaac_moveit launch/config expects this robot prim.
ROBOT_PRIM_PATH = "/panda"
END_EFFECTOR_PRIM_PATH = "/panda/panda_rightfinger"

JOINT_STATES_TOPIC = "/isaac_joint_states"
JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"

PICK_BLOCK_PRIM_PATH = "/World/PickBlock"
PICK_BLOCK_NAME = "pick_block"
PICK_BLOCK_SIZE = 0.05
PICK_BLOCK_START_POSITION = np.array([0.35, 0.0, 0.025])
PICK_BLOCK_COLOR = np.array([0.1, 0.35, 1.0])

GRIPPER_CLOSED_THRESHOLD = 0.018
GRIPPER_OPEN_THRESHOLD = 0.035
BLOCK_ATTACH_DISTANCE = 0.16
PREGRASP_ATTACH_DISTANCE = 0.26
PREGRASP_CANCEL_DISTANCE = 0.36

MOVEIT_BLOCK_ID = "pick_block"
MOVEIT_WORLD_FRAME = "panda_link0"
MOVEIT_ATTACHED_LINK = "panda_hand"
MOVEIT_TOUCH_LINKS = [
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
]

SCENE_WORLD = "world"
SCENE_PREGRASP_ATTACHED = "pregrasp_attached"
SCENE_GRASPED_ATTACHED = "grasped_attached"

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


def set_rigid_body_attrs(
    physx_rigid_body_api,
    usd_rigid_body_api,
    *,
    disable_gravity: bool,
    kinematic: bool,
) -> None:
    physx_rigid_body_api.CreateDisableGravityAttr(disable_gravity).Set(disable_gravity)
    usd_rigid_body_api.CreateKinematicEnabledAttr(kinematic).Set(kinematic)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def rotate_vector(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    vector_quat = np.array([0.0, vector[0], vector[1], vector[2]])
    rotated = quat_multiply(quat_multiply(q, vector_quat), quat_conjugate(q))
    return rotated[1:]


def make_ros_pose(position, orientation) -> Pose:
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])

    # Isaac returns quaternions scalar-first: [w, x, y, z].
    q = quat_normalize(np.asarray(orientation, dtype=np.float64))
    pose.orientation.w = float(q[0])
    pose.orientation.x = float(q[1])
    pose.orientation.y = float(q[2])
    pose.orientation.z = float(q[3])
    return pose


def make_moveit_box() -> SolidPrimitive:
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [PICK_BLOCK_SIZE, PICK_BLOCK_SIZE, PICK_BLOCK_SIZE]
    return box


class MoveItBlockSceneBridge:
    def __init__(self, block, scripted_grasp):
        self.block = block
        self.scripted_grasp = scripted_grasp
        self.enabled = False
        self.scene_state = SCENE_WORLD
        self.last_publish_time = 0.0
        self.last_status_time = 0.0

        if rclpy is None:
            log(f"MoveIt block scene bridge disabled; rclpy import failed: {ROS_IMPORT_ERROR}")
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node("datahall_block_scene_bridge")
        self.collision_object_pub = self.node.create_publisher(
            CollisionObject, "/collision_object", 10
        )
        self.attached_object_pub = self.node.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", 10
        )
        self.joint_state_pub = self.node.create_publisher(
            JointState, "/datahall/debug_pick_block_state", 10
        )
        self.enabled = True
        log("MoveIt block scene bridge enabled inside Isaac script")

    def close(self):
        if not self.enabled:
            return
        self.node.destroy_node()

    def _block_world_pose(self):
        block_position, block_orientation = self.block.get_world_pose()
        return (
            np.asarray(block_position, dtype=np.float64),
            quat_normalize(np.asarray(block_orientation, dtype=np.float64)),
        )

    def _block_pose_in_panda_link0(self) -> Pose:
        block_position, block_orientation = self._block_world_pose()
        link0_position, link0_orientation = get_world_pose("/panda/panda_link0")
        link0_position = np.asarray(link0_position, dtype=np.float64)
        link0_orientation = quat_normalize(np.asarray(link0_orientation, dtype=np.float64))

        inv_link0_orientation = quat_conjugate(link0_orientation)
        position_in_link0 = rotate_vector(
            inv_link0_orientation,
            block_position - link0_position,
        )
        orientation_in_link0 = quat_multiply(inv_link0_orientation, block_orientation)
        return make_ros_pose(position_in_link0, orientation_in_link0)

    def _block_pose_in_hand(self) -> Pose:
        block_position, block_orientation = self._block_world_pose()
        hand_position, hand_orientation = get_world_pose("/panda/panda_hand")
        hand_position = np.asarray(hand_position, dtype=np.float64)
        hand_orientation = quat_normalize(np.asarray(hand_orientation, dtype=np.float64))

        inv_hand_orientation = quat_conjugate(hand_orientation)
        position_in_hand = rotate_vector(inv_hand_orientation, block_position - hand_position)
        orientation_in_hand = quat_multiply(inv_hand_orientation, block_orientation)
        return make_ros_pose(position_in_hand, orientation_in_hand)

    def publish_world_block(self):
        obj = CollisionObject()
        obj.header.frame_id = MOVEIT_WORLD_FRAME
        obj.id = MOVEIT_BLOCK_ID
        obj.primitives = [make_moveit_box()]
        obj.primitive_poses = [self._block_pose_in_panda_link0()]
        obj.operation = CollisionObject.ADD
        self.collision_object_pub.publish(obj)

    def remove_world_block(self):
        obj = CollisionObject()
        obj.header.frame_id = MOVEIT_WORLD_FRAME
        obj.id = MOVEIT_BLOCK_ID
        obj.operation = CollisionObject.REMOVE
        self.collision_object_pub.publish(obj)

    def publish_attached_block(self):
        attached = AttachedCollisionObject()
        attached.link_name = MOVEIT_ATTACHED_LINK
        attached.touch_links = MOVEIT_TOUCH_LINKS
        attached.object.header.frame_id = MOVEIT_ATTACHED_LINK
        attached.object.id = MOVEIT_BLOCK_ID
        attached.object.primitives = [make_moveit_box()]
        attached.object.primitive_poses = [self._block_pose_in_hand()]
        attached.object.operation = CollisionObject.ADD
        self.attached_object_pub.publish(attached)

    def remove_attached_block(self):
        attached = AttachedCollisionObject()
        attached.link_name = MOVEIT_ATTACHED_LINK
        attached.object.header.frame_id = MOVEIT_ATTACHED_LINK
        attached.object.id = MOVEIT_BLOCK_ID
        attached.object.operation = CollisionObject.REMOVE
        self.attached_object_pub.publish(attached)

    def _hand_distance_to_block(self) -> float:
        block_position, _ = self._block_world_pose()
        hand_position, _ = get_world_pose("/panda/panda_hand")
        hand_position = np.asarray(hand_position, dtype=np.float64)
        return float(np.linalg.norm(block_position - hand_position))

    def update(self):
        if not self.enabled:
            return

        rclpy.spin_once(self.node, timeout_sec=0.0)

        now = time.monotonic()
        if now - self.last_publish_time < 0.2:
            return
        self.last_publish_time = now

        finger_width = self.scripted_grasp._finger_width()
        hand_distance = self._hand_distance_to_block()

        if self.scene_state == SCENE_WORLD and hand_distance <= PREGRASP_ATTACH_DISTANCE:
            self.scene_state = SCENE_PREGRASP_ATTACHED
            self.remove_world_block()
            self.publish_attached_block()
            log(
                "MoveIt pre-attached PickBlock for hand-close planning "
                f"(hand_distance={hand_distance:.4f})"
            )

        elif (
            self.scene_state == SCENE_PREGRASP_ATTACHED
            and finger_width >= GRIPPER_OPEN_THRESHOLD
            and hand_distance >= PREGRASP_CANCEL_DISTANCE
        ):
            self.scene_state = SCENE_WORLD
            self.remove_attached_block()
            self.publish_world_block()
            log(
                "MoveIt cancelled PickBlock pre-attach "
                f"(hand_distance={hand_distance:.4f})"
            )

        elif self.scripted_grasp.attached:
            if self.scene_state != SCENE_GRASPED_ATTACHED:
                self.scene_state = SCENE_GRASPED_ATTACHED
                self.remove_world_block()
                log("MoveIt confirmed PickBlock grasp")
            self.publish_attached_block()

        elif self.scene_state == SCENE_GRASPED_ATTACHED and finger_width >= GRIPPER_OPEN_THRESHOLD:
            self.scene_state = SCENE_WORLD
            self.remove_attached_block()
            self.publish_world_block()
            log("MoveIt released PickBlock back to world")

        elif self.scene_state == SCENE_WORLD:
            self.publish_world_block()

        if now - self.last_status_time > 1.0:
            self.last_status_time = now
            block_pose_in_link0 = self._block_pose_in_panda_link0()
            log(
                "MoveIt block bridge status "
                f"state={self.scene_state} "
                f"finger_width={finger_width:.4f} "
                f"hand_distance={hand_distance:.4f} "
                "block_in_panda_link0="
                f"({block_pose_in_link0.position.x:.4f}, "
                f"{block_pose_in_link0.position.y:.4f}, "
                f"{block_pose_in_link0.position.z:.4f})"
            )


class ScriptedBlockGrasp:
    def __init__(self, block, franka):
        self.block = block
        self.franka = franka
        self.physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(block.prim)
        self.usd_rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(block.prim)
        self.attached = False
        self.attach_offset = np.zeros(3)
        self.block_orientation = None

        set_rigid_body_attrs(
            self.physx_rigid_body_api,
            self.usd_rigid_body_api,
            disable_gravity=False,
            kinematic=False,
        )

    def _finger_width(self) -> float:
        joint_positions = self.franka.get_joint_positions()
        if joint_positions is None or len(joint_positions) < 9:
            return 1.0
        return float((joint_positions[7] + joint_positions[8]) / 2.0)

    def _hand_position(self):
        hand_position, hand_orientation = get_world_pose("/panda/panda_hand")
        return np.array(hand_position), hand_orientation

    def update(self) -> None:
        finger_width = self._finger_width()
        hand_position, _ = self._hand_position()
        block_position, block_orientation = self.block.get_world_pose()
        block_position = np.array(block_position)

        if not self.attached:
            distance = float(np.linalg.norm(block_position - hand_position))
            if finger_width <= GRIPPER_CLOSED_THRESHOLD and distance <= BLOCK_ATTACH_DISTANCE:
                self.attached = True
                self.attach_offset = block_position - hand_position
                self.block_orientation = block_orientation
                set_rigid_body_attrs(
                    self.physx_rigid_body_api,
                    self.usd_rigid_body_api,
                    disable_gravity=True,
                    kinematic=True,
                )
                log(
                    "Attached PickBlock to gripper "
                    f"(finger_width={finger_width:.4f}, distance={distance:.4f})"
                )
            return

        self.block.set_world_pose(
            position=hand_position + self.attach_offset,
            orientation=self.block_orientation,
        )

        if finger_width >= GRIPPER_OPEN_THRESHOLD:
            self.attached = False
            set_rigid_body_attrs(
                self.physx_rigid_body_api,
                self.usd_rigid_body_api,
                disable_gravity=False,
                kinematic=False,
            )
            log(f"Released PickBlock (finger_width={finger_width:.4f})")


def add_pick_block(my_world: World):
    log(f"Adding pick block at {PICK_BLOCK_START_POSITION.tolist()}")
    return my_world.scene.add(
        DynamicCuboid(
            name=PICK_BLOCK_NAME,
            prim_path=PICK_BLOCK_PRIM_PATH,
            position=PICK_BLOCK_START_POSITION,
            scale=np.array([PICK_BLOCK_SIZE, PICK_BLOCK_SIZE, PICK_BLOCK_SIZE]),
            size=1.0,
            color=PICK_BLOCK_COLOR,
        )
    )


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
    log("Adding default physics ground plane at z=0")
    my_world.scene.add_default_ground_plane()

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
            position=np.array([0.0, 0.0, 0.0]),
        )
    )
    my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
    my_franka.set_joints_default_state(positions=PANDA_READY_JOINT_POSITIONS)
    pick_block = add_pick_block(my_world)

    log("Resetting Isaac World")
    my_world.reset()
    my_franka.set_joint_positions(PANDA_READY_JOINT_POSITIONS)
    update_frames(60, "after World.reset")

    scripted_grasp = ScriptedBlockGrasp(pick_block, my_franka)
    moveit_block_scene_bridge = MoveItBlockSceneBridge(pick_block, scripted_grasp)

    create_moveit_joint_graph()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    log("=" * 80)
    log("Scene is running. Now launch RViz in another terminal:")
    log("ros2 launch isaac_moveit isaac_moveit.launch.py")
    log("=" * 80)

    while simulation_app.is_running():
        my_world.step(render=True)
        scripted_grasp.update()
        moveit_block_scene_bridge.update()

    timeline.stop()
    moveit_block_scene_bridge.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
