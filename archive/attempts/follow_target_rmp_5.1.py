# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb
from pxr import Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget

USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"

num_quads = 4
num_pairs = 4
num_conn_a = 2

PORT_BASE_PRIM_PATH = (
    "/World/Datacenter/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base"
    "/SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01"
)

PORT_PRIM_PATH_LIST = []
for q in range(1, num_quads + 1):
    for p in range(1, num_pairs + 1):
        for a in range(1, num_conn_a + 1):
            suffix = f"/Connector_Quad_{q:02d}/Connector_Pair_{p:02d}/QSFP_DD_Connector_A_{a:02d}"
            PORT_PRIM_PATH_LIST.append(PORT_BASE_PRIM_PATH + suffix)

# Which port in PORT_PRIM_PATH_LIST drives the target cube (even index: z -2, odd index: z +2).
CURRENT_PORT_INDEX = 0

FRANKA_WORLD_POSITION = np.array([30.0, -90.0, 150.0], dtype=np.float64)
FRANKA_EULER_DEGREES = np.array([0.0, 0.0, 180.0], dtype=np.float64)
FRANKA_WORLD_ORIENTATION = euler_angles_to_quat(FRANKA_EULER_DEGREES, degrees=True)
FRANKA_SCALE = np.array([57.0, 57.0, 57.0], dtype=np.float64)

TARGET_LOCAL_SCALE = np.array([3.0, 3.0, 3.0], dtype=np.float64)


def port_target_offset_z(port_index: int) -> np.ndarray:
    if port_index % 2 == 0:
        return np.array([0.0, 0.0, -2.0], dtype=np.float64)
    return np.array([0.0, 0.0, 2.0], dtype=np.float64)


def world_position_only(prim_path: str) -> np.ndarray:
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m.ExtractTranslation(), dtype=np.float64)


my_world = World(stage_units_in_meters=1.0)
add_reference_to_stage(USD_PATH, "/World/Datacenter")

_current_port_path = PORT_PRIM_PATH_LIST[CURRENT_PORT_INDEX]
if not is_prim_path_valid(_current_port_path):
    carb.log_error(f"Port prim not found: {_current_port_path}")
    simulation_app.close()
    raise SystemExit(1)

_target_world_position = world_position_only(_current_port_path) + port_target_offset_z(CURRENT_PORT_INDEX)

my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
my_target = my_world.scene.get_object(target_name)
my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=my_franka)
articulation_controller = my_franka.get_articulation_controller()


def apply_franka_and_target_layout():
    my_franka.set_local_scale(FRANKA_SCALE)
    my_franka.set_world_pose(position=FRANKA_WORLD_POSITION, orientation=FRANKA_WORLD_ORIENTATION)
    _, target_world_orientation = my_target.get_world_pose()
    my_target.set_world_pose(position=_target_world_position, orientation=target_world_orientation)
    my_target.set_local_scale(TARGET_LOCAL_SCALE)


apply_franka_and_target_layout()
my_world.stop()

reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            apply_franka_and_target_layout()
            reset_needed = False
        observations = my_world.get_observations()
        actions = my_controller.forward(
            target_end_effector_position=observations[target_name]["position"],
            target_end_effector_orientation=observations[target_name]["orientation"],
        )
        articulation_controller.apply_action(actions)

simulation_app.close()
