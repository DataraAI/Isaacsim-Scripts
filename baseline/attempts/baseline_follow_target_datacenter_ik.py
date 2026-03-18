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

"""Follow-target IK with datacenter USD: Franka at old robot world position; target at port world position."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb
from pxr import Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget

USD_PATH = r"C:\Users\aayus\Downloads\Datacenter_Files\Assets\DigitalTwin\Assets\Datacenter\Facilities\Stages\Data_Hall\DataHall_Full_01.usd"
PORT_PRIM_PATH = (
    "/World/Network_Switches/SN4600C_CS2FC_02/msn4600_cs2fc_01/SN4600C_A_01/msn4600_cs2fc_base/"
    "SM4600_CS2FC_01/NetworkConnectors/pcb003636_idf_01/Connector_Quad_01/Connector_Pair_01/QSFP_DD_Connector_A_01"
)
LEGACY_FRANKA_PATH = "/World/Franka"
TARGET_PRIM_PATH = "/World/TargetCube"


def world_position(prim_path: str) -> np.ndarray:
    """World-space position (stage frame) of a prim's pivot."""
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m.ExtractTranslation(), dtype=np.float64)


my_world = World(stage_units_in_meters=1.0)
add_reference_to_stage(USD_PATH, "/World")

if not is_prim_path_valid(LEGACY_FRANKA_PATH):
    carb.log_error(f"Expected Franka at {LEGACY_FRANKA_PATH} in the referenced USD.")
    simulation_app.close()
    raise SystemExit(1)
saved_franka_world_position = world_position(LEGACY_FRANKA_PATH)
if is_prim_path_valid(PORT_PRIM_PATH):
    port_world_position = world_position(PORT_PRIM_PATH)
else:
    carb.log_warn(f"Port prim not found; target stays at task default: {PORT_PRIM_PATH}")
    port_world_position = None

delete_prim(LEGACY_FRANKA_PATH)

my_task = FollowTarget(
    name="follow_target_task",
    franka_prim_path=LEGACY_FRANKA_PATH,
    target_prim_path=TARGET_PRIM_PATH,
)
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
my_target = my_world.scene.get_object(target_name)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()


def apply_franka_and_target_from_datacenter():
    """Set new Franka world position to match removed robot; set TargetCube world position to port."""
    my_franka.set_world_pose(position=saved_franka_world_position)
    if port_world_position is not None:
        my_target.set_world_pose(position=port_world_position)


apply_franka_and_target_from_datacenter()

reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            apply_franka_and_target_from_datacenter()
            reset_needed = False
        target_pos, target_orient = my_target.get_world_pose()
        actions, succ = my_controller.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_orient,
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")

simulation_app.close()
