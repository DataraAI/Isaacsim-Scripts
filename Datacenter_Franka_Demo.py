from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
 
import time
import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
    interface_config_loader,
)
 

# -----------------------------
USD_PATH    = "C:/Users/aayus/Desktop/Jonathan_Arun_Test/frana_clean.usd"
FRANKA_PATH = "/World/Franka_01"
EE_FRAME    = "panda_hand"
SETTLE_STEPS = 30
MOVE_STEPS   = 180
 
# -----------------------------
# RACK PORT POSITIONS
# Add / remove / reorder to match your rack geometry
# Each entry is (label, [X, Y, Z]) in world coordinates
# -----------------------------
RACK_PORTS = [
    ("Port-1U", np.array([0.40,  0.20, 0.50])),
    ("Port-2U", np.array([0.40,  0.10, 0.50])),
    ("Port-3U", np.array([0.40,  0.00, 0.50])),
    ("Port-4U", np.array([0.40, -0.10, 0.50])),
    ("Port-5U", np.array([0.40, -0.20, 0.50])),
]
# -----------------------------
 
 
def wait_for_stage_load():
    for _ in range(100):
        simulation_app.update()
        time.sleep(0.01)
 
 
def move_to_target(world, robot, ik_solver, lula_solver, target_pos, steps=180):
    base_pos, base_quat = robot.get_world_pose()
    lula_solver.set_robot_base_pose(base_pos, base_quat)
    current_pos, _ = ik_solver.compute_end_effector_pose()
 
    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        interp_pos = (1.0 - alpha) * current_pos + alpha * target_pos
        action, success = ik_solver.compute_inverse_kinematics(interp_pos)
        if success:
            robot.apply_action(action)
        else:
            print(f"  IK failed at step {i} targeting {target_pos}")
            break
        world.step(render=True)
 
 
def main():
    print(f"Opening stage: {USD_PATH}")
    omni.usd.get_context().open_stage(USD_PATH)
    wait_for_stage_load()
 
    world = World()
    world.reset()
 
    robot = SingleArticulation(prim_path=FRANKA_PATH, name="franka")
    robot.initialize()
 
    config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
    lula_solver = LulaKinematicsSolver(**config)
    ik_solver = ArticulationKinematicsSolver(robot, lula_solver, EE_FRAME)
 
    # Visit each port in sequence
    for label, position in RACK_PORTS:
        print(f"Moving to {label} at {position} ...")
        move_to_target(world, robot, ik_solver, lula_solver, position, steps=MOVE_STEPS)
 
        # Settle briefly at each port before moving to the next
        for _ in range(SETTLE_STEPS):
            world.step(render=True)
 
        print(f"  Reached {label}")
 
    print("All ports visited. Leaving sim open.")
    while simulation_app.is_running():
        world.step(render=True)
 
 
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        simulation_app.close()