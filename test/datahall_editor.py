# Isaac Sim 6.0 - Data Hall Editor
#
# Opens an editable copy of the Data Hall USD with no robot, controllers,
# physics setup, tables, modules, port logic, or task code.
#
# Edit the scene normally in Isaac Sim and press Ctrl+S to save your changes.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import shutil
import sys
import time

import carb
import omni.usd

try:
    from isaacsim.core.utils.viewports import set_camera_view
except Exception:
    set_camera_view = None


# -----------------------------------------------------------------------------
# Data Hall files
# -----------------------------------------------------------------------------

SOURCE_DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd"
)

# Your edits are saved here, so the original asset stays untouched.
EDITABLE_DATAHALL_USD = (
    "/home/aayush/isaacsim_assets/datacenter/Assets/DigitalTwin"
    "/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01_EDIT.usd"
)

# Change this to True only when you intentionally want to replace the existing
# editable copy with a fresh copy of the original asset.
RESET_EDITABLE_COPY = False

# Optional startup camera. Set either value to None to keep Isaac Sim's default.
STARTUP_CAMERA_EYE = [1.8809555265110907, 1.5749548281281234, 2.8130942736157993]
STARTUP_CAMERA_TARGET = [-0.40583929622548287, -0.43359704675040334, 0.7505214224942818]


# -----------------------------------------------------------------------------
# Create/open the editable stage
# -----------------------------------------------------------------------------

if not os.path.isfile(SOURCE_DATAHALL_USD):
    carb.log_error(f"Data Hall USD not found: {SOURCE_DATAHALL_USD}")
    simulation_app.close()
    sys.exit(1)

if RESET_EDITABLE_COPY or not os.path.isfile(EDITABLE_DATAHALL_USD):
    os.makedirs(os.path.dirname(EDITABLE_DATAHALL_USD), exist_ok=True)
    shutil.copy2(SOURCE_DATAHALL_USD, EDITABLE_DATAHALL_USD)
    print(f"Created editable Data Hall copy:\n  {EDITABLE_DATAHALL_USD}")
else:
    print(f"Opening existing editable Data Hall:\n  {EDITABLE_DATAHALL_USD}")

usd_context = omni.usd.get_context()
opened = usd_context.open_stage(EDITABLE_DATAHALL_USD)

if opened is False:
    carb.log_error(f"Could not open stage: {EDITABLE_DATAHALL_USD}")
    simulation_app.close()
    sys.exit(1)

# Let Isaac Sim finish loading the stage and all referenced assets.
# Isaac Sim 6.0's UsdContext does not provide is_stage_loading().
# get_stage_loading_status() returns:
#     (status_message, files_loaded, total_files)
# Wait until the loader reports no pending files for several consecutive frames.
stable_frames = 0
required_stable_frames = 15
max_wait_frames = 3600

for _ in range(max_wait_frames):
    if not simulation_app.is_running():
        break

    simulation_app.update()

    try:
        _message, files_loaded, total_files = usd_context.get_stage_loading_status()
        still_loading = bool(files_loaded or total_files)
    except AttributeError:
        # Fallback for an unexpected Kit build: allow a fixed number of frames
        # for references, materials, and viewport resources to settle.
        still_loading = False

    if still_loading:
        stable_frames = 0
    else:
        stable_frames += 1
        if stable_frames >= required_stable_frames:
            break

    time.sleep(0.01)

# Apply the same useful startup view from the original script.
if (
    set_camera_view is not None
    and STARTUP_CAMERA_EYE is not None
    and STARTUP_CAMERA_TARGET is not None
):
    try:
        set_camera_view(
            eye=STARTUP_CAMERA_EYE,
            target=STARTUP_CAMERA_TARGET,
        )
    except Exception as exc:
        carb.log_warn(f"Could not set startup camera: {exc}")

print("\nData Hall loaded with no robot or task setup.")
print("Modify the scene in Isaac Sim, then press Ctrl+S to save.")
print(f"Edits will be saved to:\n  {EDITABLE_DATAHALL_USD}\n")


# Keep Isaac Sim open until the window is closed.
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
