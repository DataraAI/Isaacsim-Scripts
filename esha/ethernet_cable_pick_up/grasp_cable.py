

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.sensors.camera import Camera
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


# ------------------------- constants -------------------------
# Same cable path / spawn location as network_connector_pickup.py so the
# numbers we get here are directly comparable to that script.
NETWORK_CABLE_USD_PATH = "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd"
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
TRACKED_PLUG_PRIM_PATH = f"{NETWORK_CABLE_ROOT_PATH}/E_crystal_head1_45"

PLUG_SPAWN_XY = np.array([0.5, 0.0], dtype=np.float64)
GROUND_CLEARANCE = 0.002
SETTLE_FRAMES = 60

# --- camera setup ---
CAMERA_PATH = "/World/TopDownCamera"
CAMERA_HEIGHT = 0.60            # meters above world origin (z=0 ground plane)
CAMERA_RESOLUTION = (640, 480)
SEMANTIC_LABEL = "cable_connector"

# --- marker sphere ---
MARKER_PATH = "/World/SensedLocationMarker"
MARKER_RADIUS = 0.006
MARKER_COLOR = (1.0, 0.0, 0.0)  # red, so it's easy to spot in the viewport


# ------------------------- USD helpers (copied unchanged from
#                            network_connector_pickup.py) -------------------------

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
    root_min, _, _, _ = get_bbox(NETWORK_CABLE_ROOT_PATH)
    _, _, plug_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH)
    root_pose = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim)
    root_t = np.array(root_pose.ExtractTranslation(), dtype=np.float64)

    delta = np.array([
        PLUG_SPAWN_XY[0] - plug_center[0],
        PLUG_SPAWN_XY[1] - plug_center[1],
        GROUND_CLEARANCE - root_min[2],
    ], dtype=np.float64)
    set_world_translate(NETWORK_CABLE_ROOT_PATH, root_t + delta)


def reload_cable():
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(NETWORK_CABLE_ROOT_PATH).IsValid():
        stage.RemovePrim(Sdf.Path(NETWORK_CABLE_ROOT_PATH))

    add_reference_to_stage(usd_path=NETWORK_CABLE_USD_PATH, prim_path=NETWORK_CABLE_ROOT_PATH)
    place_cable_on_ground()


# ------------------------- new: perception helpers -------------------------

def label_connector_for_segmentation():
    """Tag the connector prim so the segmentation annotator can pick it out.

    This is the "ground truth" shortcut: Isaac Sim already knows exactly
    which prim every pixel came from, so all we're doing is giving that
    prim a human-readable class name that will show up in the
    segmentation's idToLabels mapping.
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(TRACKED_PLUG_PRIM_PATH)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing connector head: {TRACKED_PLUG_PRIM_PATH}")
    add_update_semantics(prim, semantic_label=SEMANTIC_LABEL, type_label="class")


def spawn_marker(position, path=MARKER_PATH, radius=MARKER_RADIUS, color=MARKER_COLOR):
    """Spawn (or move, if it already exists) a small sphere at `position`."""
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(Sdf.Path(path))

    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    set_world_translate(path, position)
    return sphere


def find_connector_pixel_centroid(camera):
    """Read the current segmentation frame and return the (u, v) pixel
    centroid of every pixel tagged with SEMANTIC_LABEL.
    """
    frame = camera.get_current_frame()
    seg = frame.get("semantic_segmentation")
    if seg is None:
        raise RuntimeError(
            "No semantic_segmentation data in this frame yet - "
            "the camera may need a few more simulation steps to render."
        )

    seg_data = np.asarray(seg["data"])
    if seg_data.ndim == 3:
        seg_data = seg_data[..., 0]  # drop trailing channel dim if present

    id_to_labels = seg["info"].get("idToLabels", {})
    target_id = None
    for id_str, label_info in id_to_labels.items():
        # label_info is typically a dict like {"class": "cable_connector"}
        if isinstance(label_info, dict) and label_info.get("class") == SEMANTIC_LABEL:
            target_id = int(id_str)
            break

    if target_id is None:
        raise RuntimeError(
            f"Label '{SEMANTIC_LABEL}' not found in this frame's idToLabels: {id_to_labels}. "
            "Is the connector actually visible to the camera?"
        )

    mask = seg_data == target_id
    if not np.any(mask):
        raise RuntimeError("Found the label id, but no pixels in the frame matched it.")

    ys, xs = np.nonzero(mask)
    # (u, v) = (column, row) pixel coordinates, matching the (x, y) that
    # get_world_points_from_image_coords expects.
    centroid_px = np.array([[xs.mean(), ys.mean()]], dtype=np.float64)
    return centroid_px, int(mask.sum())


# ------------------------- scene setup -------------------------

world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)

stage = omni.usd.get_context().get_stage()
light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
light.CreateIntensityAttr(500.0)
light.CreateColorAttr((1.0, 1.0, 1.0))

world.scene.add_default_ground_plane()

reload_cable()
label_connector_for_segmentation()

camera = Camera(
    prim_path=CAMERA_PATH,
    position=np.array([PLUG_SPAWN_XY[0], PLUG_SPAWN_XY[1], CAMERA_HEIGHT]),
    frequency=20,
    resolution=CAMERA_RESOLUTION,
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # identity quaternion -> looks straight down -Z
)
camera.initialize()
camera.add_semantic_segmentation_to_frame()

world.reset()

print("[READY] Press Play to run the sensing check.")

did_run = False
was_playing = False

while simulation_app.is_running():
    world.step(render=True)
    playing = world.is_playing()

    if playing and not was_playing:
        was_playing = True

    if not playing:
        was_playing = False
        continue

    if did_run:
        continue

    # Let a few frames render/settle before trusting the segmentation data.
    if world.current_time_step_index < SETTLE_FRAMES:
        continue

    try:
        centroid_px, pixel_count = find_connector_pixel_centroid(camera)
    except RuntimeError as exc:
        print(f"[WAIT] {exc}")
        continue

    # Depth here means "distance from the camera down to the plane we're
    # projecting onto," which we already know because the cable lies flat
    # at a known height - not something a depth sensor measured for us.
    depth = np.array([CAMERA_HEIGHT - GROUND_CLEARANCE], dtype=np.float64)
    sensed_point = camera.get_world_points_from_image_coords(centroid_px, depth)[0]

    spawn_marker(sensed_point)

    _, _, truth_center, _ = get_bbox(TRACKED_PLUG_PRIM_PATH)
    error_xy_mm = np.linalg.norm(sensed_point[:2] - truth_center[:2]) * 1000.0

    print("[SENSE]")
    print(f"  pixels_matched = {pixel_count}")
    print(f"  sensed_point   = {np.round(sensed_point, 5)}")
    print(f"  truth_center   = {np.round(truth_center, 5)}")
    print(f"  xy_error_mm    = {error_xy_mm:.2f}")

    did_run = True

simulation_app.close()