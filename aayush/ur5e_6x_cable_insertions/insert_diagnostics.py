"""Record wrist wrench + PhysX/config props during align→translate insert."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pxr import UsdPhysics, UsdShade

from ur5e_6x_cable_insertions import config as cfg


def _safe_list(arr, ndigits: int = 6) -> list | None:
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        return [round(float(x), ndigits) for x in a.tolist()]
    except Exception:
        return None


def _call_first(obj: Any, names: tuple[str, ...], *args, **kwargs):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                return name, fn(*args, **kwargs)
            except TypeError:
                try:
                    return name, fn()
                except Exception:
                    continue
            except Exception:
                continue
    return None, None


def _read_material_binding(stage, prim_path: str) -> dict | None:
    """Best-effort Physics material bound on a prim (friction / restitution)."""

    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {"path": prim_path, "error": "missing"}
        binding = UsdShade.MaterialBindingAPI(prim)
        mat, _rel = binding.ComputeBoundMaterial(UsdShade.Tokens.physics)
        if not mat:
            mat, _rel = binding.ComputeBoundMaterial()
        if not mat:
            return {"path": prim_path, "material": None}
        mat_prim = mat.GetPrim()
        out: dict[str, Any] = {
            "path": prim_path,
            "material": str(mat_prim.GetPath()),
        }
        if mat_prim.HasAPI(UsdPhysics.MaterialAPI):
            api = UsdPhysics.MaterialAPI(mat_prim)
            for attr_name, key in (
                ("staticFriction", "static_friction"),
                ("dynamicFriction", "dynamic_friction"),
                ("restitution", "restitution"),
            ):
                attr = mat_prim.GetAttribute(f"physics:{attr_name}")
                if attr and attr.HasAuthoredValueOpinion():
                    out[key] = float(attr.Get())
                else:
                    # MaterialAPI getters
                    getter = getattr(api, f"Get{attr_name[0].upper() + attr_name[1:]}Attr", None)
                    if getter is not None:
                        a = getter()
                        if a:
                            val = a.Get()
                            if val is not None:
                                out[key] = float(val)
        comb = mat_prim.GetAttribute("physxMaterial:frictionCombineMode")
        if comb and comb.HasAuthoredValueOpinion():
            out["friction_combine_mode"] = str(comb.Get())
        return out
    except Exception as exc:
        return {"path": prim_path, "error": str(exc)}


def snapshot_physics_config() -> dict:
    """Static config knobs that govern grasp / insert contact."""

    return {
        "grasp_friction_static": float(cfg.GRASP_FRICTION_STATIC),
        "grasp_friction_dynamic": float(cfg.GRASP_FRICTION_DYNAMIC),
        "grasp_friction_combine": str(cfg.GRASP_FRICTION_COMBINE_MODE),
        "bezel_friction_static": float(cfg.BEZEL_FRICTION_STATIC),
        "bezel_friction_dynamic": float(cfg.BEZEL_FRICTION_DYNAMIC),
        "bezel_friction_combine": str(cfg.BEZEL_FRICTION_COMBINE_MODE),
        "worktable_friction_static": float(
            getattr(cfg, "WORKTABLE_FRICTION_STATIC", 0.0)
        ),
        "worktable_friction_dynamic": float(
            getattr(cfg, "WORKTABLE_FRICTION_DYNAMIC", 0.0)
        ),
        "worktable_friction_combine": str(
            getattr(cfg, "WORKTABLE_FRICTION_COMBINE_MODE", "min")
        ),
        "joint_friction": float(getattr(cfg, "JOINT_FRICTION", float("nan"))),
        "deformable_linear_damping": float(
            getattr(cfg, "DEFORMABLE_LINEAR_DAMPING", float("nan"))
        ),
        "deformable_contact_offset_m": float(
            getattr(cfg, "DEFORMABLE_CONTACT_OFFSET_M", float("nan"))
        ),
        "deformable_rest_offset_m": float(
            getattr(cfg, "DEFORMABLE_REST_OFFSET_M", float("nan"))
        ),
        "insert_step_m": float(cfg.INSERT_STEP_M),
        "mating_touch_gap_m": float(cfg.MATING_TOUCH_GAP_M),
        "align_insert_enable_yz": bool(getattr(cfg, "ALIGN_INSERT_ENABLE_YZ", False)),
        "finger_joint_drive": list(
            getattr(cfg, "ROBOTIQ_DRIVE_PARAMETERS", {}).get("finger_joint", ())
        ),
        "ur5e_live_arm_max_effort": float(
            getattr(cfg, "UR5E_LIVE_ARM_MAX_EFFORT", float("nan"))
        ),
        "gripper_mount_break_force": float(
            getattr(cfg, "GRIPPER_MOUNT_BREAK_FORCE", float("nan"))
        ),
    }


def sample_articulation_wrench(controller) -> dict:
    """Joint efforts / measured joint forces (and wrist 6-DoF if available)."""

    robot = getattr(controller, "_robot", None)
    out: dict[str, Any] = {"apis": {}}
    if robot is None:
        out["error"] = "no_robot"
        return out

    names = None
    try:
        names = list(robot.dof_names)
        out["dof_names"] = names
    except Exception:
        pass

    api, efforts = _call_first(
        robot,
        (
            "get_measured_joint_efforts",
            "get_applied_joint_efforts",
            "get_joint_efforts",
        ),
    )
    if efforts is not None:
        out["apis"]["efforts"] = api
        out["joint_efforts"] = _safe_list(efforts)
        if names and "finger_joint" in names:
            idx = names.index("finger_joint")
            arr = np.asarray(efforts, dtype=np.float64).reshape(-1)
            if idx < arr.size:
                out["finger_joint_effort"] = round(float(arr[idx]), 6)

    api, forces = _call_first(
        robot,
        ("get_measured_joint_forces", "get_joint_forces"),
    )
    if forces is not None:
        out["apis"]["joint_forces"] = api
        arr = np.asarray(forces, dtype=np.float64)
        out["joint_forces_shape"] = list(arr.shape)
        # Isaac: (n_links, 6) force/torque in body frame at each joint.
        if arr.ndim == 2 and arr.shape[-1] >= 6:
            # Prefer wrist_3 / ee link if we can match dof/body names.
            wrist_idx = None
            body_names = None
            for attr in ("body_names", "link_names", "_body_names"):
                body_names = getattr(robot, attr, None)
                if body_names:
                    break
            if body_names:
                out["body_names"] = list(body_names)
                for i, bn in enumerate(body_names):
                    bn_l = str(bn).lower()
                    if "wrist_3" in bn_l or bn_l.endswith("tool0") or "ee_link" in bn_l:
                        wrist_idx = i
                        break
            if wrist_idx is None:
                wrist_idx = int(arr.shape[0] - 1)
            wrench = arr[wrist_idx, :6]
            out["wrist_link_index"] = wrist_idx
            out["wrist_force_N"] = _safe_list(wrench[:3])
            out["wrist_torque_Nm"] = _safe_list(wrench[3:6])
            out["wrist_force_norm_N"] = round(float(np.linalg.norm(wrench[:3])), 6)
            out["wrist_torque_norm_Nm"] = round(float(np.linalg.norm(wrench[3:6])), 6)
            # Also dump all link force norms for peak hunting.
            norms = [round(float(np.linalg.norm(row[:3])), 6) for row in arr]
            out["link_force_norms_N"] = norms
            out["link_force_norm_max_N"] = max(norms) if norms else None
        else:
            out["joint_forces_flat"] = _safe_list(arr)

    api, vel = _call_first(robot, ("get_joint_velocities",))
    if vel is not None:
        out["joint_velocities"] = _safe_list(vel, ndigits=5)

    api, pos = _call_first(robot, ("get_joint_positions",))
    if pos is not None:
        out["joint_positions"] = _safe_list(pos, ndigits=5)

    return out


def sample_bound_materials(context) -> dict:
    """Live USD friction bindings on pads, crystal heads, and a port bezel path."""

    stage = context.services.get("stage")
    if stage is None:
        return {"error": "no_stage"}

    paths: list[str] = []
    robot_path = str(context.services.get("robot_prim_path") or "")
    spec = context.services.get("station_spec")
    if spec is not None:
        robot_path = robot_path or str(getattr(spec, "robot_prim_path", "") or "")
        for attr in ("path45", "path39", "cable_root_path", "cable_prim_path", "cable_path"):
            p = getattr(spec, attr, None)
            if p:
                paths.append(str(p))

    for token in ("left_inner_finger", "right_inner_finger"):
        if robot_path:
            paths.append(f"{robot_path}/Gripper/Robotiq_2F_85/{token}")

    for key in ("path45", "path39", "crystal_head_path", "cable_path", "cable_prim_path"):
        p = context.services.get(key)
        if p:
            paths.append(str(p))

    cable = context.services.get("cable_prim_path") or context.services.get("cable_path")
    if not cable and spec is not None:
        cable = getattr(spec, "cable_prim_path", None) or getattr(spec, "cable_path", None)
    if cable:
        for suffix in ("/E_crystal_head1_45", "/E_crystal_head2_39"):
            paths.append(f"{cable}{suffix}")

    # Work tables the trailing head can graze during insert (NegY_Top → WorkTable1).
    by_height = getattr(cfg, "WORK_TABLE_PATH_BY_HEIGHT", {}) or {}
    for p in by_height.values():
        if p:
            paths.append(str(p))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)

    return {
        "bindings": [_read_material_binding(stage, p) for p in uniq],
        "config": snapshot_physics_config(),
    }


class InsertDiagnosticsRecorder:
    """JSONL recorder for insert-phase wrench / contact / geometry samples."""

    def __init__(self, station_id: str, path: Path | None = None) -> None:
        self.station_id = str(station_id)
        self.path = Path(
            path
            if path is not None
            else getattr(cfg, "INSERT_DIAG_PATH", Path(__file__).resolve().parent / "insert_diag.jsonl")
        )
        self.samples: list[dict] = []
        self.t0 = time.monotonic()
        self._peak_force_N = 0.0
        self._peak_torque_Nm = 0.0
        self._peak_contact_impulse = 0.0
        self._header_written = False
        self.reason_end: str | None = None

    def start(self, context) -> None:
        header = {
            "type": "insert_diag_header",
            "station": self.station_id,
            "t_wall": time.time(),
            "physics": sample_bound_materials(context),
        }
        self._append_disk(header)
        self._header_written = True
        print(
            f"[INSERT DIAG {self.station_id}] recording → {self.path} "
            f"(wrench + contacts + physics materials)"
        )
        # One-line human summary of key μ / deformable knobs.
        phys = header["physics"].get("config", {})
        print(
            f"[INSERT DIAG {self.station_id}] physics cfg "
            f"grasp_μ=({phys.get('grasp_friction_static')},"
            f"{phys.get('grasp_friction_dynamic')}/{phys.get('grasp_friction_combine')}) "
            f"bezel_μ=({phys.get('bezel_friction_static')},"
            f"{phys.get('bezel_friction_dynamic')}/{phys.get('bezel_friction_combine')}) "
            f"table_μ=({phys.get('worktable_friction_static')},"
            f"{phys.get('worktable_friction_dynamic')}/{phys.get('worktable_friction_combine')}) "
            f"deform_damp={phys.get('deformable_linear_damping')} "
            f"contact_off={phys.get('deformable_contact_offset_m')} "
            f"rest_off={phys.get('deformable_rest_offset_m')} "
            f"insert_step={phys.get('insert_step_m')}m"
        )

    def sample(self, context, *, label: str | None = None, extra: dict | None = None) -> dict:
        """Capture one insert-frame sample (geometry + wrench + contacts)."""

        # Geometry is passed via services by primitives (avoids circular imports).
        controller = context.services.get("motion_controller")
        row: dict[str, Any] = {
            "type": "insert_diag_sample",
            "station": self.station_id,
            "t_rel_s": round(time.monotonic() - self.t0, 4),
            "frame": int(context.services.get("align_insert_frames", 0)),
            "label": label
            or (
                str(context.services.get("_insert_diag_label"))
                if context.services.get("_insert_diag_label")
                else None
            ),
        }

        geom = context.services.get("_insert_diag_geometry")
        if isinstance(geom, dict):
            row.update(geom)

        if controller is not None:
            row["wrench"] = sample_articulation_wrench(controller)
            w = row["wrench"]
            fn = w.get("wrist_force_norm_N")
            tn = w.get("wrist_torque_norm_Nm")
            if isinstance(fn, (int, float)):
                self._peak_force_N = max(self._peak_force_N, float(fn))
            if isinstance(tn, (int, float)):
                self._peak_torque_Nm = max(self._peak_torque_Nm, float(tn))

        monitor = context.services.get("collision_monitor")
        if monitor is not None and hasattr(monitor, "recent_contacts"):
            contacts = monitor.recent_contacts(limit=8)
            row["contacts"] = contacts
            for c in contacts:
                imp = c.get("impulse_norm")
                if isinstance(imp, (int, float)):
                    self._peak_contact_impulse = max(
                        self._peak_contact_impulse, float(imp)
                    )

        hold = context.services.get("_last_cable_hold_info")
        if isinstance(hold, dict):
            row["cable_hold"] = {
                k: hold.get(k)
                for k in (
                    "in_gripper",
                    "tip_err_m",
                    "closed_enough",
                    "fingers",
                    "both_pads",
                )
                if k in hold
            }

        if extra:
            row["extra"] = extra

        self.samples.append(row)
        self._append_disk(row)

        # Compact console pulse when force or contact spikes.
        w = row.get("wrench") or {}
        force_n = w.get("wrist_force_norm_N")
        finger_e = w.get("finger_joint_effort")
        contacts = row.get("contacts") or []
        top_imp = None
        if contacts:
            top_imp = max(
                (c.get("impulse_norm") or 0.0 for c in contacts),
                default=None,
            )
        every = max(1, int(getattr(cfg, "INSERT_DIAG_LOG_EVERY_N", 5)))
        frame = int(row.get("frame") or 0)
        spike = (
            (isinstance(force_n, (int, float)) and force_n >= float(
                getattr(cfg, "INSERT_DIAG_FORCE_LOG_N", 5.0)
            ))
            or (isinstance(top_imp, (int, float)) and top_imp >= float(
                getattr(cfg, "INSERT_DIAG_IMPULSE_LOG", 0.01)
            ))
        )
        if spike or frame % every == 0:
            tag = f" {self.station_id}" if self.station_id else ""
            print(
                f"[INSERT DIAG{tag}] "
                f"f={frame} label={row.get('label')} "
                f"F={force_n}N T={w.get('wrist_torque_norm_Nm')}Nm "
                f"finger_τ={finger_e} "
                f"gap_x={row.get('mating_gap_along_x_m')} "
                f"contacts={len(contacts)} max_imp={top_imp} "
                f"tip={row.get('tip_m')} crystal={row.get('crystal_mating_m')}"
            )
        return row

    def finish(self, context, *, reason: str) -> Path:
        self.reason_end = reason
        footer = {
            "type": "insert_diag_footer",
            "station": self.station_id,
            "reason": reason,
            "n_samples": len(self.samples),
            "duration_s": round(time.monotonic() - self.t0, 4),
            "peak_wrist_force_N": round(self._peak_force_N, 6),
            "peak_wrist_torque_Nm": round(self._peak_torque_Nm, 6),
            "peak_contact_impulse": round(self._peak_contact_impulse, 6),
            "abort_reason": context.services.get("abort_reason"),
            "port_inserted": bool(context.services.get("port_inserted")),
            # Re-snapshot materials in case something changed mid-run.
            "physics_end": sample_bound_materials(context),
        }
        self._append_disk(footer)
        print(
            f"[INSERT DIAG {self.station_id}] DONE reason={reason} "
            f"samples={len(self.samples)} "
            f"peak_F={footer['peak_wrist_force_N']}N "
            f"peak_T={footer['peak_wrist_torque_Nm']}Nm "
            f"peak_impulse={footer['peak_contact_impulse']} "
            f"file={self.path}"
        )
        return self.path

    def _append_disk(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, default=_json_default) + "\n")


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def start_insert_diagnostics(context) -> InsertDiagnosticsRecorder | None:
    if not bool(getattr(cfg, "INSERT_DIAG_ENABLE", True)):
        return None
    station = str(context.services.get("station_id") or "unknown")
    path = Path(getattr(cfg, "INSERT_DIAG_PATH", Path(__file__).resolve().parent / "insert_diag.jsonl"))
    # Fresh file per run (overwrite).
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
    rec = InsertDiagnosticsRecorder(station, path=path)
    rec.start(context)
    context.services["insert_diagnostics"] = rec
    return rec


def sample_insert_diagnostics(context, *, label: str | None = None, extra: dict | None = None) -> None:
    """Record one sample. Default: only when explicitly labeled (IK pose arrival)."""

    rec = context.services.get("insert_diagnostics")
    if rec is None:
        return
    every = int(getattr(cfg, "INSERT_DIAG_SAMPLE_EVERY_N", 0))
    frame = int(context.services.get("align_insert_frames", 0))
    force_sample = bool(label)
    if label:
        context.services["_insert_diag_label"] = label
    # every<=0 → pose-arrival / labeled samples only (no per-frame cadence).
    cadence = every > 0 and frame % every == 0
    if force_sample or cadence:
        try:
            rec.sample(context, label=label, extra=extra)
        except Exception as exc:
            print(f"[INSERT DIAG] sample failed: {exc}")


def finish_insert_diagnostics(context, *, reason: str) -> None:
    rec = context.services.pop("insert_diagnostics", None)
    monitor = context.services.get("collision_monitor")
    if monitor is not None:
        if hasattr(monitor, "set_logging"):
            monitor.set_logging(False)
        if hasattr(monitor, "clear_ignore_obstacles"):
            monitor.clear_ignore_obstacles()
    if rec is None:
        return
    try:
        rec.finish(context, reason=reason)
    except Exception as exc:
        print(f"[INSERT DIAG] finish failed: {exc}")
