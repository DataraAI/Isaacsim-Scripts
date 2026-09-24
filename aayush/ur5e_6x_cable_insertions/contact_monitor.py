"""Runtime PhysX contact reports for insert diagnostics (log only, no abort)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools

from ur5e_6x_cable_insertions import config as cfg


class DataHallCollisionMonitor:
    """Log Network-cable ↔ mesh contacts during insert (no abort).

    General ``[COLLISION]`` spam is gated by ``set_logging``. Independently,
    FIRST crystal-head-45 ↔ Mesh133/Mesh134 hits log crystal mating center vs
    port mating (YZ error + insert-axis vs −X) for waypoint / delta tuning.
    """

    def __init__(
        self,
        watched_paths: tuple[str, ...],
        *,
        skip_prefixes: tuple[str, ...] | None = None,
        cooldown_s: float | None = None,
    ) -> None:
        self._watched = tuple(str(p) for p in watched_paths if p)
        self._skip_prefixes: tuple[str, ...] = tuple(
            str(p) for p in (skip_prefixes or ()) if p
        )
        self._cooldown_s = float(
            cfg.MANEUVER_CONTACT_COOLDOWN_S if cooldown_s is None else cooldown_s
        )
        self._active_pairs: set[tuple[str, str]] = set()
        self._seen_pairs: set[tuple[str, str]] = set()
        self._last_log_time: dict[tuple[str, str], float] = {}
        self._contact_episodes = 0
        self._log_enabled = False
        self._recent: list[dict] = []
        self._recent_limit = 128
        self._services: dict[str, Any] | None = None
        self._port_hit_logged = False
        self._port_hit: dict[str, Any] | None = None
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report
        )
        print(
            f"[COLLISION] insert-diag monitor ready "
            f"(cable roots={list(self._watched)}; logging off until insert; "
            f"skip={list(self._skip_prefixes) or '-'}; "
            f"port-hit log={bool(getattr(cfg, 'INSERT_PORT_HIT_LOG', True))})"
        )

    def bind_services(self, services: dict[str, Any] | None) -> None:
        self._services = services

    def begin_insert_session(self) -> None:
        self._port_hit_logged = False
        self._port_hit = None
        if self._services is not None:
            self._services["insert_port_hit"] = None

    def set_logging(self, enabled: bool) -> None:
        self._log_enabled = bool(enabled)
        state = "ON" if self._log_enabled else "OFF"
        print(f"[COLLISION] insert cable↔mesh logging {state}")

    def set_skip_prefixes(self, prefixes: tuple[str, ...] | list[str] | None) -> None:
        self._skip_prefixes = tuple(str(p) for p in (prefixes or ()) if p)

    def set_ignore_obstacles(
        self,
        prefixes: tuple[str, ...] | list[str] | None = None,
        tokens: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        return

    def clear_ignore_obstacles(self) -> None:
        return

    @staticmethod
    def _path(path_id: int) -> str:
        try:
            return str(PhysicsSchemaTools.intToSdfPath(path_id))
        except Exception:
            return ""

    def _under(self, path: str, root: str) -> bool:
        return bool(path) and (path == root or path.startswith(f"{root}/"))

    def _is_watched(self, *paths: str) -> tuple[str, str] | None:
        best: tuple[str, str] | None = None
        for path in paths:
            if not path:
                continue
            for watched in self._watched:
                if self._under(path, watched):
                    if best is None or len(path) > len(best[1]):
                        best = (watched, path)
        return best

    def _is_skipped(self, path: str) -> bool:
        if not path:
            return True
        for watched in self._watched:
            if self._under(path, watched):
                return True
        for prefix in self._skip_prefixes:
            if self._under(path, prefix):
                return True
        return False

    @staticmethod
    def _finest(*paths: str) -> str:
        best = ""
        for path in paths:
            if path and len(path) > len(best):
                best = path
        return best

    def _path45(self) -> str:
        services = self._services or {}
        path = str(services.get("path45") or "")
        if path:
            return path
        spec = services.get("station_spec")
        if spec is not None:
            return str(getattr(spec, "path45", "") or "")
        return ""

    def _is_crystal_head45(self, path: str) -> bool:
        if not path:
            return False
        head = str(getattr(cfg, "HEAD45_NAME", "E_crystal_head1_45"))
        if f"/{head}" in path or path.endswith(head):
            return True
        path45 = self._path45()
        return bool(path45) and self._under(path, path45)

    def _is_port_jack_mesh(self, path: str) -> bool:
        if not path:
            return False
        name = path.rsplit("/", 1)[-1]
        tokens = tuple(
            str(t)
            for t in getattr(cfg, "INSERT_PORT_HIT_MESH_NAMES", ("Mesh133", "Mesh134"))
            if t
        )
        return name in tokens

    def _sample_crystal_and_port(self) -> dict[str, Any]:
        """Live crystal mating / axis and port mating (meters)."""

        out: dict[str, Any] = {}
        services = self._services or {}
        stage = services.get("stage")
        path45 = self._path45()
        neg_x = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        if stage is not None and path45:
            try:
                from insertion_features.cable_features import (
                    world_features_from_crystal_head,
                )

                head = world_features_from_crystal_head(stage, path45)
                feats = head.features
                c_mc = np.asarray(feats.mating_center, dtype=np.float64).reshape(3)
                axis = np.asarray(feats.insertion_axis, dtype=np.float64).reshape(3)
                n = float(np.linalg.norm(axis))
                if n > 1e-12:
                    axis = axis / n
                if float(np.dot(axis, neg_x)) < 0.0:
                    axis = -axis
                out["crystal_mating_m"] = c_mc
                out["crystal_insert_axis"] = axis
                out["axis_dot_neg_x"] = float(np.dot(axis, neg_x))
            except Exception as exc:
                out["crystal_error"] = str(exc)

        port_mc = services.get("insert_port_mating_center")
        if port_mc is None:
            port = services.get("port_features")
            if port is not None:
                try:
                    port_mc = port.mating_center
                except Exception:
                    port_mc = None
        if port_mc is not None:
            p = np.asarray(port_mc, dtype=np.float64).reshape(3)
            out["port_mating_m"] = p
            c_mc = out.get("crystal_mating_m")
            if c_mc is not None:
                delta = c_mc - p
                out["delta_crystal_minus_port_m"] = delta
                out["delta_yz_m"] = delta[1:3].copy()
                # Shift insert YZ path by −Δyz so crystal mating lands on port YZ.
                cur_dy = float(getattr(cfg, "INSERT_CRYSTAL_Y_DELTA_M", 0.0))
                cur_dz = float(getattr(cfg, "INSERT_CRYSTAL_Z_DELTA_M", 0.0))
                out["suggested_INSERT_CRYSTAL_Y_DELTA_M"] = cur_dy - float(delta[1])
                out["suggested_INSERT_CRYSTAL_Z_DELTA_M"] = cur_dz - float(delta[2])
                out["suggested_tip_yz_nudge_m"] = (-float(delta[1]), -float(delta[2]))
        return out

    def _maybe_log_port_hit(
        self,
        *,
        cable_mesh: str,
        obstacle_mesh: str,
        impulse_norm: float | None,
    ) -> None:
        if not bool(getattr(cfg, "INSERT_PORT_HIT_LOG", True)):
            return
        if self._port_hit_logged:
            return
        if not self._is_crystal_head45(cable_mesh):
            return
        if not self._is_port_jack_mesh(obstacle_mesh):
            return

        services = self._services or {}
        geom = self._sample_crystal_and_port()
        last_wp = services.get("insert_last_completed_waypoint")
        cur_wp = services.get("insert_current_waypoint")

        event = {
            "cable_mesh": cable_mesh,
            "obstacle": obstacle_mesh,
            "impulse_norm": impulse_norm,
            "t": time.monotonic(),
            "insert_current_waypoint": cur_wp,
            "insert_last_completed_waypoint": last_wp,
            **{
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in geom.items()
            },
        }
        self._port_hit = event
        self._port_hit_logged = True
        services["insert_port_hit"] = event

        station = ""
        spec = services.get("station_spec")
        if spec is not None:
            station = f" {getattr(spec, 'station_id', '')}"

        def _fmt_vec(v) -> str:
            if v is None:
                return "None"
            return str(np.round(np.asarray(v, dtype=np.float64), 5))

        def _wp_s(wp) -> str:
            if not isinstance(wp, dict):
                return "None"
            tip = wp.get("tip")
            mc = wp.get("mating_center")
            return (
                f"{wp.get('label', '?')} tip={_fmt_vec(tip)} mc={_fmt_vec(mc)}"
            )

        c_mc = geom.get("crystal_mating_m")
        p_mc = geom.get("port_mating_m")
        delta = geom.get("delta_crystal_minus_port_m")
        axis = geom.get("crystal_insert_axis")
        dot = geom.get("axis_dot_neg_x")
        print(
            f"[PORT HIT{station}] FIRST head45↔{obstacle_mesh.rsplit('/', 1)[-1]}\n"
            f"  cable={cable_mesh}\n"
            f"  obstacle={obstacle_mesh}\n"
            f"  impulse={impulse_norm if impulse_norm is not None else 'unknown'}\n"
            f"  crystal_mating_m={_fmt_vec(c_mc)}\n"
            f"  port_mating_m={_fmt_vec(p_mc)}\n"
            f"  Δ(crystal−port)_m={_fmt_vec(delta)} "
            f"(ΔY={None if delta is None else f'{float(delta[1]):+.5f}'}, "
            f"ΔZ={None if delta is None else f'{float(delta[2]):+.5f}'})\n"
            f"  insert_axis={_fmt_vec(axis)} "
            f"dot(−X)={None if dot is None else f'{float(dot):+.4f}'} "
            f"(want ≈ 1.0)\n"
            f"  suggested tip YZ nudge_m="
            f"{geom.get('suggested_tip_yz_nudge_m')}\n"
            f"  suggested INSERT_CRYSTAL_Y_DELTA_M="
            f"{geom.get('suggested_INSERT_CRYSTAL_Y_DELTA_M')} "
            f"Z_DELTA_M={geom.get('suggested_INSERT_CRYSTAL_Z_DELTA_M')}\n"
            f"  last_completed_wp={_wp_s(last_wp)}\n"
            f"  current_wp={_wp_s(cur_wp)}"
        )

    def _on_contact_report(self, contact_headers, contact_data) -> None:
        port_hit_on = bool(getattr(cfg, "INSERT_PORT_HIT_LOG", True))
        if not self._log_enabled and not port_hit_on:
            return

        for header in contact_headers:
            actor0 = self._path(header.actor0)
            actor1 = self._path(header.actor1)
            collider0 = self._path(header.collider0)
            collider1 = self._path(header.collider1)

            side0 = self._finest(collider0, actor0)
            side1 = self._finest(collider1, actor1)

            watched_hit = self._is_watched(side0, collider0, actor0)
            other = side1
            if watched_hit is None:
                watched_hit = self._is_watched(side1, collider1, actor1)
                other = side0
            if watched_hit is None:
                continue
            if self._is_skipped(other):
                continue

            watched_root, watched_mesh = watched_hit
            if self._is_watched(side0, collider0, actor0) is not None:
                cable_mesh = self._finest(collider0, actor0, side0, watched_mesh)
            else:
                cable_mesh = self._finest(collider1, actor1, side1, watched_mesh)
            obstacle_mesh = other

            pair = (cable_mesh, obstacle_mesh)
            if header.type == ContactEventType.CONTACT_LOST:
                self._active_pairs.discard(pair)
                continue
            if header.type not in (
                ContactEventType.CONTACT_FOUND,
                ContactEventType.CONTACT_PERSIST,
            ):
                continue
            if pair in self._active_pairs:
                continue

            self._active_pairs.add(pair)
            self._contact_episodes += 1
            first = pair not in self._seen_pairs
            self._seen_pairs.add(pair)

            impulse_norm = None
            impulse_vec = None
            normal = None
            if int(header.num_contact_data) > 0:
                contact = contact_data[int(header.contact_data_offset)]
                try:
                    impulse_vec = [float(x) for x in contact.impulse]
                    impulse_norm = float(np.linalg.norm(contact.impulse))
                except Exception:
                    impulse_norm = None
                try:
                    normal = [float(x) for x in contact.normal]
                except Exception:
                    normal = None
            event = {
                "t": time.monotonic(),
                "watched": watched_root,
                "watched_mesh": cable_mesh,
                "obstacle": obstacle_mesh,
                "first": first,
                "impulse_norm": impulse_norm,
                "impulse": impulse_vec,
                "normal": normal,
            }
            self._recent.append(event)
            if len(self._recent) > self._recent_limit:
                self._recent = self._recent[-self._recent_limit :]

            if port_hit_on:
                self._maybe_log_port_hit(
                    cable_mesh=cable_mesh,
                    obstacle_mesh=obstacle_mesh,
                    impulse_norm=impulse_norm,
                )

            if not self._log_enabled:
                continue
            now = time.monotonic()
            if now - self._last_log_time.get(pair, float("-inf")) < self._cooldown_s:
                continue
            self._last_log_time[pair] = now
            impulse_s = (
                f"{impulse_norm:.6f}" if impulse_norm is not None else "unknown"
            )
            print(
                f"[COLLISION] {'FIRST' if first else 'RECONTACT'} "
                f"cable={cable_mesh} obstacle={obstacle_mesh} impulse={impulse_s}"
            )

    def clear_hit_flag(self) -> None:
        return

    def recent_contacts(self, limit: int = 8, *, include_ignored: bool = False) -> list[dict]:
        if limit <= 0:
            return []
        return list(self._recent[-limit:])

    def clear_recent_contacts(self) -> None:
        self._recent.clear()
        self._active_pairs.clear()

    def consume_hit(self) -> tuple[str, str] | None:
        return None

    @property
    def ever_contacted(self) -> bool:
        return bool(self._seen_pairs)

    @property
    def port_hit(self) -> dict[str, Any] | None:
        return self._port_hit

    def report_summary(self) -> None:
        hit = self._port_hit
        extra = ""
        if hit is not None:
            extra = (
                f" port_hit={hit.get('obstacle')} "
                f"Δyz={hit.get('delta_yz_m')} "
                f"axis_dot={hit.get('axis_dot_neg_x')}"
            )
        print(
            f"[COLLISION] insert-diag summary "
            f"unique_pairs={len(self._seen_pairs)} episodes={self._contact_episodes}"
            f"{extra}"
        )

    def shutdown(self) -> None:
        self._subscription = None
