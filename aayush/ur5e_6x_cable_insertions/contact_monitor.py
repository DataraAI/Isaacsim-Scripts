"""Runtime PhysX contact reports for insert diagnostics (log only, no abort)."""

from __future__ import annotations

import time

import numpy as np
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools

from ur5e_6x_cable_insertions import config as cfg


class DataHallCollisionMonitor:
    """Log any Network-cable ↔ mesh contact during insert (no abort).

    Watches the cable root. When logging is on, prints every contact between a
    cable prim and any other mesh (including Ethernet/RJ45). Gripper/robot
    contacts are skipped by default so pad pinch does not flood the log.
    """

    def __init__(
        self,
        watched_paths: tuple[str, ...],
        *,
        skip_prefixes: tuple[str, ...] | None = None,
        cooldown_s: float | None = None,
    ) -> None:
        self._watched = tuple(str(p) for p in watched_paths if p)
        # Paths that are not "scene meshes of interest" (grasp pads, etc.).
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
        # Insert-only logging: off until queue_align_and_insert enables it.
        self._log_enabled = False
        self._recent: list[dict] = []
        self._recent_limit = 128
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report
        )
        print(
            f"[COLLISION] insert-diag monitor ready "
            f"(cable roots={list(self._watched)}; logging off until insert; "
            f"log any mesh; skip={list(self._skip_prefixes) or '-'})"
        )

    def set_logging(self, enabled: bool) -> None:
        """Enable/disable ``[COLLISION]`` prints (only turn on during insert)."""

        self._log_enabled = bool(enabled)
        state = "ON" if self._log_enabled else "OFF"
        print(f"[COLLISION] insert cable↔mesh logging {state}")

    def set_skip_prefixes(self, prefixes: tuple[str, ...] | list[str] | None) -> None:
        self._skip_prefixes = tuple(str(p) for p in (prefixes or ()) if p)

    # Kept for call-site compatibility (port ignore removed — log all meshes).
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
                return True  # cable↔cable self-contact
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

    def _on_contact_report(self, contact_headers, contact_data) -> None:
        if not self._log_enabled:
            return

        for header in contact_headers:
            actor0 = self._path(header.actor0)
            actor1 = self._path(header.actor1)
            collider0 = self._path(header.collider0)
            collider1 = self._path(header.collider1)

            # Prefer the collider (mesh) path when available.
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
            # Prefer the finest cable-side collider path.
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
        """No-op kept for call-site compatibility (monitor never aborts)."""

    def recent_contacts(self, limit: int = 8, *, include_ignored: bool = False) -> list[dict]:
        if limit <= 0:
            return []
        return list(self._recent[-limit:])

    def clear_recent_contacts(self) -> None:
        self._recent.clear()
        self._active_pairs.clear()

    def consume_hit(self) -> tuple[str, str] | None:
        """Never aborts — always returns None."""

        return None

    @property
    def ever_contacted(self) -> bool:
        return bool(self._seen_pairs)

    def report_summary(self) -> None:
        print(
            f"[COLLISION] insert-diag summary "
            f"unique_pairs={len(self._seen_pairs)} episodes={self._contact_episodes}"
        )

    def close(self) -> None:
        self._subscription = None
