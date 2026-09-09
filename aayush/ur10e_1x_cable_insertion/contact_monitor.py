"""Runtime contact logging for rigid cable heads against DataHall colliders."""

from __future__ import annotations

import time

import numpy as np
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools


class CableDataHallContactMonitor:
    """Log debounced cable-head contacts with geometry under DataHall."""

    DATAHALL_PREFIX = "/World/DataHall"

    def __init__(self, head_paths: tuple[str, str], cooldown_s: float = 0.5) -> None:
        self._head_paths = tuple(str(path) for path in head_paths)
        self._cooldown_s = float(cooldown_s)
        self._active_pairs: set[tuple[str, str]] = set()
        self._seen_pairs: set[tuple[str, str]] = set()
        self._last_log_time: dict[tuple[str, str], float] = {}
        self._contact_episodes = 0
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report
        )
        print(
            "[CABLE CONTACT] monitoring rigid crystal heads against /World/DataHall "
            "(deformable cable line is not covered)"
        )

    @staticmethod
    def _path(path_id: int) -> str:
        try:
            return str(PhysicsSchemaTools.intToSdfPath(path_id))
        except Exception:
            return ""

    def _head_path(self, actor: str, collider: str) -> str | None:
        for head in self._head_paths:
            if actor == head or actor.startswith(f"{head}/"):
                return head
            if collider == head or collider.startswith(f"{head}/"):
                return head
        return None

    def _datahall_path(self, actor: str, collider: str) -> str | None:
        for path in (collider, actor):
            if path == self.DATAHALL_PREFIX or path.startswith(f"{self.DATAHALL_PREFIX}/"):
                return path
        return None

    @staticmethod
    def _contact_details(header, contact_data) -> tuple[str, str]:
        if int(header.num_contact_data) <= 0:
            return "unknown", "unknown"
        contact = contact_data[int(header.contact_data_offset)]
        position = np.asarray(contact.position, dtype=np.float64).reshape(-1)
        impulse = np.asarray(contact.impulse, dtype=np.float64).reshape(-1)
        return (
            np.array2string(position, precision=4, separator=","),
            f"{float(np.linalg.norm(impulse)):.6f}",
        )

    def _on_contact_report(self, contact_headers, contact_data) -> None:
        for header in contact_headers:
            actor0 = self._path(header.actor0)
            actor1 = self._path(header.actor1)
            collider0 = self._path(header.collider0)
            collider1 = self._path(header.collider1)

            head = self._head_path(actor0, collider0)
            datahall = self._datahall_path(actor1, collider1)
            if head is None or datahall is None:
                head = self._head_path(actor1, collider1)
                datahall = self._datahall_path(actor0, collider0)
            if head is None or datahall is None:
                continue

            pair = (head, datahall)
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
            first_contact = pair not in self._seen_pairs
            self._seen_pairs.add(pair)

            now = time.monotonic()
            if now - self._last_log_time.get(pair, float("-inf")) < self._cooldown_s:
                continue
            self._last_log_time[pair] = now
            position, impulse = self._contact_details(header, contact_data)
            event = "FIRST" if first_contact else "RECONTACT"
            print(
                f"[CABLE CONTACT] {event} head={head} datahall={datahall} "
                f"position={position} impulse={impulse}"
            )

    @property
    def ever_contacted(self) -> bool:
        return bool(self._seen_pairs)

    def report_summary(self) -> None:
        print(
            f"[CABLE CONTACT] summary datahall_contact="
            f"{'yes' if self.ever_contacted else 'no'} "
            f"unique_pairs={len(self._seen_pairs)} episodes={self._contact_episodes}"
        )

    def close(self) -> None:
        self._subscription = None
