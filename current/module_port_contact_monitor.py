"""Detect when the QSFP module contacts the port / switch geometry."""

from __future__ import annotations

from isaacsim.sensors.physics import ContactSensor

NETWORK_SWITCHES_PREFIX = "/World/DataHall/Network_Switches"


class ModulePortContactMonitor:
    """Reports when a QSFP module rigid body hits port rack colliders."""

    def __init__(self, module_prim_path: str, port_prim_path: str) -> None:
        self._module_prim_path = module_prim_path
        self._port_prim_path = port_prim_path
        self._port_name = port_prim_path.rsplit("/", 1)[-1]
        sensor_path = f"{module_prim_path}/port_insert_contact_sensor"
        self.sensor = ContactSensor(
            prim_path=sensor_path,
            name=f"port_insert_contact_{module_prim_path.rsplit('_', 1)[-1]}",
            min_threshold=0,
            max_threshold=1_000_000,
            radius=-1,
        )

    def _is_module_body(self, body_path: str) -> bool:
        return self._module_prim_path in body_path

    def _is_port_rack_body(self, body_path: str) -> bool:
        if self._is_module_body(body_path):
            return False
        if NETWORK_SWITCHES_PREFIX in body_path:
            return True
        if self._port_name and self._port_name in body_path:
            return True
        return "QSFP" in body_path and "Connector" in body_path

    def in_contact(self) -> bool:
        self.sensor.add_raw_contact_data_to_frame()
        frame = self.sensor.get_current_frame()
        contacts = frame.get("contacts")
        if not contacts:
            return False
        for contact in contacts:
            body0 = str(contact.get("body0", ""))
            body1 = str(contact.get("body1", ""))
            if self._is_module_body(body0) and self._is_port_rack_body(body1):
                return True
            if self._is_module_body(body1) and self._is_port_rack_body(body0):
                return True
        return False
