"""Detect when the QSFP module contacts the port / switch geometry."""

from __future__ import annotations

NETWORK_SWITCHES_PREFIX = "/World/DataHall/Network_Switches"


def _enable_extension(extension_id: str) -> None:
    import omni.kit.app

    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    if not ext_mgr.is_extension_enabled(extension_id):
        ext_mgr.set_extension_enabled_immediate(extension_id, True)


def _create_contact_sensor(module_prim_path: str, name_suffix: str):
    sensor_path = f"{module_prim_path}/port_insert_contact_sensor"
    try:
        _enable_extension("isaacsim.sensors.experimental.physics")
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor

        sensor = ContactSensor(
            Contact.create(
                sensor_path,
                min_threshold=0,
                max_threshold=1_000_000,
                radius=-1,
            )
        )
        sensor.add_raw_contact_data_to_frame()
        return sensor, "experimental"
    except ImportError:
        pass

    _enable_extension("isaacsim.sensors.physics")
    from isaacsim.sensors.physics import ContactSensor

    sensor = ContactSensor(
        prim_path=sensor_path,
        name=f"port_insert_contact_{name_suffix}",
        min_threshold=0,
        max_threshold=1_000_000,
        radius=-1,
    )
    if hasattr(sensor, "initialize"):
        sensor.initialize()
    return sensor, "legacy"


class ModulePortContactMonitor:
    """Reports when a QSFP module rigid body hits port rack colliders."""

    def __init__(self, module_prim_path: str, port_prim_path: str) -> None:
        self._module_prim_path = module_prim_path
        self._port_prim_path = port_prim_path
        self._port_name = port_prim_path.rsplit("/", 1)[-1]
        name_suffix = module_prim_path.rsplit("_", 1)[-1]
        self.sensor, self._api = _create_contact_sensor(module_prim_path, name_suffix)

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

    def _contacts_from_sensor(self) -> list[dict]:
        if self._api == "experimental":
            self.sensor.add_raw_contact_data_to_frame()
            frame = self.sensor.get_data()
            contacts = frame.get("contacts")
            return contacts if contacts else []
        self.sensor.add_raw_contact_data_to_frame()
        frame = self.sensor.get_current_frame()
        contacts = frame.get("contacts")
        return contacts if contacts else []

    def in_contact(self) -> bool:
        for contact in self._contacts_from_sensor():
            body0 = str(contact.get("body0", ""))
            body1 = str(contact.get("body1", ""))
            if self._is_module_body(body0) and self._is_port_rack_body(body1):
                return True
            if self._is_module_body(body1) and self._is_port_rack_body(body0):
                return True
        return False
