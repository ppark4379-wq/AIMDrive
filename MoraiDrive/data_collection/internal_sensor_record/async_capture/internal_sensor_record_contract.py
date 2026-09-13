"""Pinned admission contract for MORAI 26.R1.H3 internal Sensor Record.

Only immutable build/sensor-set facts are checked here. Runtime time mode and
10 Hz cadence are checked by the scenario runner and the native-file audit.
"""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from network_config import bridge_ip
from runtime_paths import LAUNCHER_ROOT


ASSEMBLY_SHA256 = "6bd0fb11b0ccf8d4f7ca6131201eaca86c9e167d6a41b5876013d920d3769649"
LEGACY_SENSOR_SET_RAW_SHA256 = "17cb45f05f8044339d23964ca308c0810eb7e71ba10ada04769be779cb1c24b2"
# Filled from a canonical representation that excludes only the machine-local
# rosbridge hostname/IP. Geometry, rate, topic, frame, port, and enable flags
# remain immutable parts of this fingerprint.
SENSOR_SET_SHA256 = "e07a6cc30177e8ab2076b4dfd02cab4c8e40e8cea9a4850dab769e7d156c4962"
STATIC_EVIDENCE_SHA256 = "a64a7998ce79087328e3e026de5eebc62ac768177ef9f940347bf6f8a4d61772"
MODULE_ID = "b1f00d05-80bb-4e73-a7bb-2af2156e56ff"
CAPTURE_MODE = {
    "time_mode": "SYNC_MODE_TYPE_TIME_MANAGEMENT",
    "external_tick_period": 0,
    "internal_sensor_synchronization_hz": 10,
    "external_sync": False,
}
DEFAULT_ASSEMBLY = Path(
    os.environ.get("MORAI_ASSEMBLY_PATH",
        str(LAUNCHER_ROOT / "MoraiLauncher_Win_Data/Project/Simulator/"
            "Simulator_v.R1.260701.H3_Window/Simulator_Data/Managed/Assembly-CSharp.dll"))
)
DEFAULT_SENSOR_SET = Path(
    os.environ.get("MORAI_SENSOR_SET_PATH",
        str(LAUNCHER_ROOT / "MoraiLauncher_Win_Data/SaveFile/Sensor/"
            "26.R1.H3/aim_sensorset.json"))
)


def file_sha256(path, chunk_size=1024 * 1024):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def _number(value):
    return float(value)


def _close(actual, expected, tolerance=2e-6):
    return abs(_number(actual) - expected) <= tolerance


def _ros(config, *, topic, frame, expected_bridge_ip):
    ros = config["rosConfig"]
    return {
        "enabled": ros.get("RosMessageOnOff") is True,
        "transport": config.get("commType") == 3,
        "url": ros.get("RosBridgeServerUrl"),
        "ip": ros.get("RosBridgeIP"),
        "port": ros.get("RosBridgePort"),
        "topic": ros.get("Topic"),
        "frame_id": ros.get("frameID"),
        "matches": (
            ros.get("RosMessageOnOff") is True
            and config.get("commType") == 3
            and ros.get("RosBridgeServerUrl") == "ws://%s:9090" % expected_bridge_ip
            and ros.get("RosBridgeIP") == expected_bridge_ip
            and ros.get("RosBridgePort") == 9090
            and ros.get("Topic") == topic
            and ros.get("frameID") == frame
        ),
    }


def _canonical_sensor_set(data):
    """Hash sensor semantics while allowing each workstation's bridge IP."""
    def normalize(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key == "RosBridgeIP":
                    result[key] = "<MORAI_BRIDGE_IP>"
                elif key == "RosBridgeServerUrl" and isinstance(item, str):
                    parsed = urlsplit(item)
                    result[key] = urlunsplit((
                        parsed.scheme, "<MORAI_BRIDGE_IP>:%d" % (parsed.port or 9090),
                        parsed.path, parsed.query, parsed.fragment,
                    ))
                else:
                    result[key] = normalize(item)
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    return normalize(data)


def sensor_set_contract_sha256(data):
    payload = json.dumps(
        _canonical_sensor_set(data), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_sensor_set(data, expected_bridge_ip=None):
    expected_bridge_ip = expected_bridge_ip or bridge_ip()
    expected_cameras = {
        1: dict(position=(1.9, 0.0, 1.2), rotation=(0.0, 2.0, 0.0),
                resolution=(1280, 720), fov=90.0, frame="Camera-1"),
        2: dict(position=(1.15, 0.65, 1.2), rotation=(0.0, 10.0, 70.0),
                resolution=(640, 480), fov=130.0, frame="Camera-2"),
        3: dict(position=(1.15, -0.65, 1.2), rotation=(0.0, 10.0, 290.0),
                resolution=(640, 480), fov=130.0, frame="Camera-3"),
        4: dict(position=(-0.14, 0.0, 1.2), rotation=(0.0, 2.0, 180.0),
                resolution=(640, 480), fov=90.0, frame="Camera-4"),
    }
    failures = []
    camera_results = {}
    cameras = data.get("cameraList", [])
    if len(cameras) != 4:
        failures.append("camera_count_not_four")
    camera_ids = [int(row.get("m_SensorUniqueID", -1)) for row in cameras]
    if sorted(camera_ids) != sorted(expected_cameras):
        failures.append("camera_id_set_mismatch")
    for row in cameras:
        uid = int(row.get("m_SensorUniqueID", -1))
        expected = expected_cameras.get(uid)
        if expected is None:
            failures.append(f"unexpected_camera_id:{uid}")
            continue
        cfg = row["cc"]
        position = tuple(_number(row["pos"][axis]) for axis in ("x", "y", "z"))
        rotation = tuple(_number(row["rot"][axis]) for axis in ("roll", "pitch", "yaw"))
        checks = {
            "position": all(_close(a, b) for a, b in zip(position, expected["position"])),
            "rotation": all(_close(a, b) for a, b in zip(rotation, expected["rotation"])),
            "resolution": (cfg.get("cameraResWidth"), cfg.get("cameraResHeight")) == expected["resolution"],
            "fov": _close(cfg.get("cameraFOV"), expected["fov"]),
            "sensor_period": _close(cfg.get("sensorPeriod"), 0.05),
            "camera_type": cfg.get("cameraType") == 0,
            "model_type": cfg.get("modelType") == 0,
        }
        checks["ros"] = _ros(cfg, topic="/image_jpeg/compressed", frame=expected["frame"],
                             expected_bridge_ip=expected_bridge_ip)
        if not all(value if isinstance(value, bool) else value["matches"] for value in checks.values()):
            failures.append(f"camera_contract_mismatch:{uid}")
        camera_results[str(uid)] = checks

    def exactly_one(name, uid, config_name):
        rows = data.get(name, [])
        if len(rows) != 1 or int(rows[0].get("m_SensorUniqueID", -1)) != uid:
            failures.append(f"{name}_identity_mismatch")
            return None, None
        return rows[0], rows[0][config_name]

    lidar, lc = exactly_one("Lidar3DList", 5, "lc")
    gps, gc = exactly_one("GPSList", 6, "gc")
    imu, ic = exactly_one("IMUList", 7, "ic")
    sensor_results = {}
    if lidar is not None:
        checks = {
            "position": all(_close(lidar["pos"][axis], value) for axis, value in zip(("x", "y", "z"), (1.0, 0.0, 1.48))),
            "rotation": all(_close(lidar["rot"][axis], 0.0) for axis in ("roll", "pitch", "yaw")),
            "sensor_period": _close(lc.get("sensorPeriod"), 0.1),
            "intensity_type": lc.get("intensityType") == 0,
            "ros": _ros(lc, topic="/lidar3D", frame="Lidar3D-5",
                        expected_bridge_ip=expected_bridge_ip),
        }
        if not all(value if isinstance(value, bool) else value["matches"] for value in checks.values()):
            failures.append("lidar_contract_mismatch")
        sensor_results["lidar"] = checks
    if gps is not None:
        checks = {
            "sensor_period": _close(gc.get("sensorPeriod"), 0.05),
            "ros": _ros(gc, topic="/gps", frame="GPS-6",
                        expected_bridge_ip=expected_bridge_ip),
        }
        if not all(value if isinstance(value, bool) else value["matches"] for value in checks.values()):
            failures.append("gps_contract_mismatch")
        sensor_results["gps"] = checks
    if imu is not None:
        checks = {
            "sensor_period": _close(ic.get("sensorPeriod"), 0.02),
            "ros": _ros(ic, topic="/imu", frame="IMU-7",
                        expected_bridge_ip=expected_bridge_ip),
        }
        if not all(value if isinstance(value, bool) else value["matches"] for value in checks.values()):
            failures.append("imu_contract_mismatch")
        sensor_results["imu"] = checks
    for name in ("Lidar2DList", "RadarList", "Radar4DList", "UltrasonicList"):
        if data.get(name):
            failures.append(f"unexpected_enabled_sensor_list:{name}")
    return {
        "valid": not failures,
        "failures": failures,
        "cameras": camera_results,
        "sensors": sensor_results,
    }


def validate_static_evidence(report):
    findings = report.get("findings", {})
    failures = []
    if report.get("assembly_sha256", "").lower() != ASSEMBLY_SHA256:
        failures.append("evidence_assembly_mismatch")
    if report.get("module_id") != MODULE_ID:
        failures.append("evidence_module_mismatch")
    if not str(findings.get("internal_record_same_simulation_state", "")).startswith("strongly verified"):
        failures.append("same_simulation_state_not_verified")
    if findings.get("save_sensor_data_trigger_has_same_contract") is not False:
        failures.append("trigger_path_not_separated")
    return {"valid": not failures, "failures": failures}


def validate_preflight_snapshot(report):
    """Validate the immutable facts copied into a per-run capture contract."""
    hashes = report.get("sha256", {})
    checks = {
        "schema": report.get("schema") == "morai_internal_sensor_record_preflight_v1",
        "ready": report.get("ready") is True and not report.get("failures"),
        "assembly": hashes.get("assembly") == ASSEMBLY_SHA256,
        "sensor_set": (
            hashes.get("sensor_set_contract") == SENSOR_SET_SHA256
            or hashes.get("sensor_set") == LEGACY_SENSOR_SET_RAW_SHA256
        ),
        "static_evidence": hashes.get("static_evidence") == STATIC_EVIDENCE_SHA256,
        "capture_mode": report.get("expected_capture_mode") == CAPTURE_MODE,
        "sensor_set_validation": report.get("sensor_set_validation", {}).get("valid") is True,
        "static_evidence_validation": report.get("static_evidence_validation", {}).get("valid") is True,
        "not_published": report.get("training_visible") is False,
    }
    return {"valid": all(checks.values()), "checks": checks}


def capture_preflight(assembly_path, sensor_set_path, static_evidence_path):
    assembly_path = Path(assembly_path)
    sensor_set_path = Path(sensor_set_path)
    static_evidence_path = Path(static_evidence_path)
    sensor_data = json.loads(sensor_set_path.read_text())
    hashes = {
        "assembly": file_sha256(assembly_path),
        "sensor_set": file_sha256(sensor_set_path),
        "sensor_set_contract": sensor_set_contract_sha256(sensor_data),
        "static_evidence": file_sha256(static_evidence_path),
    }
    expected_bridge_ip = bridge_ip()
    sensor = validate_sensor_set(sensor_data, expected_bridge_ip=expected_bridge_ip)
    evidence = validate_static_evidence(json.loads(static_evidence_path.read_text()))
    failures = list(sensor["failures"]) + list(evidence["failures"])
    if hashes["assembly"] != ASSEMBLY_SHA256:
        failures.append("simulator_assembly_sha256_changed")
    if hashes["sensor_set_contract"] != SENSOR_SET_SHA256:
        failures.append("sensor_set_contract_sha256_changed")
    if hashes["static_evidence"] != STATIC_EVIDENCE_SHA256:
        failures.append("static_evidence_sha256_changed")
    return {
        "schema": "morai_internal_sensor_record_preflight_v1",
        "ready": not failures,
        "failures": failures,
        "paths": {
            "assembly": str(assembly_path),
            "sensor_set": str(sensor_set_path),
            "static_evidence": str(static_evidence_path),
        },
        "sha256": hashes,
        "expected_capture_mode": dict(CAPTURE_MODE),
        "expected_bridge": {"ip": expected_bridge_ip, "port": 9090},
        "sensor_set_validation": sensor,
        "static_evidence_validation": evidence,
        "scope": "immutable build and sensor-set admission; live mode and cadence require runtime checks",
        "training_visible": False,
    }
