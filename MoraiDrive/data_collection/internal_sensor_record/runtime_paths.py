"""Resolve workstation-specific paths without committing one PC's layout.

The optional JSON file below is intentionally ignored by Git:
``<repo>/MoraiDrive/.runtime/data_collection/local_machine.json``.
Environment variables always take precedence over that file.
"""
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil


PROJECT_ROOT = Path(
    os.environ.get("MORAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser().resolve()
LOCAL_MACHINE_CONFIG = Path(
    os.environ.get(
        "MORAI_LOCAL_CONFIG",
        str(PROJECT_ROOT / ".runtime/data_collection/local_machine.json"),
    )
).expanduser().resolve()


def load_local_machine_config():
    if not LOCAL_MACHINE_CONFIG.exists():
        return {}
    if not LOCAL_MACHINE_CONFIG.is_file():
        raise ValueError("MORAI local config is not a file: " + str(LOCAL_MACHINE_CONFIG))
    value = json.loads(LOCAL_MACHINE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("MORAI local config must contain one JSON object")
    return value


LOCAL_MACHINE = load_local_machine_config()


def setting(environment_name, local_key, default=""):
    value = os.environ.get(environment_name, "").strip()
    if value:
        return value
    value = LOCAL_MACHINE.get(local_key, default)
    if not isinstance(value, str):
        raise ValueError("Local machine setting must be a string: " + local_key)
    return value.strip()


def configured_path(environment_name, local_key, default):
    return Path(setting(environment_name, local_key, str(default))).expanduser().resolve()


ROS_DISTRO = os.environ.get("ROS_DISTRO", "noetic").strip() or "noetic"
ROS_ROOT = configured_path("MORAI_ROS_ROOT", "ros_root", Path("/opt/ros") / ROS_DISTRO)
CATKIN_ROOT = configured_path(
    "MORAI_CATKIN_WS", "catkin_workspace", Path.home() / "catkin_ws"
)
LAUNCHER_ROOT = configured_path(
    "MORAI_LAUNCHER_ROOT", "launcher_root", Path("/mnt/c/MoraiLauncher_Win")
)
POWERSHELL_EXE = configured_path(
    "MORAI_POWERSHELL_EXE",
    "powershell_exe",
    shutil.which("powershell.exe")
    or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
)


def windows_path_from_wsl(path):
    """Translate an absolute WSL DrvFS path into an absolute Windows path."""
    resolved = Path(path).expanduser().resolve()
    match = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", resolved.as_posix())
    if not match:
        raise ValueError(
            "Path must be on a Windows-backed /mnt/<drive> mount: " + str(resolved)
        )
    result = PureWindowsPath(
        match.group(1).upper() + ":\\",
        *(Path(match.group(2)).parts if match.group(2) else ()),
    )
    if "'" in str(result) or any(character in str(result) for character in "\r\n"):
        raise ValueError("Path contains characters unsafe for PowerShell")
    return str(result)


WINDOWS_LAUNCHER_ROOT = windows_path_from_wsl(LAUNCHER_ROOT)
WINDOWS_SENSOR_DATA_ROOT = str(
    PureWindowsPath(
        WINDOWS_LAUNCHER_ROOT,
        "MoraiLauncher_Win_Data",
        "SaveFile",
        "SensorData",
    )
)
