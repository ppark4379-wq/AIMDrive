"""Explicit environments for runner processes and the existing AIM runtime.

Project code comes only from AIM2026. Optional package directories are supplied
by the operator; the runner does not copy another project's installed packages.
"""
import json
import ipaddress
import re
import socket
import os
from pathlib import Path
import subprocess

from scenario_runner.project_paths import AIM_ROOT, PACKAGE_PARENT, SDK_SRC, SDK_PROTO, RUNTIME_ROOT


ROS_ROOT = Path(os.environ.get("MORAI_ROS_ROOT", "/opt/ros/noetic")).expanduser().resolve()
CATKIN_ROOT = Path(os.environ.get("MORAI_CATKIN_WS", str(AIM_ROOT / "Expert/ros"))).expanduser().resolve()
ASYNC_DEPS = Path(os.environ.get("AIM2026_PYTHON_DEPS") or os.environ.get(
    "MORAI_ASYNC_DEPS") or str(RUNTIME_ROOT / "python_deps")).expanduser().resolve()


def analysis_python():
    selected = os.environ.get("MORAI_ANALYSIS_PYTHON", "").strip() or os.environ.get("AIM_EXPERT_PYTHON", "").strip()
    if not selected:
        configured = Path(os.environ.get("MORAI_LOCAL_CONFIG", str(
            RUNTIME_ROOT / "data_collection/local_machine.json"))).expanduser()
        if configured.exists():
            local = json.loads(configured.read_text(encoding="utf-8"))
            if not isinstance(local, dict):
                raise ValueError("MORAI local config must contain one JSON object")
            selected = local.get("analysis_python", "")
            if not isinstance(selected, str):
                raise ValueError("Local machine setting must be a string: analysis_python")
            selected = selected.strip()
    if selected:
        import shutil
        return Path(shutil.which(selected) or selected).expanduser().resolve()
    candidates = (Path.home() / "miniconda3/envs/lmdrive/bin/python",
                  Path.home() / "miniconda3/envs/lmdrive/bin/python3.8",
                  Path("/usr/bin/python3"))
    return next((path for path in candidates if path.is_file()), candidates[-1]).resolve()


ANALYSIS_PYTHON = analysis_python()


def _unique_paths(paths):
    result = []
    for value in paths:
        if value is None or not str(value).strip():
            continue
        path = str(Path(value).expanduser().resolve())
        if path not in result:
            result.append(path)
    return result


def capture_python_paths():
    return _unique_paths([
        ASYNC_DEPS if ASYNC_DEPS.is_dir() else None,
        AIM_ROOT.parent, PACKAGE_PARENT, SDK_SRC, SDK_PROTO,
        CATKIN_ROOT / "devel/lib/python3/dist-packages",
        ROS_ROOT / "lib/python3/dist-packages",
    ])


def _environment(base=None, **updates):
    env = dict(os.environ if base is None else base)
    env.update({key: str(value) for key, value in updates.items()})
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(map(str, (CATKIN_ROOT / "devel", ROS_ROOT)))
    env["ROS_PACKAGE_PATH"] = os.pathsep.join(map(str, (CATKIN_ROOT / "src", ROS_ROOT / "share")))
    libraries = [CATKIN_ROOT / "devel/lib", ROS_ROOT / "lib"]
    for value in env.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not value.strip():
            continue
        path = Path(value).expanduser().resolve()
        if any(part in ("MoraiDrive", "VLDrive", "_support") for part in path.parts):
            continue
        libraries.append(path)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(_unique_paths(libraries))
    return env


def build_capture_environment(base=None, **updates):
    env = _environment(base, **updates)
    env["PYTHONPATH"] = os.pathsep.join(capture_python_paths())
    return env


def interpreter_package_paths(python):
    code = "import json,sysconfig;print(json.dumps([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')]))"
    result = subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError("Cannot inspect AIM Python package paths: " + result.stderr[-500:])
    return _unique_paths(json.loads(result.stdout))


def build_analysis_environment(python, base=None, **updates):
    env = _environment(base, **updates)
    system_packages = Path(os.environ.get("MORAI_SYSTEM_DIST_PACKAGES", "/usr/lib/python3/dist-packages"))
    env["PYTHONPATH"] = os.pathsep.join(_unique_paths([
        *interpreter_package_paths(python), *capture_python_paths(), system_packages,
    ]))
    return env


def ros_ip():
    """Resolve the current ROS node address; never reuse a stale inherited ROS_IP."""
    def usable(value):
        address = ipaddress.ip_address(value.strip())
        if address.version != 4 or address.is_unspecified or address.is_multicast or address.is_loopback:
            raise ValueError("Set MORAI_ROS_IP to a usable non-loopback IPv4")
        return str(address)
    explicit = os.environ.get("MORAI_ROS_IP", "").strip()
    if explicit:
        return usable(explicit)
    for command in (["ip", "-4", "route", "get", "1.1.1.1"], ["hostname", "-I"]):
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=3)
        except (OSError, subprocess.SubprocessError):
            continue
        candidates = re.findall(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])", output)
        if command[0] == "ip" and "src" in output:
            match = re.search(r"\bsrc\s+((?:[0-9]{1,3}\.){3}[0-9]{1,3})", output)
            candidates = [match.group(1)] if match else candidates
        for candidate in candidates:
            try:
                return usable(candidate)
            except ValueError:
                continue
    try:
        return usable(socket.gethostbyname(socket.gethostname()))
    except (OSError, ValueError):
        raise RuntimeError("Cannot determine the ROS IPv4; set MORAI_ROS_IP explicitly") from None
