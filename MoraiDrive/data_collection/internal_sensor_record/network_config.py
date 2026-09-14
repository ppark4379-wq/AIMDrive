"""Resolve machine-local ROS/rosbridge addresses without repository pinning."""
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess


DEFAULT_MORAI_BRIDGE_IP = "127.0.0.1"


def _normalise_ipv4(value, *, allow_loopback):
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError:
        raise ValueError("bridge address must be one IPv4 address")
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise ValueError("bridge address must be one usable IPv4 address")
    if address.is_loopback and not allow_loopback:
        raise ValueError("WSL ROS node address must be non-loopback IPv4")
    return str(address)


def validate_bridge_ip(value):
    """Validate a Windows-to-WSL rosbridge endpoint, including localhost."""
    return _normalise_ipv4(value, allow_loopback=True)


def _local_machine_bridge_ip():
    project = Path(
        os.environ.get(
            "MORAI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
        )
    ).expanduser().resolve()
    path = Path(
        os.environ.get(
            "MORAI_LOCAL_CONFIG",
            str(project / ".runtime/data_collection/local_machine.json"),
        )
    ).expanduser().resolve()
    if not path.is_file():
        return ""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read MORAI local machine config: %s" % error)
    if not isinstance(value, dict):
        raise ValueError("MORAI local machine config must contain one JSON object")
    return str(value.get("bridge_ip", "")).strip()


def _valid_wsl_ip(value):
    try:
        _normalise_ipv4(value, allow_loopback=False)
    except ValueError:
        return False
    return True


def wsl_ip():
    """Return the current WSL IPv4 used by ROS nodes inside WSL."""
    explicit = os.environ.get("MORAI_ROS_IP", "").strip()
    if explicit:
        return _normalise_ipv4(explicit, allow_loopback=False)

    commands = (["ip", "-4", "route", "get", "1.1.1.1"], ["hostname", "-I"])
    for command in commands:
        try:
            output = subprocess.check_output(
                command, text=True, stderr=subprocess.DEVNULL, timeout=3
            )
        except (OSError, subprocess.SubprocessError):
            continue
        candidates = re.findall(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])", output)
        if command[1:3] == ["-4", "route"] and "src" in output:
            match = re.search(r"\bsrc\s+((?:[0-9]{1,3}\.){3}[0-9]{1,3})", output)
            candidates = [match.group(1)] if match else candidates
        for value in candidates:
            if _valid_wsl_ip(value):
                return _normalise_ipv4(value, allow_loopback=False)

    try:
        value = socket.gethostbyname(socket.gethostname())
    except OSError:
        value = ""
    if _valid_wsl_ip(value):
        return _normalise_ipv4(value, allow_loopback=False)
    raise RuntimeError(
        "Cannot determine the WSL ROS IPv4; set MORAI_ROS_IP explicitly"
    )


def bridge_ip():
    """Return the stable address that Windows MORAI uses for rosbridge."""
    value = os.environ.get("MORAI_BRIDGE_IP", "").strip()
    if not value:
        value = _local_machine_bridge_ip() or DEFAULT_MORAI_BRIDGE_IP
    return validate_bridge_ip(value)


def ros_master_uri():
    """ROS master is local to WSL unless the operator explicitly overrides it."""
    return os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311").strip()


def runtime_environment():
    bridge_address = bridge_ip()
    ros_address = wsl_ip()
    return {
        "MORAI_BRIDGE_IP": bridge_address,
        "ROS_IP": ros_address,
        "ROS_MASTER_URI": ros_master_uri(),
    }
