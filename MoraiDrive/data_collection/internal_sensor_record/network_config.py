"""Resolve machine-local ROS/rosbridge addresses without repository pinning."""
import ipaddress
import os
import re
import socket
import subprocess


def _valid_bridge_ip(value):
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return False
    return address.version == 4 and not address.is_loopback and not address.is_unspecified


def bridge_ip():
    """Return the WSL IPv4 address that Windows MORAI must connect to."""
    # MORAI_BRIDGE_IP is the only explicit override. ROS_IP may be stale after
    # moving a checkout or restarting WSL, so it is derived again every run.
    for name in ("MORAI_BRIDGE_IP",):
        value = os.environ.get(name, "").strip()
        if value:
            if not _valid_bridge_ip(value):
                raise ValueError(name + " must be one non-loopback IPv4 address")
            return value

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
            if _valid_bridge_ip(value):
                return value

    try:
        value = socket.gethostbyname(socket.gethostname())
    except OSError:
        value = ""
    if _valid_bridge_ip(value):
        return value
    raise RuntimeError(
        "Cannot determine the WSL bridge IPv4; set MORAI_BRIDGE_IP explicitly"
    )


def ros_master_uri():
    """ROS master is local to WSL unless the operator explicitly overrides it."""
    return os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311").strip()


def runtime_environment():
    address = bridge_ip()
    return {
        "MORAI_BRIDGE_IP": address,
        "ROS_IP": address,
        "ROS_MASTER_URI": ros_master_uri(),
    }
