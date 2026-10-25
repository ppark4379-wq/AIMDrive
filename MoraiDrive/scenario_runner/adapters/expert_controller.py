import csv
import json
import hashlib
import math
import os
import signal
import subprocess
import sys
import time
from ..utils import safe_print as print


def validate_max_velocity_kmh(value):
    """Validate an explicit speed without introducing a tunnel-specific cap."""
    if isinstance(value, bool):
        raise ValueError("Expert maximum velocity must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Expert maximum velocity must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Expert maximum velocity must be finite and positive")
    return result


def _speed_contract(path, raw, requested):
    config = json.loads(raw)
    original = config["planning"]["velocity_profile"]["max_velocity"]
    resolved = validate_max_velocity_kmh(original if requested is None else requested)
    return {"schema": "aimdrive_expert_speed_source_v1", "inherited": requested is None,
            "requested_max_velocity_kmh": None if requested is None else resolved,
            "resolved_max_velocity_kmh": resolved, "expert_config_path": os.path.abspath(path),
            "expert_config_sha256": hashlib.sha256(raw).hexdigest()}


def expert_max_velocity_contract(expert_root, requested=None):
    """Read the original Expert speed and its source pin for this invocation."""
    path = os.path.join(os.path.abspath(expert_root), "src", "config", "config.json")
    with open(path, "rb") as source:
        return _speed_contract(path, source.read(), requested)


def resolve_expert_max_velocity(expert_root, requested=None):
    return expert_max_velocity_contract(expert_root, requested)["resolved_max_velocity_kmh"]


class ExpertController:
    """Export runtime inputs and launch an external expert without modifying it."""

    def __init__(
        self,
        expert_root,
        runtime_root,
        map_name,
        mgeo_root,
        route_points,
        start_link=None,
        end_link=None,
        max_velocity_kmh=None,
        traffic_light_control=False,
        python_executable=None,
        ros_remaps=None,
        is_closed_path=False,
        velocity_profile_window_size=None,
        route_signal_route_links=None,
        speed_cap_kmh=None,
        uncapped_speed_zones=(),
    ):
        self.expert_root = os.path.abspath(expert_root)
        self.runtime_root = os.path.abspath(runtime_root)
        self.map_name = str(map_name)
        self.mgeo_root = os.path.abspath(mgeo_root)
        self.route_points = route_points
        self.start_link = start_link
        self.end_link = end_link
        self.max_velocity_kmh = None if max_velocity_kmh is None else validate_max_velocity_kmh(max_velocity_kmh)
        self.requested_max_velocity_kmh = self.max_velocity_kmh
        self.speed_cap_kmh = None if speed_cap_kmh is None else validate_max_velocity_kmh(speed_cap_kmh)
        self.uncapped_speed_zones = frozenset(uncapped_speed_zones or ())
        self.traffic_light_control = bool(traffic_light_control)
        self.python_executable = python_executable or sys.executable
        self.ros_remaps = list(ros_remaps or [])
        self.is_closed_path = bool(is_closed_path)
        self.velocity_profile_window_size = velocity_profile_window_size
        if route_signal_route_links is not None and (
            not isinstance(route_signal_route_links, (list, tuple))
            or not route_signal_route_links
            or any(not isinstance(link, str) or not link.strip()
                   for link in route_signal_route_links)
        ):
            raise ValueError("Signal route links must be a nonempty ordered list of IDs")
        self.route_signal_route_links = (None if route_signal_route_links is None
                                         else list(route_signal_route_links))
        self.process = None

        self._validate_expert()
        self._prepare_runtime_inputs()

    def _validate_expert(self):
        self.expert_source = os.path.join(self.expert_root, "src")
        self.base_config_path = os.path.join(
            self.expert_source,
            "config",
            "config.json",
        )
        if not os.path.isfile(self.base_config_path):
            raise RuntimeError(
                "External AIMDrive expert not found. Expected read-only config: "
                f"{self.base_config_path}"
            )

        self.entrypoint = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "expert_process.py",
        )
        if not os.path.isfile(self.entrypoint):
            raise RuntimeError(
                f"Scenario-owned expert entrypoint is missing: {self.entrypoint}"
            )

    def _prepare_runtime_inputs(self):
        os.makedirs(self.runtime_root, exist_ok=True)

        self.runtime_path_csv = os.path.join(self.runtime_root, "path.csv")
        with open(self.runtime_path_csv, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["x", "y"])
            for point in self.route_points:
                writer.writerow([float(point[0]), float(point[1])])

        self.runtime_path_txt = os.path.join(
            self.runtime_root,
            "path_scenario_runner_current.txt",
        )
        with open(self.runtime_path_txt, "w", encoding="utf-8") as output:
            output.write("# scenario_runner current route\n")
            if self.start_link is not None or self.end_link is not None:
                output.write(f"# {self.start_link} -> {self.end_link}\n")
            output.write("# x y\n")
            for point in self.route_points:
                output.write(
                    f"{float(point[0]):.6f} {float(point[1]):.6f}\n"
                )

        with open(self.base_config_path, "rb") as source:
            source_bytes = source.read()
        config = json.loads(source_bytes)
        self.speed_contract = _speed_contract(self.base_config_path, source_bytes, self.requested_max_velocity_kmh)
        self.max_velocity_kmh = self.speed_contract["resolved_max_velocity_kmh"]

        config["map"]["name"] = self.map_name
        config["map"]["data_dir"] = self.mgeo_root
        config["map"]["use_mgeo_path"] = False
        config["map"]["traffic_light_control"] = self.traffic_light_control
        config["map"]["is_closed_path"] = self.is_closed_path
        # Bind signal permissions to this invocation, not a stale base config.
        config["map"].pop("route_signal_route_links", None)
        config["map"].pop("route_signal_targets", None)
        if self.route_signal_route_links is not None:
            config["map"]["route_signal_route_links"] = self.route_signal_route_links
        velocity_profile = config["planning"]["velocity_profile"]
        if self.requested_max_velocity_kmh is not None:
            velocity_profile["max_velocity"] = self.max_velocity_kmh
        if self.velocity_profile_window_size is not None:
            velocity_profile["window_size"] = int(
                self.velocity_profile_window_size
            )

        if self.speed_cap_kmh is not None:
            velocity_profile["max_velocity"] = min(
                validate_max_velocity_kmh(velocity_profile["max_velocity"]), self.speed_cap_kmh)
            for zone in velocity_profile.get("speed_limit_zones", []):
                if "max_velocity" in zone and zone.get("name") not in self.uncapped_speed_zones:
                    zone["max_velocity"] = min(
                        validate_max_velocity_kmh(zone["max_velocity"]), self.speed_cap_kmh)
            self.speed_contract["runtime_speed_cap_kmh"] = self.speed_cap_kmh
            print(f"[Expert] 이번 실행 목표속도 상한: {self.speed_cap_kmh:g}km/h (구간별 제한 포함)")

        self.runtime_config_path = os.path.join(
            self.runtime_root,
            "config.json",
        )
        with open(self.runtime_config_path, "w", encoding="utf-8") as output:
            json.dump(config, output, indent=2)
            output.write("\n")

        print(f"[Expert] runtime config: {self.runtime_config_path}")
        print(
            f"[Expert] runtime route: {self.runtime_path_csv}, "
            f"points={len(self.route_points)}"
        )

    def start_process(self):
        if self.process is not None and self.process.poll() is None:
            return

        environment = os.environ.copy()
        environment["MORAI_EXPERT_ROOT"] = self.expert_root
        environment["GT_BEV_CONFIG_PATH"] = self.runtime_config_path
        environment["GT_BEV_PATH_CSV"] = self.runtime_path_csv
        environment["GT_BEV_MAP_DIR"] = self.mgeo_root
        environment["GT_BEV_DIRECT_USE_ENV_CONFIG"] = "1"
        environment["SCENARIO_ROUTE_PATH_TXT"] = self.runtime_path_txt

        command = [
            self.python_executable,
            self.entrypoint,
            *self.ros_remaps,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=os.path.dirname(self.entrypoint),
            env=environment,
            start_new_session=True,
        )
        print(
            f"[Expert] process started: pid={self.process.pid}, "
            f"cmd={' '.join(command)}"
        )

    def stop_process(self):
        if self.process is None:
            return

        process_pid = self.process.pid
        if self.process.poll() is None:
            print(f"[Expert] stopping process: pid={process_pid}")
            self._terminate_process_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(signal.SIGKILL)
                self.process.wait(timeout=5.0)
        self._clear_owned_ros_registrations(process_pid)
        self.process = None

    @staticmethod
    def _owned_ros_nodes(system_state, process_pid):
        prefix = f"/morai_standard_{int(process_pid)}_"
        nodes = set()
        for registrations in system_state:
            for _, names in registrations:
                nodes.update(name for name in names if name.startswith(prefix))
        return sorted(nodes)

    def _clear_owned_ros_registrations(self, process_pid):
        """Remove only stale registrations owned by the stopped expert PID."""
        import rosgraph
        import rosnode

        master = rosgraph.Master('/scenario_runner_expert_cleanup')
        deadline = time.monotonic() + 5.0
        while True:
            owned = self._owned_ros_nodes(master.getSystemState(), process_pid)
            if not owned:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        rosnode.cleanup_master_blacklist(master, owned)
        remaining = self._owned_ros_nodes(master.getSystemState(), process_pid)
        if remaining:
            raise RuntimeError(
                "Stopped expert still owns ROS registrations: " + str(remaining)
            )

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def _terminate_process_group(self, process_signal):
        try:
            os.killpg(os.getpgid(self.process.pid), process_signal)
        except ProcessLookupError:
            pass
        except Exception as error:
            print(f"[Expert] process-group stop fallback: {error}")
            if process_signal == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()
