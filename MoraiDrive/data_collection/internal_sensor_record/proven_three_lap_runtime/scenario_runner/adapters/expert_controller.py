import csv
import json
import os
import signal
import subprocess
import sys


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
        max_velocity_kmh=60.0,
        traffic_light_control=False,
        python_executable=None,
        ros_remaps=None,
        is_closed_path=False,
        velocity_profile_window_size=None,
    ):
        self.expert_root = os.path.abspath(expert_root)
        self.runtime_root = os.path.abspath(runtime_root)
        self.map_name = str(map_name)
        self.mgeo_root = os.path.abspath(mgeo_root)
        self.route_points = route_points
        self.start_link = start_link
        self.end_link = end_link
        self.max_velocity_kmh = float(max_velocity_kmh)
        self.traffic_light_control = bool(traffic_light_control)
        self.python_executable = python_executable or sys.executable
        self.ros_remaps = list(ros_remaps or [])
        self.is_closed_path = bool(is_closed_path)
        self.velocity_profile_window_size = velocity_profile_window_size
        self.process = None

        self._validate_expert()
        self._prepare_runtime_inputs()

    def _validate_expert(self):
        self.expert_source = os.path.join(self.expert_root, "src")
        self.base_config_path = os.path.join(
            self.expert_source,
            "expert",
            "src",
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

        with open(self.base_config_path, "r", encoding="utf-8") as source:
            config = json.load(source)

        config["map"]["name"] = self.map_name
        config["map"]["data_dir"] = self.mgeo_root
        config["map"]["use_mgeo_path"] = False
        config["map"]["traffic_light_control"] = self.traffic_light_control
        config["map"]["is_closed_path"] = self.is_closed_path
        velocity_profile = config["planning"]["velocity_profile"]
        velocity_profile["max_velocity"] = self.max_velocity_kmh
        if self.velocity_profile_window_size is not None:
            velocity_profile["window_size"] = int(
                self.velocity_profile_window_size
            )

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

        if self.process.poll() is None:
            print(f"[Expert] stopping process: pid={self.process.pid}")
            self._terminate_process_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(signal.SIGKILL)
                self.process.wait(timeout=3.0)
        self.process = None

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
