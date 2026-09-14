#!/usr/bin/env python3
"""Prepare one ignored, workstation-local Sensor Record configuration."""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(
    os.environ.get("MORAI_PROJECT_ROOT", str(ROOT.parents[1]))
).expanduser().resolve()
sys.path.insert(0, str(ROOT))


def load_json(path):
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Local machine config must contain one JSON object")
    return value


def first_existing(candidates, marker=None):
    resolved = []
    for candidate in candidates:
        candidate = Path(candidate).expanduser().resolve()
        resolved.append(candidate)
        target = candidate / marker if marker else candidate
        if target.is_file():
            return candidate
    if marker:
        for candidate in resolved:
            if (candidate / "src/CMakeLists.txt").is_file():
                return candidate
    return resolved[0]


def discover_launcher_root(explicit=None, configured=None):
    """Find MORAI Launcher without baking one workstation's drive into Git."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if (
            configured_path
            / "MoraiLauncher_Win_Data/SaveFile/Sensor"
        ).is_dir():
            return configured_path
    candidates = sorted(Path("/mnt").glob("[a-zA-Z]/MoraiLauncher_Win"))
    conventional = Path("/mnt/c/MoraiLauncher_Win")
    if conventional not in candidates:
        candidates.append(conventional)
    for candidate in candidates:
        if (
            candidate
            / "MoraiLauncher_Win_Data/SaveFile/Sensor"
        ).is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def discover_catkin_workspace(project_root, explicit=None, configured=None):
    """Prefer a valid configured workspace, then the checkout's expert tree."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        if ((configured_path / "devel/setup.bash").is_file()
                or (configured_path / "src/CMakeLists.txt").is_file()):
            return configured_path
    checkout = (Path(project_root) / "expert").resolve()
    if ((checkout / "devel/setup.bash").is_file()
            or (checkout / "src/CMakeLists.txt").is_file()):
        return checkout
    return first_existing([Path.home() / "catkin_ws"], "devel/setup.bash")


def discover_sensor_set(launcher, explicit=None, configured=None):
    """Resolve the active AIM sensor set, preferring the pinned MORAI build."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("Configured sensor set is missing: " + str(path))
        return path
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
    root = launcher / "MoraiLauncher_Win_Data/SaveFile/Sensor"
    preferred = root / "26.R1.H3/aim_sensorset.json"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(root.glob("*/aim_sensorset.json"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "aim_sensorset.json을 찾지 못했습니다. MORAI에서 센서셋을 "
            "aim_sensorset 이름으로 저장하거나 --sensor-set을 지정하세요: "
            + str(root)
        )
    raise RuntimeError(
        "aim_sensorset.json이 여러 버전에 있습니다. --sensor-set으로 현재 "
        "파일을 지정하세요: " + ", ".join(map(str, candidates))
    )


def run_import_check(python, modules, environment):
    command = "import " + ",".join(modules)
    result = subprocess.run(
        [str(python), "-c", command],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return result.returncode == 0, result.stderr[-1000:]


def update_sensor_bridge(data, bridge_ip):
    expected = {
        "cameraList": (4, "cc"),
        "Lidar3DList": (1, "lc"),
        "GPSList": (1, "gc"),
        "IMUList": (1, "ic"),
    }
    changed = 0
    for key, (count, config_key) in expected.items():
        sensors = data.get(key)
        if not isinstance(sensors, list) or len(sensors) != count:
            raise ValueError(f"Unexpected {key} count: {len(sensors or [])} != {count}")
        for sensor in sensors:
            config = sensor.get(config_key)
            if not isinstance(config, dict):
                raise ValueError(f"Missing {config_key} in {key}")
            ros = config.get("rosConfig")
            if not isinstance(ros, dict):
                raise ValueError(f"Missing rosConfig in {key}")
            desired = {
                "RosBridgeIP": bridge_ip,
                "RosBridgePort": 9090,
                "RosBridgeServerUrl": f"ws://{bridge_ip}:9090",
            }
            for field, value in desired.items():
                if ros.get(field) != value:
                    ros[field] = value
                    changed += 1
    return changed


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def preserve_before_change(path, label):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(path.name + f".before_{label}_{stamp}")
    shutil.copy2(str(path), str(backup))
    return backup


def catkin_build_invocation(ros_root, workspace, environment=None):
    """Build with ROS' system Python even when the caller activated Conda."""
    build_environment = dict(
        os.environ if environment is None else environment
    )
    build_environment.pop("PYTHONHOME", None)
    build_environment.pop("PYTHONPATH", None)
    build_environment["PYTHONNOUSERSITE"] = "1"
    command = [
        "bash",
        "-c",
        (
            'source "$1"; exec catkin_make -C "$2" --force-cmake '
            '-DPYTHON_EXECUTABLE=/usr/bin/python3'
        ),
        "morai-catkin-build",
        str(Path(ros_root) / "setup.bash"),
        str(workspace),
    ]
    return command, build_environment


def probe_windows_interop(powershell_exe, environment=None):
    """Return whether WSL can execute Windows programs at all."""
    token = "morai-wsl-interop-ok"
    try:
        process = subprocess.run(
            [
                str(powershell_exe),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::Write('%s')" % token,
            ],
            env=dict(os.environ if environment is None else environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except OSError as error:
        return False, "%s: %s" % (type(error).__name__, error)
    stdout = process.stdout.decode("utf-8-sig", errors="replace")
    stderr = process.stderr.decode("utf-8-sig", errors="replace")
    if process.returncode or token not in stdout:
        return False, "exit=%d stdout=%r stderr=%r" % (
            process.returncode, stdout[-300:], stderr[-500:]
        )
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--launcher-root", type=Path)
    parser.add_argument("--catkin-workspace", type=Path)
    parser.add_argument("--ros-root", type=Path)
    parser.add_argument("--powershell-exe", type=Path)
    parser.add_argument("--analysis-python", type=Path)
    parser.add_argument("--sensor-set", type=Path)
    parser.add_argument(
        "--bridge-ip",
        help=(
            "Windows MORAI가 사용할 rosbridge IPv4. 기본 127.0.0.1은 "
            "WSL 재시작 뒤에도 유지됩니다."
        ),
    )
    parser.add_argument("--patch-sensor-set", action="store_true")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full machine report")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / ".runtime/data_collection/local_machine.json"
    existing = load_json(config_path)
    runtime_root = PROJECT_ROOT / ".runtime/data_collection"
    dependencies = runtime_root / "internal_sensor_record_deps"

    def selected(argument, key, default):
        value = argument or existing.get(key) or default
        return Path(value).expanduser().resolve()

    storage_default = (
        PROJECT_ROOT / "dataset"
        if re.match(r"^/mnt/[A-Za-z](?:/|$)", PROJECT_ROOT.as_posix())
        else Path("/mnt/c/MoraiDriveData")
    )
    catkin_workspace = discover_catkin_workspace(
        PROJECT_ROOT,
        explicit=args.catkin_workspace,
        configured=existing.get("catkin_workspace"),
    )
    launcher = discover_launcher_root(
        args.launcher_root, existing.get("launcher_root")
    )
    from network_config import validate_bridge_ip

    selected_bridge_ip = validate_bridge_ip(
        args.bridge_ip
        or os.environ.get("MORAI_BRIDGE_IP")
        or existing.get("bridge_ip")
        or "127.0.0.1"
    )
    config = {
        "bridge_ip": selected_bridge_ip,
        "storage_root": str(selected(args.storage_root, "storage_root", storage_default)),
        "launcher_root": str(launcher),
        "catkin_workspace": str(catkin_workspace),
        "ros_root": str(selected(args.ros_root, "ros_root", "/opt/ros/noetic")),
        "powershell_exe": str(selected(
            args.powershell_exe,
            "powershell_exe",
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        )),
        "async_dependencies": str(dependencies.resolve()),
    }
    configured_sensor_set = args.sensor_set or existing.get("sensor_set")
    if configured_sensor_set:
        config["sensor_set"] = str(
            Path(configured_sensor_set).expanduser().resolve()
        )

    required_files = {
        "ROS setup": Path(config["ros_root"]) / "setup.bash",
        "PowerShell": Path(config["powershell_exe"]),
    }
    missing = [f"{label}: {path}" for label, path in required_files.items() if not path.is_file()]
    if not launcher.is_dir():
        missing.append("MORAI Launcher root: " + str(launcher))
    storage = Path(config["storage_root"])
    if not re.match(r"^/mnt/[A-Za-z](?:/|$)", storage.as_posix()):
        missing.append("Windows-backed storage path required: " + str(storage))
    elif not Path(*storage.parts[:3]).is_dir():
        missing.append("Windows drive is not mounted: " + str(Path(*storage.parts[:3])))
    if missing:
        raise RuntimeError("Machine paths are not ready:\n- " + "\n- ".join(missing))

    interop_ok, interop_detail = probe_windows_interop(config["powershell_exe"])
    if not interop_ok:
        raise RuntimeError(
            "WSL_INTEROP_UNAVAILABLE: WSL cannot execute powershell.exe. "
            "Close WSL/VS Code, run `wsl --shutdown` in Windows PowerShell "
            "or Command Prompt, then reopen WSL and retry 01. If it persists, "
            "ensure /etc/wsl.conf does not set [interop] enabled=false. "
            "detail=" + interop_detail
        )

    sensor_data_root = (
        launcher / "MoraiLauncher_Win_Data/SaveFile/SensorData"
    )
    sensor_data_root.mkdir(parents=True, exist_ok=True)
    created_sensor_directories = []
    for name in (
        "CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4",
        "LIDAR_5", "GPS_6", "IMU_7",
    ):
        path = sensor_data_root / name
        if path.exists():
            if not path.is_dir():
                raise RuntimeError("SensorData entry is not a directory: " + str(path))
            continue
        path.mkdir()
        created_sensor_directories.append(str(path))

    catkin_workspace = Path(config["catkin_workspace"])
    catkin_setup = catkin_workspace / "devel/setup.bash"
    if not catkin_setup.is_file():
        catkin_source = catkin_workspace / "src/CMakeLists.txt"
        if not catkin_source.is_file():
            raise RuntimeError(
                "catkin workspace is neither built nor buildable: "
                + str(catkin_workspace)
            )
        print("[SETUP] building bundled catkin workspace: " + str(catkin_workspace), flush=True)
        command, build_environment = catkin_build_invocation(
            config["ros_root"], catkin_workspace
        )
        system_python = subprocess.run(
            ["/usr/bin/python3", "-c", "import em,catkin_pkg"],
            env=build_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if system_python.returncode:
            raise RuntimeError(
                "ROS system Python is missing em/catkin_pkg. Install "
                "python3-empy and python3-catkin-pkg before retrying: "
                + system_python.stderr[-1000:]
            )
        subprocess.run(
            command,
            env=build_environment,
            check=True,
        )
    if not catkin_setup.is_file():
        raise RuntimeError("catkin build did not create setup: " + str(catkin_setup))
    storage.mkdir(parents=True, exist_ok=True)
    # A directory permission bit alone is insufficient on DrvFS.  Exercise an
    # exclusive create/remove in the ignored spool so failures are reported by
    # setup rather than after MORAI has already reset the scenario.
    storage_probe = storage / (".morai_write_probe_" + str(os.getpid()))
    try:
        with storage_probe.open("x", encoding="ascii") as stream:
            stream.write("sensor-record-local-storage-probe\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        try:
            storage_probe.unlink()
        except FileNotFoundError:
            pass
    dependencies.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment.update(
        PYTHONNOUSERSITE="1",
        PYTHONPATH=":".join([
            str(dependencies),
            str(ROOT),
            str(PROJECT_ROOT),
            str(Path(config["catkin_workspace"]) / "devel/lib/python3/dist-packages"),
            str(Path(config["ros_root"]) / "lib/python3/dist-packages"),
        ]),
    )
    capture_modules = ("grpc", "google.protobuf", "psutil", "yaml", "numpy", "rospy", "morai_msgs")
    capture_ok, capture_error = run_import_check("/usr/bin/python3", capture_modules, environment)

    analysis_candidates = []
    if args.analysis_python:
        analysis_candidates.append(args.analysis_python)
    elif existing.get("analysis_python"):
        analysis_candidates.append(Path(existing["analysis_python"]))
    analysis_candidates.extend([
        Path.home() / "miniconda3/envs/lmdrive/bin/python",
        Path.home() / "miniconda3/envs/lmdrive/bin/python3.8",
        Path("/usr/bin/python3"),
    ])
    analysis_modules = ("cv2", "numpy", "pyproj", "scipy", "PIL")

    def choose_analysis():
        failures = []
        for candidate in analysis_candidates:
            candidate = Path(candidate).expanduser().resolve()
            if not candidate.is_file():
                continue
            ok, error = run_import_check(candidate, analysis_modules, environment)
            if ok:
                return candidate, failures
            failures.append(str(candidate) + ": " + error)
        return None, failures

    analysis_python, analysis_failures = choose_analysis()
    if (not capture_ok or analysis_python is None) and not args.no_install:
        requirement = ROOT / "requirements-runtime.txt"
        print("[SETUP] installing isolated runtime dependencies: " + str(dependencies), flush=True)
        subprocess.run(
            [
                "/usr/bin/python3", "-m", "pip", "install",
                "--disable-pip-version-check", "--upgrade", "--target",
                str(dependencies), "-r", str(requirement),
            ],
            check=True,
        )
        capture_ok, capture_error = run_import_check(
            "/usr/bin/python3", capture_modules, environment
        )
        analysis_python, analysis_failures = choose_analysis()
    if not capture_ok:
        raise RuntimeError("Capture Python imports failed: " + capture_error)
    if analysis_python is None:
        raise RuntimeError("Analysis Python imports failed:\n" + "\n".join(analysis_failures))
    config["analysis_python"] = str(analysis_python)

    backup = None
    if existing and existing != config:
        backup = preserve_before_change(config_path, "machine_setup")
    atomic_write_json(config_path, config)

    sensor_report = {
        "requested": bool(args.patch_sensor_set),
        "bridge_ip": config["bridge_ip"],
        "changed_fields": 0,
    }
    if args.patch_sensor_set:
        from async_capture.internal_sensor_record_contract import validate_sensor_set

        address = config["bridge_ip"]
        sensor_path = discover_sensor_set(
            launcher,
            explicit=args.sensor_set,
            configured=existing.get("sensor_set"),
        )
        data = json.loads(sensor_path.read_text(encoding="utf-8-sig"))
        changed = update_sensor_bridge(data, address)
        validation = validate_sensor_set(data, expected_bridge_ip=address)
        if validation["failures"]:
            raise RuntimeError("Sensor-set validation failed: " + str(validation["failures"]))
        sensor_backup = None
        if changed:
            sensor_backup = preserve_before_change(sensor_path, "bridge_ip")
            atomic_write_json(sensor_path, data)
        sensor_report.update(
            bridge_ip=address,
            path=str(sensor_path),
            changed_fields=changed,
            backup=str(sensor_backup) if sensor_backup else None,
        )
        config["sensor_set"] = str(sensor_path)
        # Persist the discovered file only after it has passed the immutable
        # contract. The local config is Git-ignored and workstation-specific.
        atomic_write_json(config_path, config)

    report = {
        "ready": True,
        "config": str(config_path),
        "config_backup": str(backup) if backup else None,
        "settings": config,
        "capture_python": "/usr/bin/python3",
        "analysis_python": str(analysis_python),
        "sensor_set": sensor_report,
        "created_sensor_directories": created_sensor_directories,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("[PASS] PC별 실행 환경 준비 완료")
        print("       bridge IP : " + config["bridge_ip"] + " (Windows 고정 연결 주소)")
        print("       저장 경로 : " + config["storage_root"])
        print("       catkin    : " + config["catkin_workspace"])
        print("       분석 Python: " + str(analysis_python))
        if created_sensor_directories:
            print("       SensorData : 누락 폴더 %d개 생성" % len(created_sensor_directories))
        if args.patch_sensor_set:
            action = (
                "%d개 주소 필드 갱신" % sensor_report["changed_fields"]
                if sensor_report["changed_fields"]
                else "이미 현재 주소와 일치"
            )
            print("       센서셋    : %s (%s)" % (sensor_report["path"], action))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("[FAIL] PC별 실행 환경 자동 준비 실패: " + str(error), file=sys.stderr)
        if os.environ.get("MORAI_SETUP_DEBUG") == "1":
            raise
        raise SystemExit(2)
