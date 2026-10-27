#!/usr/bin/env bash
# Shared Python launch helpers and ROS setup for the legacy VE tools.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../tools/project_env.sh"

# Metadata-only launchers can load these functions before their --check branch.
scenario_run_legacy_ros_python() {
    morai_load_internal_sensor_record_environment
    scenario_export_project_paths
    exec /usr/bin/python3 -B "$@"
}

# Capture launchers share environment construction and interpreter flags.
# Arguments: working directory (empty keeps cwd), then Python script or -m argv.
# Historical relative paths use the same relocation map as the runner.
scenario_run_legacy_capture_python() {
    exec /usr/bin/python3 -B - "$@" <<'PY'
import os,sys
from scenario_runner.project_paths import AIM_ROOT, REPO_ROOT, PACKAGE_ROOT, PACKAGE_PARENT, project_path
from scenario_runner.utils.environment import build_capture_environment
working_directory, *command = sys.argv[1:]
if command[0] != '-m' and not os.path.isabs(command[0]):
    command[0] = str(project_path(command[0]))
env = build_capture_environment()
if working_directory:
    os.chdir(str(project_path(working_directory)))
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B', *command], env)
PY
}

scenario_load_legacy_ve_ros_environment() {
    task_tools="$(cd "$1" && pwd)"
    task_repo="$(cd "$task_tools/../../.." && pwd)"
    morai_load_internal_sensor_record_environment
    export PYTHONPATH="$SCENARIO_AIM_ROOT/Expert/ros/devel/lib/python3/dist-packages:${PYTHONPATH:-}"
    export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
}

# Keep sourcing this file as the historical ROS setup operation. Python
# launchers use a command-scoped flag to load only their shared helpers.
if [[ "${SCENARIO_LEGACY_FUNCTIONS_ONLY:-}" != 1 ]]; then
    scenario_load_legacy_ve_ros_environment "$(dirname "${BASH_SOURCE[0]}")"
fi
