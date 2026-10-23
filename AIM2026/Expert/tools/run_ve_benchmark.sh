#!/usr/bin/env bash
set -euo pipefail
benchmark_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$benchmark_repo:$benchmark_repo/MoraiDrive:$benchmark_repo/VLDrive:$benchmark_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
# Use exactly the ROS/native dependency environment used by the existing runner.
exec /usr/bin/python3 -B - "$benchmark_repo" "$@" <<'PY'
import os,sys
from python_environment import build_capture_environment
repo=sys.argv[1]
env=build_capture_environment()
env['PYTHONPATH']=os.pathsep.join([repo,repo+'/MoraiDrive',repo+'/VLDrive',repo+'/MoraiDrive/data_collection/internal_sensor_record',env.get('PYTHONPATH','')])
os.chdir(repo+'/MoraiDrive')
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B','-m','scenario_runner.tools.ve_expert_benchmark',*sys.argv[2:]],env)
PY
