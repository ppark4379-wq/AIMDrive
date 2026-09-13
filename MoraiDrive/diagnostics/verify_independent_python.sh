#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /opt/ros/noetic/setup.bash
source expert/devel/setup.bash
probe_python="$PWD/expert/.runtime/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3"
export PYTHONPATH="$PWD/expert/src:/usr/lib/python3/dist-packages:$HOME/.local/lib/python3.8/site-packages:${PYTHONPATH:-}"
mkdir -p diagnostics/.runtime
probe_log="diagnostics/.runtime/independent-python-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$probe_log") 2>&1
printf 'Log: %s\n' "$PWD/$probe_log"
"$probe_python" -s - <<'PY'
import sys, numpy, rospy, tf, grpc
from expert.src.autonomous_driving import AutonomousDriving
from expert.src.network.ros_manager import RosManager
print('Python:', sys.executable, sys.version)
print('NumPy:', numpy.__version__, numpy.__file__)
print('gRPC:', grpc.__file__)
print('IMPORT PASS; no ROS node or simulator connection started')
PY
for probe_round in 1 2 3 4 5; do
  echo "Round $probe_round"
  "$probe_python" -s -m unittest discover -s expert/src/expert/tests -p test_project_diagnostics.py -q
  "$probe_python" -s -m unittest discover -s expert/src/expert/tests -p test_highway_lane_planning.py -q
done
echo 'ALL CHECKS PASSED'
