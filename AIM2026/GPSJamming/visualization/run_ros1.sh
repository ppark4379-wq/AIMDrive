#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
set -u
export ROS_HOME="$project_dir/artifacts/ros"
export ROS_LOG_DIR="$ROS_HOME/log"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_dir/artifacts/ros_python:${PYTHONPATH:-}"
# Small repeated LiDAR matrix operations must not fan out over every CPU.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
mkdir -p "$ROS_LOG_DIR"
case "${1:-node}" in
  audit) exec /usr/bin/python3 "$project_dir/tools/check_wall_topics.py" ;;
  test)
    cd "$project_dir"
    /usr/bin/python3 -m unittest discover -s tests -v
    ;;
  master) exec roscore ;;
  rviz) exec rviz -d "$project_dir/visualization/lidar.rviz" ;;
  node)
    if [[ $# -gt 0 ]]; then shift; fi
    exec /usr/bin/python3 "$project_dir/main.py" --ros "$@"
    ;;
  *) echo 'Usage: bash visualization/run_ros1.sh {master|node|rviz|test|audit} [ROS private parameters]' >&2; exit 2 ;;
esac
