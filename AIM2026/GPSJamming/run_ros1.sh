#!/usr/bin/env bash
set -eo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
  audit) exec /usr/bin/python3 "$project_dir/scripts/check_wall_topics.py" ;;
  test)
    cd "$project_dir"
    for test_file in test_ros1_cloud.py test_lidar.py test_intermediates.py test_walls.py test_wall_extent.py test_wall_curve.py test_wall_ema.py test_wall_width.py; do
      /usr/bin/python3 -m unittest discover -s tests -p "$test_file" -v
    done
    ;;
  master) exec roscore ;;
  rviz) exec rviz -d "$project_dir/configs/lidar_ros1.rviz" ;;
  node)
    if [[ $# -gt 0 ]]; then shift; fi
    exec /usr/bin/python3 "$project_dir/scripts/run_ros1.py" "$@"
    ;;
  *) echo 'Usage: bash run_ros1.sh {master|node|rviz|test|audit} [ROS private parameters]' >&2; exit 2 ;;
esac
