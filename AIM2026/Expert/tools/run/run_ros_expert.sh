#!/usr/bin/env bash
set -euo pipefail
expert_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${MORAI_ROS_PYTHON:-/usr/bin/python3}" -u -B "$expert_root/main.py" --transport ros "$@"
