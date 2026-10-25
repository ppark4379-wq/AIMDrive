#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTICE_SCRIPT="$SCRIPT_DIR/../../../VLDrive/tools/rviz_notice_zones.py"
NOTICE_ZONES="$SCRIPT_DIR/../../../VLDrive/configs/notice_zones/kcity.json"
if [[ -f "$NOTICE_ZONES" ]]; then
    "${NOTICE_PYTHON:-python3}" "$NOTICE_SCRIPT" --zones "$NOTICE_ZONES" &
    NOTICE_PID=$!
    trap 'kill "$NOTICE_PID" 2>/dev/null || true' EXIT
else
    echo "Notice zones are not generated yet. Confirm boundary coordinates and run VLDrive/tools/build_notice_instructions.py --points VLDrive/configs/notice_zones/kcity_points.json" >&2
fi
rviz -d "$SCRIPT_DIR/rviz/lattice_planner.rviz"
