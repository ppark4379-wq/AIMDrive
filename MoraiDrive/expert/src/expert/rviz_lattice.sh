#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rviz -d "$SCRIPT_DIR/rviz/lattice_planner.rviz"
