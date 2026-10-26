#!/usr/bin/env bash
# Current local UDP setup; object relay must be running on WSL port 17505.
set -euo pipefail
highway_tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MORAI_GRPC_TARGET="${MORAI_GRPC_TARGET:-127.0.0.1:7789}"
exec bash "$highway_tools/run_gt_udp_full_route.sh" \
  --scenario full_loop_highway --repeat \
  --sim-ip 127.0.0.1 --ego-port 9099 --object-port 17505 \
  --control-port 9093 --control-source-port 9094 "$@"
