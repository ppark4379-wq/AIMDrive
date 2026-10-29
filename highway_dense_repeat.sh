#!/bin/sh
# Run with: sh highway_dense_repeat.sh [--bus-only | --truck-only]
repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
cd "$repo" || exit 1

traffic_label="22 NPCs: 8 buses (2 merge-adjacent), 2 trucks, 12 passenger vehicles; 5 approach NPCs moved downstream; 20-100 km/h"
if [ "$#" -gt 1 ]; then
  echo "Choose only one: --bus-only or --truck-only" >&2
  exit 2
fi
case "${1-}" in
  "") set -- --highway-bus-heavy ;;
  --bus-only)
    traffic_label="20 Universe buses"
    set -- --highway-npc-model 2016_Hyundai_Universe ;;
  --truck-only)
    traffic_label="20 Xcient P520 trucks"
    set -- --highway-npc-model '2013_Hyundai_Xcient(P520)' ;;
  --help|-h)
    echo "Usage: sh highway_dense_repeat.sh [--bus-only | --truck-only]"
    echo "Default: 22 NPCs = 8 Universe buses + 2 Xcient trucks + 12 passenger vehicles; 20-100 km/h."
    echo "Exactly 2 merge-adjacent buses; 5 northern-approach NPCs relocated to the southbound mainline."
    exit 0 ;;
  *)
    echo "Unknown option: $1 (use --help)" >&2
    exit 2 ;;
esac

trap 'exit 130' INT
trap 'exit 143' TERM

while true; do
  echo "[Highway repeat] Dense traffic: $traffic_label (Ctrl+C to stop)"
  # Keep the default position-history speed estimator for planning.
  bash "$repo/AIM2026/Expert/run_highway.sh" --highway-density dense --lane-motion-tracking "$@"
  result=$?
  case "$result" in
    130|143) exit "$result" ;;
    0|20) ;; # Completed or collision: reports saved; continue with a new run.
    *) echo "[Highway repeat] Scenario failed (exit=$result); stopping"; exit "$result" ;;
  esac
  echo "[Highway repeat] exit=$result; restarting in 1 second"
  sleep 1
done
