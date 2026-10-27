#!/usr/bin/env bash
# Lane-split avoidance test: run_static.sh with the extended static section and one obstacle on the split.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_static.sh" --avoid "$@"
