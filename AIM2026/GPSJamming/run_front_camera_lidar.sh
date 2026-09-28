#!/bin/sh
# Git Bash / WSL launcher for the existing Windows CUDA environment.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
launcher="$script_dir/run_front_camera_lidar.ps1"
if command -v cygpath >/dev/null 2>&1; then
    launcher=$(cygpath -w "$launcher")
elif command -v wslpath >/dev/null 2>&1; then
    launcher=$(wslpath -w "$launcher")
else
    echo 'Run this launcher in Git Bash or WSL on the MORAI Windows PC.' >&2
    exit 1
fi
echo 'Front camera + LiDAR: MOT, timestamp association, BEV position and relative velocity'
if command -v powershell.exe >/dev/null 2>&1; then
    exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$launcher"
fi
exec /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
    -NoProfile -ExecutionPolicy Bypass -File "$launcher"
