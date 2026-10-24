#!/usr/bin/env bash
# WSL entrypoint: CUDA stays in Windows; the caller can remain in bash.
set -euo pipefail
task_tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_repo="$(cd "$task_tools/../../.." && pwd)"
task_args=()
while (($#)); do
    case "$1" in
        --help|-h)
            echo 'Usage: bash tools/run_ve_ros_inference.sh [--seconds N] [--checkpoint PATH] [--stop]'
            echo 'Uses Windows CUDA and the WSL ROS relay; no PowerShell installation in Linux is needed.'
            exit 0 ;;
        --stop)
            mkdir -p "$task_repo/artifacts/ve_live"
            printf 'stop\n' > "$task_repo/artifacts/ve_live/inference.stop"
            echo 'Requested VE inference shutdown. MORAI and rosbridge remain running.'
            exit 0 ;;
        --seconds)
            [[ $# -ge 2 ]] || { echo '--seconds needs a value' >&2; exit 2; }
            task_args+=('-Seconds' "$2"); shift 2 ;;
        --checkpoint)
            [[ $# -ge 2 ]] || { echo '--checkpoint needs a path' >&2; exit 2; }
            task_checkpoint="$2"
            if [[ "$task_checkpoint" == /* ]]; then
                task_checkpoint="$(wslpath -m "$task_checkpoint")"
            fi
            task_args+=('-Checkpoint' "$task_checkpoint"); shift 2 ;;
        *) echo "Unknown option: $1 (use --help)" >&2; exit 2 ;;
    esac
done
task_powershell='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
if [[ ! -f "$task_powershell" ]]; then
    echo 'Windows PowerShell executable is unavailable through WSL interop.' >&2
    exit 2
fi
exec "$task_powershell" -NoProfile -ExecutionPolicy Bypass \
    -File "$(wslpath -m "$task_tools/run_ve_ros_inference.ps1")" "${task_args[@]}"
