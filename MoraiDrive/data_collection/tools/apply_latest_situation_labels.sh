#!/usr/bin/env bash
# Materialize the reviewed GT and install ordinary native episode labels.
set -euo pipefail

mode="apply"
for argument in "$@"; do
  case "$argument" in
    --preflight-only) mode="preflight" ;;
    --resume) ;; # Both stages always resume verified results safely.
    -h|--help)
      cat <<'USAGE'
Usage: bash apply_latest_situation_labels.sh [--preflight-only] [--resume]

Writes final filter-applied labeled_frames.jsonl files to each native episode's
situation_labels folder. Existing labels are backed up. Sensor/GT bytes stay intact.
--preflight-only checks source packages without applying labels.
An interrupted run can be resumed with the same command.

Optional environment variables:
  SITUATION_DATASET_ROOT   Dataset directory (default: /mnt/d/MoraiDrive/dataset)
  SITUATION_LABEL_BASE     Sealed reviewed_v1 package
  SITUATION_VISUAL_PROFILE Sealed visual_profile_v2 package
  SITUATION_APPLIED_ROOT   New applied_v3 package directory
  SITUATION_PYTHON         Python 3.8+ executable (default: python3)
USAGE
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
  esac
done

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${SITUATION_DATASET_ROOT:-/mnt/d/MoraiDrive/dataset}"
ssd_root="$(dirname -- "$dataset_root")"
label_base="${SITUATION_LABEL_BASE:-$ssd_root/instruction_labels/situation_gt_20261009_reviewed_v1}"
visual_profile="${SITUATION_VISUAL_PROFILE:-$ssd_root/instruction_labels/situation_gt_20261009_visual_profile_v2}"
applied_root="${SITUATION_APPLIED_ROOT:-$ssd_root/instruction_labels/situation_gt_20261009_applied_v3}"
python_executable="${SITUATION_PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1

materialize=("$python_executable" "$script_directory/materialize_situation_labels.py"
  --base-package "$label_base" --visual-profile "$visual_profile"
  --dataset-root "$dataset_root" --output "$applied_root")
if [[ "$mode" == "preflight" ]]; then
  "${materialize[@]}" --preflight-only
  exit 0
fi

"${materialize[@]}" --resume
"$python_executable" "$script_directory/install_situation_labels.py" \
  --labels-root "$applied_root" --dataset-root "$dataset_root" --resume
