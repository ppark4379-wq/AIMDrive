#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/prepare_dataset.sh DATASET_ROOT [split-builder arguments]
DATASET_ROOT="${1:?Usage: bash scripts/prepare_dataset.sh DATASET_ROOT [split-builder arguments]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

python3 "$REPO_ROOT/data_collection/tools/build_dataset_splits.py" "$DATASET_ROOT" "$@"
python3 "$SCRIPT_DIR/../tools/audit_dataset.py" "$DATASET_ROOT"
python3 "$REPO_ROOT/data_collection/tools/validate_lmdrive_batch.py" \
    "$DATASET_ROOT" --index-file dataset_index.txt --batch-size 2 --batches 1
python3 "$REPO_ROOT/data_collection/tools/validate_lmdrive_batch.py" \
    "$DATASET_ROOT" --index-file dataset_index_test.txt --batch-size 2 --batches 1

echo "Dataset is ready. Start training with:"
echo "  bash $SCRIPT_DIR/train.sh $DATASET_ROOT"
