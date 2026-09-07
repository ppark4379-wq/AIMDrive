#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/train.sh /path/to/job [additional training arguments]
DATASET_ROOT="${1:?Usage: bash scripts/train.sh DATASET_ROOT [training arguments]}"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
python -m torch.distributed.run --standalone --nproc_per_node="${GPU_NUM:-1}" \
    train_pretrain.py "$DATASET_ROOT" --dataset morai \
    --model memfuser_baseline_e1d3 --sched cosine --epochs 25 \
    --warmup-epochs 5 --lr 0.00075 --batch-size 4 -j 4 \
    --no-prefetcher --eval-metric l1_error --saver-decreasing \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05 \
    --clip-grad 5 --freeze-num -1 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --smoothed_l1 --experiment morai_memfuser "$@"
