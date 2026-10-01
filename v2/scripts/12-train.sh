#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

DATASET="${DATASET_DIR}"
OUTPUT="${TRAIN_OUTPUT_DIR}"
NUM_CLASSES="${TRAIN_NUM_CLASSES:-25}"
DEVICE="${TRAIN_DEVICE:-gpu:0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --output)
            OUTPUT="$2"; shift 2 ;;
        --num-classes)
            NUM_CLASSES="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --)
            shift
            EXTRA_OVERRIDES=("$@")
            break ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2 ;;
    esac
done

EXTRA_OVERRIDES=(${EXTRA_OVERRIDES[@]-})

section "PP-DocLayoutV3 training"

require_file "${TRAINING_PYTHON}"
require_file "${ROOT_DIR}/PaddleX/main.py"
require_file "${ROOT_DIR}/PaddleX/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml"
require_dir "${DATASET}"

cd "${ROOT_DIR}/PaddleX"

exec "${TRAINING_PYTHON}" main.py \
    -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir="${DATASET}" \
    -o Global.device="${DEVICE}" \
    -o Global.output="${OUTPUT}" \
    -o Train.num_classes="${NUM_CLASSES}" \
    "${EXTRA_OVERRIDES[@]}"
