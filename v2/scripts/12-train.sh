#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
PADDLEX_ROOT="${PADDLEX_ROOT:-${ROOT_DIR}/PaddleX}"

TRAINING_PYTHON="${TRAINING_VENV}/bin/python"

CONFIG="${PADDLEX_ROOT}/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml"

# Defaults
DEFAULT_DATASET="/content/glm_finetune/datasets/validation/training"
DEFAULT_OUTPUT="/content/glm_finetune/models/pplayoutv3_2"
DEFAULT_NUM_CLASSES="25"
DEFAULT_BATCH_SIZE="1"
DEFAULT_EPOCHS="100"
DEFAULT_DEVICE="gpu:0"

if [[ ! -x "${TRAINING_PYTHON}" ]]; then
    echo "[ERROR] Training environment not found:"
    echo "        ${TRAINING_VENV}"
    echo
    echo "Run:"
    echo "  ./setup.sh install-training"
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] PP-DocLayoutV3 config not found:"
    echo "        ${CONFIG}"
    exit 1
fi

echo
echo "============================================"
echo "       PP-DocLayoutV3 Fine-tuning"
echo "============================================"
echo

read -rp \
    "Dataset directory [${DEFAULT_DATASET}]: " \
    DATASET

DATASET="${DATASET:-${DEFAULT_DATASET}}"

read -rp \
    "Output directory [${DEFAULT_OUTPUT}]: " \
    OUTPUT

OUTPUT="${OUTPUT:-${DEFAULT_OUTPUT}}"

read -rp \
    "Number of classes [${DEFAULT_NUM_CLASSES}]: " \
    NUM_CLASSES

NUM_CLASSES="${NUM_CLASSES:-${DEFAULT_NUM_CLASSES}}"

read -rp \
    "Batch size [${DEFAULT_BATCH_SIZE}]: " \
    BATCH_SIZE

BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"

read -rp \
    "Epochs [${DEFAULT_EPOCHS}]: " \
    EPOCHS

EPOCHS="${EPOCHS:-${DEFAULT_EPOCHS}}"

read -rp \
    "Device [${DEFAULT_DEVICE}]: " \
    DEVICE

DEVICE="${DEVICE:-${DEFAULT_DEVICE}}"

echo
echo "============================================"
echo "Training configuration"
echo "============================================"
echo "Dataset       : ${DATASET}"
echo "Output        : ${OUTPUT}"
echo "Classes       : ${NUM_CLASSES}"
echo "Batch size    : ${BATCH_SIZE}"
echo "Epochs        : ${EPOCHS}"
echo "Device        : ${DEVICE}"
echo "Config        : ${CONFIG}"
echo "============================================"
echo

if [[ ! -d "${DATASET}" ]]; then
    echo "[ERROR] Dataset directory does not exist:"
    echo "        ${DATASET}"
    exit 1
fi

mkdir -p "${OUTPUT}"

cd "${PADDLEX_ROOT}"

exec "${TRAINING_PYTHON}" main.py \
    -c "${CONFIG}" \
    -o Global.mode=train \
    -o Global.dataset_dir="${DATASET}" \
    -o Global.device="${DEVICE}" \
    -o Global.output="${OUTPUT}" \
    -o Train.num_classes="${NUM_CLASSES}" \
    -o Train.batch_size="${BATCH_SIZE}" \
    -o Train.epochs_iters="${EPOCHS}"
