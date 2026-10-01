#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INFERENCE_VENV="${INFERENCE_VENV:-${ROOT_DIR}/.venv-inference}"
TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
PADDLEX_ROOT="${PADDLEX_ROOT:-${ROOT_DIR}/PaddleX}"

INFERENCE_PYTHON="${INFERENCE_VENV}/bin/python"
TRAINING_PYTHON="${TRAINING_VENV}/bin/python"

PADDLEX_BRANCH="${PADDLEX_BRANCH:-release/3.7}"
PADDLE_GPU_VERSION="${PADDLE_GPU_VERSION:-3.0.0}"
PADDLE_CPU_VERSION="${PADDLE_CPU_VERSION:-3.2.1}"
GLMOCR_VERSION="${GLMOCR_VERSION:-0.1.5}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.3.1}"

LAYOUT_MODEL_DIR="${LAYOUT_MODEL_DIR:-${ROOT_DIR}/output/ppdoclayoutv3_test/best_model/inference}"
DATASET_DIR="${DATASET_DIR:-/content/glm_finetune/datasets/validation/training}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/content/glm_finetune/models/pplayoutv3_2}"

section() {
    printf '\n\033[1;36m== %s ==\033[0m\n' "$1"
}

info() {
    printf '\033[1;34m[INFO]\033[0m %s\n' "$1"
}

ok() {
    printf '\033[1;32m[ OK ]\033[0m %s\n' "$1"
}

warn() {
    printf '\033[1;33m[WARN]\033[0m %s\n' "$1"
}

fail() {
    printf '\033[1;31m[FAIL]\033[0m %s\n' "$1" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "Missing file: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "Missing directory: $1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

    command -v uv >/dev/null 2>&1 || fail "uv installation completed but uv is not on PATH"
}

ensure_venv() {
    local path="$1"

    ensure_uv

    if [[ ! -x "${path}/bin/python" ]]; then
        info "Creating virtual environment: ${path}"
        uv venv --python 3.11 "${path}"
    fi
}

run_inference() {
    require_file "${INFERENCE_PYTHON}"
    "${INFERENCE_PYTHON}" "$@"
}

run_training() {
    require_file "${TRAINING_PYTHON}"
    "${TRAINING_PYTHON}" "$@"
}

run_inference_cli() {
    require_file "${ROOT_DIR}/scripts/09_run_glm.py"
    run_inference "${ROOT_DIR}/scripts/09_run_glm.py" "$@"
}

run_validation_cli() {
    require_file "${ROOT_DIR}/scripts/10_validate_layout.py"
    run_inference "${ROOT_DIR}/scripts/10_validate_layout.py" "$@"
}

run_convert_cli() {
    require_file "${ROOT_DIR}/scripts/11_convert_dataset.py"
    run_training "${ROOT_DIR}/scripts/11_convert_dataset.py" "$@"
}
