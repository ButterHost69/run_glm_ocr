#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/common.sh"

section "Starting GLM-OCR vLLM server"

[[ -x "${VENV}/bin/vllm" ]] || fail \
    "vLLM is not installed in ${VENV}. Run: ./setup.sh glmocr"

printf 'Model:       zai-org/GLM-OCR\n'
printf 'Port:        %s\n' "${VLLM_PORT}"
printf 'Model name:  %s\n' "${VLLM_MODEL_NAME}"
printf 'GPU memory:  %s\n' "${VLLM_GPU_MEMORY_UTILIZATION}"
printf 'Max length:  %s\n' "${VLLM_MAX_MODEL_LEN}"
printf '\n'

exec "${VENV}/bin/vllm" serve zai-org/GLM-OCR \
    --port "${VLLM_PORT}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
