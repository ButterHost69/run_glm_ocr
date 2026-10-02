#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

section "vLLM launcher"

require_file "${INFERENCE_PYTHON}"

cat > "${ROOT_DIR}/start_vllm.sh" <<'SH2'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_VENV="${INFERENCE_VENV:-${ROOT_DIR}/.venv-inference}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PORT="${VLLM_PORT:-8080}"
MODEL_NAME="${VLLM_MODEL_NAME:-glm-ocr}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

exec "${INFERENCE_VENV}/bin/vllm" serve zai-org/GLM-OCR \
    --port "${PORT}" \
    --served-model-name "${MODEL_NAME}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}"
SH2

chmod +x "${ROOT_DIR}/start_vllm.sh"

ok "Wrote ${ROOT_DIR}/start_vllm.sh"
