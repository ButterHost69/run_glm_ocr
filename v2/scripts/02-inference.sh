#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

section "Inference environment"

ensure_uv
ensure_venv "${INFERENCE_VENV}"

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    "paddlepaddle==${PADDLE_CPU_VERSION}"

if [[ ! -d "${PADDLEX_ROOT}/.git" ]]; then
    git clone \
        --branch "${PADDLEX_BRANCH}" \
        --depth 1 \
        https://github.com/PaddlePaddle/PaddleX.git \
        "${PADDLEX_ROOT}"
else
    info "PaddleX checkout already exists: ${PADDLEX_ROOT}"
fi

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    --no-build-isolation \
    -e "${PADDLEX_ROOT}[base]"

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    "glmocr[selfhosted]==${GLMOCR_VERSION}" \
    "vllm==${VLLM_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}"

run_inference - <<'PY'
import cv2
import glmocr
import numpy
import paddle
import paddlex
import transformers
import vllm

print("Python:       ", __import__("sys").version.split()[0])
print("NumPy:        ", numpy.__version__)
print("OpenCV:       ", cv2.__version__)
print("Paddle:       ", paddle.__version__)
print("Paddle CUDA:  ", paddle.is_compiled_with_cuda())
print("PaddleX:      ", paddlex.__version__)
print("GLM-OCR:      ", getattr(glmocr, "__version__", "unknown"))
print("Transformers: ", transformers.__version__)
print("vLLM:         ", vllm.__version__)
PY

ok "Inference environment ready: ${INFERENCE_VENV}"
