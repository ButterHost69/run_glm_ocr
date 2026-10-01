#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 02 · Inference environment
#
# Creates the isolated GLM-OCR inference environment:
#
#   Python 3.11
#   CPU PaddlePaddle
#   PaddleX release/3.7
#   GLM-OCR self-hosted
#   vLLM
#   Transformers
#
# PaddleDetection is intentionally NOT installed here.
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INFERENCE_VENV="${INFERENCE_VENV:-${ROOT_DIR}/.venv-inference}"
PADDLEX_ROOT="${PADDLEX_ROOT:-${ROOT_DIR}/PaddleX}"
PADDLEX_BRANCH="${PADDLEX_BRANCH:-release/3.7}"

PADDLE_VERSION="${PADDLE_VERSION:-3.0.0}"
GLMOCR_VERSION="${GLMOCR_VERSION:-0.1.5}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.3.1}"

echo
echo "== Inference environment =="

# ------------------------------------------------------------
# 1. Check prerequisites
# ------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed."
    exit 1
fi

if [[ ! -f /usr/include/python3.11/Python.h ]]; then
    echo "[ERROR] Python 3.11 development headers are missing:"
    echo "        /usr/include/python3.11/Python.h"
    echo
    echo "Install them with:"
    echo "  apt-get update && apt-get install -y python3.11-dev"
    exit 1
fi

# ------------------------------------------------------------
# 2. Create / reuse inference environment
# ------------------------------------------------------------

if [[ ! -x "${INFERENCE_VENV}/bin/python" ]]; then
    echo "[INFO] Creating virtual environment:"
    echo "       ${INFERENCE_VENV}"

    uv venv \
        --python python3.11 \
        "${INFERENCE_VENV}"
else
    echo "[INFO] Reusing virtual environment:"
    echo "       ${INFERENCE_VENV}"
fi

INFERENCE_PYTHON="${INFERENCE_VENV}/bin/python"

echo
echo "[INFO] Python:"
"${INFERENCE_PYTHON}" -V

# ------------------------------------------------------------
# 3. Build tooling
#
# Required before editable PaddleX installation because uv venv
# does not seed setuptools.
# ------------------------------------------------------------

echo
echo "[INFO] Installing build tooling..."

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    "setuptools==79.0.1" \
    wheel \
    "setuptools_scm>=8"

# ------------------------------------------------------------
# 4. PaddlePaddle CPU
# ------------------------------------------------------------

echo
echo "[INFO] Installing PaddlePaddle ${PADDLE_VERSION}..."

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    "paddlepaddle==${PADDLE_VERSION}"

# ------------------------------------------------------------
# 5. PaddleX checkout
# ------------------------------------------------------------

if [[ ! -d "${PADDLEX_ROOT}/.git" ]]; then
    echo
    echo "[INFO] Cloning PaddleX ${PADDLEX_BRANCH}..."

    git clone \
        --branch "${PADDLEX_BRANCH}" \
        --depth 1 \
        https://github.com/PaddlePaddle/PaddleX.git \
        "${PADDLEX_ROOT}"
else
    echo
    echo "[INFO] PaddleX checkout already exists:"
    echo "       ${PADDLEX_ROOT}"
fi

# ------------------------------------------------------------
# 6. Install PaddleX
# ------------------------------------------------------------

echo
echo "[INFO] Installing PaddleX..."

cd "${PADDLEX_ROOT}"

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    --no-build-isolation \
    -e ".[base]"

# ------------------------------------------------------------
# 7. GLM-OCR self-hosted
# ------------------------------------------------------------

echo
echo "[INFO] Installing GLM-OCR ${GLMOCR_VERSION}..."

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    "glmocr[selfhosted]==${GLMOCR_VERSION}"

# ------------------------------------------------------------
# 8. vLLM + Transformers
# ------------------------------------------------------------

echo
echo "[INFO] Installing vLLM ${VLLM_VERSION}..."

uv pip install \
    --python "${INFERENCE_PYTHON}" \
    "vllm==${VLLM_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}"

# ------------------------------------------------------------
# 9. Verification
# ------------------------------------------------------------

echo
echo "[INFO] Checking inference environment..."

"${INFERENCE_PYTHON}" - <<'PY'
import sys

import numpy
import paddle
import paddlex
import glmocr
import transformers
import vllm

print()
print("============================================")
print("        INFERENCE ENVIRONMENT CHECK")
print("============================================")
print("Python:        ", sys.version.split()[0])
print("NumPy:         ", numpy.__version__)
print("Paddle:        ", paddle.__version__)
print("PaddleX:       ", paddlex.__version__)
print("GLM-OCR:       ", getattr(glmocr, "__version__", "unknown"))
print("Transformers:  ", transformers.__version__)
print("vLLM:          ", vllm.__version__)
print("Paddle device: ", paddle.device.get_device())
print("CUDA compiled: ", paddle.is_compiled_with_cuda())
print("============================================")

if paddle.is_compiled_with_cuda():
    raise SystemExit(
        "ERROR: Inference environment unexpectedly has CUDA-enabled PaddlePaddle."
    )
PY

echo
echo "[OK] Inference environment ready:"
echo "     ${INFERENCE_VENV}"
