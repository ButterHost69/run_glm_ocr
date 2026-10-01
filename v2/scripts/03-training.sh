#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
PADDLEX_ROOT="${PADDLEX_ROOT:-${ROOT_DIR}/PaddleX}"
PADDLEX_BRANCH="${PADDLEX_BRANCH:-release/3.7}"

PADDLE_VERSION="${PADDLE_VERSION:-3.0.0}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
OPENCV_VERSION="${OPENCV_VERSION:-4.5.5.64}"

PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"

echo
echo "== Training environment =="

# ------------------------------------------------------------
# 1. Python / virtual environment
# ------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed."
    exit 1
fi

if [[ ! -f /usr/include/python3.11/Python.h ]]; then
    echo "[ERROR] Python 3.11 development headers are missing:"
    echo "        /usr/include/python3.11/Python.h"
    exit 1
fi

if [[ ! -x "${TRAINING_VENV}/bin/python" ]]; then
    echo "[INFO] Creating virtual environment: ${TRAINING_VENV}"

    uv venv \
        --python python3.11 \
        "${TRAINING_VENV}"
else
    echo "[INFO] Reusing virtual environment: ${TRAINING_VENV}"
fi

TRAINING_PYTHON="${TRAINING_VENV}/bin/python"

echo "[INFO] Python:"
"${TRAINING_PYTHON}" -V

# ------------------------------------------------------------
# 2. Build tooling
# ------------------------------------------------------------

echo "[INFO] Installing build tooling..."

uv pip install \
    --python "${TRAINING_PYTHON}" \
    "setuptools==79.0.1" \
    wheel

# ------------------------------------------------------------
# 3. PaddlePaddle GPU
# ------------------------------------------------------------

echo "[INFO] Installing PaddlePaddle GPU ${PADDLE_VERSION}..."

uv pip install \
    --python "${TRAINING_PYTHON}" \
    "paddlepaddle-gpu==${PADDLE_VERSION}" \
    --index-url "${PADDLE_INDEX}"

# ------------------------------------------------------------
# 4. Keep the training environment compatible with
#    PaddleDetection / imgaug
# ------------------------------------------------------------

echo "[INFO] Installing training-compatible NumPy/OpenCV..."

uv pip install \
    --python "${TRAINING_PYTHON}" \
    "numpy==${NUMPY_VERSION}" \
    "opencv-python==${OPENCV_VERSION}"

# ------------------------------------------------------------
# 5. Verify CUDA BEFORE installing PaddleX
# ------------------------------------------------------------

echo "[INFO] Checking Paddle CUDA..."

"${TRAINING_PYTHON}" - <<'PY'
import paddle

print("Paddle version     :", paddle.__version__)
print("CUDA enabled       :", paddle.is_compiled_with_cuda())
print("Paddle device      :", paddle.device.get_device())

if not paddle.is_compiled_with_cuda():
    raise SystemExit(
        "ERROR: PaddlePaddle is not CUDA-enabled."
    )
PY

# ------------------------------------------------------------
# 6. PaddleX source
# ------------------------------------------------------------

if [[ ! -d "${PADDLEX_ROOT}/.git" ]]; then
    echo "[INFO] Cloning PaddleX ${PADDLEX_BRANCH}..."

    git clone \
        --branch "${PADDLEX_BRANCH}" \
        --depth 1 \
        https://github.com/PaddlePaddle/PaddleX.git \
        "${PADDLEX_ROOT}"
else
    echo "[INFO] PaddleX already exists:"
    echo "       ${PADDLEX_ROOT}"
fi

# ------------------------------------------------------------
# 7. Install PaddleX
# ------------------------------------------------------------

echo "[INFO] Installing PaddleX..."

cd "${PADDLEX_ROOT}"

uv pip install \
    --python "${TRAINING_PYTHON}" \
    --no-build-isolation \
    -e ".[base]"

# ------------------------------------------------------------
# 8. Install PaddleDetection through PaddleX
# ------------------------------------------------------------

echo "[INFO] Installing PaddleDetection plugin..."

if [[ ! -x "${TRAINING_VENV}/bin/paddlex" ]]; then
    echo "[ERROR] PaddleX CLI was not installed:"
    echo "        ${TRAINING_VENV}/bin/paddlex"
    exit 1
fi

"${TRAINING_VENV}/bin/paddlex" --install PaddleDetection
