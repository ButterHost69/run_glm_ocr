#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

section "Training environment"

ensure_uv
ensure_venv "${TRAINING_VENV}"

# Paddle's official CUDA 12.6 index for PaddlePaddle 3.0.0.
uv pip install \
    --python "${TRAINING_PYTHON}" \
    "paddlepaddle-gpu==${PADDLE_GPU_VERSION}" \
    --index-url "https://www.paddlepaddle.org.cn/packages/stable/cu126/"

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
    --python "${TRAINING_PYTHON}" \
    --no-build-isolation \
    -e "${PADDLEX_ROOT}[base]"

# PaddleX installs PaddleDetection through its official plugin command.
# Run the command from the isolated training environment only.
PATH="${TRAINING_VENV}/bin:${PATH}" \
VIRTUAL_ENV="${TRAINING_VENV}" \
"${TRAINING_VENV}/bin/paddlex" --install PaddleDetection

# Keep the legacy PaddleDetection stack isolated from the inference stack.
# These versions avoid the imgaug np.sctypes / NumPy 2.x incompatibility and
# satisfy PaddleDetection's opencv-python <= 4.6 requirement.
uv pip uninstall \
    --python "${TRAINING_PYTHON}" \
    -y \
    opencv-python \
    opencv-python-headless \
    numpy

uv pip install \
    --python "${TRAINING_PYTHON}" \
    "numpy==1.26.4" \
    "opencv-python==4.5.5.64"

run_training - <<'PY'
import cv2
import numpy
import paddle
import paddlex
import ppdet

print("Python:       ", __import__("sys").version.split()[0])
print("NumPy:        ", numpy.__version__)
print("OpenCV:       ", cv2.__version__)
print("Paddle:       ", paddle.__version__)
print("Paddle CUDA:  ", paddle.is_compiled_with_cuda())
print("Paddle device:", paddle.device.get_device())
print("PaddleX:      ", paddlex.__version__)
print("PaddleDet:    ", ppdet.__file__)

if not paddle.is_compiled_with_cuda():
    raise SystemExit("ERROR: training PaddlePaddle is not CUDA-enabled")
PY

ok "Training environment ready: ${TRAINING_VENV}"
