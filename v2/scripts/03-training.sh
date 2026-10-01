#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 03 · Training environment
#
# Creates an isolated PaddleX + PaddleDetection training env.
#
# Target:
#   Python 3.11
#   PaddlePaddle GPU 3.0.0
#   PaddleX release/3.7
#   PaddleDetection
#   NumPy 1.26.4
#   OpenCV 4.5.5.64
# ============================================================

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
    echo
    echo "Install them with:"
    echo "  apt-get update && apt-get install -y python3.11-dev"
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
# 3. NumPy / OpenCV versions required by PaddleDetection
# ------------------------------------------------------------

echo "[INFO] Installing training-compatible NumPy / OpenCV..."

uv pip install \
    --python "${TRAINING_PYTHON}" \
    "numpy==${NUMPY_VERSION}" \
    "opencv-python==${OPENCV_VERSION}"

# ------------------------------------------------------------
# 4. PaddlePaddle GPU
# ------------------------------------------------------------

echo "[INFO] Installing PaddlePaddle GPU ${PADDLE_VERSION}..."

uv pip install \
    --python "${TRAINING_PYTHON}" \
    "paddlepaddle-gpu==${PADDLE_VERSION}" \
    --index-url "${PADDLE_INDEX}"

# ------------------------------------------------------------
# 5. Verify Paddle CUDA support BEFORE PaddleX
# ------------------------------------------------------------

echo "[INFO] Checking Paddle CUDA..."

"${TRAINING_PYTHON}" - <<'PY'
import paddle

print("Paddle version        :", paddle.__version__)
print("Compiled with CUDA    :", paddle.is_compiled_with_cuda())
print("Paddle device         :", paddle.device.get_device())

if not paddle.is_compiled_with_cuda():
    raise SystemExit(
        "ERROR: PaddlePaddle is not CUDA-enabled."
    )
PY

# ------------------------------------------------------------
# 6. PaddleX checkout
# ------------------------------------------------------------

if [[ ! -d "${PADDLEX_ROOT}/.git" ]]; then
    echo "[INFO] Cloning PaddleX ${PADDLEX_BRANCH}..."

    git clone \
        --branch "${PADDLEX_BRANCH}" \
        --depth 1 \
        https://github.com/PaddlePaddle/PaddleX.git \
        "${PADDLEX_ROOT}"
else
    echo "[INFO] PaddleX checkout already exists:"
    echo "       ${PADDLEX_ROOT}"
fi

# ------------------------------------------------------------
# 7. Install PaddleX
#
# Keep NumPy constrained to 1.26.4.  Without this constraint,
# the resolver can upgrade NumPy to 2.x.
# ------------------------------------------------------------

CONSTRAINTS_FILE="${TRAINING_VENV}/training-constraints.txt"

cat > "${CONSTRAINTS_FILE}" <<EOF
numpy==${NUMPY_VERSION}
opencv-python==${OPENCV_VERSION}
EOF

echo "[INFO] Installing PaddleX..."

uv pip install \
    --python "${TRAINING_PYTHON}" \
    --constraint "${CONSTRAINTS_FILE}" \
    --no-build-isolation \
    -e "${PADDLEX_ROOT}[base]"

# ------------------------------------------------------------
# 8. PaddleDetection plugin
# ------------------------------------------------------------

echo "[INFO] Installing PaddleDetection..."

if "${TRAINING_PYTHON}" -c "import ppdet" 2>/dev/null; then
    echo "[INFO] PaddleDetection already importable."
else
    (
        cd "${PADDLEX_ROOT}"

        uv pip \
            install --no-deps \
            --editable \
            paddlex/repo_manager/repos/PaddleDetection
    )
fi

# ------------------------------------------------------------
# 9. Re-assert the versions after all PaddleX dependencies
# ------------------------------------------------------------

echo "[INFO] Re-checking NumPy / OpenCV..."

uv pip install \
    --python "${TRAINING_PYTHON}" \
    --constraint "${CONSTRAINTS_FILE}" \
    "numpy==${NUMPY_VERSION}" \
    "opencv-python==${OPENCV_VERSION}"

# ------------------------------------------------------------
# 10. Final sanity check
# ------------------------------------------------------------

"${TRAINING_PYTHON}" - <<'PY'
import cv2
import numpy
import paddle
import paddlex
import ppdet

print()
print("============================================")
print("        TRAINING ENVIRONMENT CHECK")
print("============================================")
print("Python:             ", __import__("sys").version.split()[0])
print("NumPy:              ", numpy.__version__)
print("OpenCV:             ", cv2.__version__)
print("Paddle:             ", paddle.__version__)
print("PaddleX:            ", paddlex.__version__)
print("PaddleDetection:    ", ppdet.__file__)
print("CUDA enabled:       ", paddle.is_compiled_with_cuda())
print("Paddle device:      ", paddle.device.get_device())
print("============================================")

if numpy.__version__ != "1.26.4":
    raise SystemExit(
        f"ERROR: Expected NumPy 1.26.4, got {numpy.__version__}"
    )

if not paddle.is_compiled_with_cuda():
    raise SystemExit(
        "ERROR: PaddlePaddle is not CUDA-enabled."
    )
PY

echo
echo "[OK] Training environment ready:"
echo "     ${TRAINING_VENV}"
