#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

section "Training sanity"

require_file "${TRAINING_PYTHON}"
require_dir "${PADDLEX_ROOT}"
require_file "${PADDLEX_ROOT}/paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml"

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

if numpy.__version__.split('.')[0] != '1':
    raise SystemExit("ERROR: training environment must use NumPy 1.x")
if paddle.is_compiled_with_cuda() is not True:
    raise SystemExit("ERROR: training PaddlePaddle is not CUDA-enabled")
PY

cd "${PADDLEX_ROOT}"
"${TRAINING_PYTHON}" main.py \
    -c paddlex/configs/modules/layout_analysis/PP-DocLayoutV3.yaml \
    --help >/dev/null

ok "Training sanity passed"
