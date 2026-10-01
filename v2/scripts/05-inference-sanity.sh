#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

section "Inference sanity"

require_file "${INFERENCE_PYTHON}"
require_file "${ROOT_DIR}/our_glm.yaml"
require_file "${ROOT_DIR}/start_vllm.sh"
require_file "${ROOT_DIR}/scripts/09_run_glm.py"
require_file "${ROOT_DIR}/scripts/10_validate_layout.py"

run_inference - <<'PY'
import numpy
import paddle
import paddlex
import glmocr
import transformers
import vllm

print("Python:       ", __import__("sys").version.split()[0])
print("NumPy:        ", numpy.__version__)
print("Paddle:       ", paddle.__version__)
print("Paddle CUDA:  ", paddle.is_compiled_with_cuda())
print("PaddleX:      ", paddlex.__version__)
print("GLM-OCR:      ", getattr(glmocr, "__version__", "unknown"))
print("Transformers: ", transformers.__version__)
print("vLLM:         ", vllm.__version__)

if paddle.is_compiled_with_cuda():
    raise SystemExit("ERROR: inference environment should use CPU Paddle")
PY

"${INFERENCE_PYTHON}" "${ROOT_DIR}/scripts/09_run_glm.py" --help >/dev/null
"${INFERENCE_PYTHON}" "${ROOT_DIR}/scripts/10_validate_layout.py" --help >/dev/null

ok "Inference sanity passed"
