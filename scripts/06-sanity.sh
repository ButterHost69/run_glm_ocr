#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

section "06 · Sanity checks"
activate_venv

python - <<'PY'
import paddle
import paddlex
import ppdet
import glmocr
import transformers
import vllm

print()
print("========== INSTALLATION CHECK ==========")
print("Paddle:         ", paddle.__version__)
print("PaddleX:        ", paddlex.__version__)
print("PaddleDetection:", ppdet.__file__)
print("GLM-OCR:        ", getattr(glmocr, "__version__", "unknown"))
print("Transformers:   ", transformers.__version__)
print("vLLM:           ", vllm.__version__)
print("Paddle device:  ", paddle.device.get_device())
print("CUDA enabled:   ", paddle.device.is_compiled_with_cuda())
print("========================================")
PY

[[ -f "${WORK_ROOT}/our_glm.yaml" ]] || fail "Missing config: ${WORK_ROOT}/our_glm.yaml"
[[ -x "${WORK_ROOT}/start_vllm.sh" ]] || fail "Missing launcher: ${WORK_ROOT}/start_vllm.sh"

ok "All sanity checks passed."
