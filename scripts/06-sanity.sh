#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

section "06 · Sanity checks"

# ============================================================
# Python / package checks
# ============================================================

python - <<'PY'
import paddle
import paddlex
import ppdet
import glmocr
import transformers
import vllm

print()
print("========== INSTALLATION CHECK ==========")
print("Paddle:          ", paddle.__version__)
print("PaddleX:         ", paddlex.__version__)
print("PaddleDetection: ", ppdet.__file__)
print("GLM-OCR:         ", getattr(glmocr, "__version__", "unknown"))
print("Transformers:    ", transformers.__version__)
print("vLLM:            ", vllm.__version__)
print("Paddle device:   ", paddle.device.get_device())
print("CUDA enabled:    ", paddle.device.is_compiled_with_cuda())
print("========================================")
PY

# ============================================================
# Core configuration
# ============================================================

[[ -f "${WORK_ROOT}/our_glm.yaml" ]] \
    || fail "Missing config: ${WORK_ROOT}/our_glm.yaml"

[[ -x "${WORK_ROOT}/start_vllm.sh" ]] \
    || fail "Missing launcher: ${WORK_ROOT}/start_vllm.sh"

# ============================================================
# Runtime scripts
# ============================================================

SCRIPT_09="$(
    find "${WORK_ROOT}/scripts" \
        -maxdepth 1 \
        -type f \
        -name '09*.py' \
        -print \
    | sort \
    | head -n 1
)"

[[ -n "${SCRIPT_09}" ]] \
    || fail "Missing 09*.py runtime script"

[[ -f "${WORK_ROOT}/scripts/10_validate_pipeline.py" ]] \
    || fail "Missing: ${WORK_ROOT}/scripts/10_validate_pipeline.py"

[[ -f "${WORK_ROOT}/scripts/11_convert_dataset.py" ]] \
    || fail "Missing: ${WORK_ROOT}/scripts/11_convert_dataset.py"

ok "Found runtime script 09: ${SCRIPT_09}"
ok "Found runtime script 10: ${WORK_ROOT}/scripts/10_validate_pipeline.py"
ok "Found runtime script 11: ${WORK_ROOT}/scripts/11_convert_dataset.py"

# ============================================================
# Verify CLI interfaces
# ============================================================

echo
echo "Checking runtime CLI interfaces..."

python "${SCRIPT_09}" --help >/dev/null \
    || fail "09 script failed --help"

python "${WORK_ROOT}/scripts/10_validate_pipeline.py" --help >/dev/null \
    || fail "10_validate_pipeline.py failed --help"

python "${WORK_ROOT}/scripts/11_convert_dataset.py" --help >/dev/null \
    || fail "11_convert_dataset.py failed --help"

ok "09 CLI is valid"
ok "10 CLI is valid"
ok "11 CLI is valid"

# ============================================================
# Verify 11's new interface
# ============================================================

echo
echo "Checking dataset converter interface..."

CONVERTER_HELP="$(
    python "${WORK_ROOT}/scripts/11_convert_dataset.py" --help
)"

grep -q -- "--dataset" <<< "${CONVERTER_HELP}" \
    || fail "11 converter is missing --dataset"

grep -q -- "--output" <<< "${CONVERTER_HELP}" \
    || fail "11 converter is missing --output"

grep -q -- "--val-ratio" <<< "${CONVERTER_HELP}" \
    || fail "11 converter is missing --val-ratio"

grep -q -- "--seed" <<< "${CONVERTER_HELP}" \
    || fail "11 converter is missing --seed"

grep -q -- "--smoke-test" <<< "${CONVERTER_HELP}" \
    || fail "11 converter is missing --smoke-test"

if grep -q -- "--images-root" <<< "${CONVERTER_HELP}"; then
    fail "11 converter still exposes obsolete --images-root"
fi

if grep -q -- "--input" <<< "${CONVERTER_HELP}"; then
    fail "11 converter still exposes obsolete --input"
fi

ok "11 converter uses the new --dataset interface"

# ============================================================
# Configuration sanity
# ============================================================

echo
echo "Checking GLM-OCR configuration..."

grep -q "layout:" "${WORK_ROOT}/our_glm.yaml" \
    || fail "GLM-OCR config is missing layout section"

grep -q "ocr_api:" "${WORK_ROOT}/our_glm.yaml" \
    || fail "GLM-OCR config is missing ocr_api section"

grep -q "device: cpu" "${WORK_ROOT}/our_glm.yaml" \
    || warn "GLM-OCR config does not currently contain 'device: cpu'"

ok "GLM-OCR configuration exists"

# ============================================================
# Complete
# ============================================================

ok "All sanity checks passed."
