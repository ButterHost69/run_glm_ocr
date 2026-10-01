#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

section "04 · GLM-OCR + vLLM"
activate_venv

log "Installing GLM-OCR self-hosted SDK ${GLMOCR_VERSION}..."
run python -m pip install "glmocr[selfhosted]==${GLMOCR_VERSION}"

# The original bootstrap installed these without version constraints. Keep the
# configured values explicit so the environment is inspectable and reproducible.
log "Installing Transformers..."
run python -m pip install "transformers"

log "Installing vLLM..."
run python -m pip install "vllm"

python - <<'PY'
import glmocr
import transformers
import vllm
print("GLM-OCR:", getattr(glmocr, "__version__", "unknown"))
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__version__)
PY

ok "GLM-OCR stack is installed."
