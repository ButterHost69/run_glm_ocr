#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

section "03 · PaddleX + PaddleDetection"
activate_venv

mkdir -p "$(dirname "${PADDLEX_ROOT}")"

if [[ ! -d "${PADDLEX_ROOT}/.git" ]]; then
    log "Cloning PaddleX ${PADDLEX_BRANCH}..."
    run git clone --branch "${PADDLEX_BRANCH}" --depth 1 \
        https://github.com/PaddlePaddle/PaddleX.git "${PADDLEX_ROOT}"
else
    ok "PaddleX checkout already exists: ${PADDLEX_ROOT}"
fi

cd "${PADDLEX_ROOT}"
log "Installing PaddleX in editable mode..."
run python -m pip install --no-build-isolation -e ".[base]"

log "Installing PaddlePaddle GPU ${PADDLE_VERSION} (CUDA 12.6 wheel index)..."
run python -m pip install \
    "paddlepaddle-gpu==${PADDLE_VERSION}" \
    -i "https://www.paddlepaddle.org.cn/packages/stable/cu126/"

if python -c 'import ppdet' >/dev/null 2>&1; then
    ok "PaddleDetection is already importable."
else
    log "Installing PaddleDetection plugin..."
    run paddlex --install PaddleDetection
fi

python - <<'PY'
import paddlex
import ppdet
print("PaddleX:", paddlex.__version__)
print("PaddleX path:", paddlex.__file__)
print("PaddleDetection path:", ppdet.__file__)
PY

warn "Do not install numba==0.56.4 on Python 3.11; it is not required for PP-DocLayoutV3."
ok "PaddleX stack is ready."
