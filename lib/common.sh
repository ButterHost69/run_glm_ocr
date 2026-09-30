#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/config/defaults.env}"

# Load defaults first; environment variables already supplied by the caller win.
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

if [[ -f "${ROOT_DIR}/config/local.env" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/config/local.env"
fi

if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_DIM=$'\033[2m'
    C_BLUE=$'\033[34m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_RED=$'\033[31m'
    C_CYAN=$'\033[36m'
else
    C_RESET=''; C_DIM=''; C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''
fi

log()      { printf '%s[%s]%s %s\n' "${C_BLUE}" "*" "${C_RESET}" "$*"; }
ok()       { printf '%s[%s]%s %s\n' "${C_GREEN}" "✓" "${C_RESET}" "$*"; }
warn()     { printf '%s[%s]%s %s\n' "${C_YELLOW}" "!" "${C_RESET}" "$*"; }
fail()     { printf '%s[%s]%s %s\n' "${C_RED}" "✗" "${C_RESET}" "$*" >&2; exit 1; }
section()  { printf '\n%s%s%s\n' "${C_CYAN}" "════════════════════════════════════════════════════════════" "${C_RESET}"; printf '%s%s%s\n' "${C_CYAN}" "$*" "${C_RESET}"; }

run() {
    printf '%s$%s %s\n' "${C_DIM}" "${C_RESET}" "$*"
    "$@"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

activate_venv() {
    [[ -x "${VENV}/bin/python" ]] || fail "Virtual environment not found: ${VENV}. Run: ./setup.sh venv"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
}

python_check_import() {
    local module="$1"
    python -c "import ${module}" >/dev/null 2>&1
}

show_config() {
    section "Configuration"
    printf 'PADDLEX_ROOT   : %s\n' "${PADDLEX_ROOT}"
    printf 'WORK_ROOT      : %s\n' "${WORK_ROOT}"
    printf 'VENV           : %s\n' "${VENV}"
    printf 'PADDLEX_BRANCH : %s\n' "${PADDLEX_BRANCH}"
    printf 'Paddle         : %s\n' "${PADDLE_VERSION}"
    printf 'GLM-OCR        : %s\n' "${GLMOCR_VERSION}"
    printf 'vLLM           : %s\n' "${VLLM_VERSION}"
    printf 'Transformers   : %s\n' "${TRANSFORMERS_VERSION}"
    printf 'Layout model   : %s\n' "${LAYOUT_MODEL_DIR}"
}
