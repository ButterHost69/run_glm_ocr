#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/common.sh"

STAGES=(system venv paddlex glmocr config sanity)

usage() {
    cat <<USAGE
Usage:
  ./setup.sh                 Interactive menu
  ./setup.sh all             Run every stage
  ./setup.sh system          Install system prerequisites
  ./setup.sh venv            Create/check Python 3.11 venv
  ./setup.sh paddlex         Install PaddleX + PaddleDetection
  ./setup.sh glmocr          Install GLM-OCR + Transformers + vLLM
  ./setup.sh config          Generate GLM-OCR config + vLLM launcher
  ./setup.sh vllm            Start the GLM-OCR vLLM server
  ./setup.sh sanity          Run installation checks
  ./setup.sh status          Show configuration
  ./setup.sh help            Show this help

Environment overrides can be placed in:
  ${ROOT_DIR}/config/local.env

The original bootstrap does not train models or convert Label Studio data.
USAGE
}

run_stage() {
    case "$1" in
        system)  "${ROOT_DIR}/scripts/01-system.sh" ;;
        venv)    "${ROOT_DIR}/scripts/02-venv.sh" ;;
        paddlex) "${ROOT_DIR}/scripts/03-paddlex.sh" ;;
        glmocr)  "${ROOT_DIR}/scripts/04-glmocr.sh" ;;
        config)  "${ROOT_DIR}/scripts/05-config.sh" ;;
        vllm)    "${ROOT_DIR}/scripts/07-vllm.sh" ;;
        sanity)  "${ROOT_DIR}/scripts/06-sanity.sh" ;;
        *) fail "Unknown stage: $1" ;;
    esac
}

run_all() {
    local stage
    for stage in "${STAGES[@]}"; do
        run_stage "${stage}"
    done
}

menu() {
    while true; do
        clear 2>/dev/null || true
        printf '%s\n' "════════════════════════════════════════════════════════════"
        printf ' GLM-OCR + PaddleX bootstrap\n'
        printf '%s\n' "════════════════════════════════════════════════════════════"
        printf '  1) Run all stages\n'
        printf '  2) System prerequisites\n'
        printf '  3) Python virtual environment\n'
        printf '  4) PaddleX + PaddleDetection\n'
        printf '  5) GLM-OCR + vLLM\n'
        printf '  6) Generate configuration\n'
        printf '  7) Start vLLM server\n'
        printf '  8) Sanity checks\n'
        printf '  9) Show configuration\n'
        printf '  q) Quit\n\n'
        read -r -p 'Select: ' choice

        case "${choice}" in
            1) run_all ;;
            2) run_stage system ;;
            3) run_stage venv ;;
            4) run_stage paddlex ;;
            5) run_stage glmocr ;;
            6) run_stage config ;;
            7) run_stage vllm ;;
            8) run_stage sanity ;;
            9) show_config ;;
            q|Q) exit 0 ;;
            *) warn "Invalid selection." ;;
        esac

        printf '\nPress Enter to continue...'
        read -r
    done
}

case "${1:-menu}" in
    all) run_all ;;
    system|venv|paddlex|glmocr|config|vllm|sanity) run_stage "$1" ;;
    status) show_config ;;
    help|-h|--help) usage ;;
    menu) menu ;;
    *) usage; exit 2 ;;
esac

if [[ "${1:-}" == "all" ]]; then
    section "Bootstrap complete"
    ok "Environment installation finished."
    printf '\nActivate:\n  source %s/bin/activate\n' "${VENV}"
    printf '\nPaddleX:\n  cd %s\n' "${PADDLEX_ROOT}"
    printf '\nGLM-OCR config:\n  %s/our_glm.yaml\n' "${WORK_ROOT}"
    printf '\nStart vLLM:\n  ./setup.sh vllm\n'    
    printf '\nSingle-T4 layout placement: pipeline.layout.device: cpu\n'
    printf 'PP-DocLayoutV3 model: %s\n' "${LAYOUT_MODEL_DIR}"
fi
