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
  ./setup.sh all             Run every installation stage
  ./setup.sh system          Install system prerequisites
  ./setup.sh venv            Create/check Python 3.11 venv
  ./setup.sh paddlex         Install PaddleX + PaddleDetection
  ./setup.sh glmocr          Install GLM-OCR + Transformers + vLLM
  ./setup.sh config          Generate GLM-OCR config + vLLM launcher
  ./setup.sh vllm            Start the GLM-OCR vLLM server
  ./setup.sh sanity          Run installation checks
  ./setup.sh 09              Run GLM-OCR inference
  ./setup.sh 10              Validate the complete GLM-OCR pipeline
  ./setup.sh 11              Convert Label Studio dataset
  ./setup.sh status          Show configuration
  ./setup.sh help            Show this help

Environment overrides can be placed in:
  ${ROOT_DIR}/config/local.env

Notes:
  - 09, 10 and 11 are runtime/data stages and are not included in "all".
  - 09 requires the GLM-OCR vLLM server to be running.
  - 10 expects a PaddleX layout dataset.
  - 11 converts a Label Studio dataset into PaddleX COCO format.

USAGE
}

run_stage() {
    local script

    case "$1" in
        system)
            script="${ROOT_DIR}/scripts/01-system.sh"
            ;;
        venv)
            script="${ROOT_DIR}/scripts/02-venv.sh"
            ;;
        paddlex)
            script="${ROOT_DIR}/scripts/03-paddlex.sh"
            ;;
        glmocr)
            script="${ROOT_DIR}/scripts/04-glmocr.sh"
            ;;
        config)
            script="${ROOT_DIR}/scripts/05-config.sh"
            ;;
        sanity)
            script="${ROOT_DIR}/scripts/06-sanity.sh"
            ;;
        vllm)
            script="${ROOT_DIR}/scripts/07-vllm.sh"
            ;;
        *)
            fail "Unknown stage: $1"
            ;;
    esac

    if [[ ! -f "${script}" ]]; then
        fail "Stage script not found: ${script}"
    fi

    chmod +x "${script}"
    "${script}"
}

run_all() {
    local stage

    for stage in "${STAGES[@]}"; do
        run_stage "${stage}"
    done
}

# ============================================================
# Locate runtime scripts
# ============================================================

find_script_09() {
    local script

    script="$(
        find "${ROOT_DIR}/scripts" \
            -maxdepth 1 \
            -type f \
            -name '09*.py' \
            -print \
        | sort \
        | head -n 1
    )"

    if [[ -z "${script}" ]]; then
        fail "Could not find a 09*.py script in ${ROOT_DIR}/scripts"
    fi

    printf '%s\n' "${script}"
}

find_script_10() {
    local script

    script="$(
        find "${ROOT_DIR}/scripts" \
            -maxdepth 1 \
            -type f \
            -name '10*.py' \
            -print \
        | sort \
        | head -n 1
    )"

    if [[ -z "${script}" ]]; then
        fail "Could not find a 10*.py script in ${ROOT_DIR}/scripts"
    fi

    printf '%s\n' "${script}"
}

find_script_11() {
    local script="${ROOT_DIR}/scripts/11_convert_dataset.py"

    if [[ ! -f "${script}" ]]; then
        fail "Could not find: ${script}"
    fi

    printf '%s\n' "${script}"
}

# ============================================================
# Interactive helpers
# ============================================================

prompt_default() {
    local prompt="$1"
    local default="$2"
    local value

    if [[ -n "${default}" ]]; then
        read -r -p "${prompt} [${default}]: " value
        printf '%s\n' "${value:-$default}"
    else
        read -r -p "${prompt}: " value
        printf '%s\n' "${value}"
    fi
}

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-y}"
    local value

    if [[ "${default}" == "y" ]]; then
        read -r -p "${prompt} [Y/n]: " value
        value="${value:-y}"
    else
        read -r -p "${prompt} [y/N]: " value
        value="${value:-n}"
    fi

    [[ "${value}" =~ ^[Yy]([Ee][Ss])?$ ]]
}

# ============================================================
# 09 — GLM-OCR inference
# ============================================================

run_09() {
    local script
    local config
    local image
    local output

    script="$(find_script_09)"

    echo
    printf '%s\n' "════════════════════════════════════════════════════════════"
    printf '%s\n' " 09 — GLM-OCR inference"
    printf '%s\n' "════════════════════════════════════════════════════════════"
    printf 'Script: %s\n\n' "${script}"

    config="$(
        prompt_default \
            "Config" \
            "${WORK_ROOT}/our_glm.yaml"
    )"

    image="$(
        prompt_default \
            "Image" \
            "${WORK_ROOT}/page1-test1/images/79e111f2-image_1.png"
    )"

    output="$(
        prompt_default \
            "Output directory" \
            "${WORK_ROOT}/output/glmocr_custom"
    )"

    echo
    printf '%s\n' "Configuration:"
    printf '  Config : %s\n' "${config}"
    printf '  Image  : %s\n' "${image}"
    printf '  Output : %s\n' "${output}"
    echo

    if [[ ! -f "${config}" ]]; then
        fail "Config file does not exist: ${config}"
    fi

    if [[ ! -f "${image}" ]]; then
        fail "Image file does not exist: ${image}"
    fi

    mkdir -p "${output}"

    python "${script}" \
        --config "${config}" \
        --image "${image}" \
        --output "${output}"
}

# ============================================================
# 10 — Full pipeline validation
# ============================================================

run_10() {
    local script
    local config
    local dataset
    local output
    local iou_threshold
    local min_f1
    local min_iou
    local min_label_accuracy

    script="$(find_script_10)"

    echo
    printf '%s\n' "════════════════════════════════════════════════════════════"
    printf '%s\n' " 10 — Full pipeline validation"
    printf '%s\n' "════════════════════════════════════════════════════════════"
    printf 'Script: %s\n\n' "${script}"

    config="$(
        prompt_default \
            "Config" \
            "${WORK_ROOT}/our_glm.yaml"
    )"

    dataset="$(
        prompt_default \
            "Dataset" \
            "${WORK_ROOT}/page1-test1"
    )"

    output="$(
        prompt_default \
            "Validation output directory" \
            "${WORK_ROOT}/output/validation"
    )"

    iou_threshold="$(
        prompt_default \
            "IoU threshold" \
            "0.5"
    )"

    min_f1="$(
        prompt_default \
            "Minimum F1" \
            "0.5"
    )"

    min_iou="$(
        prompt_default \
            "Minimum mean IoU" \
            "0.5"
    )"

    min_label_accuracy="$(
        prompt_default \
            "Minimum label accuracy" \
            "0.5"
    )"

    echo
    printf '%s\n' "Validation configuration:"
    printf '  Config           : %s\n' "${config}"
    printf '  Dataset          : %s\n' "${dataset}"
    printf '  Output           : %s\n' "${output}"
    printf '  IoU threshold    : %s\n' "${iou_threshold}"
    printf '  Minimum F1       : %s\n' "${min_f1}"
    printf '  Minimum mean IoU : %s\n' "${min_iou}"
    printf '  Label accuracy   : %s\n' "${min_label_accuracy}"
    echo

    if [[ ! -f "${config}" ]]; then
        fail "Config file does not exist: ${config}"
    fi

    if [[ ! -d "${dataset}" ]]; then
        fail "Dataset directory does not exist: ${dataset}"
    fi

    mkdir -p "${output}"

    python "${script}" \
        --config "${config}" \
        --dataset "${dataset}" \
        --output "${output}" \
        --iou-threshold "${iou_threshold}" \
        --min-f1 "${min_f1}" \
        --min-iou "${min_iou}" \
        --min-label-accuracy "${min_label_accuracy}"
}

# ============================================================
# 11 — Label Studio → PaddleX dataset conversion
# ============================================================

run_11() {
    local script
    local dataset
    local output
    local val_ratio
    local seed
    local smoke_test_args=()

    script="$(find_script_11)"

    echo
    printf '%s\n' "════════════════════════════════════════════════════════════"
    printf '%s\n' " 11 — Label Studio → PaddleX dataset conversion"
    printf '%s\n' "════════════════════════════════════════════════════════════"
    printf 'Script: %s\n\n' "${script}"

    dataset="$(
        prompt_default \
            "Dataset directory" \
            "${WORK_ROOT}/page1-test1"
    )"

    output="$(
        prompt_default \
            "Dataset output directory" \
            "${WORK_ROOT}/output/page1-test1"
    )"

    val_ratio="$(
        prompt_default \
            "Validation ratio" \
            "0.1"
    )"

    seed="$(
        prompt_default \
            "Random seed" \
            "42"
    )"

    echo

    if prompt_yes_no "Use smoke-test mode?" "n"; then
        smoke_test_args+=(--smoke-test)
    fi

    echo
    printf '%s\n' "Conversion configuration:"
    printf '  Dataset : %s\n' "${dataset}"
    printf '  Output  : %s\n' "${output}"
    printf '  Val ratio: %s\n' "${val_ratio}"
    printf '  Seed    : %s\n' "${seed}"

    if [[ "${#smoke_test_args[@]}" -gt 0 ]]; then
        printf '  Smoke test: yes\n'
    else
        printf '  Smoke test: no\n'
    fi

    echo

    if [[ ! -d "${dataset}" ]]; then
        fail "Dataset directory does not exist: ${dataset}"
    fi

    mkdir -p "${output}"

    python "${script}" \
        --dataset "${dataset}" \
        --output "${output}" \
        --val-ratio "${val_ratio}" \
        --seed "${seed}" \
        "${smoke_test_args[@]}"
}

# ============================================================
# Interactive menu
# ============================================================

menu() {
    while true; do
        clear 2>/dev/null || true

        printf '%s\n' "════════════════════════════════════════════════════════════"
        printf '%s\n' " GLM-OCR + PaddleX bootstrap"
        printf '%s\n' "════════════════════════════════════════════════════════════"
        printf '  1) Run all installation stages\n'
        printf '  2) System prerequisites\n'
        printf '  3) Python virtual environment\n'
        printf '  4) PaddleX + PaddleDetection\n'
        printf '  5) GLM-OCR + vLLM\n'
        printf '  6) Generate configuration\n'
        printf '  7) Start vLLM server\n'
        printf '  8) Sanity checks\n'
        printf '  9) Run GLM-OCR inference\n'
        printf ' 10) Validate complete pipeline\n'
        printf ' 11) Convert Label Studio dataset\n'
        printf ' 12) Show configuration\n'
        printf '  q) Quit\n\n'

        read -r -p 'Select: ' choice

        case "${choice}" in
            1)  run_all ;;
            2)  run_stage system ;;
            3)  run_stage venv ;;
            4)  run_stage paddlex ;;
            5)  run_stage glmocr ;;
            6)  run_stage config ;;
            7)  run_stage vllm ;;
            8)  run_stage sanity ;;
            9)  run_09 ;;
            10) run_10 ;;
            11) run_11 ;;
            12) show_config ;;
            q|Q) exit 0 ;;
            *) warn "Invalid selection." ;;
        esac

        printf '\nPress Enter to continue...'
        read -r
    done
}

# ============================================================
# Main
# ============================================================

case "${1:-menu}" in
    all)
        run_all
        ;;
    system|venv|paddlex|glmocr|config|vllm|sanity)
        run_stage "$1"
        ;;
    09)
        run_09
        ;;
    10)
        run_10
        ;;
    11)
        run_11
        ;;
    status)
        show_config
        ;;
    help|-h|--help)
        usage
        ;;
    menu)
        menu
        ;;
    *)
        usage
        exit 2
        ;;
esac

# ============================================================
# Completion message
# ============================================================

if [[ "${1:-}" == "all" ]]; then
    section "Bootstrap complete"

    ok "Environment installation finished."

    printf '\nActivate:\n  source %s/bin/activate\n' "${VENV}"

    printf '\nPaddleX:\n  cd %s\n' "${PADDLEX_ROOT}"

    printf '\nGLM-OCR config:\n  %s/our_glm.yaml\n' "${WORK_ROOT}"

    printf '\nStart vLLM:\n  ./setup.sh vllm\n'

    printf '\nConvert Label Studio dataset:\n  ./setup.sh 11\n'

    printf '\nRun GLM-OCR:\n  ./setup.sh 09\n'

    printf '\nValidate pipeline:\n  ./setup.sh 10\n'

    printf '\nSingle-T4 layout placement: pipeline.layout.device: cpu\n'

    printf 'PP-DocLayoutV3 model: %s\n' "${LAYOUT_MODEL_DIR}"
fi
