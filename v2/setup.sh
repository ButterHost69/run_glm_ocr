#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GLM-OCR / PaddleX master setup
#
# Interactive:
#   ./setup.sh
#
# Direct:
#   ./setup.sh inference ...
#   ./setup.sh validate ...
#   ./setup.sh convert ...
#   ./setup.sh train ...
#   ./setup.sh vllm
#
# All project-relative defaults are derived from ROOT_DIR.
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/common.sh"

export ROOT_DIR
export INFERENCE_VENV
export TRAINING_VENV
export PADDLEX_ROOT


# ============================================================
# Helpers
# ============================================================

pause() {
    echo
    read -rp "Press Enter to continue..." _
}

prompt_default() {
    local __resultvar="$1"
    local prompt="$2"
    local default="$3"
    local value

    if [[ -n "${default}" ]]; then
        read -rp "${prompt} [${default}]: " value
        value="${value:-${default}}"
    else
        read -rp "${prompt}: " value
    fi

    printf -v "${__resultvar}" '%s' "${value}"
}

prompt_required() {
    local __resultvar="$1"
    local prompt="$2"
    local value

    while true; do
        read -rp "${prompt}: " value

        if [[ -n "${value}" ]]; then
            printf -v "${__resultvar}" '%s' "${value}"
            return
        fi

        echo "Value is required."
    done
}

prompt_existing_file() {
    local __resultvar="$1"
    local prompt="$2"
    local default="$3"

    local value

    while true; do
        if [[ -n "${default}" ]]; then
            read -rp "${prompt} [${default}]: " value
            value="${value:-${default}}"
        else
            read -rp "${prompt}: " value
        fi

        if [[ -f "${value}" ]]; then
            printf -v "${__resultvar}" '%s' "${value}"
            return
        fi

        echo "File does not exist: ${value}"
    done
}

prompt_existing_dir() {
    local __resultvar="$1"
    local prompt="$2"
    local default="$3"

    local value

    while true; do
        if [[ -n "${default}" ]]; then
            read -rp "${prompt} [${default}]: " value
            value="${value:-${default}}"
        else
            read -rp "${prompt}: " value
        fi

        if [[ -d "${value}" ]]; then
            printf -v "${__resultvar}" '%s' "${value}"
            return
        fi

        echo "Directory does not exist: ${value}"
    done
}

prompt_yes_no() {
    local __resultvar="$1"
    local prompt="$2"
    local default="$3"

    local value

    while true; do
        read -rp "${prompt} [${default}]: " value
        value="${value:-${default}}"

        case "${value,,}" in
            y|yes)
                printf -v "${__resultvar}" '%s' "yes"
                return
                ;;
            n|no)
                printf -v "${__resultvar}" '%s' "no"
                return
                ;;
            *)
                echo "Please enter y or n."
                ;;
        esac
    done
}

show_header() {
    local title="$1"

    echo
    echo "============================================"
    printf "             %s\n" "${title}"
    echo "============================================"
    echo
}


# ============================================================
# Usage
# ============================================================

usage() {
    cat <<EOF
Usage:
  ./setup.sh tui

  ./setup.sh install-inference
  ./setup.sh install-training

  ./setup.sh inference [args...]
  ./setup.sh validate [args...]
  ./setup.sh convert [args...]
  ./setup.sh train [args...]

  ./setup.sh vllm
  ./setup.sh config

  ./setup.sh sanity-inference
  ./setup.sh sanity-training

  ./setup.sh system
  ./setup.sh show-config

Interactive behavior:
  Commands without arguments prompt for their parameters.

Examples:

  ./setup.sh

  ./setup.sh inference

  ./setup.sh inference \\
    --config /path/to/our_glm.yaml \\
    --image /path/to/image.png

  ./setup.sh validate

  ./setup.sh validate \\
    --config /path/to/our_glm.yaml \\
    --dataset /path/to/paddlex/dataset

  ./setup.sh convert

  ./setup.sh convert \\
    --input /path/to/result.json \\
    --output /path/to/output

  ./setup.sh train

  ./setup.sh train \\
    --dataset /path/to/training \\
    --output /path/to/model \\
    --num-classes 25 \\
    --device gpu:0

  ./setup.sh vllm
EOF
}


# ============================================================
# Generic script runner
# ============================================================

run_script() {
    local script="$1"
    shift

    local path="${ROOT_DIR}/scripts/${script}"

    require_file "${path}"

    chmod +x "${path}"

    "${path}" "$@"
}


# ============================================================
# Installation
# ============================================================

install_inference() {
    run_script 01-system.sh
    run_script 02-inference.sh
    run_script 04-config.sh
    run_script 07-vllm.sh
    run_script 05-inference-sanity.sh
}

install_training() {
    run_script 01-system.sh
    run_script 03-training.sh
    run_script 06-training-sanity.sh
}


# ============================================================
# Validation
# ============================================================

run_validation_cli() {

    # --------------------------------------------------------
    # Direct CLI mode
    #
    # If arguments are supplied, pass them straight through.
    # --------------------------------------------------------

    if (( $# > 0 )); then
        require_file "${INFERENCE_VENV}/bin/python"
        require_file "${ROOT_DIR}/scripts/10_validate_layout.py"

        "${INFERENCE_VENV}/bin/python" \
            "${ROOT_DIR}/scripts/10_validate_layout.py" \
            "$@"

        return
    fi


    # --------------------------------------------------------
    # Interactive mode
    # --------------------------------------------------------

    local default_config="${ROOT_DIR}/our_glm.yaml"
    local default_output="${ROOT_DIR}/output/glmocr_validation"

    local config=""
    local mode=""
    local dataset=""
    local image=""
    local ground_truth=""
    local expected=""

    local iou_threshold="0.50"
    local min_f1="0.50"
    local min_iou="0.50"
    local min_label_accuracy="0.50"


    show_header "Validate Layout"

    prompt_default \
        config \
        "Config path" \
        "${default_config}"


    echo "Validation mode:"
    echo "1) Single image"
    echo "2) PaddleX dataset"
    echo

    while true; do
        read -rp "Select [1]: " mode
        mode="${mode:-1}"

        case "${mode}" in
            1|2)
                break
                ;;
            *)
                echo "Invalid selection. Enter 1 or 2."
                ;;
        esac
    done


    if [[ "${mode}" == "1" ]]; then

        prompt_required \
            image \
            "Image path"

        prompt_required \
            ground_truth \
            "Ground-truth annotation file"

    else

        prompt_required \
            dataset \
            "PaddleX dataset directory"

    fi


    echo

    prompt_default \
        expected \
        "Expected OCR JSON" \
        ""


    prompt_default \
        output \
        "Output directory" \
        "${default_output}"


    prompt_default \
        iou_threshold \
        "IoU threshold" \
        "${iou_threshold}"

    prompt_default \
        min_f1 \
        "Minimum F1" \
        "${min_f1}"

    prompt_default \
        min_iou \
        "Minimum mean IoU" \
        "${min_iou}"

    prompt_default \
        min_label_accuracy \
        "Minimum label accuracy" \
        "${min_label_accuracy}"


    local args=(
        --config "${config}"
        --output "${output}"
        --iou-threshold "${iou_threshold}"
        --min-f1 "${min_f1}"
        --min-iou "${min_iou}"
        --min-label-accuracy "${min_label_accuracy}"
    )


    if [[ "${mode}" == "1" ]]; then

        args+=(
            --image "${image}"
            --ground-truth "${ground_truth}"
        )

    else

        args+=(
            --dataset "${dataset}"
        )

    fi


    if [[ -n "${expected}" ]]; then
        args+=(
            --expected "${expected}"
        )
    fi


    echo
    echo "============================================"
    echo "Validation configuration"
    echo "============================================"
    echo "Config:              ${config}"

    if [[ "${mode}" == "1" ]]; then
        echo "Image:               ${image}"
        echo "Ground truth:        ${ground_truth}"
    else
        echo "Dataset:             ${dataset}"
    fi

    if [[ -n "${expected}" ]]; then
        echo "Expected OCR:        ${expected}"
    else
        echo "Expected OCR:        automatic/default"
    fi

    echo "Output:              ${output}"
    echo "IoU threshold:       ${iou_threshold}"
    echo "Minimum F1:          ${min_f1}"
    echo "Minimum mean IoU:    ${min_iou}"
    echo "Minimum label acc.:  ${min_label_accuracy}"
    echo "============================================"
    echo


    require_file "${INFERENCE_VENV}/bin/python"
    require_file "${ROOT_DIR}/scripts/10_validate_layout.py"

    "${INFERENCE_VENV}/bin/python" \
        "${ROOT_DIR}/scripts/10_validate_layout.py" \
        "${args[@]}"
}


# ============================================================
# Inference
# ============================================================

run_inference_cli() {

    # Direct CLI mode
    if (( $# > 0 )); then

        require_file "${ROOT_DIR}/scripts/09_run_glm.py"

        run_inference \
            "${ROOT_DIR}/scripts/09_run_glm.py" \
            "$@"

        return
    fi


    # Interactive mode

    local default_config="${ROOT_DIR}/our_glm.yaml"
    local default_image="${ROOT_DIR}/page1-test1/images/79e111f2-image_1.png"

    local config=""
    local image=""


    show_header "GLM-OCR Inference"

    prompt_default \
        config \
        "Config path" \
        "${default_config}"

    prompt_default \
        image \
        "Image path" \
        "${default_image}"


    echo
    echo "Config: ${config}"
    echo "Image:  ${image}"
    echo


    require_file "${ROOT_DIR}/scripts/09_run_glm.py"

    run_inference \
        "${ROOT_DIR}/scripts/09_run_glm.py" \
        --config "${config}" \
        --image "${image}"
}


# ============================================================
# Dataset conversion
# ============================================================

run_convert_cli() {

    # --------------------------------------------------------
    # Direct CLI mode
    # --------------------------------------------------------

    if (( $# > 0 )); then

        require_file "${ROOT_DIR}/scripts/11_convert_dataset.py"

        "${TRAINING_VENV}/bin/python" \
            "${ROOT_DIR}/scripts/11_convert_dataset.py" \
            "$@"

        return
    fi


    # --------------------------------------------------------
    # Interactive mode
    # --------------------------------------------------------

    local default_input="${ROOT_DIR}/result.json"
    local default_output="${ROOT_DIR}/output/paddle_dataset"

    local input=""
    local dataset=""
    local output=""
    local format=""
    local args=()


    show_header "Label Studio Dataset Conversion"

    echo "Input type:"
    echo "1) Label Studio JSON"
    echo "2) Existing dataset directory"
    echo


    while true; do

        read -rp "Select [1]: " format
        format="${format:-1}"

        case "${format}" in
            1|2)
                break
                ;;
            *)
                echo "Invalid selection. Enter 1 or 2."
                ;;
        esac

    done


    echo

    if [[ "${format}" == "1" ]]; then

        prompt_existing_file \
            input \
            "Label Studio JSON" \
            "${default_input}"

        args+=(
            --input "${input}"
        )

    else

        prompt_existing_dir \
            dataset \
            "Dataset directory" \
            "${ROOT_DIR}"

        args+=(
            --dataset "${dataset}"
        )

    fi


    prompt_default \
        output \
        "Output directory" \
        "${default_output}"

    args+=(
        --output "${output}"
    )


    echo
    echo "============================================"
    echo "Conversion configuration"
    echo "============================================"

    if [[ "${format}" == "1" ]]; then
        echo "Input:       ${input}"
    else
        echo "Dataset:     ${dataset}"
    fi

    echo "Output:      ${output}"
    echo "============================================"
    echo


    prompt_yes_no \
        confirm \
        "Run conversion?" \
        "Y"


    if [[ "${confirm}" != "yes" ]]; then
        echo "Conversion cancelled."
        return
    fi


    require_file "${TRAINING_VENV}/bin/python"
    require_file "${ROOT_DIR}/scripts/11_convert_dataset.py"

    "${TRAINING_VENV}/bin/python" \
        "${ROOT_DIR}/scripts/11_convert_dataset.py" \
        "${args[@]}"
}


# ============================================================
# Training
# ============================================================

run_train_cli() {
    # Direct arguments: pass them through.
    if (( $# > 0 )); then
        require_file "${ROOT_DIR}/scripts/12-train.sh"
        run_script 12-train.sh "$@"
        return
    fi

    # No arguments: let the training script handle its own prompts.
    require_file "${ROOT_DIR}/scripts/12-train.sh"
    run_script 12-train.sh
}


# ============================================================
# Inference menu
# ============================================================

run_inference_menu() {

    while true; do

        clear 2>/dev/null || true

        echo "================ INFERENCE ================"
        echo "1) Install / setup inference"
        echo "2) Configure GLM-OCR"
        echo "3) Start vLLM"
        echo "4) Inference"
        echo "5) Validate layout"
        echo "6) Inference sanity"
        echo "b) Back"
        echo

        read -rp "Select: " choice


        case "${choice}" in

            1)
                install_inference
                pause
                ;;

            2)
                run_script 04-config.sh
                pause
                ;;

            3)
                run_script 07-vllm.sh
                ;;

            4)
                run_inference_cli
                pause
                ;;

            5)
                run_validation_cli
                pause
                ;;

            6)
                run_script 05-inference-sanity.sh
                pause
                ;;

            b|B)
                return
                ;;

            *)
                echo "Invalid selection."
                sleep 1
                ;;

        esac

    done
}


# ============================================================
# Training menu
# ============================================================

run_training_menu() {

    while true; do

        clear 2>/dev/null || true

        echo "================= TRAINING ================="
        echo "1) Install / setup training"
        echo "2) Convert Label Studio dataset"
        echo "3) Fine-tune PP-DocLayoutV3"
        echo "4) Training sanity"
        echo "b) Back"
        echo

        read -rp "Select: " choice


        case "${choice}" in

            1)
                install_training
                pause
                ;;

            2)
                run_convert_cli
                pause
                ;;

            3)
                run_train_cli
                pause
                ;;

            4)
                run_script 06-training-sanity.sh
                pause
                ;;

            b|B)
                return
                ;;

            *)
                echo "Invalid selection."
                sleep 1
                ;;

        esac

    done
}


# ============================================================
# Other menu
# ============================================================

run_other_menu() {

    while true; do

        clear 2>/dev/null || true

        echo "=================== OTHER =================="
        echo "1) System prerequisites"
        echo "2) Show config"
        echo "3) Install both environments"
        echo "b) Back"
        echo

        read -rp "Select: " choice


        case "${choice}" in

            1)
                run_script 01-system.sh
                pause
                ;;

            2)
                show_config
                pause
                ;;

            3)
                install_inference
                install_training
                pause
                ;;

            b|B)
                return
                ;;

            *)
                echo "Invalid selection."
                sleep 1
                ;;

        esac

    done
}


# ============================================================
# Show generated configuration
# ============================================================

show_config() {

    if [[ -f "${ROOT_DIR}/our_glm.yaml" ]]; then
        cat "${ROOT_DIR}/our_glm.yaml"
    else
        echo "Config not found:"
        echo "  ${ROOT_DIR}/our_glm.yaml"
    fi
}


# ============================================================
# Main TUI
# ============================================================

tui() {

    while true; do

        clear 2>/dev/null || true

        echo "============================================"
        echo "        GLM-OCR / PaddleX Setup"
        echo "============================================"
        echo "1) Inference"
        echo "2) Training"
        echo "3) Other"
        echo "q) Quit"
        echo

        read -rp "Select: " choice


        case "${choice}" in

            1)
                run_inference_menu
                ;;

            2)
                run_training_menu
                ;;

            3)
                run_other_menu
                ;;

            q|Q)
                exit 0
                ;;

            *)
                echo "Invalid selection."
                sleep 1
                ;;

        esac

    done
}


# ============================================================
# Command dispatch
# ============================================================

cmd="${1:-tui}"
shift || true


case "${cmd}" in

    tui)
        tui
        ;;

    system)
        run_script 01-system.sh "$@"
        ;;

    install-inference)
        install_inference
        ;;

    install-training)
        install_training
        ;;

    inference)
        run_inference_cli "$@"
        ;;

    validate)
        run_validation_cli "$@"
        ;;

    convert)
        run_convert_cli "$@"
        ;;

    train)
        run_train_cli "$@"
        ;;

    vllm)
        run_script 07-vllm.sh "$@"
        ;;

    config)
        run_script 04-config.sh "$@"
        ;;

    sanity-inference)
        run_script 05-inference-sanity.sh "$@"
        ;;

    sanity-training)
        run_script 06-training-sanity.sh "$@"
        ;;

    show-config)
        show_config
        ;;

    help|-h|--help)
        usage
        ;;

    *)
        echo "Unknown command: ${cmd}" >&2
        echo
        usage >&2
        exit 2
        ;;

esac
