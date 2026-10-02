#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/lib/common.sh"

export ROOT_DIR INFERENCE_VENV TRAINING_VENV PADDLEX_ROOT


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

Examples:
  ./setup.sh inference

  ./setup.sh inference \
    --config /content/run_glm_ocr/v2/our_glm.yaml \
    --image /content/glm_finetune/datasets/validation/images/example.png

  ./setup.sh validate \
    --config /content/run_glm_ocr/v2/our_glm.yaml \
    --dataset /content/glm_finetune/datasets/validation/training

  ./setup.sh convert

  ./setup.sh train \
    --dataset /content/glm_finetune/datasets/validation/training \
    --output /content/glm_finetune/models/pplayoutv3_2 \
    --num-classes 25 \
    --device gpu:0
EOF
}


run_script() {
    local script="$1"
    shift

    chmod +x "${ROOT_DIR}/scripts/${script}"
    "${ROOT_DIR}/scripts/${script}" "$@"
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

run_validation_cli() {
    # Preserve direct CLI usage:
    #
    #   ./setup.sh validate --image ... --ground-truth ...
    #
    if (( $# > 0 )); then
        require_file "${INFERENCE_VENV}/bin/python"
        require_file "${ROOT_DIR}/scripts/10_validate_layout.py"

        "${INFERENCE_VENV}/bin/python" \
            "${ROOT_DIR}/scripts/10_validate_layout.py" \
            "$@"
        return
    fi

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

    echo
    echo "============================================"
    echo "             Validate Layout"
    echo "============================================"
    echo

    read -rp \
        "Config path [${default_config}]: " \
        config
    config="${config:-${default_config}}"

    echo
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
        echo

        read -rp "Image path: " image
        if [[ -z "${image}" ]]; then
            echo "[ERROR] Image path is required."
            return 1
        fi

        read -rp "Ground-truth annotation file: " ground_truth
        if [[ -z "${ground_truth}" ]]; then
            echo "[ERROR] Ground-truth annotation file is required."
            return 1
        fi
    else
        echo

        read -rp "PaddleX dataset directory: " dataset
        if [[ -z "${dataset}" ]]; then
            echo "[ERROR] Dataset directory is required."
            return 1
        fi
    fi

    echo

    read -rp \
        "Expected OCR JSON [none]: " \
        expected

    read -rp \
        "Output directory [${default_output}]: " \
        output
    output="${output:-${default_output}}"

    echo

    read -rp \
        "IoU threshold [${iou_threshold}]: " \
        iou_threshold
    iou_threshold="${iou_threshold:-0.50}"

    read -rp \
        "Minimum F1 [${min_f1}]: " \
        min_f1
    min_f1="${min_f1:-0.50}"

    read -rp \
        "Minimum mean IoU [${min_iou}]: " \
        min_iou
    min_iou="${min_iou:-0.50}"

    read -rp \
        "Minimum label accuracy [${min_label_accuracy}]: " \
        min_label_accuracy
    min_label_accuracy="${min_label_accuracy:-0.50}"

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
    # Explicit CLI arguments:
    #
    #   ./setup.sh inference --config ... --image ...
    #
    # pass straight through without prompting.
    if (( $# > 0 )); then
        require_file "${ROOT_DIR}/scripts/09_run_glm.py"
        run_inference \
            "${ROOT_DIR}/scripts/09_run_glm.py" \
            "$@"
        return
    fi

    local default_config="${ROOT_DIR}/our_glm.yaml"
    local default_image="${ROOT_DIR}/page1-test1/images/79e111f2-image_1.png"

    echo
    echo "============================================"
    echo "              GLM-OCR Inference"
    echo "============================================"
    echo

    read -rp \
        "Config path [${default_config}]: " \
        config

    config="${config:-${default_config}}"

    read -rp \
        "Image path [${default_image}]: " \
        image

    image="${image:-${default_image}}"

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
# Menus
# ============================================================

run_inference_menu() {
    while true; do
        clear 2>/dev/null || true

        echo "================ INFERENCE ================"
        echo "1) Install / setup inference"
        echo "2) Configure GLM-OCR"
        echo "3) Generate vLLM launcher"
        echo "4) Inference"
        echo "5) Validate layout"
        echo "6) Inference sanity"
        echo "b) Back"
        echo

        read -rp "Select: " choice

        case "${choice}" in
            1)
                install_inference
                read -rp "Press Enter..." _
                ;;

            2)
                run_script 04-config.sh
                read -rp "Press Enter..." _
                ;;

            3)
                run_script 07-vllm.sh
                read -rp "Press Enter..." _
                ;;

            4)
                run_inference_cli
                read -rp "Press Enter..." _
                ;;

            5)
                run_validation_cli
                read -rp "Press Enter..." _
                ;;

            6)
                run_script 05-inference-sanity.sh
                read -rp "Press Enter..." _
                ;;

            b|B)
                return
                ;;

            *)
                echo "Invalid selection"
                sleep 1
                ;;
        esac
    done
}


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
                read -rp "Press Enter..." _
                ;;

            2)
                run_convert_cli
                read -rp "Press Enter..." _
                ;;

            3)
                run_script 12-train.sh
                read -rp "Press Enter..." _
                ;;

            4)
                run_script 06-training-sanity.sh
                read -rp "Press Enter..." _
                ;;

            b|B)
                return
                ;;

            *)
                echo "Invalid selection"
                sleep 1
                ;;
        esac
    done
}


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
                read -rp "Press Enter..." _
                ;;

            2)
                show_config
                read -rp "Press Enter..." _
                ;;

            3)
                install_inference
                install_training
                read -rp "Press Enter..." _
                ;;

            b|B)
                return
                ;;

            *)
                echo "Invalid selection"
                sleep 1
                ;;
        esac
    done
}


show_config() {
    if [[ -f "${ROOT_DIR}/our_glm.yaml" ]]; then
        cat "${ROOT_DIR}/our_glm.yaml"
    else
        echo "Config not found: ${ROOT_DIR}/our_glm.yaml"
    fi
}


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
                echo "Invalid selection"
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
        run_script 12-train.sh "$@"
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
        usage >&2
        exit 2
        ;;
esac
