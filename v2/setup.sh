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
  ./setup.sh train [--dataset DIR] [--output DIR] [--num-classes N] [--device DEVICE]
  ./setup.sh vllm
  ./setup.sh config
  ./setup.sh sanity-inference
  ./setup.sh sanity-training
  ./setup.sh system
  ./setup.sh show-config
EOF
}

run_script() {
    local script="$1"
    shift
    chmod +x "${ROOT_DIR}/scripts/${script}"
    "${ROOT_DIR}/scripts/${script}" "$@"
}

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
            1) install_inference; read -rp "Press Enter..." _ ;;
            2) run_script 04-config.sh; read -rp "Press Enter..." _ ;;
            3) run_script 07-vllm.sh; read -rp "Press Enter..." _ ;;
            4) run_inference_cli; read -rp "Press Enter..." _ ;;
            5) run_validation_cli; read -rp "Press Enter..." _ ;;
            6) run_script 05-inference-sanity.sh; read -rp "Press Enter..." _ ;;
            b|B) return ;;
            *) echo "Invalid selection"; sleep 1 ;;
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
            1) install_training; read -rp "Press Enter..." _ ;;
            2) run_convert_cli; read -rp "Press Enter..." _ ;;
            3) run_script 12-train.sh; read -rp "Press Enter..." _ ;;
            4) run_script 06-training-sanity.sh; read -rp "Press Enter..." _ ;;
            b|B) return ;;
            *) echo "Invalid selection"; sleep 1 ;;
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
            1) run_script 01-system.sh; read -rp "Press Enter..." _ ;;
            2) show_config; read -rp "Press Enter..." _ ;;
            3) install_inference; install_training; read -rp "Press Enter..." _ ;;
            b|B) return ;;
            *) echo "Invalid selection"; sleep 1 ;;
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
            1) run_inference_menu ;;
            2) run_training_menu ;;
            3) run_other_menu ;;
            q|Q) exit 0 ;;
            *) echo "Invalid selection"; sleep 1 ;;
        esac
    done
}

cmd="${1:-tui}"
shift || true

case "${cmd}" in
    tui) tui ;;
    system) run_script 01-system.sh "$@" ;;
    install-inference) install_inference ;;
    install-training) install_training ;;
    inference) run_inference_cli "$@" ;;
    validate) run_validation_cli "$@" ;;
    convert) run_convert_cli "$@" ;;
    train) run_script 12-train.sh "$@" ;;
    vllm) run_script 07-vllm.sh "$@" ;;
    config) run_script 04-config.sh "$@" ;;
    sanity-inference) run_script 05-inference-sanity.sh "$@" ;;
    sanity-training) run_script 06-training-sanity.sh "$@" ;;
    show-config) show_config ;;
    help|-h|--help) usage ;;
    *)
        echo "Unknown command: ${cmd}" >&2
        usage >&2
        exit 2
        ;;
esac
