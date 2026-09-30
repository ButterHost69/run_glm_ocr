#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

section "05 · GLM-OCR configuration"

mkdir -p "${WORK_ROOT}"

log "Writing ${WORK_ROOT}/our_glm.yaml..."
cat > "${WORK_ROOT}/our_glm.yaml" <<YAML
pipeline:
  maas:
    enabled: false

  ocr_api:
    api_host: 127.0.0.1
    api_port: ${VLLM_PORT}
    api_path: /v1/chat/completions
    api_mode: openai
    model: ${VLLM_MODEL_NAME}
    verify_ssl: false

  layout:
    model_dir: ${LAYOUT_MODEL_DIR}
    threshold: 0.3
    batch_size: 1
    workers: 1
    cuda_visible_devices: "0"
    device: cpu
    use_polygon: true

    id2label:
      0: abstract
      1: algorithm
      2: aside_text
      3: chart
      4: content
      5: display_formula
      6: doc_title
      7: figure_title
      8: footer
      9: footer_image
      10: footnote
      11: formula_number
      12: header
      13: header_image
      14: image
      15: inline_formula
      16: number
      17: paragraph_title
      18: reference
      19: reference_content
      20: seal
      21: table
      22: text
      23: vertical_text
      24: vision_footnote

    label_task_mapping:
      text:
        - abstract
        - algorithm
        - content
        - doc_title
        - figure_title
        - paragraph_title
        - reference_content
        - text
        - vertical_text
        - vision_footnote
        - seal
        - formula_number

      table:
        - table

      formula:
        - display_formula
        - inline_formula

      skip:
        - chart
        - image

      abandon:
        - header
        - footer
        - number
        - footnote
        - aside_text
        - reference
        - footer_image
        - header_image
YAML

printf 'Config:  %s\n' "${WORK_ROOT}/our_glm.yaml"
