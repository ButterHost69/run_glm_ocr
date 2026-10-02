#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

CONFIG_PATH="${ROOT_DIR}/our_glm.yaml"

section "GLM-OCR configuration"

mkdir -p "${ROOT_DIR}"

cat > "${CONFIG_PATH}" <<YAML
pipeline:

  maas:
    enabled: false

  ocr_api:
    api_host: 127.0.0.1
    api_port: 8081
    api_path: /v1/chat/completions
    api_mode: openai
    model: glm-ocr
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

  page_loader:
    max_tokens: 4096
YAML

ok "Wrote ${CONFIG_PATH}"
