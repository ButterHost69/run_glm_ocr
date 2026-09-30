#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

section "01 · System prerequisites"

require_cmd apt-get

log "Installing Python 3.11, development headers, venv, and build tools..."
run apt-get update -qq
run apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    git \
    curl \
    wget

[[ -f /usr/include/python3.11/Python.h ]] || fail "/usr/include/python3.11/Python.h is missing. Install python3.11-dev."
ok "System prerequisites are ready."
