#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/lib/common.sh"

section "System prerequisites"

if [[ "${EUID}" -eq 0 ]]; then
    APT="apt-get"
else
    fail "Run this script as root or with sudo."
fi

${APT} update -qq
${APT} install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    git \
    curl \
    wget

PY_INCLUDE="/usr/include/python3.11"
[[ -f "${PY_INCLUDE}/Python.h" ]] || fail "Python.h not found at ${PY_INCLUDE}/Python.h"

ensure_uv

ok "Python 3.11 + development headers installed"
ok "uv available: $(uv --version)"
