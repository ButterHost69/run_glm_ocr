#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../lib/common.sh"

section "02 · Python 3.11 virtual environment"

if [[ ! -x "${VENV}/bin/python" ]]; then
    log "Creating virtual environment: ${VENV}"
    run python3.11 -m venv "${VENV}"
else
    ok "Reusing existing virtual environment: ${VENV}"
fi

activate_venv
python -V
printf 'Executable: '; python -c 'import sys; print(sys.executable)'

PY_INCLUDE="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
printf 'Python include: %s\n' "${PY_INCLUDE}"
[[ -f "${PY_INCLUDE}/Python.h" ]] || fail "Python.h not found: ${PY_INCLUDE}/Python.h"

ok "Python environment is ready."
