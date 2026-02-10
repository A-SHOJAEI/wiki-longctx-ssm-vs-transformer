#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: venv python not found at ${VENV_DIR}/bin/python" >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/pip" ]]; then
  GETPIP_URL="${GETPIP_URL:-https://bootstrap.pypa.io/get-pip.py}"
  GETPIP_PATH="${VENV_DIR}/get-pip.py"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${GETPIP_URL}" -o "${GETPIP_PATH}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${GETPIP_PATH}" "${GETPIP_URL}"
  else
    echo "ERROR: need curl or wget to download get-pip.py" >&2
    exit 1
  fi

  if [[ -n "${GETPIP_SHA256:-}" ]]; then
    echo "${GETPIP_SHA256}  ${GETPIP_PATH}" | sha256sum -c -
  else
    echo "NOTE: GETPIP_SHA256 not set; skipping checksum verification for get-pip.py" >&2
  fi

  "${VENV_DIR}/bin/python" "${GETPIP_PATH}" --disable-pip-version-check --no-warn-script-location
fi

"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install -r requirements.txt

