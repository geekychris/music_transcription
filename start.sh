#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: virtualenv not found at $VENV_DIR"
    echo "Run install_mac.sh first."
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR"
exec "$VENV_DIR/bin/python" -m app.main "$@"
