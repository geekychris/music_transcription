#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Music Transposer — Raspberry Pi Install ==="

# System dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-dev python3-venv python3-pip \
    portaudio19-dev \
    libasound2-dev \
    lilypond \
    libatlas-base-dev \
    libopenblas-dev \
    ffmpeg

# Check Python
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies (this may take a while on Pi)..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Pre-download Basic Pitch model
echo "Pre-downloading Basic Pitch model..."
python3 -c "
from basic_pitch import ICASSP_2022_MODEL_PATH
print(f'Model path: {ICASSP_2022_MODEL_PATH}')
print('Basic Pitch model ready.')
"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To run:"
echo "  source venv/bin/activate"
echo "  python -m app.main"
echo ""
echo "Then open http://<pi-ip>:5050 in your browser."
