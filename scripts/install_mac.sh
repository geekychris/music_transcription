#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Music Transposer — Mac Install ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.11+ first."
    exit 1
fi

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
echo "Installing Python dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Install LilyPond for PDF export (optional)
if command -v brew &>/dev/null; then
    if ! command -v lilypond &>/dev/null; then
        echo "Installing LilyPond via Homebrew (for PDF export)..."
        brew install lilypond
    else
        echo "LilyPond already installed."
    fi
else
    echo "Warning: Homebrew not found. Install LilyPond manually for PDF export."
    echo "  brew install lilypond"
fi

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
echo "Then open http://localhost:5050 in your browser."
