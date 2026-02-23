"""Application configuration."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
PLUGINS_DIR = BASE_DIR / "plugins"
OUTPUT_DIR = BASE_DIR / "output_files"

# Audio settings
SAMPLE_RATE = 22050
CHUNK_DURATION = 2.0  # seconds per analysis window
CHUNK_HOP = 0.5  # seconds between chunk starts (overlap = CHUNK_DURATION - CHUNK_HOP)
MIC_BLOCK_SIZE = 1024  # samples per mic callback block

# Transcription defaults
DEFAULT_ONSET_THRESHOLD = 0.3
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MIN_NOTE_LENGTH_MS = 127  # milliseconds
DEFAULT_MIN_FREQUENCY = 27.5  # A0
DEFAULT_MAX_FREQUENCY = 4186.0  # C8

# MIDI
VIRTUAL_MIDI_PORT_NAME = "MusicTransposer"
DEFAULT_MIDI_TEMPO = 120  # BPM

# Web server
HOST = os.environ.get("MT_HOST", "0.0.0.0")
PORT = int(os.environ.get("MT_PORT", "5050"))
DEBUG = os.environ.get("MT_DEBUG", "false").lower() == "true"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
