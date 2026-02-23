"""Load audio files in various formats."""

import logging
from pathlib import Path

import numpy as np
import librosa

from app.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file and return (mono_audio, sample_rate).

    Resamples to the target sample rate and converts to mono.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {path.suffix}. Supported: {SUPPORTED_EXTENSIONS}")

    logger.info("Loading audio: %s (target sr=%d)", path, sr)
    audio, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    logger.info("Loaded %.2f seconds of audio", len(audio) / actual_sr)
    return audio, actual_sr


def get_duration(path: str | Path) -> float:
    """Get audio file duration in seconds without loading entire file."""
    return librosa.get_duration(path=str(path))
