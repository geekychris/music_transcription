"""Audio preprocessing utilities."""

import numpy as np

from app.config import SAMPLE_RATE, CHUNK_DURATION, CHUNK_HOP


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def chunk_audio(audio: np.ndarray, sr: int = SAMPLE_RATE,
                chunk_duration: float = CHUNK_DURATION,
                hop_duration: float = CHUNK_HOP) -> list[tuple[np.ndarray, float]]:
    """Split audio into overlapping chunks.

    Returns list of (chunk_array, start_time_seconds).
    """
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(hop_duration * sr)
    chunks = []
    pos = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]
        # Pad last chunk if too short
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")
        start_time = pos / sr
        chunks.append((chunk, start_time))
        pos += hop_samples
        if end >= len(audio):
            break
    return chunks
