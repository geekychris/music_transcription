"""Basic Pitch plugin — Spotify's lightweight polyphonic AMT model."""

import logging
import tempfile
import os

import numpy as np
import soundfile as sf

from app.transcription.base import TranscriberPlugin, TranscriptionResult, NoteEvent
from app.transcription.chord_classifier import group_notes_into_chords
from app.config import (
    DEFAULT_ONSET_THRESHOLD, DEFAULT_FRAME_THRESHOLD,
    DEFAULT_MIN_NOTE_LENGTH_MS, DEFAULT_MIN_FREQUENCY, DEFAULT_MAX_FREQUENCY,
)

logger = logging.getLogger(__name__)

# Lazy-load basic_pitch to avoid slow import at startup
_predict_fn = None
_model_path = None


def _ensure_loaded():
    global _predict_fn, _model_path
    if _predict_fn is None:
        logger.info("Loading Basic Pitch model (first use)...")
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _predict_fn = predict
        _model_path = ICASSP_2022_MODEL_PATH
        logger.info("Basic Pitch model loaded.")


class BasicPitchPlugin(TranscriberPlugin):
    """Polyphonic note transcription using Spotify's Basic Pitch.

    Works best on single-instrument recordings (piano, guitar).
    Detects individual notes then groups them into chords.
    """

    def __init__(self, onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
                 frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
                 min_note_length_ms: float = DEFAULT_MIN_NOTE_LENGTH_MS,
                 min_freq: float = DEFAULT_MIN_FREQUENCY,
                 max_freq: float = DEFAULT_MAX_FREQUENCY):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_length_ms = min_note_length_ms
        self.min_freq = min_freq
        self.max_freq = max_freq

    def name(self) -> str:
        return "Basic Pitch"

    def description(self) -> str:
        return ("Spotify's lightweight polyphonic note transcription model. "
                "Best for single-instrument recordings (piano, guitar). "
                "Detects notes then classifies chords via template matching.")

    def supports_realtime(self) -> bool:
        # Basic Pitch can process chunks but is not optimized for real-time
        return True

    def transcribe(self, audio: np.ndarray, sr: int) -> TranscriptionResult:
        _ensure_loaded()

        # Basic Pitch expects a file path — write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(tmp_path, audio, sr)

        try:
            model_output, midi_data, note_events = _predict_fn(
                tmp_path,
                model_or_model_path=_model_path,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_note_length=self.min_note_length_ms,
                minimum_frequency=self.min_freq,
                maximum_frequency=self.max_freq,
            )
        finally:
            os.unlink(tmp_path)

        # Convert note_events to our format
        # note_events is a list of (start_time_s, end_time_s, pitch_midi, velocity, pitch_bend)
        notes = []
        for event in note_events:
            start_t, end_t = float(event[0]), float(event[1])
            pitch = int(event[2])
            # event[3] is amplitude 0.0-1.0 — scale to MIDI velocity 0-127
            velocity = int(min(event[3] * 127, 127))
            velocity = max(velocity, 30)  # ensure audible minimum
            notes.append(NoteEvent(
                start_time=start_t,
                end_time=end_t,
                midi_pitch=pitch,
                velocity=velocity,
            ))

        # Estimate tempo from MIDI data
        tempo = None
        if midi_data and midi_data.estimate_tempo():
            tempo = midi_data.estimate_tempo()

        # Group notes into chords
        chords = group_notes_into_chords(notes)

        return TranscriptionResult(notes=notes, chords=chords, tempo=tempo)

    def transcribe_chunk(self, chunk: np.ndarray, sr: int,
                         time_offset: float = 0.0) -> TranscriptionResult:
        result = self.transcribe(chunk, sr)
        if time_offset > 0:
            result = result.shift_time(time_offset)
        return result
