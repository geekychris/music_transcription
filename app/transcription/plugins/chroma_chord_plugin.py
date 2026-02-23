"""Chroma-based chord detection plugin — lightweight, suitable for real-time."""

import logging

import numpy as np
import librosa

from app.transcription.base import TranscriberPlugin, TranscriptionResult, NoteEvent, ChordEvent
from app.transcription.chord_classifier import CHORD_TEMPLATES, NOTE_NAMES, _COMPLEXITY_PENALTY

logger = logging.getLogger(__name__)

# Pre-compute template vectors for efficiency
_TEMPLATE_VECTORS = {}
for _name, _intervals in CHORD_TEMPLATES.items():
    v = np.zeros(12, dtype=np.float32)
    for i in _intervals:
        v[i % 12] = 1.0
    _TEMPLATE_VECTORS[_name] = v


class ChromaChordPlugin(TranscriberPlugin):
    """Real-time chord detection using librosa CQT chromagram.

    Computes a chromagram, segments by beat or fixed frames, and matches
    each frame against chord templates. Also extracts approximate note
    pitches from chroma peaks.
    """

    def __init__(self, hop_length: int = 256, n_chroma: int = 12,
                 frame_duration: float = 0.5, min_confidence: float = 0.4):
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.frame_duration = frame_duration  # seconds per chord frame
        self.min_confidence = min_confidence

    def name(self) -> str:
        return "Chroma Chords"

    def description(self) -> str:
        return ("Lightweight chord detection using CQT chromagram and template matching. "
                "Fast enough for real-time use. Detects chords and approximate pitches.")

    def supports_realtime(self) -> bool:
        return True

    def transcribe(self, audio: np.ndarray, sr: int) -> TranscriptionResult:
        # Compute CQT chromagram
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=sr, hop_length=self.hop_length, n_chroma=self.n_chroma
        )

        # Time axis
        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]), sr=sr, hop_length=self.hop_length
        )

        # Segment into fixed-duration frames
        frame_samples = max(1, int(self.frame_duration / (self.hop_length / sr)))
        n_frames = chroma.shape[1]

        chords = []
        notes = []

        for start_idx in range(0, n_frames, frame_samples):
            end_idx = min(start_idx + frame_samples, n_frames)
            segment = chroma[:, start_idx:end_idx]

            # Average chroma across the segment
            avg_chroma = np.mean(segment, axis=1)

            # Find chord
            chord_name, root, confidence = self._match_chord(avg_chroma)
            if confidence >= self.min_confidence:
                start_time = times[start_idx] if start_idx < len(times) else 0
                end_time = times[min(end_idx, len(times) - 1)] if end_idx <= len(times) else times[-1]

                # Extract active pitches from chroma peaks
                threshold = 0.4 * np.max(avg_chroma) if np.max(avg_chroma) > 0 else 0
                active_pcs = np.where(avg_chroma > threshold)[0]
                # Map pitch classes to MIDI (assume octave 4 = middle)
                midi_pitches = [int(pc) + 60 for pc in active_pcs]

                chords.append(ChordEvent(
                    start_time=start_time,
                    end_time=end_time,
                    chord_name=chord_name,
                    root=root,
                    pitches=midi_pitches,
                    confidence=confidence,
                ))

                # Also emit individual notes
                for pc in active_pcs:
                    midi_pitch = int(pc) + 60
                    velocity = int(min(avg_chroma[pc] * 100, 127))
                    notes.append(NoteEvent(
                        start_time=start_time,
                        end_time=end_time,
                        midi_pitch=midi_pitch,
                        velocity=max(velocity, 30),
                        confidence=float(avg_chroma[pc]),
                    ))

        # Smooth consecutive chords: merge adjacent frames with the same chord name
        chords = self._smooth_chords(chords)

        # Build preliminary result and merge same-pitch notes across frames
        result = TranscriptionResult(notes=notes, chords=chords)
        result = result.merge_notes(max_gap=self.frame_duration * 0.5)
        notes = result.notes

        # Estimate tempo
        tempo = None
        try:
            tempo_est, _ = librosa.beat.beat_track(y=audio, sr=sr)
            if isinstance(tempo_est, np.ndarray):
                tempo_est = float(tempo_est[0])
            tempo = float(tempo_est)
        except Exception:
            pass

        return TranscriptionResult(notes=notes, chords=chords, tempo=tempo)

    @staticmethod
    def _smooth_chords(chords: list[ChordEvent]) -> list[ChordEvent]:
        """Merge consecutive frames that detected the same chord.

        Also applies a simple majority-vote smoothing: if a single
        different chord appears between two runs of the same chord,
        replace it with the surrounding chord (reduces noise from
        melodic passages like trills).
        """
        if len(chords) < 2:
            return chords

        # Pass 1: majority-vote de-noise — replace isolated 1-frame outliers
        names = [c.chord_name for c in chords]
        for i in range(1, len(names) - 1):
            if names[i - 1] == names[i + 1] and names[i] != names[i - 1]:
                names[i] = names[i - 1]
                chords[i] = ChordEvent(
                    start_time=chords[i].start_time,
                    end_time=chords[i].end_time,
                    chord_name=names[i],
                    root=chords[i - 1].root,
                    pitches=chords[i - 1].pitches,
                    confidence=min(chords[i - 1].confidence, chords[i + 1].confidence),
                )

        # Pass 2: merge consecutive same-name chords
        merged: list[ChordEvent] = [chords[0]]
        for c in chords[1:]:
            prev = merged[-1]
            if c.chord_name == prev.chord_name:
                # Extend the previous chord
                merged[-1] = ChordEvent(
                    start_time=prev.start_time,
                    end_time=c.end_time,
                    chord_name=prev.chord_name,
                    root=prev.root,
                    pitches=prev.pitches,
                    confidence=max(prev.confidence, c.confidence),
                )
            else:
                merged.append(c)
        return merged

    def _match_chord(self, chroma_vector: np.ndarray) -> tuple[str, int, float]:
        """Match a 12-bin chroma vector against all chord templates.

        Applies the same simplicity bias as the classifier so that basic
        triads are preferred over complex chords when scores are close.
        """
        best_score = -1.0
        best_name = "N"
        best_root = 0

        chroma_norm = np.linalg.norm(chroma_vector)
        if chroma_norm == 0:
            return "N", 0, 0.0

        for root in range(12):
            rotated = np.roll(chroma_vector, -root)
            for tmpl_name, tmpl_vec in _TEMPLATE_VECTORS.items():
                dot = np.dot(rotated, tmpl_vec)
                norm = chroma_norm * np.linalg.norm(tmpl_vec)
                score = dot / norm if norm > 0 else 0
                # Simplicity bias — prefer basic triads
                score -= _COMPLEXITY_PENALTY.get(tmpl_name, 0.05)
                if score > best_score:
                    best_score = score
                    best_name = f"{NOTE_NAMES[root]}{tmpl_name}"
                    best_root = root

        return best_name, best_root, float(best_score)

    def transcribe_chunk(self, chunk: np.ndarray, sr: int,
                         time_offset: float = 0.0) -> TranscriptionResult:
        result = self.transcribe(chunk, sr)
        if time_offset > 0:
            result = result.shift_time(time_offset)
        return result
