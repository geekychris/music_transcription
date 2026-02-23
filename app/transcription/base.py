"""Base classes for the transcription plugin system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoteEvent:
    """A single detected note."""
    start_time: float  # seconds
    end_time: float    # seconds
    midi_pitch: int    # 0-127
    velocity: int = 80 # 0-127
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def note_name(self) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{names[self.midi_pitch % 12]}{self.midi_pitch // 12 - 1}"


@dataclass
class ChordEvent:
    """A detected chord."""
    start_time: float       # seconds
    end_time: float         # seconds
    chord_name: str         # e.g. "Cmaj", "Am7"
    root: int               # pitch class 0-11
    pitches: list[int] = field(default_factory=list)  # constituent MIDI pitches
    confidence: float = 1.0


@dataclass
class TranscriptionResult:
    """Output from a transcription plugin."""
    notes: list[NoteEvent] = field(default_factory=list)
    chords: list[ChordEvent] = field(default_factory=list)
    tempo: float | None = None  # estimated BPM
    time_offset: float = 0.0    # offset for chunk-based results

    def shift_time(self, offset: float) -> "TranscriptionResult":
        """Return a new result with all times shifted by offset."""
        return TranscriptionResult(
            notes=[NoteEvent(n.start_time + offset, n.end_time + offset,
                             n.midi_pitch, n.velocity, n.confidence)
                   for n in self.notes],
            chords=[ChordEvent(c.start_time + offset, c.end_time + offset,
                               c.chord_name, c.root, c.pitches, c.confidence)
                    for c in self.chords],
            tempo=self.tempo,
            time_offset=self.time_offset + offset,
        )

    def merge_notes(self, max_gap: float = 0.05) -> "TranscriptionResult":
        """Return a new result with consecutive same-pitch notes merged.

        If two notes of the same MIDI pitch are separated by at most
        ``max_gap`` seconds, they are combined into a single longer note.
        This fixes the "machine-gun" effect where sustained tones are
        chopped into many tiny notes by frame-based detection.
        """
        if len(self.notes) < 2:
            return self

        # Sort by pitch then start time
        by_pitch: dict[int, list[NoteEvent]] = {}
        for n in self.notes:
            by_pitch.setdefault(n.midi_pitch, []).append(n)

        merged: list[NoteEvent] = []
        for pitch, pnotes in by_pitch.items():
            pnotes.sort(key=lambda n: n.start_time)
            cur = pnotes[0]
            for nxt in pnotes[1:]:
                if nxt.start_time <= cur.end_time + max_gap:
                    # Merge: extend current note, keep max velocity/confidence
                    cur = NoteEvent(
                        start_time=cur.start_time,
                        end_time=max(cur.end_time, nxt.end_time),
                        midi_pitch=pitch,
                        velocity=max(cur.velocity, nxt.velocity),
                        confidence=max(cur.confidence, nxt.confidence),
                    )
                else:
                    merged.append(cur)
                    cur = nxt
            merged.append(cur)

        merged.sort(key=lambda n: n.start_time)
        return TranscriptionResult(
            notes=merged, chords=self.chords,
            tempo=self.tempo, time_offset=self.time_offset,
        )

    def to_dict(self) -> dict:
        """Serialize for JSON / WebSocket transmission."""
        return {
            "notes": [
                {"start": n.start_time, "end": n.end_time, "pitch": n.midi_pitch,
                 "velocity": n.velocity, "confidence": n.confidence, "name": n.note_name}
                for n in self.notes
            ],
            "chords": [
                {"start": c.start_time, "end": c.end_time, "name": c.chord_name,
                 "root": c.root, "pitches": c.pitches, "confidence": c.confidence}
                for c in self.chords
            ],
            "tempo": self.tempo,
            "time_offset": self.time_offset,
        }


class TranscriberPlugin(ABC):
    """Abstract base class for transcription plugins."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Brief description of the plugin."""
        ...

    @abstractmethod
    def supports_realtime(self) -> bool:
        """Whether this plugin supports chunk-by-chunk real-time processing."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int) -> TranscriptionResult:
        """Transcribe a complete audio buffer.

        Args:
            audio: mono audio signal, float32, normalized to [-1, 1]
            sr: sample rate in Hz
        """
        ...

    def transcribe_chunk(self, chunk: np.ndarray, sr: int,
                         time_offset: float = 0.0) -> TranscriptionResult:
        """Transcribe a single chunk for real-time use.

        Default implementation falls back to transcribe() with time offset.
        Override for optimized streaming behavior.
        """
        result = self.transcribe(chunk, sr)
        if time_offset > 0:
            result = result.shift_time(time_offset)
        return result

    def info(self) -> dict:
        """Return plugin metadata for the UI."""
        return {
            "name": self.name(),
            "description": self.description(),
            "supports_realtime": self.supports_realtime(),
        }
