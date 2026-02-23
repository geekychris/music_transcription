"""Export transcription results to MusicXML."""

import logging
from pathlib import Path

from music21 import stream, note, chord, tempo, metadata, meter

from app.transcription.base import TranscriptionResult
from app.config import OUTPUT_DIR, DEFAULT_MIDI_TEMPO

logger = logging.getLogger(__name__)

# Valid quarter-length values that LilyPond can express
_VALID_QUARTER_LENGTHS = [
    0.125, 0.1875, 0.25, 0.375, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
]


def _quantize_quarter_length(ql: float) -> float:
    """Snap a quarterLength to the nearest standard musical duration."""
    ql = max(0.125, ql)
    return min(_VALID_QUARTER_LENGTHS, key=lambda v: abs(v - ql))


def to_music21_stream(result: TranscriptionResult,
                      title: str = "Transcription") -> stream.Score:
    """Convert a TranscriptionResult to a music21 Score."""
    score = stream.Score()

    # Metadata
    md = metadata.Metadata()
    md.title = title
    md.composer = "Music Transposer"
    score.metadata = md

    part = stream.Part()
    part.partName = "Transcription"

    # Time signature (default 4/4)
    part.append(meter.TimeSignature("4/4"))

    # Tempo
    bpm = result.tempo or DEFAULT_MIDI_TEMPO
    part.append(tempo.MetronomeMark(number=bpm))

    if result.notes:
        # Group notes by start time to detect chords
        from collections import defaultdict
        time_groups = defaultdict(list)
        for n in result.notes:
            # Quantize start time to nearest 0.05s to group simultaneous notes
            quantized = round(n.start_time * 20) / 20
            time_groups[quantized].append(n)

        for t in sorted(time_groups.keys()):
            group = time_groups[t]
            if len(group) == 1:
                n = group[0]
                m21_note = note.Note(n.midi_pitch)
                m21_note.quarterLength = _quantize_quarter_length(n.duration * (bpm / 60))
                m21_note.volume.velocity = n.velocity
                m21_note.offset = t * (bpm / 60)
                part.append(m21_note)
            else:
                # Multiple simultaneous notes = chord
                pitches = [n.midi_pitch for n in group]
                duration = max(n.duration for n in group)
                m21_chord = chord.Chord(pitches)
                m21_chord.quarterLength = _quantize_quarter_length(duration * (bpm / 60))
                m21_chord.volume.velocity = max(n.velocity for n in group)
                m21_chord.offset = t * (bpm / 60)
                part.append(m21_chord)

    score.append(part)
    return score


def write_musicxml(result: TranscriptionResult, path: str | Path | None = None,
                   title: str = "Transcription") -> Path:
    """Write transcription to a MusicXML file.

    Returns path to the written file.
    """
    if path is None:
        path = OUTPUT_DIR / "transcription.musicxml"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    score = to_music21_stream(result, title)
    score.write("musicxml", fp=str(path))
    logger.info("MusicXML written: %s", path)
    return path
