"""Write transcription results to standard MIDI files."""

import logging
from pathlib import Path

import mido

from app.transcription.base import TranscriptionResult
from app.config import DEFAULT_MIDI_TEMPO, OUTPUT_DIR

logger = logging.getLogger(__name__)


def write_midi(result: TranscriptionResult, path: str | Path | None = None,
               tempo: float | None = None) -> Path:
    """Write a TranscriptionResult to a MIDI file.

    Args:
        result: transcription result with notes
        path: output file path (defaults to output_files/transcription.mid)
        tempo: BPM override (uses result.tempo or default if not specified)

    Returns:
        Path to the written MIDI file.
    """
    if path is None:
        path = OUTPUT_DIR / "transcription.mid"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bpm = tempo or result.tempo or DEFAULT_MIDI_TEMPO
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    us_per_beat = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))

    # Track name
    track.append(mido.MetaMessage("track_name", name="Music Transposer", time=0))

    if not result.notes:
        logger.warning("No notes to write to MIDI")
        mid.save(str(path))
        return path

    # Convert note events to MIDI messages sorted by absolute time
    events = []
    for note in result.notes:
        tick_on = int(mido.second2tick(note.start_time, 480, us_per_beat))
        tick_off = int(mido.second2tick(note.end_time, 480, us_per_beat))
        pitch = int(note.midi_pitch) & 0x7F
        vel = max(1, int(note.velocity) & 0x7F)  # ensure velocity > 0 for note_on
        events.append((tick_on, "note_on", pitch, vel))
        events.append((tick_off, "note_off", pitch, 0))

    # Sort by absolute tick time
    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    # Convert to delta times
    last_tick = 0
    for tick, msg_type, pitch, vel in events:
        delta = max(0, tick - last_tick)
        track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta))
        last_tick = tick

    mid.save(str(path))
    logger.info("MIDI file written: %s (%d notes, %.0f BPM)", path, len(result.notes), bpm)
    return path
