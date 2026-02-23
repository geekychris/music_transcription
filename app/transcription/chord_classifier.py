"""Chord classification from detected notes using template matching."""

import numpy as np
from app.transcription.base import NoteEvent, ChordEvent

# Chord templates: intervals from root (in semitones)
# Display names use standard musician notation:
#   Major triad = just the root ("C"), minor = "m" ("Am"), etc.
CHORD_TEMPLATES = {
    "":       [0, 4, 7],         # major triad: C, D, G
    "m":      [0, 3, 7],         # minor triad: Am, Dm
    "dim":    [0, 3, 6],         # diminished: Bdim
    "aug":    [0, 4, 8],         # augmented: Caug
    "sus2":   [0, 2, 7],         # suspended 2nd: Csus2
    "sus4":   [0, 5, 7],         # suspended 4th: Csus4
    "7":      [0, 4, 7, 10],     # dominant 7th: G7
    "maj7":   [0, 4, 7, 11],     # major 7th: Cmaj7
    "m7":     [0, 3, 7, 10],     # minor 7th: Am7
    "dim7":   [0, 3, 6, 9],      # diminished 7th: Bdim7
    "m7b5":   [0, 3, 6, 10],     # half-diminished: Bm7b5
    "add9":   [0, 2, 4, 7],      # add 9: Cadd9
    "6":      [0, 4, 7, 9],      # major 6th: C6
    "m6":     [0, 3, 7, 9],      # minor 6th: Am6
    "9":      [0, 2, 4, 7, 10],  # dominant 9th: C9
}

# Simplicity bias: basic triads are preferred over complex chords.
# The penalty is subtracted from the cosine similarity score.
# This prevents sus2/sus4/dim7/add9 from winning when a basic triad
# is an equally good (or nearly as good) explanation.
_COMPLEXITY_PENALTY = {
    "":     0.00,   # major — no penalty
    "m":    0.00,   # minor — no penalty
    "7":    0.03,   # dom7 — very small penalty (common)
    "dim":  0.04,
    "aug":  0.04,
    "sus2": 0.06,
    "sus4": 0.06,
    "maj7": 0.05,
    "m7":   0.04,
    "dim7": 0.06,
    "m7b5": 0.06,
    "add9": 0.07,
    "6":    0.05,
    "m6":   0.05,
    "9":    0.08,
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_class_vector(pitches: list[int]) -> np.ndarray:
    """Convert a list of MIDI pitches to a 12-bin pitch class vector."""
    pcv = np.zeros(12, dtype=np.float32)
    for p in pitches:
        pcv[p % 12] += 1.0
    return pcv


def _template_vector(intervals: list[int]) -> np.ndarray:
    """Convert chord template intervals to a 12-bin vector."""
    v = np.zeros(12, dtype=np.float32)
    for i in intervals:
        v[i % 12] = 1.0
    return v


def classify_chord(pitches: list[int], min_notes: int = 2) -> tuple[str, int, float] | None:
    """Classify a set of MIDI pitches as a chord.

    Returns (chord_name, root_pitch_class, confidence) or None if < min_notes.
    Uses cosine similarity with a simplicity bias that prefers basic triads.
    """
    if len(pitches) < min_notes:
        return None

    pcv = _pitch_class_vector(pitches)
    best_score = -1.0
    best_name = "?"
    best_root = 0

    for root in range(12):
        # Rotate the pitch class vector so root is at index 0
        rotated = np.roll(pcv, -root)
        for tmpl_name, intervals in CHORD_TEMPLATES.items():
            tmpl = _template_vector(intervals)
            # Cosine similarity
            dot = np.dot(rotated, tmpl)
            norm = np.linalg.norm(rotated) * np.linalg.norm(tmpl)
            if norm == 0:
                continue
            score = dot / norm
            # Apply simplicity bias — prefer basic triads
            score -= _COMPLEXITY_PENALTY.get(tmpl_name, 0.05)
            if score > best_score:
                best_score = score
                best_name = f"{NOTE_NAMES[root]}{tmpl_name}"
                best_root = root

    if best_score < 0.5:
        return None

    return best_name, best_root, float(best_score)


def group_notes_into_chords(notes: list[NoteEvent],
                            time_window: float = 0.5,
                            min_notes: int = 2,
                            min_pitch_classes: int = 3) -> list[ChordEvent]:
    """Group notes into chords by onset-time clustering.

    Notes whose onset times fall within ``time_window`` of the group's
    first onset are considered part of the same chord.  Additionally,
    notes that are still sustaining from earlier groups are included
    to capture arpeggiated chords and bass-melody combinations.

    Args:
        time_window: max seconds between first and last onset in a group.
        min_notes: minimum number of notes in a group to attempt classification.
        min_pitch_classes: minimum distinct pitch classes required.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    chords = []

    # Build onset-based groups first
    groups: list[list[NoteEvent]] = []
    current_group: list[NoteEvent] = [sorted_notes[0]]
    group_onset = sorted_notes[0].start_time

    for note in sorted_notes[1:]:
        if note.start_time <= group_onset + time_window:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
            group_onset = note.start_time
    groups.append(current_group)

    # For each group, also include notes from earlier groups that are still
    # sustaining at the current group's onset time.  This captures arpeggiated
    # bass notes and held tones that define the harmonic context.
    for group in groups:
        group_start = min(n.start_time for n in group)
        enriched_pitches = set()
        # Add sustaining notes from ALL previous notes
        for n in sorted_notes:
            if n.start_time > group_start:
                break
            if n.end_time > group_start:  # still sounding
                enriched_pitches.add(n.midi_pitch)
        # Add the group's own notes
        for n in group:
            enriched_pitches.add(n.midi_pitch)

        pitches = list(enriched_pitches)
        distinct_pcs = len({p % 12 for p in pitches})
        if len(pitches) < min_notes or distinct_pcs < min_pitch_classes:
            continue

        result = classify_chord(pitches, min_notes)
        if result:
            chord_name, root, confidence = result
            start = min(n.start_time for n in group)
            end = max(n.end_time for n in group)
            chords.append(ChordEvent(
                start_time=start, end_time=end,
                chord_name=chord_name, root=root,
                pitches=pitches, confidence=confidence,
            ))

    return chords


def _emit_chord(group: list[NoteEvent], chords: list[ChordEvent],
                min_notes: int, min_pitch_classes: int = 2):
    """Classify a group of notes and append chord if valid."""
    pitches = [n.midi_pitch for n in group]
    # Require enough distinct pitch classes to identify a real chord
    distinct_pcs = len({p % 12 for p in pitches})
    if distinct_pcs < min_pitch_classes:
        return
    result = classify_chord(pitches, min_notes)
    if result:
        chord_name, root, confidence = result
        start = min(n.start_time for n in group)
        end = max(n.end_time for n in group)
        chords.append(ChordEvent(
            start_time=start,
            end_time=end,
            chord_name=chord_name,
            root=root,
            pitches=pitches,
            confidence=confidence,
        ))
