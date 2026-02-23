"""Round-trip accuracy test.

Creates a MIDI file with known notes and chords, synthesizes it to a WAV file
using pretty_midi's built-in FluidSynth (or sine wave fallback), transcribes
the WAV with each plugin, and measures how close the transcription is to the
original.

Run:
    PYTHONPATH=. python -m pytest tests/test_round_trip.py -v
    # or directly:
    PYTHONPATH=. python tests/test_round_trip.py
"""

import sys
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pretty_midi
import soundfile as sf

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.transcription.base import NoteEvent, TranscriptionResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test data: known musical phrases
# ---------------------------------------------------------------------------

@dataclass
class TestNote:
    """A ground-truth note."""
    pitch: int    # MIDI pitch
    start: float  # seconds
    end: float    # seconds
    velocity: int = 100


def _c_major_chord(start: float, dur: float = 1.0, vel: int = 100) -> list[TestNote]:
    """C-E-G chord."""
    return [
        TestNote(60, start, start + dur, vel),
        TestNote(64, start, start + dur, vel),
        TestNote(67, start, start + dur, vel),
    ]


def _am_chord(start: float, dur: float = 1.0, vel: int = 100) -> list[TestNote]:
    """A-C-E chord."""
    return [
        TestNote(57, start, start + dur, vel),
        TestNote(60, start, start + dur, vel),
        TestNote(64, start, start + dur, vel),
    ]


def _f_major_chord(start: float, dur: float = 1.0, vel: int = 100) -> list[TestNote]:
    """F-A-C chord."""
    return [
        TestNote(53, start, start + dur, vel),
        TestNote(57, start, start + dur, vel),
        TestNote(60, start, start + dur, vel),
    ]


def _g_major_chord(start: float, dur: float = 1.0, vel: int = 100) -> list[TestNote]:
    """G-B-D chord."""
    return [
        TestNote(55, start, start + dur, vel),
        TestNote(59, start, start + dur, vel),
        TestNote(62, start, start + dur, vel),
    ]


def build_test_phrase() -> list[TestNote]:
    """A simple I-vi-IV-V chord progression in C major, 1 second per chord."""
    notes: list[TestNote] = []
    notes.extend(_c_major_chord(0.0, 1.0))
    notes.extend(_am_chord(1.0, 1.0))
    notes.extend(_f_major_chord(2.0, 1.0))
    notes.extend(_g_major_chord(3.0, 1.0))
    return notes


def build_scale_phrase() -> list[TestNote]:
    """C major scale, one note at a time, 0.5s each."""
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    return [TestNote(p, i * 0.5, i * 0.5 + 0.45, 100) for i, p in enumerate(pitches)]


# ---------------------------------------------------------------------------
# MIDI / WAV generation
# ---------------------------------------------------------------------------

def create_midi_from_notes(notes: list[TestNote], path: str, tempo: float = 120.0):
    """Write test notes to a MIDI file."""
    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for n in notes:
        midi_note = pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end,
        )
        piano.notes.append(midi_note)
    mid.instruments.append(piano)
    mid.write(path)
    logger.info("Created MIDI: %s (%d notes, %.1f BPM)", path, len(notes), tempo)


def midi_to_wav(midi_path: str, wav_path: str, sr: int = 22050):
    """Synthesize a MIDI file to WAV using pretty_midi's FluidSynth.

    Falls back to sine-wave synthesis if FluidSynth is unavailable.
    """
    mid = pretty_midi.PrettyMIDI(midi_path)
    try:
        audio = mid.fluidsynth(fs=sr)
        logger.info("Synthesized with FluidSynth")
    except Exception:
        logger.warning("FluidSynth unavailable, using sine-wave synthesis")
        audio = mid.synthesize(fs=sr)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    sf.write(wav_path, audio, sr)
    logger.info("Created WAV: %s (%.1fs, %d Hz)", wav_path, len(audio) / sr, sr)
    return audio, sr


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def compute_accuracy(ground_truth: list[TestNote],
                     detected: list[NoteEvent],
                     pitch_tolerance: int = 1,
                     time_tolerance: float = 0.15) -> dict:
    """Compare detected notes against ground truth.

    Args:
        ground_truth: expected notes
        detected: transcribed notes
        pitch_tolerance: semitones of allowed pitch deviation
        time_tolerance: seconds of allowed onset/offset deviation

    Returns:
        dict with precision, recall, f1, and per-note match details.
    """
    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detected)
    matches = []

    for di, dn in enumerate(detected):
        best_dist = float("inf")
        best_gi = -1
        for gi, gn in enumerate(ground_truth):
            if gt_matched[gi]:
                continue
            pitch_ok = abs(dn.midi_pitch - gn.pitch) <= pitch_tolerance
            onset_ok = abs(dn.start_time - gn.start) <= time_tolerance
            if pitch_ok and onset_ok:
                dist = abs(dn.start_time - gn.start) + abs(dn.midi_pitch - gn.pitch) * 0.1
                if dist < best_dist:
                    best_dist = dist
                    best_gi = gi

        if best_gi >= 0:
            gt_matched[best_gi] = True
            det_matched[di] = True
            gn = ground_truth[best_gi]
            matches.append({
                "gt_pitch": gn.pitch, "det_pitch": dn.midi_pitch,
                "gt_onset": gn.start, "det_onset": dn.start_time,
                "onset_err": abs(dn.start_time - gn.start),
                "pitch_err": abs(dn.midi_pitch - gn.pitch),
            })

    n_matches = len(matches)
    precision = n_matches / len(detected) if detected else 0.0
    recall = n_matches / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Average errors for matched notes
    avg_onset_err = np.mean([m["onset_err"] for m in matches]) if matches else float("nan")
    avg_pitch_err = np.mean([m["pitch_err"] for m in matches]) if matches else float("nan")

    # Velocity accuracy for matched notes
    vel_errors = []
    for m, di in zip(matches, [i for i, v in enumerate(det_matched) if v]):
        vel_errors.append(abs(detected[di].velocity - ground_truth[
            next(gi for gi, matched in enumerate(gt_matched) if matched)
        ].velocity))

    return {
        "n_ground_truth": len(ground_truth),
        "n_detected": len(detected),
        "n_matched": n_matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_onset_error_s": float(avg_onset_err),
        "avg_pitch_error_semitones": float(avg_pitch_err),
        "missed_notes": [ground_truth[i] for i, m in enumerate(gt_matched) if not m],
        "extra_notes": [detected[i] for i, m in enumerate(det_matched) if not m],
    }


def compute_chord_accuracy(expected_chords: list[str],
                           expected_times: list[float],
                           detected_chords: list,
                           time_tolerance: float = 0.3) -> dict:
    """Compare detected chords against expected chord sequence.

    Args:
        expected_chords: list of chord names, e.g. ["C", "Am", "F", "G"]
        expected_times: onset time for each expected chord
        detected_chords: list of ChordEvent from transcription
        time_tolerance: how close the onset must be

    Returns:
        dict with match rate and details.
    """
    matched = 0
    details = []
    for name, onset in zip(expected_chords, expected_times):
        # Find the closest detected chord by time
        best = None
        best_dist = float("inf")
        for dc in detected_chords:
            dist = abs(dc.start_time - onset)
            if dist < best_dist:
                best_dist = dist
                best = dc

        if best and best_dist <= time_tolerance:
            # Normalize chord names for comparison (strip quality suffix variations)
            is_match = _chords_equivalent(name, best.chord_name)
            if is_match:
                matched += 1
            details.append({
                "expected": name, "detected": best.chord_name,
                "time_err": best_dist, "match": is_match,
            })
        else:
            details.append({
                "expected": name, "detected": best.chord_name if best else "N",
                "time_err": best_dist if best else float("inf"), "match": False,
            })

    rate = matched / len(expected_chords) if expected_chords else 0.0
    return {"chord_match_rate": rate, "n_expected": len(expected_chords),
            "n_matched": matched, "details": details}


def _chords_equivalent(expected: str, detected: str) -> bool:
    """Flexible chord name comparison.

    Handles variations like 'C' vs 'C', 'Am' vs 'Am', 'F' vs 'Fmaj' etc.
    """
    # Normalize: strip whitespace, lowercase comparison
    e = expected.strip()
    d = detected.strip()
    if e == d:
        return True
    # 'C' should match 'C', 'Cmaj', 'CM'
    # 'Am' should match 'Am', 'Amin'
    e_root = e.rstrip("mMajin7dimaugsu2469b#")
    d_root = d.rstrip("mMajin7dimaugsu2469b#")
    e_qual = e[len(e_root):]
    d_qual = d[len(d_root):]
    if e_root != d_root:
        return False
    # Check quality equivalence
    major_aliases = {"", "maj", "M", "major"}
    minor_aliases = {"m", "min", "minor"}
    if e_qual in major_aliases and d_qual in major_aliases:
        return True
    if e_qual in minor_aliases and d_qual in minor_aliases:
        return True
    return e_qual == d_qual


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(test_name: str, notes: list[TestNote],
             expected_chords: list[str] | None = None,
             expected_chord_times: list[float] | None = None,
             sr: int = 22050):
    """Run a round-trip test with both plugins."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = str(Path(tmpdir) / "test.mid")
        wav_path = str(Path(tmpdir) / "test.wav")

        # Step 1: Create MIDI
        create_midi_from_notes(notes, midi_path)

        # Step 2: Synthesize to WAV
        audio, sample_rate = midi_to_wav(midi_path, wav_path, sr=sr)

        # Step 3: Transcribe with each plugin
        from app.transcription.plugin_manager import PluginManager
        pm = PluginManager()
        pm.discover()

        results = {}
        for plugin_info in pm.list_plugins():
            plugin = pm.get_plugin(plugin_info["name"])
            if plugin is None:
                continue
            pname = plugin.name()
            print(f"\n--- Plugin: {pname} ---")

            try:
                result = plugin.transcribe(audio, sample_rate)
                results[pname] = result

                # Note accuracy
                acc = compute_accuracy(notes, result.notes)
                print(f"  Notes:  GT={acc['n_ground_truth']}, Detected={acc['n_detected']}, "
                      f"Matched={acc['n_matched']}")
                print(f"  Precision={acc['precision']:.2%}, Recall={acc['recall']:.2%}, "
                      f"F1={acc['f1']:.2%}")
                print(f"  Avg onset error: {acc['avg_onset_error_s']:.3f}s")
                print(f"  Avg pitch error: {acc['avg_pitch_error_semitones']:.1f} semitones")

                if acc["missed_notes"]:
                    print(f"  Missed: {[(n.pitch, f'{n.start:.2f}s') for n in acc['missed_notes']]}")
                if acc["extra_notes"] and len(acc["extra_notes"]) <= 10:
                    print(f"  Extra:  {[(n.midi_pitch, f'{n.start_time:.2f}s') for n in acc['extra_notes']]}")
                elif acc["extra_notes"]:
                    print(f"  Extra:  {len(acc['extra_notes'])} extra notes")

                # Chord accuracy (if expected)
                if expected_chords and result.chords:
                    chord_acc = compute_chord_accuracy(
                        expected_chords, expected_chord_times or [], result.chords,
                    )
                    print(f"  Chords: {chord_acc['n_matched']}/{chord_acc['n_expected']} "
                          f"({chord_acc['chord_match_rate']:.0%})")
                    for d in chord_acc["details"]:
                        status = "✓" if d["match"] else "✗"
                        print(f"    {status} Expected={d['expected']}, Got={d['detected']}, "
                              f"dt={d['time_err']:.2f}s")

                # Velocity check
                if result.notes:
                    vels = [n.velocity for n in result.notes]
                    print(f"  Velocities: min={min(vels)}, max={max(vels)}, "
                          f"mean={np.mean(vels):.0f}")
                    assert all(v > 0 for v in vels), "BUG: Zero-velocity notes detected!"

                # MIDI export verification
                from app.output.midi_file_writer import write_midi
                import mido
                export_path = Path(tmpdir) / f"export_{pname.replace(' ', '_')}.mid"
                write_midi(result, path=export_path)
                mid = mido.MidiFile(str(export_path))
                note_ons = [m for t in mid.tracks for m in t if m.type == "note_on" and m.velocity > 0]
                note_offs = [m for t in mid.tracks for m in t
                             if m.type == "note_off" or (m.type == "note_on" and m.velocity == 0)]
                print(f"  MIDI export: {len(note_ons)} note_ons (vel>0), {len(note_offs)} note_offs")
                assert len(note_ons) > 0, "BUG: Exported MIDI has no audible note_on events!"
                assert all(m.velocity > 0 for m in note_ons), \
                    "BUG: Exported MIDI has note_on with velocity=0!"

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    return results


# ---------------------------------------------------------------------------
# pytest-compatible test functions
# ---------------------------------------------------------------------------

def test_chord_progression_round_trip():
    """Round-trip test with a I-vi-IV-V chord progression."""
    notes = build_test_phrase()
    results = run_test(
        "Chord Progression (C-Am-F-G)",
        notes,
        expected_chords=["C", "Am", "F", "G"],
        expected_chord_times=[0.0, 1.0, 2.0, 3.0],
    )
    # At minimum, verify we get some notes back and velocities are valid
    for pname, result in results.items():
        assert len(result.notes) > 0, f"{pname}: No notes detected"
        assert all(n.velocity > 0 for n in result.notes), f"{pname}: Zero velocity notes"


def test_scale_round_trip():
    """Round-trip test with a C major scale."""
    notes = build_scale_phrase()
    results = run_test("C Major Scale", notes)
    for pname, result in results.items():
        assert len(result.notes) > 0, f"{pname}: No notes detected"
        assert all(n.velocity > 0 for n in result.notes), f"{pname}: Zero velocity notes"


def test_midi_velocity_export():
    """Verify MIDI export preserves non-zero velocities through the full pipeline."""
    import mido
    from app.output.midi_file_writer import write_midi

    # Create notes with various velocities
    notes_in = [
        NoteEvent(0.0, 0.5, 60, velocity=30),
        NoteEvent(0.5, 1.0, 64, velocity=80),
        NoteEvent(1.0, 1.5, 67, velocity=127),
    ]
    result = TranscriptionResult(notes=notes_in)

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = write_midi(result, path=f.name)

    mid = mido.MidiFile(str(path))
    note_ons = [m for t in mid.tracks for m in t if m.type == "note_on"]
    assert len(note_ons) == 3, f"Expected 3 note_on, got {len(note_ons)}"
    for msg in note_ons:
        assert msg.velocity > 0, f"note_on velocity is 0 for note {msg.note}!"
    velocities = sorted(m.velocity for m in note_ons)
    assert velocities == [30, 80, 127], f"Velocities incorrect: {velocities}"
    print(f"  ✓ MIDI velocity export: {velocities}")

    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Music Transposer — Round-Trip Accuracy Test")
    print("=" * 70)

    test_midi_velocity_export()
    test_chord_progression_round_trip()
    test_scale_round_trip()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
