"""Test chord detection with Für Elise opening section.

Creates a MIDI file of the famous opening theme, synthesizes to WAV,
runs both plugins, and compares detected chords against known harmony.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.transcription.plugin_manager import PluginManager
from tests.test_round_trip import compute_accuracy, compute_chord_accuracy, TestNote


def create_fur_elise_midi(path: str, tempo: float = 72.0):
    """Create the opening ~16 bars of Für Elise as MIDI.

    Key: A minor, 3/8 time
    Known chord progression:
      Am - E - Am - E - Am - C - G - Am - E - Am
    """
    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")

    # Für Elise opening — note data as (pitch, start_beat, duration_beats, velocity)
    # Using eighth-note = 1 beat unit in 3/8 time
    # BPM is per quarter note, so eighth note duration = 60/tempo/2 seconds
    eighth = 60.0 / tempo / 2.0  # duration of one eighth note in seconds

    # ---- Section A (bars 1-8): Am - E - Am - E - Am ----
    # Bar 1-2: The famous trill motif (pickup + bar 1-2)
    # E5-D#5-E5-D#5-E5-B4-D5-C5 (all eighth notes, pickup into Am)
    motif_notes = [
        # Pickup + Bar 1: E5-D#5-E5-D#5-E5-B4-D5-C5
        (76, 0, 1, 80),    # E5
        (75, 1, 1, 75),    # D#5
        (76, 2, 1, 80),    # E5
        (75, 3, 1, 75),    # D#5
        (76, 4, 1, 80),    # E5
        (71, 5, 1, 75),    # B4
        (74, 6, 1, 78),    # D5
        (72, 7, 1, 78),    # C5
        # Bar 2: Am chord - A4-C5-E5 (resolve)
        (69, 8, 3, 85),    # A4 (dotted quarter)
        # Left hand: A2-E3-A3
        (45, 8, 3, 60),    # A2
        (52, 9, 1, 55),    # E3
        (57, 10, 1, 55),   # A3

        # Bar 3: E chord bass + melody continues
        # C4-E4-B4 melody pickup
        (48, 11, 1, 55),   # C4 (melody pickup)
        (52, 12, 1, 60),   # E4
        (71, 12, 3, 80),   # B4 (held)
        # Left hand: E2-G#3-B3
        (40, 12, 3, 60),   # E2
        (56, 13, 1, 55),   # G#3
        (59, 14, 1, 55),   # B3

        # Bar 4: melody - C5
        (72, 15, 3, 82),   # C5
        # Left hand: A2-E3-A3
        (45, 15, 3, 60),   # A2
        (52, 16, 1, 55),   # E3
        (57, 17, 1, 55),   # A3

        # Bar 5-6: Repeat of trill motif
        (76, 18, 1, 80),   # E5
        (75, 19, 1, 75),   # D#5
        (76, 20, 1, 80),   # E5
        (75, 21, 1, 75),   # D#5
        (76, 22, 1, 80),   # E5
        (71, 23, 1, 75),   # B4
        (74, 24, 1, 78),   # D5
        (72, 25, 1, 78),   # C5

        # Bar 6: Am resolve
        (69, 26, 3, 85),   # A4
        (45, 26, 3, 60),   # A2
        (52, 27, 1, 55),   # E3
        (57, 28, 1, 55),   # A3

        # Bar 7: E resolve
        (48, 29, 1, 55),   # C4
        (52, 30, 1, 60),   # E4
        (69, 30, 3, 80),   # A4
        (40, 30, 3, 60),   # E2
        (56, 31, 1, 55),   # G#3
        (59, 32, 1, 55),   # B3

        # Bar 8: Am final resolve of A section
        (69, 33, 3, 85),   # A4
        (45, 33, 3, 60),   # A2
        (52, 34, 1, 55),   # E3
        (57, 35, 1, 55),   # A3

        # ---- Section B (bars 9-12): C - G - Am - E ----
        # Bar 9: C major — E4-G4-C5
        (71, 36, 1, 78),   # B4 pickup
        (72, 37, 3, 85),   # C5
        (64, 37, 3, 70),   # E4
        (67, 37, 3, 70),   # G4
        (48, 37, 3, 60),   # C3
        (52, 38, 1, 55),   # E3
        (55, 39, 1, 55),   # G3

        # Bar 10: G major — D5
        (74, 40, 3, 82),   # D5
        (62, 40, 3, 70),   # D4
        (67, 40, 3, 70),   # G4
        (43, 40, 3, 60),   # G2
        (50, 41, 1, 55),   # D3
        (55, 42, 1, 55),   # G3

        # Bar 11: Am — E5
        (76, 43, 3, 85),   # E5
        (64, 43, 3, 70),   # E4
        (69, 43, 3, 70),   # A4
        (45, 43, 3, 60),   # A2
        (52, 44, 1, 55),   # E3
        (57, 45, 1, 55),   # A3

        # Bar 12: E — D5-C5-B4 descending
        (74, 46, 1, 78),   # D5
        (72, 47, 1, 78),   # C5
        (71, 48, 3, 80),   # B4
        (40, 48, 3, 60),   # E2
        (56, 49, 1, 55),   # G#3
        (59, 50, 1, 55),   # B3

        # ---- Return to A (bars 13-16): Am - E - Am ----
        # Bar 13-14: trill motif again
        (76, 51, 1, 80),   # E5
        (75, 52, 1, 75),   # D#5
        (76, 53, 1, 80),   # E5
        (75, 54, 1, 75),   # D#5
        (76, 55, 1, 80),   # E5
        (71, 56, 1, 75),   # B4
        (74, 57, 1, 78),   # D5
        (72, 58, 1, 78),   # C5

        # Bar 15: Am
        (69, 59, 3, 85),   # A4
        (45, 59, 3, 60),   # A2
        (52, 60, 1, 55),   # E3
        (57, 61, 1, 55),   # A3

        # Bar 16: E -> Am resolve
        (48, 62, 1, 55),   # C4
        (52, 63, 1, 60),   # E4
        (69, 63, 3, 80),   # A4
        (40, 63, 3, 60),   # E2
        (56, 64, 1, 55),   # G#3
        (59, 65, 1, 55),   # B3

        # Final Am
        (69, 66, 6, 90),   # A4 (held)
        (45, 66, 6, 65),   # A2 (held)
        (57, 66, 6, 60),   # A3 (held)
        (60, 66, 6, 60),   # C4 (held)
        (64, 66, 6, 60),   # E4 (held)
    ]

    for pitch, start_beat, dur_beats, vel in motif_notes:
        start_s = start_beat * eighth
        end_s = (start_beat + dur_beats) * eighth
        piano.notes.append(pretty_midi.Note(
            velocity=vel, pitch=pitch, start=start_s, end=end_s,
        ))

    mid.instruments.append(piano)
    mid.write(path)

    total_notes = len(motif_notes)
    duration = (66 + 6) * eighth
    print(f"Created Für Elise MIDI: {total_notes} notes, {duration:.1f}s, {tempo} BPM")
    return mid


def get_ground_truth_notes(eighth: float) -> list[TestNote]:
    """Return TestNote versions matching the MIDI we created."""
    motif_notes = [
        (76, 0, 1, 80), (75, 1, 1, 75), (76, 2, 1, 80), (75, 3, 1, 75),
        (76, 4, 1, 80), (71, 5, 1, 75), (74, 6, 1, 78), (72, 7, 1, 78),
        (69, 8, 3, 85), (45, 8, 3, 60), (52, 9, 1, 55), (57, 10, 1, 55),
        (48, 11, 1, 55), (52, 12, 1, 60), (71, 12, 3, 80),
        (40, 12, 3, 60), (56, 13, 1, 55), (59, 14, 1, 55),
        (72, 15, 3, 82), (45, 15, 3, 60), (52, 16, 1, 55), (57, 17, 1, 55),
        (76, 18, 1, 80), (75, 19, 1, 75), (76, 20, 1, 80), (75, 21, 1, 75),
        (76, 22, 1, 80), (71, 23, 1, 75), (74, 24, 1, 78), (72, 25, 1, 78),
        (69, 26, 3, 85), (45, 26, 3, 60), (52, 27, 1, 55), (57, 28, 1, 55),
        (48, 29, 1, 55), (52, 30, 1, 60), (69, 30, 3, 80),
        (40, 30, 3, 60), (56, 31, 1, 55), (59, 32, 1, 55),
        (69, 33, 3, 85), (45, 33, 3, 60), (52, 34, 1, 55), (57, 35, 1, 55),
        (71, 36, 1, 78), (72, 37, 3, 85), (64, 37, 3, 70), (67, 37, 3, 70),
        (48, 37, 3, 60), (52, 38, 1, 55), (55, 39, 1, 55),
        (74, 40, 3, 82), (62, 40, 3, 70), (67, 40, 3, 70),
        (43, 40, 3, 60), (50, 41, 1, 55), (55, 42, 1, 55),
        (76, 43, 3, 85), (64, 43, 3, 70), (69, 43, 3, 70),
        (45, 43, 3, 60), (52, 44, 1, 55), (57, 45, 1, 55),
        (74, 46, 1, 78), (72, 47, 1, 78), (71, 48, 3, 80),
        (40, 48, 3, 60), (56, 49, 1, 55), (59, 50, 1, 55),
        (76, 51, 1, 80), (75, 52, 1, 75), (76, 53, 1, 80), (75, 54, 1, 75),
        (76, 55, 1, 80), (71, 56, 1, 75), (74, 57, 1, 78), (72, 58, 1, 78),
        (69, 59, 3, 85), (45, 59, 3, 60), (52, 60, 1, 55), (57, 61, 1, 55),
        (48, 62, 1, 55), (52, 63, 1, 60), (69, 63, 3, 80),
        (40, 63, 3, 60), (56, 64, 1, 55), (59, 65, 1, 55),
        (69, 66, 6, 90), (45, 66, 6, 65), (57, 66, 6, 60),
        (60, 66, 6, 60), (64, 66, 6, 60),
    ]
    return [
        TestNote(pitch=p, start=s * eighth, end=(s + d) * eighth, velocity=v)
        for p, s, d, v in motif_notes
    ]


# Expected chord progression with approximate onset times (in eighth-note beats)
EXPECTED_CHORDS = [
    ("Am", 8),    # Bar 2
    ("E",  12),   # Bar 3
    ("Am", 15),   # Bar 4
    ("Am", 26),   # Bar 6
    ("E",  30),   # Bar 7
    ("Am", 33),   # Bar 8
    ("C",  37),   # Bar 9
    ("G",  40),   # Bar 10
    ("Am", 43),   # Bar 11
    ("E",  48),   # Bar 12
    ("Am", 59),   # Bar 15
    ("E",  63),   # Bar 16
    ("Am", 66),   # Final
]


def run_fur_elise_test():
    """Full Für Elise chord detection test."""
    tempo = 72.0
    eighth = 60.0 / tempo / 2.0

    with tempfile.TemporaryDirectory() as tmpdir:
        midi_path = str(Path(tmpdir) / "fur_elise.mid")
        wav_path = str(Path(tmpdir) / "fur_elise.wav")

        # 1. Create MIDI
        create_fur_elise_midi(midi_path, tempo=tempo)

        # 2. Synthesize to WAV
        mid = pretty_midi.PrettyMIDI(midi_path)
        try:
            audio = mid.fluidsynth(fs=22050)
            print("Synthesized with FluidSynth")
        except Exception:
            print("FluidSynth unavailable, using sine-wave synthesis")
            audio = mid.synthesize(fs=22050)

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        sf.write(wav_path, audio, 22050)
        print(f"WAV: {len(audio)/22050:.1f}s\n")

        # 3. Load plugins
        pm = PluginManager()
        pm.discover()

        expected_chord_names = [c[0] for c in EXPECTED_CHORDS]
        expected_chord_times = [c[1] * eighth for c in EXPECTED_CHORDS]
        gt_notes = get_ground_truth_notes(eighth)

        # 4. Test each plugin
        for plugin_info in pm.list_plugins():
            plugin = pm.get_plugin(plugin_info["name"])
            if plugin is None:
                continue
            pname = plugin.name()
            print(f"\n{'='*60}")
            print(f"Plugin: {pname}")
            print(f"{'='*60}")

            try:
                result = plugin.transcribe(audio, 22050)

                # Note accuracy
                acc = compute_accuracy(gt_notes, result.notes, time_tolerance=0.2)
                print(f"\nNote Detection:")
                print(f"  Ground truth: {acc['n_ground_truth']} notes")
                print(f"  Detected:     {acc['n_detected']} notes")
                print(f"  Matched:      {acc['n_matched']}")
                print(f"  Precision:    {acc['precision']:.1%}")
                print(f"  Recall:       {acc['recall']:.1%}")
                print(f"  F1:           {acc['f1']:.1%}")
                print(f"  Onset error:  {acc['avg_onset_error_s']:.3f}s avg")
                print(f"  Pitch error:  {acc['avg_pitch_error_semitones']:.1f} semitones avg")

                # Chord accuracy
                if result.chords:
                    chord_acc = compute_chord_accuracy(
                        expected_chord_names, expected_chord_times,
                        result.chords, time_tolerance=0.5,
                    )
                    print(f"\nChord Detection:")
                    print(f"  Matched: {chord_acc['n_matched']}/{chord_acc['n_expected']} "
                          f"({chord_acc['chord_match_rate']:.0%})")
                    for d in chord_acc["details"]:
                        status = "✓" if d["match"] else "✗"
                        print(f"    {status} Expected={d['expected']:4s}  Got={d['detected']:8s}  "
                              f"Δt={d['time_err']:.2f}s")

                    # Show all detected chords for analysis
                    print(f"\n  All detected chords ({len(result.chords)}):")
                    for c in result.chords:
                        print(f"    {c.start_time:5.2f}s - {c.end_time:5.2f}s  "
                              f"{c.chord_name:8s}  conf={c.confidence:.2f}")
                else:
                    print("\n  No chords detected!")

                # Velocity stats
                if result.notes:
                    vels = [n.velocity for n in result.notes]
                    print(f"\nVelocity: min={min(vels)}, max={max(vels)}, mean={np.mean(vels):.0f}")
                    zero_vel = sum(1 for v in vels if v == 0)
                    if zero_vel:
                        print(f"  WARNING: {zero_vel} notes with velocity=0!")

                # MIDI export check
                from app.output.midi_file_writer import write_midi
                import mido
                export_path = Path(tmpdir) / f"fur_elise_{pname.replace(' ', '_')}.mid"
                write_midi(result, path=export_path)
                mid_out = mido.MidiFile(str(export_path))
                note_ons = [m for t in mid_out.tracks for m in t
                            if m.type == "note_on" and m.velocity > 0]
                print(f"\nMIDI Export: {len(note_ons)} note_ons with vel>0 ✓"
                      if all(m.velocity > 0 for m in note_ons)
                      else f"\nMIDI Export: VELOCITY BUG — {sum(1 for m in note_ons if m.velocity == 0)} with vel=0")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    print("Für Elise — Chord Detection Accuracy Test")
    print("=" * 60)
    run_fur_elise_test()
