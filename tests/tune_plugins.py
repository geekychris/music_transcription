#!/usr/bin/env python3
"""Automated plugin tuning against Für Elise ground truth.

Sweeps key parameters for each plugin, measures note + chord accuracy,
and reports the best configuration found.

Run:
    PYTHONPATH=. python tests/tune_plugins.py
"""

import sys
import tempfile
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_fur_elise import (
    create_fur_elise_midi, get_ground_truth_notes, EXPECTED_CHORDS,
)
from tests.test_round_trip import compute_accuracy, compute_chord_accuracy


# ---------------------------------------------------------------------------
# Prepare audio once, reuse for all parameter sweeps
# ---------------------------------------------------------------------------

def prepare_test_audio(tempo: float = 72.0, sr: int = 22050):
    """Create Für Elise WAV and ground-truth data."""
    eighth = 60.0 / tempo / 2.0
    gt_notes = get_ground_truth_notes(eighth)
    expected_chord_names = [c[0] for c in EXPECTED_CHORDS]
    expected_chord_times = [c[1] * eighth for c in EXPECTED_CHORDS]

    tmpdir = tempfile.mkdtemp()
    midi_path = str(Path(tmpdir) / "fur_elise.mid")
    wav_path = str(Path(tmpdir) / "fur_elise.wav")

    create_fur_elise_midi(midi_path, tempo=tempo)
    mid = pretty_midi.PrettyMIDI(midi_path)
    try:
        audio = mid.fluidsynth(fs=sr)
        synth = "FluidSynth"
    except Exception:
        audio = mid.synthesize(fs=sr)
        synth = "sine"

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    sf.write(wav_path, audio, sr)
    print(f"Audio: {len(audio)/sr:.1f}s, {sr}Hz, synth={synth}")
    return audio, sr, gt_notes, expected_chord_names, expected_chord_times


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Score:
    params: dict
    n_detected: int
    note_recall: float
    note_precision: float
    note_f1: float
    chord_match_rate: float
    avg_onset_err: float
    mean_duration: float
    short_pct: float  # % notes < 100ms

    @property
    def combined(self) -> float:
        """Weighted score for ranking configurations."""
        return (
            self.note_f1 * 0.4
            + self.note_recall * 0.25
            + self.chord_match_rate * 0.25
            + (1.0 - min(self.short_pct, 1.0)) * 0.1  # penalize machine-gun
        )


def evaluate(plugin, audio, sr, gt_notes, chord_names, chord_times,
             params: dict) -> Score:
    """Run a plugin and score the results."""
    result = plugin.transcribe(audio, sr)
    # Apply export-time merge like the real export path
    result = result.merge_notes(max_gap=0.25)

    acc = compute_accuracy(gt_notes, result.notes, pitch_tolerance=1,
                           time_tolerance=0.2)
    chord_rate = 0.0
    if result.chords:
        ca = compute_chord_accuracy(chord_names, chord_times, result.chords,
                                    time_tolerance=0.5)
        chord_rate = ca["chord_match_rate"]

    durs = [n.end_time - n.start_time for n in result.notes] if result.notes else [0]
    short = sum(1 for d in durs if d < 0.1) / max(len(durs), 1)

    return Score(
        params=params,
        n_detected=len(result.notes),
        note_recall=acc["recall"],
        note_precision=acc["precision"],
        note_f1=acc["f1"],
        chord_match_rate=chord_rate,
        avg_onset_err=acc["avg_onset_error_s"],
        mean_duration=sum(durs) / len(durs),
        short_pct=short,
    )


# ---------------------------------------------------------------------------
# Basic Pitch sweep
# ---------------------------------------------------------------------------

def tune_basic_pitch(audio, sr, gt_notes, chord_names, chord_times):
    """Sweep Basic Pitch parameters."""
    from app.transcription.plugins.basic_pitch_plugin import BasicPitchPlugin

    param_grid = {
        "onset_threshold":    [0.3, 0.4, 0.5, 0.6],
        "frame_threshold":    [0.2, 0.3, 0.4, 0.5],
        "min_note_length_ms": [50, 80, 127],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"\n{'='*70}")
    print(f"TUNING: Basic Pitch ({len(combos)} combinations)")
    print(f"{'='*70}")

    scores: list[Score] = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        plugin = BasicPitchPlugin(
            onset_threshold=params["onset_threshold"],
            frame_threshold=params["frame_threshold"],
            min_note_length_ms=params["min_note_length_ms"],
        )
        score = evaluate(plugin, audio, sr, gt_notes, chord_names, chord_times, params)
        scores.append(score)
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  [{i+1}/{len(combos)}] Best so far: combined={max(scores, key=lambda s: s.combined).combined:.3f}")

    scores.sort(key=lambda s: s.combined, reverse=True)
    print(f"\nTop 5 Basic Pitch configurations:")
    for i, s in enumerate(scores[:5]):
        print(f"  {i+1}. combined={s.combined:.3f}  F1={s.note_f1:.1%}  "
              f"recall={s.note_recall:.1%}  prec={s.note_precision:.1%}  "
              f"chords={s.chord_match_rate:.0%}  notes={s.n_detected}  "
              f"onset_err={s.avg_onset_err:.3f}s  "
              f"dur={s.mean_duration:.3f}s  short={s.short_pct:.0%}")
        print(f"     params: {s.params}")

    return scores[0]


# ---------------------------------------------------------------------------
# Chroma Chords sweep
# ---------------------------------------------------------------------------

def tune_chroma_chords(audio, sr, gt_notes, chord_names, chord_times):
    """Sweep Chroma Chords parameters."""
    from app.transcription.plugins.chroma_chord_plugin import ChromaChordPlugin

    param_grid = {
        "frame_duration":  [0.25, 0.35, 0.5, 0.75],
        "min_confidence":  [0.4, 0.5, 0.55, 0.65],
        "hop_length":      [256, 512, 1024],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"\n{'='*70}")
    print(f"TUNING: Chroma Chords ({len(combos)} combinations)")
    print(f"{'='*70}")

    scores: list[Score] = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        plugin = ChromaChordPlugin(
            frame_duration=params["frame_duration"],
            min_confidence=params["min_confidence"],
            hop_length=params["hop_length"],
        )
        score = evaluate(plugin, audio, sr, gt_notes, chord_names, chord_times, params)
        scores.append(score)
        if (i + 1) % 10 == 0 or i == len(combos) - 1:
            print(f"  [{i+1}/{len(combos)}] Best so far: combined={max(scores, key=lambda s: s.combined).combined:.3f}")

    scores.sort(key=lambda s: s.combined, reverse=True)
    print(f"\nTop 5 Chroma Chords configurations:")
    for i, s in enumerate(scores[:5]):
        print(f"  {i+1}. combined={s.combined:.3f}  F1={s.note_f1:.1%}  "
              f"recall={s.note_recall:.1%}  prec={s.note_precision:.1%}  "
              f"chords={s.chord_match_rate:.0%}  notes={s.n_detected}  "
              f"onset_err={s.avg_onset_err:.3f}s  "
              f"dur={s.mean_duration:.3f}s  short={s.short_pct:.0%}")
        print(f"     params: {s.params}")

    return scores[0]


# ---------------------------------------------------------------------------
# Apply best parameters
# ---------------------------------------------------------------------------

def suggest_changes(bp_best: Score, cc_best: Score):
    """Print suggested config/code changes based on best parameters."""
    print(f"\n{'='*70}")
    print("SUGGESTED PARAMETER CHANGES")
    print(f"{'='*70}")

    bp = bp_best.params
    print(f"\nBasic Pitch (combined={bp_best.combined:.3f}, F1={bp_best.note_f1:.1%}, chords={bp_best.chord_match_rate:.0%}):")
    print(f"  app/config.py:")
    print(f"    DEFAULT_ONSET_THRESHOLD = {bp['onset_threshold']}")
    print(f"    DEFAULT_FRAME_THRESHOLD = {bp['frame_threshold']}")
    print(f"    DEFAULT_MIN_NOTE_LENGTH_MS = {bp['min_note_length_ms']}")

    cc = cc_best.params
    print(f"\nChroma Chords (combined={cc_best.combined:.3f}, F1={cc_best.note_f1:.1%}, chords={cc_best.chord_match_rate:.0%}):")
    print(f"  app/transcription/plugins/chroma_chord_plugin.py __init__:")
    print(f"    frame_duration = {cc['frame_duration']}")
    print(f"    min_confidence = {cc['min_confidence']}")
    print(f"    hop_length = {cc['hop_length']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Music Transposer — Automated Plugin Tuning")
    print("=" * 70)
    print("Ground truth: Für Elise opening (92 notes, 13 chord changes)\n")

    audio, sr, gt_notes, chord_names, chord_times = prepare_test_audio()

    bp_best = tune_basic_pitch(audio, sr, gt_notes, chord_names, chord_times)
    cc_best = tune_chroma_chords(audio, sr, gt_notes, chord_names, chord_times)

    suggest_changes(bp_best, cc_best)

    # Ask to apply
    print(f"\nApply these parameters? [y/N] ", end="", flush=True)
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "y"  # auto-apply when not interactive

    if answer == "y":
        # Write updated config
        config_path = Path(__file__).resolve().parent.parent / "app" / "config.py"
        config = config_path.read_text()

        bp = bp_best.params
        import re
        config = re.sub(r"DEFAULT_ONSET_THRESHOLD = [\d.]+",
                        f"DEFAULT_ONSET_THRESHOLD = {bp['onset_threshold']}", config)
        config = re.sub(r"DEFAULT_FRAME_THRESHOLD = [\d.]+",
                        f"DEFAULT_FRAME_THRESHOLD = {bp['frame_threshold']}", config)
        config = re.sub(r"DEFAULT_MIN_NOTE_LENGTH_MS = [\d.]+",
                        f"DEFAULT_MIN_NOTE_LENGTH_MS = {bp['min_note_length_ms']}", config)
        config_path.write_text(config)
        print(f"  ✓ Updated {config_path}")

        # Update chroma plugin defaults
        cc = cc_best.params
        plugin_path = (Path(__file__).resolve().parent.parent /
                       "app/transcription/plugins/chroma_chord_plugin.py")
        plugin_src = plugin_path.read_text()
        plugin_src = re.sub(r"frame_duration: float = [\d.]+",
                            f"frame_duration: float = {cc['frame_duration']}", plugin_src)
        plugin_src = re.sub(r"min_confidence: float = [\d.]+",
                            f"min_confidence: float = {cc['min_confidence']}", plugin_src)
        plugin_src = re.sub(r"hop_length: int = \d+",
                            f"hop_length: int = {cc['hop_length']}", plugin_src)
        plugin_path.write_text(plugin_src)
        print(f"  ✓ Updated {plugin_path}")

        print("\nDone! Restart the server to use the new parameters.")
    else:
        print("Skipped. You can apply the changes manually.")
