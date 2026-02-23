"""REST API routes."""

import logging
import tempfile
from pathlib import Path

from flask import Blueprint, request, jsonify, send_from_directory, send_file

from app.config import STATIC_DIR, OUTPUT_DIR
from app.transcription.plugin_manager import PluginManager
from app.audio.file_loader import load_audio
from app.audio.processor import normalize
from app.output.midi_file_writer import write_midi
from app.output.musicxml_export import write_musicxml
from app.output.pdf_export import write_pdf, is_lilypond_available
from app.audio.capture import MicCapture

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__)

# Shared state — initialized in main.py and set here
plugin_manager: PluginManager | None = None
mic_capture: MicCapture | None = None


def init_shared_state(pm: PluginManager, mc: MicCapture):
    """Set shared state from main.py."""
    global plugin_manager, mic_capture
    plugin_manager = pm
    mic_capture = mc


@bp.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@bp.route("/api/plugins", methods=["GET"])
def list_plugins():
    if plugin_manager is None:
        return jsonify({"error": "Not initialized"}), 500
    return jsonify({"plugins": plugin_manager.list_plugins()})


@bp.route("/api/devices", methods=["GET"])
def list_devices():
    try:
        devices = MicCapture.list_devices()
        return jsonify({"devices": devices})
    except Exception as e:
        return jsonify({"devices": [], "error": str(e)})


@bp.route("/api/transcribe/file", methods=["POST"])
def transcribe_file():
    """Upload and transcribe an audio file."""
    if plugin_manager is None:
        return jsonify({"error": "Not initialized"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    plugin_name = request.form.get("plugin", "")

    plugin = plugin_manager.get_plugin(plugin_name) or plugin_manager.default_plugin
    if plugin is None:
        return jsonify({"error": "No plugins available"}), 500

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "audio.wav").suffix,
                                      delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        audio, sr = load_audio(tmp_path)
        audio = normalize(audio)
        result = plugin.transcribe(audio, sr)
        return jsonify({"result": result.to_dict(), "plugin": plugin.name()})
    except Exception as e:
        logger.exception("Transcription failed")
        return jsonify({"error": str(e)}), 500
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@bp.route("/api/export/midi", methods=["POST"])
def export_midi():
    """Export last transcription as MIDI file."""
    data = request.get_json()
    if not data or "result" not in data:
        return jsonify({"error": "No transcription data"}), 400

    from app.transcription.base import TranscriptionResult, NoteEvent, ChordEvent
    result = _result_from_dict(data["result"])
    # Merge same-pitch notes that span chunk boundaries
    result = result.merge_notes(max_gap=0.25)

    path = write_midi(result)
    return send_file(str(path), as_attachment=True, download_name="transcription.mid")


@bp.route("/api/export/musicxml", methods=["POST"])
def export_musicxml():
    data = request.get_json()
    if not data or "result" not in data:
        return jsonify({"error": "No transcription data"}), 400

    result = _result_from_dict(data["result"])
    result = result.merge_notes(max_gap=0.25)
    path = write_musicxml(result)
    return send_file(str(path), as_attachment=True, download_name="transcription.musicxml")


@bp.route("/api/export/pdf", methods=["POST"])
def export_pdf():
    if not is_lilypond_available():
        return jsonify({"error": "LilyPond not installed"}), 500

    data = request.get_json()
    if not data or "result" not in data:
        return jsonify({"error": "No transcription data"}), 400

    result = _result_from_dict(data["result"])
    result = result.merge_notes(max_gap=0.25)
    path = write_pdf(result)
    if path is None:
        return jsonify({"error": "PDF generation failed"}), 500
    return send_file(str(path), as_attachment=True, download_name="transcription.pdf")


@bp.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "running",
        "plugins": plugin_manager.list_plugins() if plugin_manager else [],
        "lilypond_available": is_lilypond_available(),
        "mic_running": mic_capture.is_running if mic_capture else False,
    })


def _result_from_dict(data: dict):
    """Reconstruct a TranscriptionResult from a JSON dict."""
    from app.transcription.base import TranscriptionResult, NoteEvent, ChordEvent

    notes = [
        NoteEvent(
            start_time=float(n["start"]), end_time=float(n["end"]),
            midi_pitch=int(n["pitch"]), velocity=int(n.get("velocity", 80)),
            confidence=float(n.get("confidence", 1.0)),
        )
        for n in data.get("notes", [])
    ]
    chords = [
        ChordEvent(
            start_time=c["start"], end_time=c["end"],
            chord_name=c["name"], root=c.get("root", 0),
            pitches=c.get("pitches", []), confidence=c.get("confidence", 1.0),
        )
        for c in data.get("chords", [])
    ]
    return TranscriptionResult(
        notes=notes, chords=chords,
        tempo=data.get("tempo"), time_offset=data.get("time_offset", 0),
    )
