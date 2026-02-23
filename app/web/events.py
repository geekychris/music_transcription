"""Socket.IO event handlers for real-time transcription."""

import logging
import time

import eventlet
from flask_socketio import emit

from app.web.server import socketio
from app.audio.capture import MicCapture
from app.output.midi_stream import MidiStream
from app.transcription.plugin_manager import PluginManager
from app.transcription.base import TranscriberPlugin, NoteEvent, TranscriptionResult

logger = logging.getLogger(__name__)

# Shared state — set from main.py
_plugin_manager: PluginManager | None = None
_mic_capture: MicCapture | None = None
_midi_stream: MidiStream | None = None
_active_plugin: TranscriberPlugin | None = None
_midi_streaming_enabled: bool = False
_chunk_counter: int = 0
_recording_start_time: float = 0.0
# Trailing notes from the previous chunk, keyed by midi_pitch.
# Used to merge sustained notes that span chunk boundaries.
_prev_chunk_trailing: dict[int, NoteEvent] = {}


def init_realtime(pm: PluginManager, mc: MicCapture, ms: MidiStream):
    """Initialize shared state for real-time events."""
    global _plugin_manager, _mic_capture, _midi_stream, _active_plugin
    _plugin_manager = pm
    _mic_capture = mc
    _midi_stream = ms
    _active_plugin = pm.default_plugin


def _start_chunk_consumer():
    """Background task that polls the mic queue and processes chunks.

    Runs inside the eventlet context so socketio.emit() works correctly.
    """
    global _chunk_counter
    logger.info("Chunk consumer started")
    while True:
        if _mic_capture is None or not _mic_capture.is_running:
            eventlet.sleep(0.2)
            continue

        result = _mic_capture.get_chunk(timeout=0.1)
        if result is None:
            eventlet.sleep(0.05)
            continue

        chunk, sr = result
        if _active_plugin is None:
            continue

        try:
            _chunk_counter += 1
            # Compute wall-clock offset so notes are placed at the right time
            wall_offset = time.time() - _recording_start_time
            t0 = time.time()
            transcription = _active_plugin.transcribe_chunk(
                chunk, sr, time_offset=wall_offset
            )
            elapsed = time.time() - t0
            data = transcription.to_dict()

            note_count = len(transcription.notes)
            chord_count = len(transcription.chords)
            logger.info("Chunk #%d processed in %.2fs: %d notes, %d chords",
                        _chunk_counter, elapsed, note_count, chord_count)

            # Emit to all connected WebSocket clients
            socketio.emit("transcription_result", data)

            # Forward to virtual MIDI port if enabled
            if _midi_streaming_enabled and _midi_stream and _midi_stream.is_open:
                _midi_stream.send_result(transcription)

        except Exception:
            logger.exception("Error processing audio chunk #%d", _chunk_counter)

        # Yield to other greenlets
        eventlet.sleep(0)


def _merge_across_chunks(transcription: TranscriptionResult,
                         max_gap: float = 0.15) -> TranscriptionResult:
    """Merge notes that continue from the previous chunk.

    If a note in the new chunk has the same pitch as a trailing note from
    the previous chunk and the gap is small, extend the previous note
    rather than creating a new one.  The extended note is emitted with
    ``start_time`` set to the *new* chunk's start so that the JS client
    can update the existing note's end time.
    """
    global _prev_chunk_trailing

    merged_notes: list[NoteEvent] = []
    used_prev = set()  # pitches that were extended

    for note in transcription.notes:
        prev = _prev_chunk_trailing.get(note.midi_pitch)
        if prev and note.start_time <= prev.end_time + max_gap:
            # This note is a continuation — extend it
            merged_notes.append(NoteEvent(
                start_time=prev.start_time,
                end_time=max(prev.end_time, note.end_time),
                midi_pitch=note.midi_pitch,
                velocity=max(prev.velocity, note.velocity),
                confidence=max(prev.confidence, note.confidence),
            ))
            used_prev.add(note.midi_pitch)
        else:
            merged_notes.append(note)

    # Update trailing notes for the next chunk.
    # Keep the latest note per pitch from this chunk's output.
    new_trailing: dict[int, NoteEvent] = {}
    for note in merged_notes:
        existing = new_trailing.get(note.midi_pitch)
        if existing is None or note.end_time > existing.end_time:
            new_trailing[note.midi_pitch] = note
    _prev_chunk_trailing = new_trailing

    return TranscriptionResult(
        notes=merged_notes, chords=transcription.chords,
        tempo=transcription.tempo, time_offset=transcription.time_offset,
    )


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    emit("status", {
        "mic_running": _mic_capture.is_running if _mic_capture else False,
        "active_plugin": _active_plugin.name() if _active_plugin else None,
        "midi_streaming": _midi_streaming_enabled,
    })


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")


@socketio.on("start_mic")
def handle_start_mic(data=None):
    """Start microphone capture."""
    global _active_plugin
    if _mic_capture is None or _plugin_manager is None:
        emit("error", {"message": "Not initialized"})
        return

    # Optionally switch plugin
    if data and "plugin" in data:
        plugin = _plugin_manager.get_plugin(data["plugin"])
        if plugin:
            _active_plugin = plugin

    global _recording_start_time, _chunk_counter, _prev_chunk_trailing
    _recording_start_time = time.time()
    _chunk_counter = 0
    _prev_chunk_trailing = {}
    _mic_capture.start()
    emit("status", {"mic_running": True, "active_plugin": _active_plugin.name() if _active_plugin else None})
    logger.info("Mic started via WebSocket")


@socketio.on("stop_mic")
def handle_stop_mic():
    """Stop microphone capture."""
    if _mic_capture:
        _mic_capture.stop()
    emit("status", {"mic_running": False})
    logger.info("Mic stopped via WebSocket")


@socketio.on("set_plugin")
def handle_set_plugin(data):
    """Switch the active transcription plugin."""
    global _active_plugin
    if _plugin_manager is None:
        return

    name = data.get("plugin", "")
    plugin = _plugin_manager.get_plugin(name)
    if plugin:
        _active_plugin = plugin
        emit("status", {"active_plugin": plugin.name()})
        logger.info("Plugin switched to: %s", plugin.name())
    else:
        emit("error", {"message": f"Plugin not found: {name}"})


@socketio.on("enable_midi_stream")
def handle_enable_midi(data=None):
    """Open virtual MIDI port and enable streaming."""
    global _midi_streaming_enabled
    if _midi_stream is None:
        emit("error", {"message": "MIDI stream not available"})
        return

    _midi_stream.open()
    _midi_streaming_enabled = True
    emit("status", {"midi_streaming": True})
    logger.info("MIDI streaming enabled")


@socketio.on("disable_midi_stream")
def handle_disable_midi(data=None):
    """Close virtual MIDI port."""
    global _midi_streaming_enabled
    _midi_streaming_enabled = False
    if _midi_stream:
        _midi_stream.close()
    emit("status", {"midi_streaming": False})
    logger.info("MIDI streaming disabled")
