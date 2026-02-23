"""Virtual MIDI port for real-time note streaming."""

import logging
import time
import threading

import rtmidi

from app.transcription.base import TranscriptionResult, NoteEvent
from app.config import VIRTUAL_MIDI_PORT_NAME

logger = logging.getLogger(__name__)


class MidiStream:
    """Creates a virtual MIDI output port and sends note events in real-time."""

    def __init__(self, port_name: str = VIRTUAL_MIDI_PORT_NAME):
        self.port_name = port_name
        self._midi_out: rtmidi.MidiOut | None = None
        self._active_notes: set[int] = set()
        self._lock = threading.Lock()

    def open(self):
        """Open the virtual MIDI port."""
        if self._midi_out is not None:
            return
        self._midi_out = rtmidi.MidiOut()
        self._midi_out.open_virtual_port(self.port_name)
        logger.info("Virtual MIDI port opened: %s", self.port_name)

    def close(self):
        """Close the virtual MIDI port and send all-notes-off."""
        if self._midi_out is None:
            return
        self.all_notes_off()
        self._midi_out.close_port()
        self._midi_out = None
        logger.info("Virtual MIDI port closed: %s", self.port_name)

    @property
    def is_open(self) -> bool:
        return self._midi_out is not None

    def note_on(self, pitch: int, velocity: int = 80, channel: int = 0):
        """Send a MIDI note-on message."""
        if self._midi_out is None:
            return
        with self._lock:
            self._midi_out.send_message([0x90 | channel, pitch & 0x7F, velocity & 0x7F])
            self._active_notes.add(pitch)

    def note_off(self, pitch: int, channel: int = 0):
        """Send a MIDI note-off message."""
        if self._midi_out is None:
            return
        with self._lock:
            self._midi_out.send_message([0x80 | channel, pitch & 0x7F, 0])
            self._active_notes.discard(pitch)

    def all_notes_off(self, channel: int = 0):
        """Send note-off for all currently active notes."""
        if self._midi_out is None:
            return
        with self._lock:
            for pitch in list(self._active_notes):
                self._midi_out.send_message([0x80 | channel, pitch & 0x7F, 0])
            self._active_notes.clear()

    def send_result(self, result: TranscriptionResult):
        """Send all notes from a transcription result immediately.

        For real-time use: turns on all notes, waits, turns them off.
        Best used for chunk-by-chunk results where timing is approximate.
        """
        if self._midi_out is None:
            return

        # Turn off previous notes
        self.all_notes_off()

        # Send new notes
        for note in result.notes:
            self.note_on(note.midi_pitch, note.velocity)

    def send_result_timed(self, result: TranscriptionResult):
        """Send notes with proper timing in a background thread.

        Respects the relative timing of note events. Best for file-based results.
        """
        if not result.notes:
            return

        def _play():
            sorted_notes = sorted(result.notes, key=lambda n: n.start_time)
            base_time = sorted_notes[0].start_time

            for note in sorted_notes:
                wait = note.start_time - base_time
                if wait > 0:
                    time.sleep(wait)
                    base_time = note.start_time
                self.note_on(note.midi_pitch, note.velocity)

                # Schedule note-off
                def _off(p=note.midi_pitch, d=note.duration):
                    time.sleep(d)
                    self.note_off(p)
                threading.Thread(target=_off, daemon=True).start()

        threading.Thread(target=_play, daemon=True).start()

    @staticmethod
    def list_ports() -> list[str]:
        """List available MIDI output ports."""
        out = rtmidi.MidiOut()
        ports = out.get_ports()
        del out
        return ports
