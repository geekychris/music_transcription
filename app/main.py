"""Music Transposer — main entry point."""

import logging
import sys

# Eventlet must be monkey-patched before anything else
import eventlet
eventlet.monkey_patch()

from app.config import HOST, PORT, DEBUG
from app.transcription.plugin_manager import PluginManager
from app.audio.capture import MicCapture
from app.output.midi_stream import MidiStream
from app.web.server import create_app, socketio
from app.web.routes import init_shared_state
from app.web.events import init_realtime, _start_chunk_consumer


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger("music_transposer")

    logger.info("=== Music Transposer ===")

    # Discover plugins
    plugin_manager = PluginManager()
    plugin_manager.discover()

    if not plugin_manager.plugins:
        logger.error("No transcription plugins found! Check your installation.")
        sys.exit(1)

    # Initialize audio capture
    mic_capture = MicCapture()
    logger.info("Available audio devices: %s", MicCapture.list_devices())

    # Initialize MIDI stream
    midi_stream = MidiStream()

    # Wire up shared state
    init_shared_state(plugin_manager, mic_capture)
    init_realtime(plugin_manager, mic_capture, midi_stream)

    # Create and run Flask app
    app = create_app()

    # Start background chunk consumer (runs in eventlet context)
    socketio.start_background_task(_start_chunk_consumer)

    logger.info("Starting server on http://%s:%d", HOST, PORT)
    socketio.run(app, host=HOST, port=PORT, debug=DEBUG, use_reloader=False)


if __name__ == "__main__":
    main()
