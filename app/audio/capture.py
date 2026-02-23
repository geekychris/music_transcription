"""Microphone audio capture using sounddevice."""

import threading
import logging
import queue

import numpy as np
import sounddevice as sd

from app.config import SAMPLE_RATE, MIC_BLOCK_SIZE, CHUNK_DURATION, CHUNK_HOP

logger = logging.getLogger(__name__)


class MicCapture:
    """Captures audio from the microphone and provides overlapping chunks via a queue.

    Uses a thread-safe queue so that the PortAudio callback thread never calls
    into eventlet-patched code.  Consumers should poll get_chunk() or iterate
    over chunk_iter() from an eventlet-friendly context.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = MIC_BLOCK_SIZE,
                 chunk_duration: float = CHUNK_DURATION, chunk_hop: float = CHUNK_HOP,
                 device: int | str | None = None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.hop_samples = int(chunk_hop * sample_rate)
        self.device = device

        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._running = False
        # Thread-safe queue — PortAudio thread pushes, eventlet greenlet pops
        self.chunk_queue: queue.Queue[tuple[np.ndarray, int]] = queue.Queue(maxsize=64)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Mic status: %s", status)
        audio = indata[:, 0].copy()  # mono
        with self._lock:
            self._buffer = np.concatenate([self._buffer, audio])
            while len(self._buffer) >= self.chunk_samples:
                chunk = self._buffer[:self.chunk_samples].copy()
                self._buffer = self._buffer[self.hop_samples:]
                try:
                    self.chunk_queue.put_nowait((chunk, self.sample_rate))
                except queue.Full:
                    pass  # drop chunk if consumer is too slow

    def start(self):
        if self._running:
            return
        logger.info("Starting mic capture (sr=%d, device=%s)", self.sample_rate, self.device)
        self._buffer = np.zeros(0, dtype=np.float32)
        # Drain any stale chunks
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True

    def stop(self):
        if not self._running:
            return
        logger.info("Stopping mic capture")
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)

    def get_chunk(self, timeout: float = 0.1):
        """Get next chunk from queue.  Returns (chunk, sr) or None."""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def list_devices() -> list[dict]:
        """Return available input audio devices."""
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
        return result
