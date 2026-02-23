"""Export transcription results to PDF via music21 + LilyPond."""

import logging
import shutil
from pathlib import Path

from app.transcription.base import TranscriptionResult
from app.output.musicxml_export import to_music21_stream
from app.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def is_lilypond_available() -> bool:
    """Check if LilyPond is installed and accessible."""
    return shutil.which("lilypond") is not None


def write_pdf(result: TranscriptionResult, path: str | Path | None = None,
              title: str = "Transcription") -> Path | None:
    """Write transcription to a PDF file via LilyPond.

    Returns path to the PDF, or None if LilyPond is not available.
    """
    if not is_lilypond_available():
        logger.error("LilyPond not found. Install it for PDF export. "
                     "(Mac: brew install lilypond, Pi: sudo apt install lilypond)")
        return None

    if path is None:
        path = OUTPUT_DIR / "transcription.pdf"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    score = to_music21_stream(result, title)

    try:
        # music21 uses LilyPond under the hood for PDF rendering
        written_path = score.write("lily.pdf", fp=str(path))
        logger.info("PDF written: %s", written_path)
        return Path(written_path)
    except Exception:
        logger.exception("Failed to write PDF. Is LilyPond properly installed?")
        return None
