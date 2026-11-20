"""Audio utilities for format validation, preprocessing, and FFmpeg conversion."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
    ".aac",
    ".wma",
    ".opus",
}

# 50 MB default max file size limit
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_AUDIO_FILE_SIZE_BYTES", 50 * 1024 * 1024))


class AudioValidationError(Exception):
    """Raised when an audio file fails format or size validation."""
    pass


class AudioConversionError(Exception):
    """Raised when FFmpeg audio conversion fails."""
    pass


def validate_file_metadata(filename: str, file_size: int) -> None:
    """Validate file extension and size constraints.

    Args:
        filename: Name of the uploaded file.
        file_size: Size of the file in bytes.

    Raises:
        AudioValidationError: If format is unsupported or file size is invalid.
    """
    if not filename:
        raise AudioValidationError("Filename cannot be empty.")

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported file format '{ext}'. Supported formats: {supported}"
        )

    if file_size <= 0:
        raise AudioValidationError("Uploaded file is empty (0 bytes).")

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise AudioValidationError(
            f"File size exceeds the maximum allowed limit of {max_mb:.1f} MB."
        )


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg executable is available in PATH."""
    return shutil.which("ffmpeg") is not None


def convert_audio_to_wav(input_path: str, output_path: str | None = None) -> str:
    """Convert any audio file to standard 16kHz mono 16-bit PCM WAV using FFmpeg.

    Faster-Whisper processes audio most efficiently when pre-sampled to 16,000 Hz mono.

    Args:
        input_path: Path to the source audio file.
        output_path: Optional destination path. If not provided, a temp file is created.

    Returns:
        Path to the converted WAV file.

    Raises:
        AudioConversionError: If FFmpeg is missing or conversion fails.
    """
    if not is_ffmpeg_available():
        raise AudioConversionError(
            "FFmpeg executable not found on the system. Please install FFmpeg to enable audio preprocessing."
        )

    if not os.path.exists(input_path):
        raise AudioConversionError(f"Input audio file does not exist: {input_path}")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    # Standardize to 16kHz, mono channel, 16-bit PCM WAV
    cmd = [
        "ffmpeg",
        "-y",               # Overwrite output
        "-v", "error",      # Only log errors
        "-i", input_path,   # Input file
        "-ar", "16000",     # 16kHz sample rate
        "-ac", "1",         # Mono channel
        "-c:a", "pcm_s16le",# Standard 16-bit PCM
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr_msg = result.stderr.strip() or "Unknown FFmpeg error."
            raise AudioConversionError(
                f"FFmpeg conversion failed (exit code {result.returncode}): {stderr_msg}"
            )
    except Exception as exc:
        if isinstance(exc, AudioConversionError):
            raise
        raise AudioConversionError(f"Failed to execute FFmpeg: {str(exc)}") from exc

    return output_path
