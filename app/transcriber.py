"""Faster-Whisper transcription engine wrapper."""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("audio_api.transcriber")


class WhisperTranscriber:
    """Wrapper around Faster-Whisper for efficient speech-to-text inference."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        cpu_threads: int = 4,
    ):
        self.model_size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
        self.device = device or os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = compute_type or os.getenv(
            "WHISPER_COMPUTE_TYPE", "int8" if self.device == "cpu" else "float16"
        )
        self.cpu_threads = cpu_threads
        self._model = None

    def load_model(self) -> None:
        """Load the Whisper model into memory if not already loaded."""
        if self._model is not None:
            return

        logger.info(
            "Loading Faster-Whisper model '%s' (device=%s, compute_type=%s)...",
            self.model_size,
            self.device,
            self.compute_type,
        )

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            logger.info("Faster-Whisper model loaded successfully.")
        except ImportError as exc:
            logger.error("faster-whisper is not installed.")
            raise RuntimeError(
                "faster-whisper library is required. Install via 'pip install faster-whisper'."
            ) from exc
        except Exception as exc:
            logger.error("Failed to load Faster-Whisper model: %s", exc)
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if model has been loaded into memory."""
        return self._model is not None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
    ) -> Dict[str, Any]:
        """Transcribe an audio file and generate timestamped segments.

        Args:
            audio_path: Path to the preprocessed audio file.
            language: Optional language code (e.g. 'en', 'es'). Auto-detected if None.
            beam_size: Beam search width (default: 5).

        Returns:
            Dictionary containing language, full_text, duration, and segment list.
        """
        if self._model is None:
            self.load_model()

        logger.info("Transcribing audio file: %s (language=%s)", audio_path, language)

        try:
            raw_segments, info = self._model.transcribe(
                audio_path,
                beam_size=beam_size,
                language=language,
                task="transcribe",
                vad_filter=True,  # Filter out background noise & silence
            )

            segments: List[Dict[str, Any]] = []
            segment_texts: List[str] = []

            for seg in raw_segments:
                cleaned_text = seg.text.strip()
                if cleaned_text:
                    segment_texts.append(cleaned_text)
                    segments.append(
                        {
                            "start": round(seg.start, 2),
                            "end": round(seg.end, 2),
                            "text": cleaned_text,
                        }
                    )

            full_text = " ".join(segment_texts)

            return {
                "language": info.language if hasattr(info, "language") else (language or "unknown"),
                "language_probability": round(float(info.language_probability), 4)
                if hasattr(info, "language_probability")
                else 1.0,
                "duration": round(float(info.duration), 2)
                if hasattr(info, "duration")
                else 0.0,
                "full_text": full_text,
                "segments": segments,
            }
        except Exception as exc:
            logger.exception("Transcription failed: %s", exc)
            raise RuntimeError(f"Speech transcription failed: {str(exc)}") from exc
