"""FastAPI application entry point for Audio Transcription API."""

import utils.net
import utils.clients
import utils.http_session
import utils.transport
import utils.session
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.audio_utils import (
    AudioConversionError,
    AudioValidationError,
    convert_audio_to_wav,
    is_ffmpeg_available,
    validate_file_metadata,
)
from app.transcriber import WhisperTranscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audio_api")

# Global transcriber instance
transcriber = WhisperTranscriber()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Initializing Audio Transcription API...")
    # Optional eager model loading on startup if configured
    if os.getenv("EAGER_MODEL_LOAD", "false").lower() in ("true", "1", "yes"):
        try:
            transcriber.load_model()
        except Exception as err:
            logger.warning("Eager model loading failed (will retry on first request): %s", err)
    yield
    logger.info("Shutting down Audio Transcription API.")


app = FastAPI(
    title="Audio Transcription API",
    description="High-performance speech-to-text transcription service powered by Faster-Whisper and FFmpeg.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Schemas
class SegmentResponse(BaseModel):
    start: float = Field(..., description="Segment start timestamp in seconds", json_schema_extra={"example": 0.0})
    end: float = Field(..., description="Segment end timestamp in seconds", json_schema_extra={"example": 1.25})
    text: str = Field(..., description="Transcribed segment text", json_schema_extra={"example": "Hello,"})


class TranscriptionResponse(BaseModel):
    language: str = Field(..., description="Detected or specified spoken language code", json_schema_extra={"example": "en"})
    full_text: str = Field(
        ...,
        description="Full consolidated transcription of the audio",
        json_schema_extra={"example": "Hello, welcome to our application."},
    )
    segments: List[SegmentResponse] = Field(
        ..., description="List of timestamped speech segments"
    )
    duration: Optional[float] = Field(
        None, description="Audio duration in seconds", json_schema_extra={"example": 2.8}
    )
    language_probability: Optional[float] = Field(
        None, description="Model confidence score for the detected language", json_schema_extra={"example": 0.985}
    )


class HealthResponse(BaseModel):
    status: str
    model_size: str
    device: str
    model_loaded: bool
    ffmpeg_available: bool


@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    """Root endpoint returning API service status and quick links."""
    return {
        "service": "Audio Transcription API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "transcribe_url": "/transcribe",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Check service health and dependency readiness."""
    return HealthResponse(
        status="healthy",
        model_size=transcriber.model_size,
        device=transcriber.device,
        model_loaded=transcriber.is_loaded,
        ffmpeg_available=is_ffmpeg_available(),
    )


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Transcription"],
    summary="Transcribe uploaded audio file",
    description="Upload an audio file (WAV, MP3, M4A, FLAC, OGG, WebM, etc.) and receive segment timestamps and full transcript.",
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Query(
        None,
        description="Optional ISO language code (e.g., 'en', 'es', 'fr', 'ur'). Auto-detected if omitted.",
        openapi_examples={"example": {"value": "en"}},
    ),
) -> TranscriptionResponse:
    """Handle audio upload, conversion via FFmpeg, and Faster-Whisper transcription."""
    temp_input_path: Optional[str] = None
    temp_converted_path: Optional[str] = None

    try:
        # Read content and inspect size
        content = await file.read()
        file_size = len(content)

        # 1. Validate file format and size
        try:
            validate_file_metadata(file.filename or "", file_size)
        except AudioValidationError as val_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(val_err),
            )

        # 2. Save incoming stream to a temporary file
        file_ext = Path(file.filename or "audio").suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_in:
            tmp_in.write(content)
            temp_input_path = tmp_in.name

        # 3. Preprocess audio with FFmpeg (standardize to 16kHz mono WAV)
        audio_for_transcription = temp_input_path
        if is_ffmpeg_available():
            try:
                temp_converted_path = convert_audio_to_wav(temp_input_path)
                audio_for_transcription = temp_converted_path
            except AudioConversionError as conv_err:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Audio preprocessing failed: {conv_err}",
                )
        else:
            logger.warning(
                "FFmpeg not available on host. Attempting direct transcription on original file."
            )

        # 4. Perform AI Speech-to-Text transcription
        try:
            result = transcriber.transcribe(
                audio_path=audio_for_transcription,
                language=language,
            )
        except Exception as trans_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription processing error: {trans_err}",
            )

        return TranscriptionResponse(**result)

    finally:
        # Cleanup temporary files to prevent disk leaks
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass
        if temp_converted_path and os.path.exists(temp_converted_path):
            try:
                os.remove(temp_converted_path)
            except OSError:
                pass
