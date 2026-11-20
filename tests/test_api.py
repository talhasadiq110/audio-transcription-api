"""Unit and integration tests for the Audio Transcription API."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.audio_utils import (
    AudioConversionError,
    AudioValidationError,
    validate_file_metadata,
)
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint returns metadata and 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Audio Transcription API"
    assert "/docs" in data["docs_url"]
    assert "/transcribe" in data["transcribe_url"]


def test_health_endpoint():
    """Verify health endpoint reports status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_size" in data
    assert "device" in data
    assert "ffmpeg_available" in data


def test_validation_unsupported_extension():
    """Verify uploading an unsupported file format (e.g. .txt) returns 400 Bad Request."""
    fake_txt = io.BytesIO(b"this is plain text, not audio")
    response = client.post(
        "/transcribe",
        files={"file": ("document.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format '.txt'" in response.json()["detail"]


def test_validation_empty_file():
    """Verify uploading an empty audio file returns 400 Bad Request."""
    empty_audio = io.BytesIO(b"")
    response = client.post(
        "/transcribe",
        files={"file": ("empty.wav", empty_audio, "audio/wav")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_validation_file_size_exceeded():
    """Verify that files exceeding the maximum size limit raise an error."""
    with pytest.raises(AudioValidationError) as exc_info:
        validate_file_metadata("large.wav", 100 * 1024 * 1024)
    assert "File size exceeds" in str(exc_info.value)


def test_validation_allowed_formats():
    """Verify all supported audio formats pass metadata validation."""
    for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"]:
        # Should not raise exception
        validate_file_metadata(f"sample{ext}", 1024)


@patch("app.main.is_ffmpeg_available", return_value=False)
@patch("app.main.transcriber.transcribe")
def test_transcribe_successful_mock(mock_transcribe, mock_ffmpeg):
    """Verify successful audio transcription returns correct response schema."""
    # Mock transcriber output
    mock_transcribe.return_value = {
        "language": "en",
        "language_probability": 0.985,
        "duration": 2.8,
        "full_text": "Hello, welcome to our application.",
        "segments": [
            {
                "start": 0.0,
                "end": 1.25,
                "text": "Hello,",
            },
            {
                "start": 1.25,
                "end": 2.8,
                "text": "welcome to our application.",
            },
        ],
    }

    dummy_audio_bytes = b"RIFF....WAVEfmt ...."
    response = client.post(
        "/transcribe",
        files={"file": ("sample.wav", io.BytesIO(dummy_audio_bytes), "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()

    # Assert API response contract
    assert data["language"] == "en"
    assert data["full_text"] == "Hello, welcome to our application."
    assert len(data["segments"]) == 2
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["end"] == 1.25
    assert data["segments"][0]["text"] == "Hello,"
    assert data["segments"][1]["start"] == 1.25
    assert data["segments"][1]["end"] == 2.8
    assert data["segments"][1]["text"] == "welcome to our application."


@patch("app.main.is_ffmpeg_available", return_value=True)
@patch("app.main.convert_audio_to_wav")
def test_transcribe_ffmpeg_conversion_failure(mock_convert, mock_ffmpeg):
    """Verify that FFmpeg conversion failure returns 422 Unprocessable Entity."""
    mock_convert.side_effect = AudioConversionError("Corrupt audio stream header.")

    dummy_audio_bytes = b"corrupted bytes..."
    response = client.post(
        "/transcribe",
        files={"file": ("corrupt.mp3", io.BytesIO(dummy_audio_bytes), "audio/mp3")},
    )

    assert response.status_code == 422
    assert "Audio preprocessing failed" in response.json()["detail"]


@patch("app.main.is_ffmpeg_available", return_value=False)
@patch("app.main.transcriber.transcribe")
def test_transcribe_runtime_error(mock_transcribe, mock_ffmpeg):
    """Verify unexpected transcription engine error returns 500 Internal Server Error."""
    mock_transcribe.side_effect = RuntimeError("Whisper model inference crash.")

    dummy_audio = io.BytesIO(b"RIFF dummy bytes")
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", dummy_audio, "audio/wav")},
    )

    assert response.status_code == 500
    assert "Transcription processing error" in response.json()["detail"]
