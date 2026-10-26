"""Tests for video metadata extraction, corrupt file handling, and format inspection."""

from pathlib import Path
import pytest
from src.synthetic_generator import SyntheticVideoGenerator
from src.video_metadata import VideoMetadataExtractor


@pytest.fixture
def test_video(tmp_path: Path) -> Path:
    gen = SyntheticVideoGenerator(output_dir=tmp_path)
    video_path = tmp_path / "test_cam.mp4"
    gen.generate_camera_video(
        file_path=video_path,
        camera_name="Test Camera",
        duration_sec=3.0,
        fps=24.0,
        true_offset_sec=0.0,
        true_rate_drift=1.0,
        global_event_times_sec=[1.5]
    )
    return video_path


def test_valid_video_metadata_extraction(test_video: Path):
    extractor = VideoMetadataExtractor()
    meta = extractor.extract(test_video, camera_name="Test Camera", video_id="CAM_TEST", sha256_hash="abc123")

    assert meta.is_valid is True
    assert meta.error_message is None
    assert meta.camera_name == "Test Camera"
    assert meta.width == 640
    assert meta.height == 360
    assert abs(meta.nominal_fps - 24.0) < 0.5
    assert meta.total_frames == 72
    assert abs(meta.duration_sec - 3.0) < 0.2


def test_nonexistent_file_handling():
    extractor = VideoMetadataExtractor()
    meta = extractor.extract("non_existent_file.mp4", camera_name="Ghost Cam", video_id="CAM_GHOST")

    assert meta.is_valid is False
    assert "does not exist" in meta.error_message.lower()


def test_empty_zero_byte_video_handling(tmp_path: Path):
    extractor = VideoMetadataExtractor()
    empty_file = tmp_path / "zero_bytes.mp4"
    empty_file.write_bytes(b"")

    meta = extractor.extract(empty_file, camera_name="Zero Cam", video_id="CAM_ZERO")

    assert meta.is_valid is False
    assert "0 bytes" in meta.error_message.lower()


def test_corrupt_header_handling(tmp_path: Path):
    extractor = VideoMetadataExtractor()
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"RANDOM_CORRUPT_HEADER_BYTES_NOT_A_REAL_VIDEO")

    meta = extractor.extract(corrupt_file, camera_name="Corrupt Cam", video_id="CAM_CORRUPT")

    assert meta.is_valid is False
    assert "could not open stream" in meta.error_message.lower() or "unreadable" in meta.error_message.lower()
