"""Tests for evidence integrity, cryptographic hashing, and audit trails."""

import hashlib
from pathlib import Path
import pytest
from src.evidence_integrity import EvidenceManager, EvidenceRecord, SOFTWARE_VERSION


def test_sha256_computation(tmp_path: Path):
    test_file = tmp_path / "sample_evidence.bin"
    raw_content = b"FORENSIC_EVIDENCE_RECORD_SAMPLE_2026"
    test_file.write_bytes(raw_content)

    expected_hash = hashlib.sha256(raw_content).hexdigest().lower()
    computed_hash = EvidenceManager.compute_sha256(test_file)

    assert computed_hash == expected_hash
    assert len(computed_hash) == 64


def test_evidence_ingestion_and_vault(tmp_path: Path):
    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    src_file = tmp_path / "cctv_orig.mp4"
    src_file.write_bytes(b"VIDEO_HEADER_DATA_1234567890")

    record = em.register_evidence_file(src_file, camera_name="Camera 1 - Front")

    assert record.camera_name == "Camera 1 - Front"
    assert record.original_filename == "cctv_orig.mp4"
    assert Path(record.stored_path).is_file()
    assert record.file_size_bytes == len(b"VIDEO_HEADER_DATA_1234567890")
    assert record.is_tampered is False
    assert record.software_version == SOFTWARE_VERSION

    # Verify audit log recorded ingestion
    logs = em.load_audit_log()
    assert len(logs) >= 1
    assert logs[0]["action"] == "EVIDENCE_INGESTED"
    assert logs[0]["details"]["sha256"] == record.sha256_hash


def test_tamper_detection(tmp_path: Path):
    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    src_file = tmp_path / "evidence_file.mp4"
    src_file.write_bytes(b"AUTHENTIC_VIDEO_STREAM_DATA")

    record = em.register_evidence_file(src_file, camera_name="Cam Test")

    # Initial integrity check should pass
    assert em.verify_integrity(record.evidence_id) is True

    # Intentionally modify stored file to simulate tampering
    stored = Path(record.stored_path)
    # Ensure write permissions to simulate unauthorized tampering
    import stat, os
    os.chmod(stored, stat.S_IWRITE | stat.S_IREAD)
    with open(stored, "ab") as f:
        f.write(b"_TAMPERED_BYTES")

    # Tamper detection must catch the corruption
    assert em.verify_integrity(record.evidence_id) is False
    assert record.is_tampered is True

    # Audit log should record the verification failure
    logs = em.load_audit_log()
    tamper_logs = [l for l in logs if l["action"] == "INTEGRITY_VERIFICATION"]
    assert len(tamper_logs) == 2
    assert tamper_logs[-1]["details"]["is_valid"] is False


def test_empty_evidence_rejection(tmp_path: Path):
    em = EvidenceManager(base_evidence_dir=tmp_path / "vault", audit_log_path=tmp_path / "audit.jsonl")
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="0 bytes"):
        em.register_evidence_file(empty_file, camera_name="Empty Cam")


def test_evidence_paths_ignored_by_git():
    """Verify that private evidence directories, video files, reports, and logs are excluded by git."""
    import shutil
    import subprocess

    root_dir = Path(__file__).parent.parent
    gitignore_path = root_dir / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore must exist"

    gitignore_content = gitignore_path.read_text(encoding="utf-8")
    # Verify core exclusion patterns exist in .gitignore
    assert "data/" in gitignore_content
    assert "*.mp4" in gitignore_content
    assert "reports/*" in gitignore_content
    assert "audit_log.jsonl" in gitignore_content

    # If git executable is available, verify git check-ignore behavior directly
    if shutil.which("git"):
        evidence_paths = [
            "data/evidence/cam1.mp4",
            "data/evidence/audit_log.jsonl",
            "data/cache/temp_upload.mp4",
            "evidence_vault/confidential_cam.mp4",
            "reports/unified_timeline.csv",
            "reports/evidence_manifest.json",
            "sample_data/cam1_traffic_pole_reference.mp4",
            "sensitive_surveillance.mov",
            "crash_dashcam.avi",
            "witness_phone.mkv",
            "temp_processing.tmp",
        ]
        res = subprocess.run(
            ["git", "check-ignore"] + evidence_paths,
            cwd=str(root_dir),
            capture_output=True,
            text=True
        )
        ignored_lines = {line.replace("\\", "/").strip() for line in res.stdout.strip().splitlines() if line}
        for ep in evidence_paths:
            normalized_ep = ep.replace("\\", "/")
            assert normalized_ep in ignored_lines, f"Evidence path '{ep}' is NOT excluded by git!"

        # Ensure essential code, tests, and documentation are NOT excluded
        essential_paths = [
            "app.py",
            "requirements.txt",
            "README.md",
            "src/evidence_integrity.py",
            "tests/test_evidence_integrity.py",
            "docs/TECHNICAL_AUDIT.md",
        ]
        res_essential = subprocess.run(
            ["git", "check-ignore"] + essential_paths,
            cwd=str(root_dir),
            capture_output=True,
            text=True
        )
        unintended_ignored = [line for line in res_essential.stdout.strip().splitlines() if line]
        assert not unintended_ignored, f"Essential files unintentionally ignored: {unintended_ignored}"


def test_safe_ingest_uploaded_stream_valid(tmp_path: Path):
    """Verify safe ingestion of valid video stream into vault with temporary cleanup."""
    import io

    vault_dir = tmp_path / "vault"
    temp_dir = tmp_path / "temp"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(
        base_evidence_dir=vault_dir,
        audit_log_path=audit_file,
        temp_upload_dir=temp_dir
    )

    # Valid MP4 header: 4-byte box size + b"ftyp" + brand
    mp4_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 200
    stream = io.BytesIO(mp4_bytes)

    record, err = em.safe_ingest_uploaded_stream(
        file_name="patrol_cam.mp4",
        file_bytes=stream,
        camera_name="Patrol Dashcam"
    )

    assert err is None
    assert record is not None
    assert record.camera_name == "Patrol Dashcam"
    assert record.original_filename == "patrol_cam.mp4"
    assert Path(record.stored_path).is_file()
    assert record.file_size_bytes == len(mp4_bytes)

    # Verify temp directory was cleaned up and has no lingering files
    temp_files = list(temp_dir.iterdir())
    assert len(temp_files) == 0

    # Verify audit log contains EVIDENCE_INGESTED
    logs = em.load_audit_log()
    upload_logs = [l for l in logs if l["action"] == "EVIDENCE_INGESTED"]
    assert len(upload_logs) == 1
    assert upload_logs[0]["details"]["original_filename"] == "patrol_cam.mp4"


def test_safe_ingest_uploaded_stream_empty(tmp_path: Path):
    """Verify that empty 0-byte stream is rejected."""
    import io

    em = EvidenceManager(
        base_evidence_dir=tmp_path / "vault",
        audit_log_path=tmp_path / "audit.jsonl"
    )
    empty_stream = io.BytesIO(b"")

    record, err = em.safe_ingest_uploaded_stream("empty.mp4", empty_stream, "Cam")
    assert record is None
    assert err is not None
    assert "0 bytes" in err


def test_safe_ingest_uploaded_stream_invalid_magic_bytes(tmp_path: Path):
    """Verify that streams with invalid magic bytes / non-video headers are rejected."""
    import io

    temp_dir = tmp_path / "temp"
    em = EvidenceManager(
        base_evidence_dir=tmp_path / "vault",
        audit_log_path=tmp_path / "audit.jsonl",
        temp_upload_dir=temp_dir
    )
    fake_stream = io.BytesIO(b"NOT_A_VIDEO_FILE_CONTENT_JUST_TEXT_DATA_1234567890")

    record, err = em.safe_ingest_uploaded_stream("fake.mp4", fake_stream, "Cam")
    assert record is None
    assert err is not None
    assert "video container signature" in err

    # Temp file must be cleaned up even when header validation fails
    if temp_dir.exists():
        assert list(temp_dir.iterdir()) == []


def test_safe_ingest_uploaded_stream_oversized(tmp_path: Path):
    """Verify that uploads exceeding max_size_bytes are rejected and cleaned up."""
    import io

    temp_dir = tmp_path / "temp"
    em = EvidenceManager(
        base_evidence_dir=tmp_path / "vault",
        audit_log_path=tmp_path / "audit.jsonl",
        temp_upload_dir=temp_dir
    )
    large_stream = io.BytesIO(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 500)

    record, err = em.safe_ingest_uploaded_stream("large.mp4", large_stream, "Cam", max_bytes=200)
    assert record is None
    assert err is not None
    assert "exceeds maximum allowed size" in err

    # Temp file must be cleaned up
    if temp_dir.exists():
        assert list(temp_dir.iterdir()) == []


def test_camera_name_illegal_characters_sanitized(tmp_path: Path):
    """Verify that camera names with Windows illegal path characters are sanitized without failing."""
    vault_dir = tmp_path / "vault"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=tmp_path / "audit.jsonl")

    src_file = tmp_path / "cam_source.mp4"
    src_file.write_bytes(b"VIDEO_PAYLOAD_TEST_DATA")

    # Windows illegal characters: / \ : * ? " < > |
    illegal_camera_name = 'Cam: 1 / Front \\ Street * ? "Traffic" <East> | Post'
    record = em.register_evidence_file(src_file, camera_name=illegal_camera_name)

    assert record.camera_name == illegal_camera_name  # Preserved for display
    assert Path(record.stored_path).is_file()  # Safely written to disk
    stored_name = Path(record.stored_path).name
    for char in [":", "/", "\\", "*", "?", '"', "<", ">", "|"]:
        assert char not in stored_name


def test_is_valid_video_container_mp4_mov_atom_locations():
    """Verify that MP4/MOV files are recognized regardless of atom ordering, while corrupt/empty files are rejected."""
    # 1. Standard MP4 with ftyp
    mp4_ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 100
    assert EvidenceManager.is_valid_video_container(mp4_ftyp) is True

    # 2. Faststart MOV with moov at start
    mov_moov_start = b"\x00\x00\x00\x20moov\x00\x00\x00\x18mvhd" + b"\x00" * 100
    assert EvidenceManager.is_valid_video_container(mov_moov_start) is True

    # 3. Non-faststart MP4/MOV with mdat at start and moov at tail
    mov_moov_tail = b"\x00\x00\x01\x00mdat" + b"\x00" * 200 + b"\x00\x00\x00\x20moov\x00\x00\x00\x18mvhd"
    assert EvidenceManager.is_valid_video_container(mov_moov_tail) is True

    # 4. QuickTime file starting with 'wide' atom then 'mdat'
    qt_wide_mdat = b"\x00\x00\x00\x08wide\x00\x00\x00\x80mdat" + b"\x00" * 100
    assert EvidenceManager.is_valid_video_container(qt_wide_mdat) is True

    # 5. AVI container
    avi_data = b"RIFF\x00\x00\x01\x00AVI LIST\x00\x00\x00\x50hdrl" + b"\x00" * 50
    assert EvidenceManager.is_valid_video_container(avi_data) is True

    # 6. MKV container
    mkv_data = b"\x1a\x45\xdf\xa3\x93B\x82\x88matroska" + b"\x00" * 50
    assert EvidenceManager.is_valid_video_container(mkv_data) is True

    # 7. Zero-byte stream -> False
    assert EvidenceManager.is_valid_video_container(b"") is False

    # 8. Corrupt / non-video files -> False
    assert EvidenceManager.is_valid_video_container(b"\x00" * 100) is False
    assert EvidenceManager.is_valid_video_container(b"THIS_IS_A_TEXT_FILE_NOT_VIDEO") is False
    assert EvidenceManager.is_valid_video_container(b"\xff\xd8\xff\xe0\x00\x10JFIF") is False


def test_temporary_files_removed_on_success_and_failure(tmp_path: Path):
    """Verify that temporary files are cleaned up in both successful and failed ingest operations."""
    import io

    vault_dir = tmp_path / "vault"
    temp_dir = tmp_path / "temp_isolated"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file, temp_upload_dir=temp_dir)

    # 1. Successful upload
    valid_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 100
    rec, err = em.safe_ingest_uploaded_stream("good.mp4", io.BytesIO(valid_bytes), "Cam Good", temp_dir=temp_dir)
    assert err is None
    assert rec is not None
    assert list(temp_dir.iterdir()) == []  # Cleaned up

    # 2. Failed upload (corrupt bytes)
    rec_bad, err_bad = em.safe_ingest_uploaded_stream("corrupt.mp4", io.BytesIO(b"CORRUPT_BYTES_DATA"), "Cam Bad", temp_dir=temp_dir)
    assert rec_bad is None
    assert err_bad is not None
    assert list(temp_dir.iterdir()) == []  # Cleaned up


def test_original_evidence_file_remains_unmodified(tmp_path: Path):
    """Verify that original evidence file remains byte-for-byte immutable during registration."""
    vault_dir = tmp_path / "vault"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=tmp_path / "audit.jsonl")

    original_file = tmp_path / "original_dashcam.mp4"
    original_payload = b"ORIGINAL_EVIDENCE_PAYLOAD_IMMUTABLE_HASH_2026"
    original_file.write_bytes(original_payload)

    initial_hash = EvidenceManager.compute_sha256(original_file)
    initial_stat = original_file.stat()

    # Ingest into custody vault
    record = em.register_evidence_file(original_file, camera_name="Patrol Dashcam", copy_to_evidence_vault=True)

    # Re-verify original file hash and size
    post_hash = EvidenceManager.compute_sha256(original_file)
    post_stat = original_file.stat()

    assert post_hash == initial_hash
    assert post_stat.st_size == initial_stat.st_size
    assert original_file.read_bytes() == original_payload
    assert record.sha256_hash == initial_hash


def test_audit_log_records_examiner_operator_timestamp_and_reason(tmp_path: Path):
    """Verify that manual overrides and decisions persist operator, timestamp, and reason."""
    from datetime import datetime

    audit_file = tmp_path / "audit_test.jsonl"
    em = EvidenceManager(base_evidence_dir=tmp_path / "vault", audit_log_path=audit_file)

    # Log examiner manual offset override
    em.record_audit(
        action="MANUAL_OFFSET_APPLIED",
        details={
            "camera_id": "cam_2",
            "offset": 4.5,
            "rate": 1.00005,
            "reason": "Corroborated by pedestrian crossing signal timing"
        },
        operator="Det. Miller #4092"
    )

    # Log examiner candidate rejection
    em.record_audit(
        action="CANDIDATE_REJECTED",
        details={
            "match_id": "MATCH-04",
            "reason": "Vehicle make in Camera 2 does not match reference suspect vehicle"
        },
        operator="Det. Miller #4092"
    )

    logs = em.load_audit_log()
    assert len(logs) == 2

    # Check entry 0
    e0 = logs[0]
    assert e0["operator"] == "Det. Miller #4092"
    assert e0["action"] == "MANUAL_OFFSET_APPLIED"
    assert e0["details"]["reason"] == "Corroborated by pedestrian crossing signal timing"
    assert "timestamp_utc" in e0
    # Must parse as ISO timestamp
    ts0 = datetime.fromisoformat(e0["timestamp_utc"])
    assert ts0.year >= 2026

    # Check entry 1
    e1 = logs[1]
    assert e1["operator"] == "Det. Miller #4092"
    assert e1["action"] == "CANDIDATE_REJECTED"
    assert e1["details"]["reason"] == "Vehicle make in Camera 2 does not match reference suspect vehicle"


def test_safe_ingest_uploaded_stream_end_to_end_real_video_zero_byte_and_corrupt(tmp_path: Path):
    """Regression test: verify safe upload stream ingest on real generated MP4, zero-byte, and corrupt video."""
    import io
    from src.synthetic_generator import SyntheticVideoGenerator
    from src.video_metadata import VideoMetadataExtractor

    vault_dir = tmp_path / "vault"
    temp_dir = tmp_path / "temp"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file, temp_upload_dir=temp_dir)
    extractor = VideoMetadataExtractor()

    # 1. Real valid video
    gen = SyntheticVideoGenerator(output_dir=tmp_path / "gen")
    real_video_path = tmp_path / "gen" / "source_stream.mp4"
    gen.generate_camera_video(
        file_path=real_video_path,
        camera_name="Live Camera 1",
        duration_sec=2.0,
        fps=20.0,
        true_offset_sec=0.0,
        true_rate_drift=1.0,
        global_event_times_sec=[1.0]
    )
    real_bytes = real_video_path.read_bytes()

    record, err = em.safe_ingest_uploaded_stream(
        file_name="uploaded_stream.mp4",
        file_bytes=io.BytesIO(real_bytes),
        camera_name="Live Camera 1",
        temp_dir=temp_dir
    )
    assert err is None
    assert record is not None
    assert Path(record.stored_path).is_file()
    assert record.file_size_bytes == len(real_bytes)
    assert list(temp_dir.iterdir()) == []  # Temp file cleaned up

    # Extract metadata from vault copy
    meta = extractor.extract(record.stored_path, camera_name=record.camera_name, video_id="CAM_LIVE", sha256_hash=record.sha256_hash)
    assert meta.is_valid is True
    assert meta.total_frames == 40
    assert abs(meta.nominal_fps - 20.0) < 0.5
    assert meta.sha256_hash == EvidenceManager.compute_sha256(real_video_path)

    # 2. Zero-byte stream
    rec_zero, err_zero = em.safe_ingest_uploaded_stream(
        file_name="zero.mp4",
        file_bytes=io.BytesIO(b""),
        camera_name="Zero Camera",
        temp_dir=temp_dir
    )
    assert rec_zero is None
    assert err_zero is not None
    assert "0 bytes" in err_zero
    assert list(temp_dir.iterdir()) == []

    # 3. Corrupt video stream
    corrupt_bytes = b"CORRUPT_HEADER_AND_NOISE_DATA_STREAM_TEST_987654321"
    rec_corrupt, err_corrupt = em.safe_ingest_uploaded_stream(
        file_name="corrupt.mp4",
        file_bytes=io.BytesIO(corrupt_bytes),
        camera_name="Corrupt Camera",
        temp_dir=temp_dir
    )
    assert rec_corrupt is None
    assert err_corrupt is not None
    assert "video container signature" in err_corrupt
    assert list(temp_dir.iterdir()) == []
