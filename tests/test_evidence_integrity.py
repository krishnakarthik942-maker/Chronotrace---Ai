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
