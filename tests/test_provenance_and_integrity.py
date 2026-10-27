"""Comprehensive regression tests for Phase 2C: Report Traceability and Evidence Integrity.

Tests:
1. All three data classifications (MEASURED_FACT, ALGORITHMIC_ESTIMATE, EXAMINER_OVERRIDE).
2. Inferred events on the reference camera (classified as ALGORITHMIC_ESTIMATE, not MEASURED_FACT).
3. Missing provenance values handled gracefully without inventing fabricated values.
4. Candidate versus confirmed status in CSV and JSON exports.
5. Unsafe paths (path traversal) and sanitization collisions.
6. Failed ingestion and partial-file cleanup.
7. Hash verification and original-file immutability (read-only locking and tamper detection).
8. Audit records for manual overrides and examiner decisions with cryptographic hash chaining.
"""

from __future__ import annotations

import csv
import io
import json
import os
import stat
from pathlib import Path

import pytest

from src.evidence_integrity import AuditLogEntry, EvidenceManager, EvidenceRecord, SOFTWARE_VERSION
from src.event_detector import VisualEvent
from src.report_exporter import ForensicReportExporter
from src.synthetic_generator import SyntheticVideoGenerator
from src.time_aligner import CameraAlignmentResult, TimeAligner, TimeAnchorPoint
from src.timeline_builder import TimelineBuilder, UnifiedTimelineEvent
from src.video_metadata import VideoMetadata, VideoMetadataExtractor
from src.visual_matcher import CandidateMatch


def test_three_data_classifications(tmp_path: Path):
    """Test that MEASURED_FACT, ALGORITHMIC_ESTIMATE, and EXAMINER_OVERRIDE are
    consistently assigned and preserved in both CSV and JSON exports."""
    reports_dir = tmp_path / "reports"
    exporter = ForensicReportExporter(reports_dir=reports_dir)

    metadata = {
        "cam_ref": VideoMetadata(
            video_id="cam_ref", camera_name="Reference Cam", file_path="cam_ref.mp4",
            file_name="cam_ref.mp4", file_size_bytes=1000, sha256_hash="hash_ref",
            duration_sec=10.0, nominal_fps=30.0, total_frames=300
        ),
        "cam_alg": VideoMetadata(
            video_id="cam_alg", camera_name="Algorithmic Cam", file_path="cam_alg.mp4",
            file_name="cam_alg.mp4", file_size_bytes=1000, sha256_hash="hash_alg",
            duration_sec=10.0, nominal_fps=30.0, total_frames=300
        ),
        "cam_man": VideoMetadata(
            video_id="cam_man", camera_name="Manual Cam", file_path="cam_man.mp4",
            file_name="cam_man.mp4", file_size_bytes=1000, sha256_hash="hash_man",
            duration_sec=10.0, nominal_fps=30.0, total_frames=300
        ),
    }

    alignments = {
        "cam_ref": CameraAlignmentResult(
            camera_id="cam_ref", camera_name="Reference Cam", is_reference=True,
            offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED", method_description="REFERENCE_CAMERA"
        ),
        "cam_alg": CameraAlignmentResult(
            camera_id="cam_alg", camera_name="Algorithmic Cam", is_reference=False,
            offset_sec=2.5, rate_drift=1.0, status="SYNCHRONIZED", method_description="RANSAC_LINEAR_FIT",
            supporting_match_ids=["ANCHOR-01", "ANCHOR-02"]
        ),
        "cam_man": CameraAlignmentResult(
            camera_id="cam_man", camera_name="Manual Cam", is_reference=False,
            offset_sec=5.0, rate_drift=1.0, status="MANUAL_OVERRIDE", method_description="EXAMINER_MANUAL_CALIBRATION",
            override_reason="Corroborated vehicle impact audio pop at t=5.0s", audit_ref="AUDIT-20261027-0001"
        ),
    }

    events = [
        VisualEvent("EVT-REF", "cam_ref", "Reference Cam", "cam_ref.mp4", 30, 1.0, "direct_spike", 0.95),
        VisualEvent("EVT-ALG", "cam_alg", "Algorithmic Cam", "cam_alg.mp4", 30, 1.0, "direct_spike", 0.88),
        VisualEvent("EVT-MAN", "cam_man", "Manual Cam", "cam_man.mp4", 30, 1.0, "direct_spike", 0.91),
    ]

    tb = TimelineBuilder(metadata_dict=metadata, alignments=alignments)
    unified = tb.build_unified_events(events, confirmed_matches=[], candidate_matches=[])

    ev_ref = next(e for e in unified if e.event_id == "EVT-REF")
    ev_alg = next(e for e in unified if e.event_id == "EVT-ALG")
    ev_man = next(e for e in unified if e.event_id == "EVT-MAN")

    assert ev_ref.data_classification == "MEASURED_FACT"
    assert ev_alg.data_classification == "ALGORITHMIC_ESTIMATE"
    assert ev_man.data_classification == "EXAMINER_OVERRIDE"
    assert ev_man.override_reason == "Corroborated vehicle impact audio pop at t=5.0s"
    assert ev_man.audit_ref == "AUDIT-20261027-0001"

    # Export to CSV and JSON
    csv_path = exporter.export_unified_timeline_csv(unified)
    json_path = exporter.export_unified_timeline_json(unified)

    # Check CSV content
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        row_ref = next(r for r in reader if r["event_id"] == "EVT-REF")
        row_alg = next(r for r in reader if r["event_id"] == "EVT-ALG")
        row_man = next(r for r in reader if r["event_id"] == "EVT-MAN")

        assert row_ref["data_classification"] == "MEASURED_FACT"
        assert row_alg["data_classification"] == "ALGORITHMIC_ESTIMATE"
        assert row_man["data_classification"] == "EXAMINER_OVERRIDE"
        assert row_man["override_reason"] == "Corroborated vehicle impact audio pop at t=5.0s"
        assert row_man["audit_ref"] == "AUDIT-20261027-0001"

    # Check JSON content
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
        assert doc["report_type"] == "FORENSIC_UNIFIED_TIMELINE_REPORT"
        assert doc["software_name"] == "ChronoTrace AI"
        events_json = doc["events"]
        assert len(events_json) == 3

        j_ref = next(e for e in events_json if e["event_id"] == "EVT-REF")
        j_alg = next(e for e in events_json if e["event_id"] == "EVT-ALG")
        j_man = next(e for e in events_json if e["event_id"] == "EVT-MAN")

        assert j_ref["data_classification"] == "MEASURED_FACT"
        assert j_alg["data_classification"] == "ALGORITHMIC_ESTIMATE"
        assert j_man["data_classification"] == "EXAMINER_OVERRIDE"
        assert j_man["override_reason"] == "Corroborated vehicle impact audio pop at t=5.0s"
        assert j_man["audit_ref"] == "AUDIT-20261027-0001"


def test_inferred_event_on_reference_camera(tmp_path: Path):
    """An inferred or interpolated event on the reference camera must be classified
    as ALGORITHMIC_ESTIMATE, NOT automatically as MEASURED_FACT."""
    metadata = {
        "cam_ref": VideoMetadata(
            video_id="cam_ref", camera_name="Reference Cam", file_path="cam_ref.mp4",
            file_name="cam_ref.mp4", file_size_bytes=1000, sha256_hash="hash_ref",
            duration_sec=10.0, nominal_fps=30.0, total_frames=300
        ),
    }

    alignments = {
        "cam_ref": CameraAlignmentResult(
            camera_id="cam_ref", camera_name="Reference Cam", is_reference=True,
            offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"
        ),
    }

    # Direct observation vs inferred events on the reference camera
    direct_evt = VisualEvent("EVT-DIRECT", "cam_ref", "Reference Cam", "cam_ref.mp4", 30, 1.0, "visual_spike", 0.95)
    inferred_flag = VisualEvent("EVT-INF-FLAG", "cam_ref", "Reference Cam", "cam_ref.mp4", 45, 1.5, "interpolated_spike", 0.70, is_inferred=True)
    inferred_type = VisualEvent("EVT-INF-TYPE", "cam_ref", "Reference Cam", "cam_ref.mp4", 60, 2.0, "inferred", 0.75)
    inferred_class = VisualEvent("EVT-INF-CLASS", "cam_ref", "Reference Cam", "cam_ref.mp4", 75, 2.5, "motion_gap", 0.80, data_classification="ALGORITHMIC_ESTIMATE")

    events = [direct_evt, inferred_flag, inferred_type, inferred_class]

    tb = TimelineBuilder(metadata_dict=metadata, alignments=alignments)
    unified = tb.build_unified_events(events, confirmed_matches=[], candidate_matches=[])

    ev_direct = next(e for e in unified if e.event_id == "EVT-DIRECT")
    ev_flag = next(e for e in unified if e.event_id == "EVT-INF-FLAG")
    ev_type = next(e for e in unified if e.event_id == "EVT-INF-TYPE")
    ev_class = next(e for e in unified if e.event_id == "EVT-INF-CLASS")

    # Direct observation is factual
    assert ev_direct.data_classification == "MEASURED_FACT"

    # All inferred events on reference camera MUST be algorithmic estimates
    assert ev_flag.data_classification == "ALGORITHMIC_ESTIMATE"
    assert ev_type.data_classification == "ALGORITHMIC_ESTIMATE"
    assert ev_class.data_classification == "ALGORITHMIC_ESTIMATE"

    # Verify in export
    exporter = ForensicReportExporter(reports_dir=tmp_path / "reports")
    csv_path = exporter.export_unified_timeline_csv(unified)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        for row in reader:
            if row["event_id"] == "EVT-DIRECT":
                assert row["data_classification"] == "MEASURED_FACT"
            else:
                assert row["data_classification"] == "ALGORITHMIC_ESTIMATE"


def test_missing_provenance_values_handled_gracefully(tmp_path: Path):
    """Missing or uncalculated provenance fields must produce explicit null/unknown
    placeholders rather than crashing, raising formatting errors, or inventing values."""
    exporter = ForensicReportExporter(reports_dir=tmp_path / "reports")

    # Event on unaligned camera with unknown/missing fields
    ev = UnifiedTimelineEvent(
        event_id="EVT-UNKNOWN",
        camera_id="cam_uncal",
        camera_name="Uncalibrated Cam",
        source_filename="unknown.mp4",
        frame_number=100,
        source_timestamp_sec=3.3333,
        corrected_timestamp_sec=None,  # Unaligned
        event_type="motion",
        visual_category="DETECTED_EVENT",
        matched_with_camera=None,
        matched_event_id=None,
        confidence=None,  # No confidence genuinely calculated
        review_status="UNREVIEWED",
        uncertainty_sec=None,  # Unknown uncertainty
        data_classification="MEASURED_FACT",
        calibration_method=None,
        synchronization_status=None,
        supporting_anchors=[],
        override_reason=None,
        audit_ref=None,
        notes="Uncalibrated camera event"
    )

    csv_path = exporter.export_unified_timeline_csv([ev])
    json_path = exporter.export_unified_timeline_json([ev])

    # Check CSV representation
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["corrected_timestamp_sec"] == "UNALIGNED"
        assert row["confidence"] == "NONE"
        assert row["uncertainty_sec"] == "UNKNOWN"
        assert row["calibration_method"] == "NONE"
        assert row["synchronization_status"] == "UNALIGNED"
        assert row["supporting_anchors"] == "NONE"
        assert row["override_reason"] == "N/A"
        assert row["audit_ref"] == "N/A"

    # Check JSON representation
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
        event_json = doc["events"][0]
        assert event_json["corrected_timestamp_sec"] is None
        assert event_json["confidence"] is None
        assert event_json["uncertainty_sec"] is None
        assert event_json["matched_with_camera"] is None
        assert event_json["matched_event_id"] is None
        assert event_json["calibration_method"] is None
        assert event_json["override_reason"] is None
        assert event_json["audit_ref"] is None
        assert event_json["supporting_anchors"] == []


def test_candidate_versus_confirmed_status_in_csv_and_json(tmp_path: Path):
    """Candidate matches must never be falsely marked as confirmed in CSV or JSON exports."""
    exporter = ForensicReportExporter(reports_dir=tmp_path / "reports")

    candidate_match = CandidateMatch(
        match_id="MATCH-CANDIDATE-01",
        camera_a_id="cam_1",
        camera_a_name="Camera 1",
        event_a_id="EVT-01",
        frame_a=30,
        time_a_sec=1.0,
        thumb_a=None,
        camera_b_id="cam_2",
        camera_b_name="Camera 2",
        event_b_id="EVT-02",
        frame_b=45,
        time_b_sec=1.5,
        thumb_b=None,
        similarity_score=0.78,
        phash_distance=12,
        color_similarity=0.82,
        review_status="CANDIDATE_UNVERIFIED",
        is_anchor=False
    )

    confirmed_match = CandidateMatch(
        match_id="MATCH-CONFIRMED-02",
        camera_a_id="cam_1",
        camera_a_name="Camera 1",
        event_a_id="EVT-03",
        frame_a=90,
        time_a_sec=3.0,
        thumb_a=None,
        camera_b_id="cam_2",
        camera_b_name="Camera 2",
        event_b_id="EVT-04",
        frame_b=105,
        time_b_sec=3.5,
        thumb_b=None,
        similarity_score=0.92,
        phash_distance=4,
        color_similarity=0.95,
        review_status="CONFIRMED_BY_USER",
        is_anchor=True
    )

    # 1. Test event_matches.csv
    matches_csv = exporter.export_event_matches_csv([candidate_match, confirmed_match])
    with open(matches_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2
        r_cand = next(r for r in rows if r["match_id"] == "MATCH-CANDIDATE-01")
        r_conf = next(r for r in rows if r["match_id"] == "MATCH-CONFIRMED-02")

        assert r_cand["review_status"] == "CANDIDATE_UNVERIFIED"
        assert r_cand["is_anchor"] == "FALSE"

        assert r_conf["review_status"] == "CONFIRMED_BY_USER"
        assert r_conf["is_anchor"] == "TRUE"

    # 2. Test TimelineBuilder and unified exports
    metadata = {
        "cam_1": VideoMetadata("cam_1", "Camera 1", "c1.mp4", "c1.mp4", 100, "h1", duration_sec=10, nominal_fps=30, total_frames=300),
        "cam_2": VideoMetadata("cam_2", "Camera 2", "c2.mp4", "c2.mp4", 100, "h2", duration_sec=10, nominal_fps=30, total_frames=300),
    }
    alignments = {
        "cam_1": CameraAlignmentResult("cam_1", "Camera 1", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_2": CameraAlignmentResult("cam_2", "Camera 2", is_reference=False, offset_sec=0.5, rate_drift=1.0, status="SYNCHRONIZED"),
    }

    events = [
        VisualEvent("EVT-01", "cam_1", "Camera 1", "c1.mp4", 30, 1.0, "motion", 0.8),
        VisualEvent("EVT-02", "cam_2", "Camera 2", "c2.mp4", 45, 1.5, "motion", 0.8),
        VisualEvent("EVT-03", "cam_1", "Camera 1", "c1.mp4", 90, 3.0, "motion", 0.9),
        VisualEvent("EVT-04", "cam_2", "Camera 2", "c2.mp4", 105, 3.5, "motion", 0.9),
    ]

    tb = TimelineBuilder(metadata_dict=metadata, alignments=alignments)
    unified = tb.build_unified_events(
        events,
        confirmed_matches=[confirmed_match],
        candidate_matches=[candidate_match]
    )

    ev_01 = next(e for e in unified if e.event_id == "EVT-01")
    ev_03 = next(e for e in unified if e.event_id == "EVT-03")

    # Unverified candidate MUST remain CANDIDATE_UNVERIFIED
    assert ev_01.review_status == "CANDIDATE_UNVERIFIED"
    assert ev_01.visual_category == "SUGGESTED_MATCH"

    # Confirmed match MUST be CONFIRMED
    assert ev_03.review_status == "CONFIRMED"
    assert ev_03.visual_category == "VERIFIED_ANCHOR"

    # Verify in JSON export
    json_path = exporter.export_unified_timeline_json(unified)
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
        events_json = doc["events"]
        j_01 = next(e for e in events_json if e["event_id"] == "EVT-01")
        j_03 = next(e for e in events_json if e["event_id"] == "EVT-03")
        assert j_01["review_status"] == "CANDIDATE_UNVERIFIED"
        assert j_03["review_status"] == "CONFIRMED"


def test_unsafe_paths_and_sanitization_collisions(tmp_path: Path):
    """Test path traversal neutralization and collision detection in evidence vaulting."""
    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    # 1. Path traversal neutralization
    assert "/" not in em.sanitize_path_component("../../etc/passwd")
    assert "\\" not in em.sanitize_path_component("..\\..\\windows\\system32\\cmd.exe")
    assert ".." not in em.sanitize_path_component("../../../evil_cam")
    assert em.sanitize_path_component("camera:01*?<>|") == "camera_01"
    assert em.sanitize_path_component("   ") != ""  # Safe fallback

    # 2. Collision detection after sanitization: two distinct files that sanitize to the same name
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    file_a = src_dir / "cam_file.mp4"
    file_a.write_bytes(b"DATA_FILE_A_CONTENT_1234567890")

    file_b = src_dir / "cam_file_other.mp4"
    file_b.write_bytes(b"DATA_FILE_B_CONTENT_DIFFERENT_HASH")

    # Register file A
    rec_a = em.register_evidence_file(
        file_a,
        camera_name="Cam#1",
        original_filename="footage.mp4"
    )

    # Register file B with same sanitized camera name and display name
    rec_b = em.register_evidence_file(
        file_b,
        camera_name="Cam/1",  # Sanitizes to "Cam_1", same as "Cam#1"
        original_filename="footage.mp4"
    )

    assert rec_a.stored_path != rec_b.stored_path
    assert Path(rec_a.stored_path).is_file()
    assert Path(rec_b.stored_path).is_file()
    assert Path(rec_a.stored_path).read_bytes() == b"DATA_FILE_A_CONTENT_1234567890"
    assert Path(rec_b.stored_path).read_bytes() == b"DATA_FILE_B_CONTENT_DIFFERENT_HASH"


def test_failed_ingestion_and_partial_file_cleanup(tmp_path: Path):
    """Failed ingestion must clean up temporary files, leave no partial vault files,
    and avoid writing misleading custody records."""
    vault_dir = tmp_path / "vault"
    temp_dir = tmp_path / "temp"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file, temp_upload_dir=temp_dir)

    # 1. Ingestion of non-video bytes fails signature check
    record_bad, err_bad = em.safe_ingest_uploaded_stream(
        file_name="malicious.exe",
        file_bytes=b"MZ\x90\x00NotAVideo",
        camera_name="Camera 1",
        temp_dir=temp_dir
    )
    assert record_bad is None
    assert err_bad is not None
    assert "video container signature" in err_bad
    assert not temp_dir.exists() or list(temp_dir.iterdir()) == []
    assert not vault_dir.exists() or list(vault_dir.iterdir()) == []

    # 2. Ingestion of corrupt/undecodable container with validate_decodable=True
    # Craft an ISO box container that has magic 'ftyp' but invalid/undecodable stream data
    fake_mp4_header = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08mdat" + b"\x00" * 1024
    record_corrupt, err_corrupt = em.safe_ingest_uploaded_stream(
        file_name="corrupt_stream.mp4",
        file_bytes=fake_mp4_header,
        camera_name="Camera Corrupt",
        temp_dir=temp_dir,
        validate_decodable=True
    )
    assert record_corrupt is None
    assert err_corrupt is not None
    assert "cannot be opened or decoded" in err_corrupt

    # Verify no temporary files or vault files remain
    assert not temp_dir.exists() or list(temp_dir.iterdir()) == []
    assert not vault_dir.exists() or list(vault_dir.iterdir()) == []

    # Verify audit log contains NO evidence ingested record
    audit_entries = em.load_audit_log()
    ingested_actions = [a for a in audit_entries if a.get("action") == "EVIDENCE_INGESTED"]
    assert len(ingested_actions) == 0


def test_hash_verification_and_original_file_immutability(tmp_path: Path):
    """Verify SHA-256 calculation, original file immutability, read-only vault protection,
    and tamper detection."""
    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    # Generate a real valid synthetic video
    gen = SyntheticVideoGenerator(output_dir=tmp_path / "gen")
    src_video = tmp_path / "gen" / "original_source.mp4"
    gen.generate_camera_video(
        file_path=src_video,
        camera_name="Source Cam",
        duration_sec=2.0,
        fps=20.0,
        true_offset_sec=0.0,
        true_rate_drift=1.0,
        global_event_times_sec=[1.0]
    )

    src_hash_before = em.compute_sha256(src_video)
    rec = em.register_evidence_file(src_video, camera_name="Source Cam", copy_to_evidence_vault=True)

    # 1. Source file remains completely unmodified
    src_hash_after = em.compute_sha256(src_video)
    assert src_hash_before == src_hash_after

    # 2. Vault copy matches original hash
    vault_path = Path(rec.stored_path)
    assert vault_path.is_file()
    assert em.compute_sha256(vault_path) == src_hash_before

    # 3. Vault copy is marked read-only
    vault_mode = vault_path.stat().st_mode
    assert (vault_mode & stat.S_IWRITE) == 0, "Vaulted evidence file should not have write permission."

    # 4. Tamper detection: unlock file, modify content, verify tamper alert
    em.unlock_file(vault_path)
    with open(vault_path, "ab") as f:
        f.write(b"TAMPER_PAYLOAD")
    em.set_read_only(vault_path)

    is_intact = em.verify_integrity(rec.evidence_id)
    assert is_intact is False
    assert rec.is_tampered is True


def test_audit_records_for_manual_overrides_and_examiner_decisions(tmp_path: Path):
    """Examiner manual clock adjustments, rejected candidates, and verified anchors
    must persist with UTC timestamps, operator, and reason, with cryptographic integrity."""
    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    # 1. Record manual offset adjustment
    entry1 = em.record_audit(
        action="MANUAL_OFFSET_APPLIED",
        details={
            "camera_id": "cam_2",
            "offset_sec": 3.75,
            "rate_drift": 1.00005,
            "reason": "Corroborated vehicle impact audible on Cam 1 at 14:02:05 UTC"
        },
        operator="Examiner_Det_Miller"
    )

    # 2. Record rejected candidate match
    entry2 = em.record_audit(
        action="CANDIDATE_REJECTED",
        details={
            "match_id": "CAND-0042",
            "reason": "False visual correspondence caused by headlight glare reflection"
        },
        operator="Examiner_Det_Miller"
    )

    # 3. Record anchor confirmation rejection
    entry3 = em.record_audit(
        action="ANCHOR_CONFIRMATION_REJECTED",
        details={
            "match_id": "CAND-0099",
            "reason": "Neither camera is linked to reference timeline (unanchored rejection)"
        },
        operator="Examiner_Det_Miller"
    )

    # 4. Record verified anchor addition
    entry4 = em.record_audit(
        action="VERIFIED_ANCHOR_ADDED",
        details={
            "anchor_id": "ANCHOR-MANUAL-001",
            "camera_id": "cam_3",
            "src_time": 12.4,
            "ref_time": 15.9,
            "reason": "Pedestrian crossing street at crosswalk mark visible on both feeds"
        },
        operator="Examiner_Det_Miller"
    )

    # Verify audit entries
    entries = em.load_audit_log()
    assert len(entries) == 4

    for item in entries:
        assert item["operator"] == "Examiner_Det_Miller"
        assert item["timestamp_utc"].endswith("Z") or "+00:00" in item["timestamp_utc"] or "T" in item["timestamp_utc"]
        assert "reason" in item["details"]
        assert len(item["details"]["reason"]) > 10

    # Verify cryptographic hash chain
    is_valid, msg = em.verify_audit_log_integrity()
    assert is_valid is True
    assert "verified intact (4 entries)" in msg

    # Tamper with the audit log: alter reason in second line
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    tampered_entry = json.loads(lines[1])
    tampered_entry["details"]["reason"] = "TAMPERED_REASON_NO_HASH_MATCH"
    lines[1] = json.dumps(tampered_entry)
    audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    is_valid_after_tamper, tamper_msg = em.verify_audit_log_integrity()
    assert is_valid_after_tamper is False
    assert "tampered with" in tamper_msg or "Hash-chain broken" in tamper_msg


def test_audit_log_tamper_scenarios_deleted_reordered_truncated(tmp_path: Path):
    """Verify hash-chain validation detects edited, deleted, reordered, and truncated log entries."""
    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    for i in range(5):
        em.record_audit(f"ACTION_{i}", {"step": i, "reason": f"Operational step {i}"}, operator="Examiner_1")

    assert em.verify_audit_log_integrity()[0] is True
    valid_lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(valid_lines) == 5
    head_hash = json.loads(valid_lines[-1])["entry_hash"]

    # Scenario 1: Deletion of genesis entry (line 0)
    audit_file.write_text("\n".join(valid_lines[1:]) + "\n", encoding="utf-8")
    ok, msg = em.verify_audit_log_integrity()
    assert ok is False
    assert "Hash-chain broken" in msg

    # Scenario 2: Deletion of middle entry (line 2)
    lines_deleted_mid = valid_lines[:2] + valid_lines[3:]
    audit_file.write_text("\n".join(lines_deleted_mid) + "\n", encoding="utf-8")
    ok, msg = em.verify_audit_log_integrity()
    assert ok is False
    assert "Hash-chain broken" in msg

    # Scenario 3: Reordering entries (swap line 1 and line 2)
    lines_reordered = [valid_lines[0], valid_lines[2], valid_lines[1], valid_lines[3], valid_lines[4]]
    audit_file.write_text("\n".join(lines_reordered) + "\n", encoding="utf-8")
    ok, msg = em.verify_audit_log_integrity()
    assert ok is False
    assert "Hash-chain broken" in msg

    # Scenario 4: Partial line truncation / corrupt JSON in transit
    corrupt_lines = valid_lines[:4] + ['{"log_id": "AUDIT-CORRUPT", "timestamp_utc": "2026-10-27T10:00:00Z"']
    audit_file.write_text("\n".join(corrupt_lines) + "\n", encoding="utf-8")
    ok, msg = em.verify_audit_log_integrity()
    assert ok is False
    assert "corrupted, or truncated" in msg

    # Scenario 5: Silent tail truncation detected via expected count / manifest head hash
    audit_file.write_text("\n".join(valid_lines[:3]) + "\n", encoding="utf-8")
    # Prefix alone is a valid chain from genesis:
    ok_prefix, _ = em.verify_audit_log_integrity()
    assert ok_prefix is True
    # BUT verifying against the expected count or manifest head hash detects the truncation:
    ok_count, msg_count = em.verify_audit_log_integrity(expected_count=5)
    assert ok_count is False
    assert "entry count mismatch" in msg_count

    ok_head, msg_head = em.verify_audit_log_integrity(expected_tail_hash=head_hash)
    assert ok_head is False
    assert "tail hash mismatch" in msg_head


def test_concurrent_audit_writes_thread_safety(tmp_path: Path):
    """Verify concurrent audit writes do not interleave or corrupt the hash chain."""
    import concurrent.futures

    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit_concurrent.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    num_threads = 8
    writes_per_thread = 10
    total_expected = num_threads * writes_per_thread

    def worker(worker_id: int):
        for step in range(writes_per_thread):
            em.record_audit(
                action=f"THREAD_ACTION_{worker_id}",
                details={"worker": worker_id, "step": step, "reason": f"Thread {worker_id} write {step}"},
                operator=f"WorkerThread_{worker_id}"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, tid) for tid in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    entries = em.load_audit_log()
    assert len(entries) == total_expected

    # Hash chain must be completely unbroken
    ok, msg = em.verify_audit_log_integrity()
    assert ok is True
    assert f"verified intact ({total_expected} entries)" in msg


def test_evidence_registry_multi_camera_collision_isolation(tmp_path: Path):
    """Verify that multiple cameras sharing the same file or colliding names
    do not overwrite each other in the evidence registry."""
    vault_dir = tmp_path / "vault"
    audit_file = tmp_path / "audit.jsonl"
    em = EvidenceManager(base_evidence_dir=vault_dir, audit_log_path=audit_file)

    src_file = tmp_path / "shared_source.mp4"
    src_file.write_bytes(b"SHARED_VIDEO_BYTES_CONTENT_12345")

    # Camera 1 registers the shared file
    rec1 = em.register_evidence_file(src_file, camera_name="Camera East", copy_to_evidence_vault=True, operator="Examiner A")

    # Camera 2 registers the exact same video file
    rec2 = em.register_evidence_file(src_file, camera_name="Camera West", copy_to_evidence_vault=True, operator="Examiner B")

    # Both records must be preserved independently in the registry
    all_recs = em.get_all_records()
    assert len(all_recs) == 2
    assert rec1.evidence_id != rec2.evidence_id
    assert em.get_record(rec1.evidence_id).camera_name == "Camera East"
    assert em.get_record(rec2.evidence_id).camera_name == "Camera West"
    assert em.get_record(rec1.evidence_id).sha256_hash == em.get_record(rec2.evidence_id).sha256_hash
