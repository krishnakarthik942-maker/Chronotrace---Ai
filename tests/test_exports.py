"""Tests for forensic report exporters, schemas, and evidence integrity declarations."""

import csv
import json
from pathlib import Path
import pytest
from src.evidence_integrity import EvidenceManager, EvidenceRecord
from src.report_exporter import (
    FORENSIC_LIMITATIONS_DISCLOSURE,
    FORENSIC_METHOD_EXPLANATION,
    ForensicReportExporter
)
from src.time_aligner import CameraAlignmentResult
from src.timeline_builder import UnifiedTimelineEvent
from src.video_metadata import VideoMetadata
from src.visual_matcher import CandidateMatch


def test_export_all_artifacts(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    exporter = ForensicReportExporter(reports_dir=reports_dir)

    # Mock inputs
    metadata = {
        "cam_1": VideoMetadata(
            video_id="cam_1",
            camera_name="Camera 1 - Reference",
            file_path="cam1.mp4",
            file_name="cam1.mp4",
            file_size_bytes=10240,
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            duration_sec=10.0,
            nominal_fps=30.0,
            total_frames=300,
            width=640,
            height=360
        )
    }

    alignments = {
        "cam_1": CameraAlignmentResult(
            camera_id="cam_1",
            camera_name="Camera 1 - Reference",
            is_reference=True,
            offset_sec=0.0,
            rate_drift=1.0,
            status="SYNCHRONIZED",
            uncertainty_sec=0.0
        )
    }

    events = [
        UnifiedTimelineEvent(
            event_id="EVT-01",
            camera_id="cam_1",
            camera_name="Camera 1 - Reference",
            source_filename="cam1.mp4",
            frame_number=90,
            source_timestamp_sec=3.0,
            corrected_timestamp_sec=3.0,
            event_type="visual_spike",
            visual_category="VERIFIED_ANCHOR",
            matched_with_camera="Camera 2",
            matched_event_id="EVT-02",
            confidence=0.95,
            review_status="CONFIRMED",
            uncertainty_sec=0.033
        )
    ]

    matches = [
        CandidateMatch(
            match_id="MATCH-01",
            camera_a_id="cam_1",
            camera_a_name="Camera 1 - Reference",
            event_a_id="EVT-01",
            frame_a=90,
            time_a_sec=3.0,
            thumb_a=None,
            camera_b_id="cam_2",
            camera_b_name="Camera 2 - Dashcam",
            event_b_id="EVT-02",
            frame_b=150,
            time_b_sec=6.5,
            thumb_b=None,
            similarity_score=0.92,
            phash_distance=4,
            color_similarity=0.88,
            review_status="CONFIRMED_BY_USER",
            is_anchor=True
        )
    ]

    evidence_records = [
        EvidenceRecord(
            evidence_id="EVID-TEST01",
            camera_name="Camera 1 - Reference",
            original_filename="cam1.mp4",
            stored_path="vault/cam1.mp4",
            file_size_bytes=10240,
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ingest_time_utc="2026-10-26T12:00:00Z"
        )
    ]

    paths = exporter.export_all(
        unified_events=events,
        matches=matches,
        alignments=alignments,
        metadata=metadata,
        evidence_records=evidence_records,
        evidence_manager=None,
        case_id="CASE-UNIT-TEST",
        investigator_name="Det. Inspector"
    )

    # 1. Verify unified_timeline.csv
    csv_timeline = Path(paths["unified_timeline_csv"])
    assert csv_timeline.is_file()
    with open(csv_timeline, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["camera_id"] == "cam_1"
        assert rows[0]["source_filename"] == "cam1.mp4"
        assert rows[0]["frame_number"] == "90"
        assert rows[0]["source_timestamp_sec"] == "3.0000"
        assert rows[0]["corrected_timestamp_sec"] == "3.0000"
        assert rows[0]["matching_evidence"] == "Camera 2:EVT-02"
        assert rows[0]["data_classification"] == "DERIVED_RESULT"
        assert rows[0]["review_status"] == "CONFIRMED"

    # 2. Verify event_matches.csv
    csv_matches = Path(paths["event_matches_csv"])
    assert csv_matches.is_file()
    with open(csv_matches, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["match_id"] == "MATCH-01"
        assert rows[0]["is_anchor"] == "TRUE"
        assert "heuristic candidate generator" in rows[0]["forensic_disclaimer"]

    # 3. Verify synchronization_report.json
    sync_report_path = Path(paths["synchronization_report_json"])
    assert sync_report_path.is_file()
    with open(sync_report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["case_id"] == "CASE-UNIT-TEST"
        assert data["methodology"]["description"] == FORENSIC_METHOD_EXPLANATION
        assert data["methodology"]["limitations_disclosure"] == FORENSIC_LIMITATIONS_DISCLOSURE
        assert "cam_1" in data["camera_alignments"]

    # 4. Verify evidence_manifest.json
    manifest_path = Path(paths["evidence_manifest_json"])
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        assert manifest["case_id"] == "CASE-UNIT-TEST"
        assert manifest["evidence_files_count"] == 1
        assert manifest["evidence_catalog"][0]["sha256_hash"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
