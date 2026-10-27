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


def test_dynamic_data_classification_and_review_status_preservation(tmp_path: Path):
    """Verify that export distinguishes MEASURED_FACT, ALGORITHMIC_ESTIMATE, and EXAMINER_OVERRIDE,
    and preserves CANDIDATE_UNVERIFIED for unreviewed candidate matches."""
    from src.event_detector import VisualEvent
    from src.timeline_builder import TimelineBuilder

    metadata = {
        "cam_ref": VideoMetadata(
            video_id="cam_ref", camera_name="Reference Cam", file_path="cam_ref.mp4",
            file_name="cam_ref.mp4", file_size_bytes=1000, sha256_hash="hash_ref",
            duration_sec=10.0, nominal_fps=30.0, total_frames=300
        ),
        "cam_alg": VideoMetadata(
            video_id="cam_alg", camera_name="Alg Cam", file_path="cam_alg.mp4",
            file_name="cam_alg.mp4", file_size_bytes=1000, sha256_hash="hash_alg",
            duration_sec=10.0, nominal_fps=30.0, total_frames=300
        ),
        "cam_man": VideoMetadata(
            video_id="cam_man", camera_name="Man Cam", file_path="cam_man.mp4",
            file_name="cam_man.mp4", file_size_bytes=1000, sha256_hash="hash_man",
            duration_sec=10.0, nominal_fps=30.0, total_frames=300
        ),
    }

    alignments = {
        "cam_ref": CameraAlignmentResult(camera_id="cam_ref", camera_name="Reference Cam", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_alg": CameraAlignmentResult(camera_id="cam_alg", camera_name="Alg Cam", is_reference=False, offset_sec=2.5, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_man": CameraAlignmentResult(camera_id="cam_man", camera_name="Man Cam", is_reference=False, offset_sec=4.0, rate_drift=1.0, status="MANUAL_OVERRIDE"),
    }

    events = [
        VisualEvent("EVT-REF", "cam_ref", "Reference Cam", "cam_ref.mp4", 30, 1.0, "visual_spike", 0.9),
        VisualEvent("EVT-ALG", "cam_alg", "Alg Cam", "cam_alg.mp4", 30, 1.0, "visual_spike", 0.9),
        VisualEvent("EVT-MAN", "cam_man", "Man Cam", "cam_man.mp4", 30, 1.0, "visual_spike", 0.9),
    ]

    unverified_candidate = CandidateMatch(
        match_id="CAND-01", camera_a_id="cam_ref", camera_a_name="Reference Cam", event_a_id="EVT-REF",
        frame_a=30, time_a_sec=1.0, thumb_a=None, camera_b_id="cam_alg", camera_b_name="Alg Cam",
        event_b_id="EVT-ALG", frame_b=30, time_b_sec=1.0, thumb_b=None, similarity_score=0.8,
        phash_distance=10, color_similarity=0.75, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )

    tb = TimelineBuilder(metadata_dict=metadata, alignments=alignments)
    # Pass empty confirmed list and unverified candidate in candidate list
    unified = tb.build_unified_events(events, confirmed_matches=[], candidate_matches=[unverified_candidate])

    ev_ref = next(e for e in unified if e.event_id == "EVT-REF")
    ev_alg = next(e for e in unified if e.event_id == "EVT-ALG")
    ev_man = next(e for e in unified if e.event_id == "EVT-MAN")

    # 1. Unverified match must NOT be fabricated as CONFIRMED
    assert ev_ref.review_status == "CANDIDATE_UNVERIFIED"
    assert ev_ref.visual_category == "SUGGESTED_MATCH"

    # 2. Dynamic Data Classification checks
    assert ev_ref.data_classification == "MEASURED_FACT"
    assert ev_alg.data_classification == "ALGORITHMIC_ESTIMATE"
    assert ev_man.data_classification == "EXAMINER_OVERRIDE"

    # 3. Verify in exported CSV
    exporter = ForensicReportExporter(reports_dir=tmp_path / "reports_dyn")
    csv_path = exporter.export_unified_timeline_csv(unified)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = {r["event_id"]: r for r in csv.DictReader(f)}

    assert rows["EVT-REF"]["data_classification"] == "MEASURED_FACT"
    assert rows["EVT-REF"]["review_status"] == "CANDIDATE_UNVERIFIED"
    assert rows["EVT-ALG"]["data_classification"] == "ALGORITHMIC_ESTIMATE"
    assert rows["EVT-MAN"]["data_classification"] == "EXAMINER_OVERRIDE"


def test_candidate_matches_cannot_become_confirmed_when_passed_in_confirmed_matches_list():
    """Verify that candidate matches cannot become CONFIRMED simply by being passed in confirmed_matches argument."""
    from src.event_detector import VisualEvent
    from src.timeline_builder import TimelineBuilder

    metadata = {
        "cam_1": VideoMetadata("cam_1", "Cam 1", "cam1.mp4", "cam1.mp4", 100, "h1", 10.0, 30.0, 300),
        "cam_2": VideoMetadata("cam_2", "Cam 2", "cam2.mp4", "cam2.mp4", 100, "h2", 10.0, 30.0, 300)
    }
    alignments = {
        "cam_1": CameraAlignmentResult("cam_1", "Cam 1", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_2": CameraAlignmentResult("cam_2", "Cam 2", is_reference=False, offset_sec=2.0, rate_drift=1.0, status="SYNCHRONIZED"),
    }
    events = [
        VisualEvent("E1", "cam_1", "Cam 1", "cam1.mp4", 30, 1.0, "motion", 0.9),
        VisualEvent("E2", "cam_2", "Cam 2", "cam2.mp4", 30, 1.0, "motion", 0.9)
    ]
    unverified_candidate = CandidateMatch(
        match_id="CAND-TEST", camera_a_id="cam_1", camera_a_name="Cam 1", event_a_id="E1",
        frame_a=30, time_a_sec=1.0, thumb_a=None, camera_b_id="cam_2", camera_b_name="Cam 2",
        event_b_id="E2", frame_b=30, time_b_sec=1.0, thumb_b=None, similarity_score=0.88,
        phash_distance=8, color_similarity=0.8, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )

    tb = TimelineBuilder(metadata_dict=metadata, alignments=alignments)

    # Pass the unverified candidate directly into confirmed_matches!
    unified = tb.build_unified_events(events, confirmed_matches=[unverified_candidate], candidate_matches=[])

    ev1 = next(e for e in unified if e.event_id == "E1")
    ev2 = next(e for e in unified if e.event_id == "E2")

    # The unverified candidate must NOT become CONFIRMED
    assert ev1.review_status != "CONFIRMED"
    assert ev1.visual_category != "CONFIRMED_MATCH"
    assert ev2.review_status != "CONFIRMED"
    assert ev2.visual_category != "CONFIRMED_MATCH"


def test_rejected_match_cannot_be_confirmed_or_suggested():
    """Verify that a rejected match remains REJECTED and is not treated as suggested or confirmed."""
    from src.event_detector import VisualEvent
    from src.timeline_builder import TimelineBuilder

    metadata = {
        "cam_1": VideoMetadata("cam_1", "Cam 1", "cam1.mp4", "cam1.mp4", 100, "h1", 10.0, 30.0, 300),
        "cam_2": VideoMetadata("cam_2", "Cam 2", "cam2.mp4", "cam2.mp4", 100, "h2", 10.0, 30.0, 300)
    }
    alignments = {
        "cam_1": CameraAlignmentResult("cam_1", "Cam 1", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_2": CameraAlignmentResult("cam_2", "Cam 2", is_reference=False, offset_sec=2.0, rate_drift=1.0, status="SYNCHRONIZED"),
    }
    events = [
        VisualEvent("E1", "cam_1", "Cam 1", "cam1.mp4", 30, 1.0, "motion", 0.9),
        VisualEvent("E2", "cam_2", "Cam 2", "cam2.mp4", 30, 1.0, "motion", 0.9)
    ]
    rejected_match = CandidateMatch(
        match_id="REJ-TEST", camera_a_id="cam_1", camera_a_name="Cam 1", event_a_id="E1",
        frame_a=30, time_a_sec=1.0, thumb_a=None, camera_b_id="cam_2", camera_b_name="Cam 2",
        event_b_id="E2", frame_b=30, time_b_sec=1.0, thumb_b=None, similarity_score=0.4,
        phash_distance=25, color_similarity=0.3, review_status="REJECTED_BY_USER", is_anchor=False
    )

    tb = TimelineBuilder(metadata_dict=metadata, alignments=alignments)

    # Pass in both lists to ensure it is thoroughly rejected
    unified = tb.build_unified_events(events, confirmed_matches=[rejected_match], candidate_matches=[rejected_match])

    ev1 = next(e for e in unified if e.event_id == "E1")
    assert ev1.review_status != "CONFIRMED"
    assert ev1.visual_category != "CONFIRMED_MATCH"
    assert ev1.visual_category != "SUGGESTED_MATCH"
