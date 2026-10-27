"""Forensic report export engine for ChronoTrace AI.

Generates court-admissible forensic artifacts:
1. unified_timeline.csv
2. event_matches.csv
3. synchronization_report.json
4. evidence_manifest.json

Strictly adheres to forensic evidence integrity standards:
- Derived results explicitly labeled
- Methodological limitations documented
- Zero fabricated timestamps or confidence scores
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evidence_integrity import EvidenceManager, EvidenceRecord, SOFTWARE_VERSION
from src.time_aligner import CameraAlignmentResult, FORENSIC_STATUS_CRITERIA
from src.timeline_builder import UnifiedTimelineEvent
from src.video_metadata import VideoMetadata
from src.visual_matcher import CandidateMatch


FORENSIC_METHOD_EXPLANATION = (
    "ChronoTrace AI employs robust multi-point linear calibration and RANSAC (Random Sample Consensus) "
    "to model clock disparity between disparate video recording streams. The temporal relationship is "
    "modeled as: T_corrected = offset + rate * t_source, where 'rate' captures clock crystal oscillator drift "
    "or frame-drop timing skew, and 'offset' represents clock initialization discrepancy. When multiple visual "
    "or manual anchors are available, RANSAC statistically identifies genuine temporal correspondences (inliers) "
    "while rejecting false visual correlations (outliers). 95% confidence intervals are computed from inlier residuals."
)

FORENSIC_LIMITATIONS_DISCLOSURE = [
    "Image similarity metrics (pHash, dHash, HSV histogram) serve strictly as candidate generators and do NOT constitute conclusive forensic identification.",
    "Single-anchor synchronizations assume zero clock drift (rate = 1.000000) and are subject to unquantified cumulative drift over extended intervals.",
    "Variable frame rate (VFR) containers and dropped frames may introduce local timing deviations that deviate from the constant linear drift model.",
    "All derived timestamps and offsets represent mathematical estimates subject to the stated confidence bounds (uncertainty_sec).",
    "Independent investigator verification is required before introducing any derived timeline as evidence."
]


class ForensicReportExporter:
    """Exports structured, compliant forensic data packages."""

    def __init__(self, reports_dir: str | Path = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        unified_events: List[UnifiedTimelineEvent],
        matches: List[CandidateMatch],
        alignments: Dict[str, CameraAlignmentResult],
        metadata: Dict[str, VideoMetadata],
        evidence_records: List[EvidenceRecord],
        evidence_manager: Optional[EvidenceManager] = None,
        case_id: str = "CASE-2026-CHRONOTRACE",
        investigator_name: str = "Forensic Examiner"
    ) -> Dict[str, str]:
        """Export all mandatory forensic artifacts."""

        timeline_csv = self.export_unified_timeline_csv(unified_events)
        timeline_json = self.export_unified_timeline_json(unified_events, case_id, investigator_name)
        matches_csv = self.export_event_matches_csv(matches)
        sync_json = self.export_synchronization_report_json(
            alignments, metadata, matches, case_id, investigator_name
        )
        manifest_json = self.export_evidence_manifest_json(
            evidence_records, metadata, evidence_manager, case_id, investigator_name
        )

        return {
            "unified_timeline_csv": str(timeline_csv.resolve()),
            "unified_timeline_json": str(timeline_json.resolve()),
            "event_matches_csv": str(matches_csv.resolve()),
            "synchronization_report_json": str(sync_json.resolve()),
            "evidence_manifest_json": str(manifest_json.resolve())
        }

    def export_unified_timeline_csv(
        self,
        events: List[UnifiedTimelineEvent],
        filename: str = "unified_timeline.csv"
    ) -> Path:
        """Export unified_timeline.csv containing all forensic events with provenance."""
        out_path = self.reports_dir / filename
        now_utc = datetime.now(timezone.utc).isoformat()
        fieldnames = [
            "event_id",
            "camera_id",
            "camera_name",
            "source_filename",
            "frame_number",
            "source_timestamp_sec",
            "corrected_timestamp_sec",
            "event_type",
            "visual_category",
            "matching_evidence",
            "confidence",
            "review_status",
            "uncertainty_sec",
            "data_classification",
            "notes",
            "calibration_method",
            "synchronization_status",
            "supporting_anchors",
            "override_reason",
            "audit_ref",
            "software_version",
            "export_timestamp_utc"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ev in events:
                matching_ev = ""
                if ev.matched_with_camera and ev.matched_event_id:
                    matching_ev = f"{ev.matched_with_camera}:{ev.matched_event_id}"

                corr_str = f"{ev.corrected_timestamp_sec:.4f}" if ev.corrected_timestamp_sec is not None else "UNALIGNED"
                unc_str = f"{ev.uncertainty_sec:.4f}" if ev.uncertainty_sec is not None else "UNKNOWN"
                conf_str = f"{ev.confidence:.3f}" if ev.confidence is not None else "NONE"
                anchors_str = ";".join(ev.supporting_anchors) if getattr(ev, "supporting_anchors", None) else "NONE"

                writer.writerow({
                    "event_id": ev.event_id,
                    "camera_id": ev.camera_id,
                    "camera_name": ev.camera_name,
                    "source_filename": ev.source_filename,
                    "frame_number": ev.frame_number,
                    "source_timestamp_sec": f"{ev.source_timestamp_sec:.4f}",
                    "corrected_timestamp_sec": corr_str,
                    "event_type": ev.event_type,
                    "visual_category": ev.visual_category,
                    "matching_evidence": matching_ev,
                    "confidence": conf_str,
                    "review_status": ev.review_status,
                    "uncertainty_sec": unc_str,
                    "data_classification": getattr(ev, "data_classification", "DERIVED_RESULT"),
                    "notes": ev.notes,
                    "calibration_method": getattr(ev, "calibration_method", None) or "NONE",
                    "synchronization_status": getattr(ev, "synchronization_status", None) or "UNALIGNED",
                    "supporting_anchors": anchors_str,
                    "override_reason": getattr(ev, "override_reason", None) or "N/A",
                    "audit_ref": getattr(ev, "audit_ref", None) or "N/A",
                    "software_version": SOFTWARE_VERSION,
                    "export_timestamp_utc": now_utc
                })

        return out_path

    def export_unified_timeline_json(
        self,
        events: List[UnifiedTimelineEvent],
        case_id: str = "CASE-2026-CHRONOTRACE",
        investigator_name: str = "Forensic Examiner",
        filename: str = "unified_timeline.json"
    ) -> Path:
        """Export structured unified_timeline.json containing all forensic events and full provenance."""
        out_path = self.reports_dir / filename
        now_utc = datetime.now(timezone.utc).isoformat()

        serialized_events = []
        for ev in events:
            serialized_events.append({
                "event_id": ev.event_id,
                "camera_id": ev.camera_id,
                "camera_name": ev.camera_name,
                "source_filename": ev.source_filename,
                "frame_number": ev.frame_number,
                "source_timestamp_sec": ev.source_timestamp_sec,
                "corrected_timestamp_sec": ev.corrected_timestamp_sec,
                "event_type": ev.event_type,
                "visual_category": ev.visual_category,
                "matched_with_camera": ev.matched_with_camera,
                "matched_event_id": ev.matched_event_id,
                "confidence": ev.confidence,
                "review_status": ev.review_status,
                "uncertainty_sec": None if (ev.uncertainty_sec is None or math.isinf(ev.uncertainty_sec)) else ev.uncertainty_sec,
                "data_classification": getattr(ev, "data_classification", "DERIVED_RESULT"),
                "calibration_method": getattr(ev, "calibration_method", None),
                "synchronization_status": getattr(ev, "synchronization_status", "UNALIGNED"),
                "supporting_anchors": getattr(ev, "supporting_anchors", []),
                "override_reason": getattr(ev, "override_reason", None),
                "audit_ref": getattr(ev, "audit_ref", None),
                "thumbnail_path": ev.thumbnail_path,
                "notes": ev.notes
            })

        doc = {
            "report_type": "FORENSIC_UNIFIED_TIMELINE_REPORT",
            "software_name": "ChronoTrace AI",
            "software_version": SOFTWARE_VERSION,
            "generated_utc": now_utc,
            "case_id": case_id,
            "investigator": investigator_name,
            "total_events": len(serialized_events),
            "events": serialized_events
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

        return out_path

    def export_event_matches_csv(
        self,
        matches: List[CandidateMatch],
        filename: str = "event_matches.csv"
    ) -> Path:
        """Export event_matches.csv documenting all cross-camera candidate and anchor matches."""
        out_path = self.reports_dir / filename
        fieldnames = [
            "match_id",
            "camera_a",
            "camera_a_name",
            "event_a_id",
            "frame_a",
            "time_a_sec",
            "camera_b",
            "camera_b_name",
            "event_b_id",
            "frame_b",
            "time_b_sec",
            "raw_time_delta_sec",
            "similarity_score",
            "phash_distance",
            "color_similarity",
            "review_status",
            "is_anchor",
            "notes",
            "forensic_disclaimer"
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for m in matches:
                writer.writerow({
                    "match_id": m.match_id,
                    "camera_a": m.camera_a_id,
                    "camera_a_name": m.camera_a_name,
                    "event_a_id": m.event_a_id,
                    "frame_a": m.frame_a,
                    "time_a_sec": f"{m.time_a_sec:.4f}",
                    "camera_b": m.camera_b_id,
                    "camera_b_name": m.camera_b_name,
                    "event_b_id": m.event_b_id,
                    "frame_b": m.frame_b,
                    "time_b_sec": f"{m.time_b_sec:.4f}",
                    "raw_time_delta_sec": f"{m.raw_time_delta_sec:.4f}",
                    "similarity_score": f"{m.similarity_score:.3f}",
                    "phash_distance": m.phash_distance,
                    "color_similarity": f"{m.color_similarity:.3f}",
                    "review_status": m.review_status,
                    "is_anchor": "TRUE" if m.is_anchor else "FALSE",
                    "notes": m.notes,
                    "forensic_disclaimer": m.forensic_disclaimer
                })

        return out_path

    def export_synchronization_report_json(
        self,
        alignments: Dict[str, CameraAlignmentResult],
        metadata: Dict[str, VideoMetadata],
        matches: List[CandidateMatch],
        case_id: str,
        investigator_name: str,
        filename: str = "synchronization_report.json"
    ) -> Path:
        """Export synchronization_report.json detailing calibration models, errors, and limitations."""
        out_path = self.reports_dir / filename
        now_utc = datetime.now(timezone.utc).isoformat()

        cameras_report: Dict[str, Any] = {}
        for cam_id, align in alignments.items():
            meta = metadata.get(cam_id)
            cameras_report[cam_id] = {
                "camera_name": align.camera_name,
                "source_file": meta.file_name if meta else "unknown",
                "is_reference": align.is_reference,
                "status": align.status,
                "linear_model": {
                    "formula": "corrected_time = offset + rate * source_time",
                    "offset_sec": align.offset_sec,
                    "rate_drift": align.rate_drift,
                    "drift_rate_ppm": round((align.rate_drift - 1.0) * 1e6, 2)
                },
                "uncertainty_metrics": {
                    "confidence_level": "95%",
                    "uncertainty_margin_sec": None if align.uncertainty_sec == float("inf") else align.uncertainty_sec,
                    "inlier_rmse_sec": align.rmse_residual_sec,
                    "drift_standard_error": align.drift_std_error
                },
                "supporting_match_ids": align.supporting_match_ids,
                "rejected_match_ids": align.rejected_match_ids,
                "derivation_label": align.derivation_label,
                "method_description": align.method_description,
                "forensic_notes": align.forensic_notes
            }

        report = {
            "report_type": "FORENSIC_TIMELINE_SYNCHRONIZATION_REPORT",
            "software_name": "ChronoTrace AI",
            "software_version": SOFTWARE_VERSION,
            "generated_utc": now_utc,
            "case_id": case_id,
            "investigator": investigator_name,
            "methodology": {
                "algorithm": "RANSAC Robust Multi-Point Temporal Calibration",
                "mathematical_model": "T_corrected = offset + rate * t_source",
                "description": FORENSIC_METHOD_EXPLANATION,
                "limitations_disclosure": FORENSIC_LIMITATIONS_DISCLOSURE,
                "status_criteria": FORENSIC_STATUS_CRITERIA
            },
            "total_cameras": len(metadata),
            "camera_alignments": cameras_report,
            "total_candidate_matches": len(matches),
            "total_confirmed_anchors": sum(1 for m in matches if m.is_anchor or m.review_status in ("CONFIRMED", "CONFIRMED_BY_USER"))
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return out_path

    def export_evidence_manifest_json(
        self,
        evidence_records: List[EvidenceRecord],
        metadata: Dict[str, VideoMetadata],
        evidence_manager: Optional[EvidenceManager],
        case_id: str,
        investigator_name: str,
        filename: str = "evidence_manifest.json"
    ) -> Path:
        """Export evidence_manifest.json with cryptographic hashes, file catalog, and audit log."""
        out_path = self.reports_dir / filename
        now_utc = datetime.now(timezone.utc).isoformat()

        files_catalog: List[Dict[str, Any]] = []
        for rec in evidence_records:
            cam_meta = next((m for m in metadata.values() if m.sha256_hash == rec.sha256_hash), None)
            files_catalog.append({
                "evidence_id": rec.evidence_id,
                "camera_name": rec.camera_name,
                "original_filename": rec.original_filename,
                "stored_path": rec.stored_path,
                "sha256_hash": rec.sha256_hash,
                "file_size_bytes": rec.file_size_bytes,
                "ingest_time_utc": rec.ingest_time_utc,
                "software_version": rec.software_version,
                "integrity_status": "TAMPERED" if rec.is_tampered else "VERIFIED_INTACT",
                "stream_parameters": {
                    "duration_sec": cam_meta.duration_sec if cam_meta else None,
                    "nominal_fps": cam_meta.nominal_fps if cam_meta else None,
                    "total_frames": cam_meta.total_frames if cam_meta else None,
                    "resolution": f"{cam_meta.width}x{cam_meta.height}" if cam_meta else None,
                    "codec": cam_meta.codec_name if cam_meta else None,
                    "container_timestamp_utc": cam_meta.creation_time_utc if cam_meta else None
                }
            })

        audit_entries = evidence_manager.load_audit_log() if evidence_manager else []

        manifest = {
            "manifest_type": "FORENSIC_DIGITAL_EVIDENCE_MANIFEST",
            "software_name": "ChronoTrace AI",
            "software_version": SOFTWARE_VERSION,
            "generated_utc": now_utc,
            "case_id": case_id,
            "investigator": investigator_name,
            "hash_algorithm": "SHA-256",
            "evidence_files_count": len(files_catalog),
            "evidence_catalog": files_catalog,
            "audit_trail": audit_entries
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return out_path
