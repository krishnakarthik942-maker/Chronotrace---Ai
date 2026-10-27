"""Unified forensic timeline construction and multi-camera query engine.

Integrates camera calibrations, maps all observations to a global reference scale,
calculates camera coverage spans and temporal overlaps, and facilitates synchronized
multi-camera frame lookups.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.evidence_integrity import SOFTWARE_VERSION
from src.event_detector import VisualEvent
from src.time_aligner import CameraAlignmentResult
from src.video_metadata import VideoMetadata
from src.visual_matcher import CandidateMatch


@dataclass
class UnifiedTimelineEvent:
    """An event placed on the unified forensic reference timeline."""

    event_id: str
    camera_id: str
    camera_name: str
    source_filename: str
    frame_number: int
    source_timestamp_sec: float
    corrected_timestamp_sec: Optional[float]
    event_type: str
    visual_category: str  # "VERIFIED_ANCHOR", "SUGGESTED_MATCH", "DETECTED_EVENT", "MANUAL_ADJUSTMENT"
    matched_with_camera: Optional[str] = None
    matched_event_id: Optional[str] = None
    confidence: Optional[float] = 1.0
    review_status: str = "UNREVIEWED"  # "CONFIRMED", "REJECTED", "UNREVIEWED"
    uncertainty_sec: Optional[float] = None
    thumbnail_path: Optional[str] = None
    notes: str = ""
    data_classification: str = "DERIVED_RESULT"
    calibration_method: Optional[str] = None
    synchronization_status: Optional[str] = None
    supporting_anchors: List[str] = field(default_factory=list)
    override_reason: Optional[str] = None
    audit_ref: Optional[str] = None
    software_version: str = SOFTWARE_VERSION
    export_timestamp_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("uncertainty_sec") is not None and math.isinf(d["uncertainty_sec"]):
            d["uncertainty_sec"] = None
        return d


@dataclass
class CameraTimelineSpan:
    """Temporal coverage span of a camera on the global reference timeline."""

    camera_id: str
    camera_name: str
    source_duration_sec: float
    global_start_sec: Optional[float]
    global_end_sec: Optional[float]
    alignment_status: str
    offset_sec: float
    rate_drift: float
    uncertainty_sec: Optional[float]

    def contains_global_time(self, global_time_sec: float) -> bool:
        if self.global_start_sec is None or self.global_end_sec is None:
            return False
        return self.global_start_sec <= global_time_sec <= self.global_end_sec


@dataclass
class SynchronizedCameraLookup:
    """Result of querying all cameras at a specific global timeline instant."""

    global_timestamp_sec: float
    camera_id: str
    camera_name: str
    source_timestamp_sec: Optional[float]
    approx_frame_idx: Optional[int]
    is_recording: bool
    status_message: str


class TimelineBuilder:
    """Builds and queries the multi-camera forensic timeline."""

    def __init__(
        self,
        metadata_dict: Dict[str, VideoMetadata],
        alignments: Dict[str, CameraAlignmentResult]
    ):
        self.metadata = metadata_dict
        self.alignments = alignments

    def compute_camera_spans(self) -> Dict[str, CameraTimelineSpan]:
        """Calculate global reference time boundaries for each camera."""
        spans: Dict[str, CameraTimelineSpan] = {}

        for cam_id, meta in self.metadata.items():
            align = self.alignments.get(cam_id)
            if not align or align.status == "UNALIGNED":
                spans[cam_id] = CameraTimelineSpan(
                    camera_id=cam_id,
                    camera_name=meta.camera_name,
                    source_duration_sec=meta.duration_sec,
                    global_start_sec=None,
                    global_end_sec=None,
                    alignment_status="UNALIGNED",
                    offset_sec=0.0,
                    rate_drift=1.0,
                    uncertainty_sec=None
                )
            else:
                g_start = align.to_corrected_time(0.0)
                g_end = align.to_corrected_time(meta.duration_sec)
                unc = None if math.isinf(align.uncertainty_sec) else align.uncertainty_sec

                spans[cam_id] = CameraTimelineSpan(
                    camera_id=cam_id,
                    camera_name=meta.camera_name,
                    source_duration_sec=meta.duration_sec,
                    global_start_sec=min(g_start, g_end),
                    global_end_sec=max(g_start, g_end),
                    alignment_status=align.status,
                    offset_sec=align.offset_sec,
                    rate_drift=align.rate_drift,
                    uncertainty_sec=unc
                )

        return spans

    def build_unified_events(
        self,
        all_events: List[VisualEvent],
        confirmed_matches: List[CandidateMatch],
        candidate_matches: List[CandidateMatch]
    ) -> List[UnifiedTimelineEvent]:
        """Map all detected events and cross-camera matches into unified events."""
        unified: List[UnifiedTimelineEvent] = []

        # Index matches by (camera_id, event_id)
        confirmed_map: Dict[Tuple[str, str], CandidateMatch] = {}
        for m in confirmed_matches:
            if m.is_anchor or m.review_status in ("CONFIRMED", "CONFIRMED_BY_USER"):
                confirmed_map[(m.camera_a_id, m.event_a_id)] = m
                confirmed_map[(m.camera_b_id, m.event_b_id)] = m

        suggested_map: Dict[Tuple[str, str], CandidateMatch] = {}
        for m in candidate_matches:
            if m.review_status == "CANDIDATE_UNVERIFIED":
                suggested_map[(m.camera_a_id, m.event_a_id)] = m
                suggested_map[(m.camera_b_id, m.event_b_id)] = m

        for ev in all_events:
            align = self.alignments.get(ev.camera_id)
            meta = self.metadata.get(ev.camera_id)

            if align and align.status != "UNALIGNED":
                corr_time = align.to_corrected_time(ev.source_timestamp_sec)
                unc = None if math.isinf(align.uncertainty_sec) else align.uncertainty_sec
            else:
                corr_time = None
                unc = None

            matched_cam = None
            matched_evt = None
            conf = ev.score
            rev_status = ev.review_status

            # Check if this event participates in a confirmed anchor or match
            key = (ev.camera_id, ev.event_id)
            if key in confirmed_map:
                m = confirmed_map[key]
                if m.is_anchor:
                    category = "VERIFIED_ANCHOR"
                elif align and align.status == "MANUAL_OVERRIDE":
                    category = "VERIFIED_ANCHOR"
                else:
                    category = "CONFIRMED_MATCH"
                rev_status = "CONFIRMED"
                conf = m.similarity_score
                if ev.camera_id == m.camera_a_id:
                    matched_cam = m.camera_b_name
                    matched_evt = m.event_b_id
                else:
                    matched_cam = m.camera_a_name
                    matched_evt = m.event_a_id
            elif key in suggested_map:
                m = suggested_map[key]
                category = "SUGGESTED_MATCH"
                rev_status = "CANDIDATE_UNVERIFIED"
                conf = m.similarity_score
                if ev.camera_id == m.camera_a_id:
                    matched_cam = m.camera_b_name
                    matched_evt = m.event_b_id
                else:
                    matched_cam = m.camera_a_name
                    matched_evt = m.event_a_id
            elif align and align.status == "MANUAL_OVERRIDE":
                category = "MANUAL_ADJUSTMENT"
            else:
                category = "DETECTED_EVENT"

            # Determine forensic evidentiary classification
            # Provenance-based classifications:
            # - MEASURED_FACT: a direct observation from source evidence, with its source identified.
            # - ALGORITHMIC_ESTIMATE: a calculated or inferred timestamp, match, or calibration.
            # - EXAMINER_OVERRIDE: a value explicitly changed by an examiner.
            # Note: Inferred events on the reference camera are ALGORITHMIC_ESTIMATE, not MEASURED_FACT.
            explicit_classification = getattr(ev, "data_classification", None)
            is_inferred_event = (
                explicit_classification == "ALGORITHMIC_ESTIMATE"
                or getattr(ev, "is_inferred", False)
                or ev.event_type.lower() in ("inferred", "inferred_event", "interpolated")
            )

            if explicit_classification == "EXAMINER_OVERRIDE" or (align and align.status == "MANUAL_OVERRIDE") or category == "MANUAL_ADJUSTMENT":
                classification = "EXAMINER_OVERRIDE"
            elif explicit_classification == "MEASURED_FACT":
                classification = "MEASURED_FACT"
            elif is_inferred_event:
                classification = "ALGORITHMIC_ESTIMATE"
            elif align and align.is_reference:
                classification = "MEASURED_FACT"
            elif corr_time is not None:
                classification = "ALGORITHMIC_ESTIMATE"
            else:
                classification = "MEASURED_FACT"

            # Provenance fields
            calib_method = align.method_description if align else "NONE"
            sync_status = align.status if align else "UNALIGNED"

            supporting_anchors: List[str] = []
            if align and align.supporting_match_ids:
                supporting_anchors.extend(align.supporting_match_ids)
            if key in confirmed_map:
                supporting_anchors.append(confirmed_map[key].match_id)
            supporting_anchors = list(dict.fromkeys(supporting_anchors))

            override_reason_val: Optional[str] = None
            if classification == "EXAMINER_OVERRIDE":
                override_reason_val = (
                    getattr(ev, "override_reason", None)
                    or getattr(align, "override_reason", None)
                    or (align.forensic_notes if align and align.status == "MANUAL_OVERRIDE" else None)
                    or "Examiner manual timeline adjustment"
                )

            audit_ref_val: Optional[str] = getattr(ev, "audit_ref", None) or (getattr(align, "audit_ref", None) if align else None)
            conf_val = round(conf, 3) if conf is not None else None

            uni_evt = UnifiedTimelineEvent(
                event_id=ev.event_id,
                camera_id=ev.camera_id,
                camera_name=ev.camera_name,
                source_filename=meta.file_name if meta else ev.source_filename,
                frame_number=ev.frame_idx,
                source_timestamp_sec=ev.source_timestamp_sec,
                corrected_timestamp_sec=corr_time,
                event_type=ev.event_type,
                visual_category=category,
                matched_with_camera=matched_cam,
                matched_event_id=matched_evt,
                confidence=conf_val,
                review_status=rev_status,
                uncertainty_sec=unc,
                thumbnail_path=ev.thumbnail_path,
                notes=ev.notes,
                data_classification=classification,
                calibration_method=calib_method,
                synchronization_status=sync_status,
                supporting_anchors=supporting_anchors,
                override_reason=override_reason_val,
                audit_ref=audit_ref_val,
                software_version=SOFTWARE_VERSION
            )
            unified.append(uni_evt)

        # Sort events: those with corrected time first by corrected time, then source time
        unified.sort(key=lambda x: (
            0 if x.corrected_timestamp_sec is not None else 1,
            x.corrected_timestamp_sec if x.corrected_timestamp_sec is not None else x.source_timestamp_sec
        ))
        return unified

    def query_global_time(self, query_global_sec: float) -> List[SynchronizedCameraLookup]:
        """Find the corresponding source timestamp and frame index in every camera."""
        spans = self.compute_camera_spans()
        results: List[SynchronizedCameraLookup] = []

        for cam_id, span in spans.items():
            meta = self.metadata.get(cam_id)
            align = self.alignments.get(cam_id)

            if not meta or not align or align.status == "UNALIGNED":
                results.append(SynchronizedCameraLookup(
                    global_timestamp_sec=query_global_sec,
                    camera_id=cam_id,
                    camera_name=span.camera_name,
                    source_timestamp_sec=None,
                    approx_frame_idx=None,
                    is_recording=False,
                    status_message="Camera not synchronized"
                ))
                continue

            src_time = align.to_source_time(query_global_sec)

            if src_time < 0.0:
                results.append(SynchronizedCameraLookup(
                    global_timestamp_sec=query_global_sec,
                    camera_id=cam_id,
                    camera_name=span.camera_name,
                    source_timestamp_sec=src_time,
                    approx_frame_idx=None,
                    is_recording=False,
                    status_message=f"Before recording start ({abs(src_time):.2f}s prior)"
                ))
            elif src_time > meta.duration_sec:
                results.append(SynchronizedCameraLookup(
                    global_timestamp_sec=query_global_sec,
                    camera_id=cam_id,
                    camera_name=span.camera_name,
                    source_timestamp_sec=src_time,
                    approx_frame_idx=None,
                    is_recording=False,
                    status_message=f"After recording end ({src_time - meta.duration_sec:.2f}s past)"
                ))
            else:
                fps = meta.nominal_fps if meta.nominal_fps > 0 else 30.0
                max_frame = max(0, meta.total_frames - 1) if meta.total_frames > 0 else 0
                frame_idx = min(max_frame, max(0, int(round(src_time * fps))))

                results.append(SynchronizedCameraLookup(
                    global_timestamp_sec=query_global_sec,
                    camera_id=cam_id,
                    camera_name=span.camera_name,
                    source_timestamp_sec=round(src_time, 4),
                    approx_frame_idx=frame_idx,
                    is_recording=True,
                    status_message="Synchronized active footage"
                ))

        return results
