"""ChronoTrace AI: Multi-Camera Forensic Timeline Reconstruction
Professional Streamlit Forensic Command Dashboard.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.evidence_integrity import EvidenceManager, EvidenceRecord, SOFTWARE_VERSION
from src.event_detector import EventDetector, VisualEvent
from src.frame_extractor import FrameExtractor
from src.report_exporter import ForensicReportExporter
from src.synthetic_generator import SyntheticVideoGenerator
from src.time_aligner import CameraAlignmentResult, TimeAligner, TimeAnchorPoint
from src.timeline_builder import TimelineBuilder, UnifiedTimelineEvent
from src.video_metadata import VideoMetadata, VideoMetadataExtractor
from src.visual_matcher import CandidateMatch, VisualMatcher


# --- PAGE CONFIGURATION & FORENSIC THEME ---
st.set_page_config(
    page_title="ChronoTrace AI — Forensic Timeline Reconstruction",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Dark Cyber-Forensic Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Banner */
    .forensic-header {
        background: linear-gradient(135deg, #111827 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .forensic-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .forensic-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 6px;
    }
    
    /* Badges & Chips */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-sync { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-approx { background-color: #451a03; color: #fbbf24; border: 1px solid #d97706; }
    .badge-unaligned { background-color: #312e81; color: #a5b4fc; border: 1px solid #4f46e5; }
    .badge-conflict { background-color: #4c0519; color: #f87171; border: 1px solid #e11d48; }
    .badge-manual { background-color: #1e1b4b; color: #38bdf8; border: 1px solid #0284c7; }
    
    /* Hash Badge */
    .hash-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        background-color: #1e293b;
        color: #38bdf8;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #334155;
    }
    
    /* Card Container */
    .forensic-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    
    /* Timeline Visual Bar */
    .timeline-track-container {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
    }
    
    /* Metric Card */
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- APPLICATION DIRECTORIES ---
BASE_DIR = Path(__file__).parent.resolve()
VAULT_DIR = BASE_DIR / "data" / "evidence"
CACHE_DIR = BASE_DIR / "data" / "cache"
REPORTS_DIR = BASE_DIR / "reports"
SAMPLE_DIR = BASE_DIR / "sample_data"

for d in [VAULT_DIR, CACHE_DIR, REPORTS_DIR, SAMPLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- SESSION STATE INITIALIZATION ---
if "evidence_manager" not in st.session_state:
    st.session_state.evidence_manager = EvidenceManager(
        base_evidence_dir=VAULT_DIR,
        audit_log_path=VAULT_DIR / "audit_log.jsonl"
    )

if "metadata" not in st.session_state:
    st.session_state.metadata: Dict[str, VideoMetadata] = {}

if "events" not in st.session_state:
    st.session_state.events: List[VisualEvent] = []

if "candidate_matches" not in st.session_state:
    st.session_state.candidate_matches: List[CandidateMatch] = []

if "confirmed_anchors" not in st.session_state:
    st.session_state.confirmed_anchors: List[TimeAnchorPoint] = []

if "manual_offsets" not in st.session_state:
    st.session_state.manual_offsets: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

if "reference_camera_id" not in st.session_state:
    st.session_state.reference_camera_id: Optional[str] = None

if "last_exported_files" not in st.session_state:
    st.session_state.last_exported_files: Optional[Dict[str, str]] = None


# --- HELPER FUNCTIONS ---
def get_status_badge_html(status: str) -> str:
    mapping = {
        "SYNCHRONIZED": ('badge-sync', 'SYNCHRONIZED'),
        "APPROXIMATE": ('badge-approx', 'APPROXIMATE'),
        "UNALIGNED": ('badge-unaligned', 'UNALIGNED'),
        "CONFLICTING": ('badge-conflict', 'CONFLICTING (REVIEW REQ)'),
        "MANUAL_OVERRIDE": ('badge-manual', 'MANUAL OVERRIDE')
    }
    cls, text = mapping.get(status, ('badge-unaligned', status))
    return f'<span class="status-badge {cls}">{text}</span>'


def load_demo_dataset():
    """Generate and load synthetic 3-camera forensic case."""
    with st.spinner("Generating synthetic multi-camera dataset with known ground truth..."):
        gen = SyntheticVideoGenerator(output_dir=SAMPLE_DIR)
        dataset = gen.create_forensic_hit_and_run_dataset()

        st.session_state.metadata.clear()
        st.session_state.events.clear()
        st.session_state.candidate_matches.clear()
        st.session_state.confirmed_anchors.clear()
        st.session_state.manual_offsets.clear()

        em: EvidenceManager = st.session_state.evidence_manager
        extractor = VideoMetadataExtractor()
        detector = EventDetector(cache_dir=CACHE_DIR / "events")

        for cam_id, gt in dataset.items():
            rec = em.register_evidence_file(gt.file_path, camera_name=gt.camera_name)
            meta = extractor.extract(
                rec.stored_path,
                camera_name=gt.camera_name,
                video_id=cam_id,
                sha256_hash=rec.sha256_hash
            )
            st.session_state.metadata[cam_id] = meta

            # Detect events
            evs = detector.detect_events(
                meta.file_path,
                camera_id=cam_id,
                camera_name=gt.camera_name,
                detect_motion_spikes=True
            )
            st.session_state.events.extend(evs)

        st.session_state.reference_camera_id = "cam_1"

        # Generate candidates
        matcher = VisualMatcher()
        cam_ids = list(dataset.keys())
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                c_a, c_b = cam_ids[i], cam_ids[j]
                ev_a = [e for e in st.session_state.events if e.camera_id == c_a]
                ev_b = [e for e in st.session_state.events if e.camera_id == c_b]
                matches = matcher.match_events(ev_a, ev_b)
                st.session_state.candidate_matches.extend(matches)

        # Pre-seed realistic anchors based on the impact event at T = 6.0s
        # Cam 2 local time 2.5s -> Cam 1 (Ref) 6.0s
        # Cam 3 local time 8.0s -> Cam 1 (Ref) 6.0s
        st.session_state.confirmed_anchors.extend([
            TimeAnchorPoint("ANCHOR-DEMO-01", "cam_2", source_time_sec=2.5, reference_time_sec=6.0),
            TimeAnchorPoint("ANCHOR-DEMO-02", "cam_2", source_time_sec=5.0, reference_time_sec=8.5),
            TimeAnchorPoint("ANCHOR-DEMO-03", "cam_3", source_time_sec=5.0, reference_time_sec=3.0),
            TimeAnchorPoint("ANCHOR-DEMO-04", "cam_3", source_time_sec=8.0, reference_time_sec=6.0),
        ])

    st.success("Successfully loaded 3-camera Forensic Hit-and-Run Case!")
    st.rerun()


# --- HEADER ---
st.markdown("""
<div class="forensic-header">
    <div class="forensic-title">
        <span>⏱️ ChronoTrace AI</span>
        <span style="font-size: 13px; font-weight: 600; color: #38bdf8; background: #0c4a6e; padding: 2px 8px; border-radius: 4px;">v1.0.0 FORENSIC SUITE</span>
    </div>
    <div class="forensic-subtitle">
        Multi-Camera Forensic Timeline Reconstruction & Temporal Clock Calibration Engine
    </div>
</div>
""", unsafe_allow_html=True)


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🎛️ Forensic Controls")

    st.subheader("1. Case & Quick Loader")
    case_id = st.text_input("Forensic Case ID", value="CASE-2026-HIT-AND-RUN-01")
    examiner = st.text_input("Examiner Name", value="Det. Senior Analyst")

    if st.button("🚀 Load Synthetic Demo Case (3 Cams)", use_container_width=True):
        load_demo_dataset()

    st.markdown("---")
    st.subheader("2. Evidence Ingestion")
    uploaded_files = st.file_uploader(
        "Upload 2–6 Video Files",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process & Hash Uploaded Videos", use_container_width=True):
            with st.spinner("Cryptographically hashing and ingesting videos..."):
                em: EvidenceManager = st.session_state.evidence_manager
                extractor = VideoMetadataExtractor()
                detector = EventDetector(cache_dir=CACHE_DIR / "events")

                for idx, uf in enumerate(uploaded_files):
                    cam_id = f"cam_{idx+1}"
                    cam_name = f"Camera {idx+1} ({uf.name})"

                    # Save temp copy
                    temp_path = CACHE_DIR / uf.name
                    temp_path.write_bytes(uf.getbuffer())

                    rec = em.register_evidence_file(temp_path, camera_name=cam_name)
                    meta = extractor.extract(
                        rec.stored_path,
                        camera_name=cam_name,
                        video_id=cam_id,
                        sha256_hash=rec.sha256_hash
                    )
                    st.session_state.metadata[cam_id] = meta

                    evs = detector.detect_events(
                        meta.file_path,
                        camera_id=cam_id,
                        camera_name=cam_name,
                        detect_motion_spikes=True
                    )
                    st.session_state.events.extend(evs)

                if not st.session_state.reference_camera_id and st.session_state.metadata:
                    st.session_state.reference_camera_id = list(st.session_state.metadata.keys())[0]

            st.success(f"Ingested {len(uploaded_files)} video streams!")
            st.rerun()

    st.markdown("---")
    st.subheader("3. Calibration Settings")
    if st.session_state.metadata:
        cams = list(st.session_state.metadata.keys())
        ref_cam = st.selectbox(
            "Reference Timeline Camera (Ground Truth)",
            options=cams,
            format_func=lambda c: st.session_state.metadata[c].camera_name,
            index=cams.index(st.session_state.reference_camera_id) if st.session_state.reference_camera_id in cams else 0
        )
        st.session_state.reference_camera_id = ref_cam

    st.markdown("---")
    st.markdown("""
    **Forensic Guarantee:**
    - Non-destructive original custody
    - SHA-256 tamper verification
    - Inferences marked as DERIVED_RESULT
    """)


# --- COMPUTE SYNCHRONIZATION ALIGNMENTS ---
alignments: Dict[str, CameraAlignmentResult] = {}
if st.session_state.metadata:
    ref_id = st.session_state.reference_camera_id or list(st.session_state.metadata.keys())[0]
    aligner = TimeAligner(reference_camera_id=ref_id)

    for cam_id, meta in st.session_state.metadata.items():
        manual_off, manual_r = st.session_state.manual_offsets.get(cam_id, (None, None))
        res = aligner.align_camera_from_anchors(
            camera_id=cam_id,
            camera_name=meta.camera_name,
            anchors=st.session_state.confirmed_anchors,
            nominal_fps=meta.nominal_fps,
            manual_offset=manual_off,
            manual_rate=manual_r
        )
        alignments[cam_id] = res


# --- MAIN TABS ---
tab_overview, tab_timeline, tab_matches, tab_manual, tab_reports = st.tabs([
    "📹 Evidence & Calibration Status",
    "📊 Unified Multi-Camera Timeline",
    "🔍 Event Matching & Corroboration",
    "⚙️ Manual Clock Corrections",
    "📑 Forensic Export Packages"
])


# ==========================================
# TAB 1: EVIDENCE & CALIBRATION STATUS
# ==========================================
with tab_overview:
    if not st.session_state.metadata:
        st.info("👋 Welcome! Click **'Load Synthetic Demo Case'** in the sidebar or upload 2–6 video files to begin.")
    else:
        st.subheader("Video Evidence Custody & Temporal Calibration Overview")

        # Top Metric Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="forensic-card">
                <div class="metric-label">Ingested Cameras</div>
                <div class="metric-value">{len(st.session_state.metadata)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            synced_count = sum(1 for a in alignments.values() if a.status in ("SYNCHRONIZED", "APPROXIMATE", "MANUAL_OVERRIDE"))
            st.markdown(f"""
            <div class="forensic-card">
                <div class="metric-label">Synchronized Streams</div>
                <div class="metric-value" style="color: #34d399;">{synced_count} / {len(alignments)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="forensic-card">
                <div class="metric-label">Verified Anchors</div>
                <div class="metric-value" style="color: #fbbf24;">{len(st.session_state.confirmed_anchors)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="forensic-card">
                <div class="metric-label">Detected Visual Events</div>
                <div class="metric-value">{len(st.session_state.events)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Camera Evidence & Calibration Cards")
        em: EvidenceManager = st.session_state.evidence_manager

        for cam_id, meta in st.session_state.metadata.items():
            align = alignments[cam_id]
            is_valid_hash = True
            rec = next((r for r in em.get_all_records() if r.sha256_hash == meta.sha256_hash), None)
            if rec:
                is_valid_hash = not rec.is_tampered

            with st.container():
                st.markdown(f"""
                <div class="forensic-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <span style="font-size: 18px; font-weight: 700; color: #f8fafc;">{meta.camera_name}</span>
                            {" <span style='font-size: 11px; color: #38bdf8; font-weight: 600;'>(REFERENCE TIMELINE)</span>" if align.is_reference else ""}
                        </div>
                        <div>
                            {get_status_badge_html(align.status)}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 13px;">
                        <div>
                            <span style="color: #94a3b8;">Duration:</span> <strong>{meta.duration_sec:.2f}s</strong><br>
                            <span style="color: #94a3b8;">FPS:</span> <strong>{meta.nominal_fps:.2f}</strong> | <span style="color: #94a3b8;">Frames:</span> <strong>{meta.total_frames}</strong><br>
                            <span style="color: #94a3b8;">Resolution:</span> <strong>{meta.width}x{meta.height}</strong>
                        </div>
                        <div>
                            <span style="color: #94a3b8;">Estimated Clock Offset:</span> <strong style="color: #38bdf8;">{align.offset_sec:+.4f}s</strong><br>
                            <span style="color: #94a3b8;">Drift Rate:</span> <strong>{align.rate_drift:.6f}</strong> ({(align.rate_drift-1.0)*1e6:+.1f} ppm)<br>
                            <span style="color: #94a3b8;">Uncertainty (95% CI):</span> <strong>±{align.uncertainty_sec:.4f}s</strong>
                        </div>
                        <div>
                            <span style="color: #94a3b8;">Supporting Anchors:</span> <strong>{len(align.supporting_match_ids)}</strong><br>
                            <span style="color: #94a3b8;">Rejected Outliers:</span> <strong style="color: #f87171;">{len(align.rejected_match_ids)}</strong><br>
                            <span style="color: #94a3b8;">Residual RMSE:</span> <strong>{align.rmse_residual_sec*1000.0:.1f}ms</strong>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #1f2937;">
                        <span style="color: #94a3b8; font-size: 11px;">SHA-256 HASH:</span>
                        <span class="hash-badge">{meta.sha256_hash}</span>
                        {"<span style='color: #34d399; font-size: 11px; margin-left: 8px;'>✓ VERIFIED INTACT</span>" if is_valid_hash else "<span style='color: #f87171; font-size: 11px; margin-left: 8px;'>⚠️ TAMPER ALERT</span>"}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==========================================
# TAB 2: UNIFIED MULTI-CAMERA TIMELINE
# ==========================================
with tab_timeline:
    if not st.session_state.metadata:
        st.info("Please load or ingest video files first.")
    else:
        tb = TimelineBuilder(metadata_dict=st.session_state.metadata, alignments=alignments)
        spans = tb.compute_camera_spans()
        unified_events = tb.build_unified_events(
            st.session_state.events,
            st.session_state.candidate_matches,
            st.session_state.candidate_matches
        )

        st.subheader("Interactive Forensic Multi-Track Timeline")

        # Determine global time window
        valid_starts = [s.global_start_sec for s in spans.values() if s.global_start_sec is not None]
        valid_ends = [s.global_end_sec for s in spans.values() if s.global_end_sec is not None]

        if valid_starts and valid_ends:
            g_min = min(valid_starts)
            g_max = max(valid_ends)
        else:
            g_min, g_max = 0.0, 10.0

        st.markdown(f"**Unified Timeline Scale:** Reference Time from **{g_min:.2f}s** to **{g_max:.2f}s**")

        # Visual track representation
        for cam_id, span in spans.items():
            align = alignments[cam_id]
            cam_events = [e for e in unified_events if e.camera_id == cam_id]

            if span.global_start_sec is not None and (g_max - g_min) > 0:
                left_pct = max(0.0, min(100.0, ((span.global_start_sec - g_min) / (g_max - g_min)) * 100.0))
                width_pct = max(2.0, min(100.0 - left_pct, ((span.global_end_sec - span.global_start_sec) / (g_max - g_min)) * 100.0))
            else:
                left_pct, width_pct = 0.0, 100.0

            st.markdown(f"""
            <div class="timeline-track-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="font-weight: 700; font-size: 14px;">{span.camera_name}</span>
                    <span style="font-size: 12px; color: #94a3b8;">
                        Offset: {align.offset_sec:+.2f}s | Span: [{span.global_start_sec or 0.0:.1f}s — {span.global_end_sec or 0.0:.1f}s]
                    </span>
                </div>
                <div style="background-color: #1e293b; height: 32px; border-radius: 6px; position: relative; overflow: hidden;">
                    <div style="position: absolute; left: {left_pct:.1f}%; width: {width_pct:.1f}%; height: 100%; background: linear-gradient(90deg, #1e3a8a, #0369a1); border-radius: 4px; opacity: 0.85;">
                    </div>
                </div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
                    Events Detected: {len(cam_events)} (Visual Spikes & Anchors)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🎯 Synchronized Multi-Camera Frame Inspector")
        st.caption("Scrub to any reference time to view what every camera recorded at that exact microsecond.")

        # Time scrubber slider
        if g_max > g_min:
            selected_global_time = st.slider(
                "Global Timeline Playhead (Reference Seconds)",
                min_value=float(math.floor(g_min)),
                max_value=float(math.ceil(g_max)),
                value=float(min(max(g_min, 6.0), g_max)),
                step=0.1
            )
        else:
            selected_global_time = 0.0

        lookup_results = tb.query_global_time(selected_global_time)
        fe = FrameExtractor(cache_dir=CACHE_DIR / "frames")

        cols = st.columns(len(lookup_results))
        for idx, res in enumerate(lookup_results):
            meta = st.session_state.metadata.get(res.camera_id)
            with cols[idx]:
                st.markdown(f"**{res.camera_name}**")
                if res.is_recording and meta and res.source_timestamp_sec is not None:
                    st.caption(f"Local Time: `{res.source_timestamp_sec:.3f}s` | Frame: `{res.approx_frame_idx}`")
                    frame = fe.extract_frame_at_timestamp(meta.file_path, res.source_timestamp_sec, meta.nominal_fps)
                    if frame is not None:
                        # Convert BGR to RGB for Streamlit
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb, use_container_width=True)
                    else:
                        st.warning("Frame seek failed")
                else:
                    st.info(f"⚪ {res.status_message}")


# ==========================================
# TAB 3: EVENT MATCHING & CORROBORATION
# ==========================================
with tab_matches:
    st.subheader("Cross-Camera Candidate Event Matches")
    st.markdown("""
    > [!IMPORTANT]
    > **Forensic Evidentiary Standard:** Visual similarity serves strictly as a **candidate generator**, NOT conclusive proof.
    > Investigators must manually verify or reject proposed matches before establishing clock alignment.
    """)

    if not st.session_state.candidate_matches:
        st.info("No cross-camera candidate matches generated yet. Load the demo dataset or ingest videos.")
    else:
        for idx, m in enumerate(st.session_state.candidate_matches):
            with st.container():
                st.markdown(f"""
                <div class="forensic-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: #f8fafc;">Match #{idx+1}: {m.match_id}</span>
                        <span style="color: #38bdf8; font-weight: 700;">Similarity Score: {m.similarity_score:.1%}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">
                        pHash Hamming Dist: {m.phash_distance}/64 | Color Intersection: {m.color_similarity:.2f} | Raw Delta: {m.raw_time_delta_sec:+.3f}s
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b, col_act = st.columns([2, 2, 1.5])
                with col_a:
                    st.markdown(f"**{m.camera_a_name}** (Frame {m.frame_a} @ {m.time_a_sec:.2f}s)")
                    if m.thumb_a and Path(m.thumb_a).is_file():
                        st.image(m.thumb_a, use_container_width=True)
                with col_b:
                    st.markdown(f"**{m.camera_b_name}** (Frame {m.frame_b} @ {m.time_b_sec:.2f}s)")
                    if m.thumb_b and Path(m.thumb_b).is_file():
                        st.image(m.thumb_b, use_container_width=True)
                with col_act:
                    st.markdown(f"**Status:** `{m.review_status}`")
                    if st.button(f"✓ Confirm as Anchor", key=f"confirm_{idx}"):
                        m.review_status = "CONFIRMED_BY_USER"
                        m.is_anchor = True
                        # Add to confirmed anchors
                        st.session_state.confirmed_anchors.append(
                            TimeAnchorPoint(
                                match_id=m.match_id,
                                camera_id=m.camera_b_id,
                                source_time_sec=m.time_b_sec,
                                reference_time_sec=m.time_a_sec
                            )
                        )
                        st.success("Anchor confirmed!")
                        st.rerun()

                    if st.button(f"✗ Reject Candidate", key=f"reject_{idx}"):
                        m.review_status = "REJECTED_BY_USER"
                        m.is_anchor = False
                        st.rerun()


# ==========================================
# TAB 4: MANUAL CLOCK CORRECTIONS
# ==========================================
with tab_manual:
    st.subheader("Manual Clock Offset & Verified Reference Adjustments")
    st.markdown("Investigator can enter known clock discrepancies or verified reference anchor timestamps directly.")

    if not st.session_state.metadata:
        st.info("No cameras loaded.")
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("### Manual Camera Offset Override")
            target_cam = st.selectbox(
                "Select Camera Stream to Calibrate",
                options=list(st.session_state.metadata.keys()),
                format_func=lambda c: st.session_state.metadata[c].camera_name
            )

            current_offset, current_rate = st.session_state.manual_offsets.get(target_cam, (0.0, 1.0))
            new_offset = st.number_input(
                "Manual Time Offset (seconds)",
                value=float(current_offset if current_offset is not None else 0.0),
                step=0.1,
                help="Positive value means camera clock was behind reference clock."
            )
            new_rate = st.number_input(
                "Manual Clock Rate Drift Multiplier",
                value=float(current_rate if current_rate is not None else 1.0),
                step=0.0001,
                format="%.6f",
                help="Nominal is 1.000000. E.g. 1.001 represents 1000 ppm drift."
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Apply Manual Calibration", use_container_width=True):
                    st.session_state.manual_offsets[target_cam] = (new_offset, new_rate)
                    st.session_state.evidence_manager.record_audit(
                        action="MANUAL_OFFSET_APPLIED",
                        details={"camera_id": target_cam, "offset": new_offset, "rate": new_rate}
                    )
                    st.success(f"Applied manual calibration to {target_cam}")
                    st.rerun()
            with col_btn2:
                if st.button("Reset to Algorithmic", use_container_width=True):
                    st.session_state.manual_offsets.pop(target_cam, None)
                    st.success(f"Reset {target_cam} to RANSAC calibration")
                    st.rerun()

        with col_c2:
            st.markdown("### Add Verified Time Anchor")
            st.markdown("Link an exact source timestamp in any camera to an exact ground-truth reference time.")

            anchor_cam = st.selectbox(
                "Camera Stream",
                options=list(st.session_state.metadata.keys()),
                format_func=lambda c: st.session_state.metadata[c].camera_name,
                key="anchor_cam_select"
            )
            src_time = st.number_input("Source Video Timestamp (sec)", value=0.0, step=0.1)
            ref_time = st.number_input("Known Reference Timeline Timestamp (sec)", value=0.0, step=0.1)
            anchor_note = st.text_input("Corroboration Notes", value="Audio pop / vehicle collision visible on both")

            if st.button("Add Verified Forensic Anchor", use_container_width=True):
                new_a_id = f"ANCHOR-MANUAL-{len(st.session_state.confirmed_anchors)+1:03d}"
                st.session_state.confirmed_anchors.append(
                    TimeAnchorPoint(
                        match_id=new_a_id,
                        camera_id=anchor_cam,
                        source_time_sec=src_time,
                        reference_time_sec=ref_time,
                        weight=1.0,
                        is_confirmed=True
                    )
                )
                st.session_state.evidence_manager.record_audit(
                    action="VERIFIED_ANCHOR_ADDED",
                    details={"anchor_id": new_a_id, "camera_id": anchor_cam, "src_time": src_time, "ref_time": ref_time}
                )
                st.success(f"Added anchor {new_a_id}!")
                st.rerun()


# ==========================================
# TAB 5: FORENSIC EXPORT PACKAGES
# ==========================================
with tab_reports:
    st.subheader("Court-Admissible Forensic Data Package Export")
    st.markdown("""
    Generates all four standard digital forensic packages with cryptographic validation:
    1. `unified_timeline.csv` — Unified timeline with uncertainty bounds and derived flags.
    2. `event_matches.csv` — Corroborated cross-camera candidate and anchor matches.
    3. `synchronization_report.json` — Calibration formulas, RANSAC inliers/outliers, and limitations disclosures.
    4. `evidence_manifest.json` — Cryptographic SHA-256 catalog and append-only audit trail.
    """)

    if not st.session_state.metadata:
        st.info("No data available to export.")
    else:
        tb = TimelineBuilder(metadata_dict=st.session_state.metadata, alignments=alignments)
        unified_events = tb.build_unified_events(
            st.session_state.events,
            st.session_state.candidate_matches,
            st.session_state.candidate_matches
        )

        exporter = ForensicReportExporter(reports_dir=REPORTS_DIR)

        if st.button("📦 Generate Complete Forensic Report Package", type="primary", use_container_width=True):
            with st.spinner("Compiling cryptographic manifests and reports..."):
                em: EvidenceManager = st.session_state.evidence_manager
                paths = exporter.export_all(
                    unified_events=unified_events,
                    matches=st.session_state.candidate_matches,
                    alignments=alignments,
                    metadata=st.session_state.metadata,
                    evidence_records=em.get_all_records(),
                    evidence_manager=em,
                    case_id=case_id,
                    investigator_name=examiner
                )
                st.session_state.last_exported_files = paths
                st.success("Successfully generated all forensic artifacts!")

        if st.session_state.last_exported_files:
            st.markdown("### Download Artifacts")
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                p_csv = Path(st.session_state.last_exported_files["unified_timeline_csv"])
                if p_csv.is_file():
                    st.download_button(
                        label="📥 Download unified_timeline.csv",
                        data=p_csv.read_text(encoding="utf-8"),
                        file_name="unified_timeline.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                p_matches = Path(st.session_state.last_exported_files["event_matches_csv"])
                if p_matches.is_file():
                    st.download_button(
                        label="📥 Download event_matches.csv",
                        data=p_matches.read_text(encoding="utf-8"),
                        file_name="event_matches.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with col_d2:
                p_sync = Path(st.session_state.last_exported_files["synchronization_report_json"])
                if p_sync.is_file():
                    st.download_button(
                        label="📥 Download synchronization_report.json",
                        data=p_sync.read_text(encoding="utf-8"),
                        file_name="synchronization_report.json",
                        mime="application/json",
                        use_container_width=True
                    )

                p_man = Path(st.session_state.last_exported_files["evidence_manifest_json"])
                if p_man.is_file():
                    st.download_button(
                        label="📥 Download evidence_manifest.json",
                        data=p_man.read_text(encoding="utf-8"),
                        file_name="evidence_manifest.json",
                        mime="application/json",
                        use_container_width=True
                    )
