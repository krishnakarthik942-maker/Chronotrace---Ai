"""Tests for clock correction, RANSAC drift estimation, and timing error quantification."""

import pytest
from src.time_aligner import CameraAlignmentResult, TimeAligner, TimeAnchorPoint


def test_reference_camera_calibration():
    aligner = TimeAligner(reference_camera_id="cam_ref")
    res = aligner.align_camera_from_anchors(
        camera_id="cam_ref",
        camera_name="Reference Cam",
        anchors=[]
    )

    assert res.is_reference is True
    assert res.offset_sec == 0.0
    assert res.rate_drift == 1.0
    assert res.status == "SYNCHRONIZED"
    assert res.uncertainty_sec == 0.0


def test_insufficient_anchors():
    aligner = TimeAligner(reference_camera_id="cam_ref")
    res = aligner.align_camera_from_anchors(
        camera_id="cam_2",
        camera_name="Cam 2",
        anchors=[]
    )

    assert res.status == "UNALIGNED"
    assert res.uncertainty_sec == float("inf")
    assert "Insufficient" in res.method_description


def test_single_anchor_constant_offset():
    aligner = TimeAligner(reference_camera_id="cam_ref")
    # Camera 2 local time 10.0s corresponds to Reference time 14.5s
    # offset = 14.5 - 10.0 = 4.5s
    anchors = [
        TimeAnchorPoint(
            match_id="M1",
            camera_id="cam_2",
            source_time_sec=10.0,
            reference_time_sec=14.5
        )
    ]
    res = aligner.align_camera_from_anchors(
        camera_id="cam_2",
        camera_name="Cam 2",
        anchors=anchors,
        nominal_fps=30.0
    )

    assert res.status == "APPROXIMATE"
    assert abs(res.offset_sec - 4.5) < 1e-4
    assert res.rate_drift == 1.0
    assert res.uncertainty_sec > 0.0
    assert "M1" in res.supporting_match_ids


def test_known_positive_and_negative_offsets_with_drift():
    aligner = TimeAligner(reference_camera_id="cam_ref")

    # True model for Cam Positive: T_ref = 5.0 + 1.001 * t_src
    anchors_pos = [
        TimeAnchorPoint("M1", "cam_pos", 0.0, 5.0),
        TimeAnchorPoint("M2", "cam_pos", 10.0, 5.0 + 1.001 * 10.0),
        TimeAnchorPoint("M3", "cam_pos", 20.0, 5.0 + 1.001 * 20.0),
        TimeAnchorPoint("M4", "cam_pos", 30.0, 5.0 + 1.001 * 30.0)
    ]
    res_pos = aligner.align_camera_from_anchors("cam_pos", "Cam Pos", anchors_pos)
    assert abs(res_pos.offset_sec - 5.0) < 0.01
    assert abs(res_pos.rate_drift - 1.001) < 0.0005
    assert res_pos.status == "SYNCHRONIZED"
    assert len(res_pos.supporting_match_ids) == 4

    # True model for Cam Negative: T_ref = -3.2 + 0.999 * t_src
    anchors_neg = [
        TimeAnchorPoint("N1", "cam_neg", 5.0, -3.2 + 0.999 * 5.0),
        TimeAnchorPoint("N2", "cam_neg", 15.0, -3.2 + 0.999 * 15.0),
        TimeAnchorPoint("N3", "cam_neg", 25.0, -3.2 + 0.999 * 25.0)
    ]
    res_neg = aligner.align_camera_from_anchors("cam_neg", "Cam Neg", anchors_neg)
    assert abs(res_neg.offset_sec - (-3.2)) < 0.01
    assert abs(res_neg.rate_drift - 0.999) < 0.0005
    assert res_neg.status == "SYNCHRONIZED"


def test_ransac_rejects_false_visual_matches():
    aligner = TimeAligner(reference_camera_id="cam_ref", ransac_residual_threshold_sec=0.25)

    # True model: T_ref = 2.0 + 1.0 * t_src
    # Outlier match: M_false at t_src = 12.0s with T_ref = 18.0s (a 4-second mismatch!)
    anchors = [
        TimeAnchorPoint("M1", "cam_x", 0.0, 2.0),
        TimeAnchorPoint("M2", "cam_x", 5.0, 7.0),
        TimeAnchorPoint("M3", "cam_x", 10.0, 12.0),
        TimeAnchorPoint("M_false", "cam_x", 12.0, 18.0),  # Gross false visual correlation
        TimeAnchorPoint("M4", "cam_x", 15.0, 17.0),
        TimeAnchorPoint("M5", "cam_x", 20.0, 22.0)
    ]

    res = aligner.align_camera_from_anchors("cam_x", "Cam X", anchors)

    # RANSAC must isolate M_false into rejected_match_ids
    assert "M_false" in res.rejected_match_ids
    assert len(res.supporting_match_ids) == 5
    assert abs(res.offset_sec - 2.0) < 0.05
    assert abs(res.rate_drift - 1.0) < 0.01


def test_conflicting_anchors_flagged_for_human_review():
    aligner = TimeAligner(reference_camera_id="cam_ref", min_inlier_ratio=0.6)

    # Completely contradictory anchor points (e.g. 2 anchors say +10s, 2 anchors say -10s)
    anchors = [
        TimeAnchorPoint("C1", "cam_c", 0.0, 10.0),
        TimeAnchorPoint("C2", "cam_c", 5.0, 15.0),
        TimeAnchorPoint("C3", "cam_c", 0.0, -10.0),
        TimeAnchorPoint("C4", "cam_c", 5.0, -5.0)
    ]

    res = aligner.align_camera_from_anchors("cam_c", "Cam Conflicting", anchors)
    assert res.status == "CONFLICTING"
    assert "Conflicting" in res.method_description
    assert res.uncertainty_sec > 1.0


def test_reproducibility_of_estimated_offsets():
    aligner1 = TimeAligner(reference_camera_id="cam_ref")
    aligner2 = TimeAligner(reference_camera_id="cam_ref")

    anchors = [
        TimeAnchorPoint("M1", "cam_rep", 1.0, 4.0),
        TimeAnchorPoint("M2", "cam_rep", 6.0, 9.0),
        TimeAnchorPoint("M3", "cam_rep", 11.0, 14.0),
        TimeAnchorPoint("M4", "cam_rep", 16.0, 19.0)
    ]

    res1 = aligner1.align_camera_from_anchors("cam_rep", "Cam Rep", anchors)
    res2 = aligner2.align_camera_from_anchors("cam_rep", "Cam Rep", anchors)

    assert res1.offset_sec == res2.offset_sec
    assert res1.rate_drift == res2.rate_drift
    assert res1.rmse_residual_sec == res2.rmse_residual_sec


def test_timing_error_metrics():
    estimated = {"cam_1": 0.0, "cam_2": 3.48, "cam_3": -2.02}
    ground_truth = {"cam_1": 0.0, "cam_2": 3.50, "cam_3": -2.00}

    stats = TimeAligner.evaluate_timing_error(estimated, ground_truth)
    assert stats["count"] == 3
    assert stats["median_error_sec"] == 0.02
    assert stats["p95_error_sec"] <= 0.02
    assert stats["max_error_sec"] == 0.02


def test_two_anchors_zero_dof_is_approximate():
    """Verify that exactly 2 anchors (DoF = 0) cannot be reported as authoritative SYNCHRONIZED."""
    aligner = TimeAligner(reference_camera_id="cam_ref")
    anchors = [
        TimeAnchorPoint("A1", "cam_2pt", 2.0, 5.0),
        TimeAnchorPoint("A2", "cam_2pt", 8.0, 11.0)
    ]
    res = aligner.align_camera_from_anchors("cam_2pt", "Cam 2Pt", anchors)

    assert res.status == "APPROXIMATE"
    assert "DoF=0" in res.method_description or "unverified drift" in res.method_description
    assert "Degrees of freedom = 0" in res.forensic_notes
    assert abs(res.offset_sec - 3.0) < 1e-4
    assert abs(res.rate_drift - 1.0) < 1e-4


def test_resolve_anchor_from_match_direct_ref():
    """Verify direct match resolution when one camera is the reference camera."""
    from src.visual_matcher import CandidateMatch

    # Case A: Camera A is reference
    match_a = CandidateMatch(
        match_id="M-DIR-A", camera_a_id="cam_ref", camera_a_name="Ref", event_a_id="E1",
        frame_a=30, time_a_sec=10.0, thumb_a=None, camera_b_id="cam_src", camera_b_name="Src",
        event_b_id="E2", frame_b=45, time_b_sec=12.5, thumb_b=None, similarity_score=0.9,
        phash_distance=5, color_similarity=0.9, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )
    anchor_a, msg_a = TimeAligner.resolve_anchor_from_match(match_a, "cam_ref", {})
    assert anchor_a is not None
    assert anchor_a.camera_id == "cam_src"
    assert anchor_a.source_time_sec == 12.5
    assert anchor_a.reference_time_sec == 10.0
    assert "Direct reference anchor" in msg_a

    # Case B: Camera B is reference
    match_b = CandidateMatch(
        match_id="M-DIR-B", camera_a_id="cam_src", camera_a_name="Src", event_a_id="E1",
        frame_a=30, time_a_sec=14.0, thumb_a=None, camera_b_id="cam_ref", camera_b_name="Ref",
        event_b_id="E2", frame_b=45, time_b_sec=11.0, thumb_b=None, similarity_score=0.9,
        phash_distance=5, color_similarity=0.9, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )
    anchor_b, msg_b = TimeAligner.resolve_anchor_from_match(match_b, "cam_ref", {})
    assert anchor_b is not None
    assert anchor_b.camera_id == "cam_src"
    assert anchor_b.source_time_sec == 14.0
    assert anchor_b.reference_time_sec == 11.0
    assert "Direct reference anchor" in msg_b


def test_resolve_anchor_from_match_transitive_non_reference():
    """Verify transitive mapping between two non-reference cameras via calibrated intermediate."""
    from src.visual_matcher import CandidateMatch

    # cam_calib is calibrated to reference timeline with offset=5.0s, rate=1.0
    alignments = {
        "cam_ref": CameraAlignmentResult("cam_ref", "Ref", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_calib": CameraAlignmentResult("cam_calib", "Calib", is_reference=False, offset_sec=5.0, rate_drift=1.0, status="SYNCHRONIZED"),
    }

    # Match between calibrated cam_calib (time=10.0s) and uncalibrated cam_new (time=12.0s)
    # Reference time should be: 5.0 + 1.0 * 10.0 = 15.0s
    match = CandidateMatch(
        match_id="M-TRANS", camera_a_id="cam_calib", camera_a_name="Calib", event_a_id="E1",
        frame_a=300, time_a_sec=10.0, thumb_a=None, camera_b_id="cam_new", camera_b_name="New",
        event_b_id="E2", frame_b=360, time_b_sec=12.0, thumb_b=None, similarity_score=0.9,
        phash_distance=5, color_similarity=0.9, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )
    anchor, msg = TimeAligner.resolve_anchor_from_match(match, "cam_ref", alignments)

    assert anchor is not None
    assert anchor.camera_id == "cam_new"
    assert anchor.source_time_sec == 12.0
    assert abs(anchor.reference_time_sec - 15.0) < 1e-4
    assert "Transitive anchor confirmed" in msg
    assert "10.000s -> 15.000s" in msg


def test_resolve_anchor_from_match_unanchored_safely_rejected():
    """Verify that a match between two non-reference cameras with no reference connection is safely rejected."""
    from src.visual_matcher import CandidateMatch

    alignments = {
        "cam_ref": CameraAlignmentResult("cam_ref", "Ref", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_uncal1": CameraAlignmentResult("cam_uncal1", "Uncal 1", is_reference=False, offset_sec=0.0, rate_drift=1.0, status="UNALIGNED"),
    }

    match = CandidateMatch(
        match_id="M-UNANCHORED", camera_a_id="cam_uncal1", camera_a_name="Uncal 1", event_a_id="E1",
        frame_a=30, time_a_sec=5.0, thumb_a=None, camera_b_id="cam_uncal2", camera_b_name="Uncal 2",
        event_b_id="E2", frame_b=45, time_b_sec=8.0, thumb_b=None, similarity_score=0.9,
        phash_distance=5, color_similarity=0.9, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )
    anchor, msg = TimeAligner.resolve_anchor_from_match(match, "cam_ref", alignments)

    assert anchor is None
    assert "Cannot establish anchor between non-reference cameras" in msg
    assert "cam_uncal1" in msg and "cam_uncal2" in msg
    assert "Please calibrate at least one camera against the reference timeline first." in msg


def test_resolve_anchor_from_match_invalid_timestamps():
    """Verify that non-finite or negative timestamps are rejected safely."""
    from src.visual_matcher import CandidateMatch

    bad_match = CandidateMatch(
        match_id="M-BAD", camera_a_id="cam_ref", camera_a_name="Ref", event_a_id="E1",
        frame_a=0, time_a_sec=-2.5, thumb_a=None, camera_b_id="cam_src", camera_b_name="Src",
        event_b_id="E2", frame_b=0, time_b_sec=10.0, thumb_b=None, similarity_score=0.9,
        phash_distance=5, color_similarity=0.9, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )
    anchor, msg = TimeAligner.resolve_anchor_from_match(bad_match, "cam_ref", {})
    assert anchor is None
    assert "Invalid anchor timestamps" in msg


def test_invalid_anchor_does_not_corrupt_existing_calibration():
    """Verify regression: invalid or rejected anchor operations do not corrupt existing calibrations."""
    aligner = TimeAligner(reference_camera_id="cam_ref")

    # Cam 1 has 3 valid anchors and is reliably synchronized
    valid_cam1_anchors = [
        TimeAnchorPoint("A1", "cam_1", 2.0, 5.0),
        TimeAnchorPoint("A2", "cam_1", 6.0, 9.0),
        TimeAnchorPoint("A3", "cam_1", 10.0, 13.0)
    ]
    res1_initial = aligner.align_camera_from_anchors("cam_1", "Cam 1", valid_cam1_anchors)
    assert res1_initial.status == "SYNCHRONIZED"
    assert abs(res1_initial.offset_sec - 3.0) < 1e-4

    # Now attempt alignment on Cam 2 with corrupt / invalid anchors
    corrupt_cam2_anchors = [
        TimeAnchorPoint("BAD1", "cam_2", -10.0, 5.0),
        TimeAnchorPoint("BAD2", "cam_2", float("nan"), 12.0)
    ]
    res2 = aligner.align_camera_from_anchors("cam_2", "Cam 2", corrupt_cam2_anchors)
    assert res2.status == "UNALIGNED"

    # Re-verify Cam 1 calibration is completely identical and uncorrupted
    res1_after = aligner.align_camera_from_anchors("cam_1", "Cam 1", valid_cam1_anchors)
    assert res1_after.offset_sec == res1_initial.offset_sec
    assert res1_after.rate_drift == res1_initial.rate_drift
    assert res1_after.status == res1_initial.status
    assert res1_after.rmse_residual_sec == res1_initial.rmse_residual_sec


def test_clustered_anchors_temporal_span_small():
    """Verify that anchors clustered within < 1.0 second are treated with constant offset and marked APPROXIMATE."""
    aligner = TimeAligner(reference_camera_id="cam_ref")
    clustered_anchors = [
        TimeAnchorPoint("C1", "cam_clust", 5.00, 8.00),
        TimeAnchorPoint("C2", "cam_clust", 5.20, 8.20),
        TimeAnchorPoint("C3", "cam_clust", 5.50, 8.50)
    ]
    res = aligner.align_camera_from_anchors("cam_clust", "Cam Clustered", clustered_anchors)

    assert res.status == "APPROXIMATE"
    assert "clustered anchors" in res.method_description.lower()
    assert "clustered within" in res.forensic_notes or "narrow temporal baselines" in res.forensic_notes
    assert abs(res.rate_drift - 1.0) < 1e-6  # Forced to constant offset
    assert abs(res.offset_sec - 3.0) < 1e-4


def test_outlier_filtering_and_robust_alignment():
    """Verify RANSAC outlier filtering when misleading candidate matches are present.
    Because outliers were rejected, status is documented as APPROXIMATE."""
    aligner = TimeAligner(reference_camera_id="cam_ref", ransac_residual_threshold_sec=0.35)
    # 5 true anchors at offset = 4.0s (plus tiny noise), 2 false/misleading matches
    anchors = [
        TimeAnchorPoint("T1", "cam_outlier", 2.0, 6.0),
        TimeAnchorPoint("T2", "cam_outlier", 5.0, 9.0),
        TimeAnchorPoint("T3", "cam_outlier", 8.0, 12.0),
        TimeAnchorPoint("T4", "cam_outlier", 11.0, 15.0),
        TimeAnchorPoint("T5", "cam_outlier", 14.0, 18.0),
        # Misleading outliers
        TimeAnchorPoint("F1", "cam_outlier", 4.0, 25.0),
        TimeAnchorPoint("F2", "cam_outlier", 10.0, 2.0),
    ]
    res = aligner.align_camera_from_anchors("cam_outlier", "Cam Outlier", anchors)

    # With rejected outliers, forensic status is APPROXIMATE (cautious disclosure)
    assert res.status == "APPROXIMATE"
    assert abs(res.offset_sec - 4.0) < 0.2
    assert "F1" in res.rejected_match_ids and "F2" in res.rejected_match_ids
    assert len(res.supporting_match_ids) == 5


def test_long_duration_clock_drift_estimation():
    """Verify that multi-hour drift is reliably quantified with well-spread anchors."""
    aligner = TimeAligner(reference_camera_id="cam_ref")
    # True relationship: ref_time = 12.0 + 1.000100 * src_time
    true_offset = 12.0
    true_rate = 1.000100

    src_times = [0.0, 600.0, 1800.0, 3600.0]
    anchors = [
        TimeAnchorPoint(f"D{i}", "cam_drift", t, true_offset + true_rate * t)
        for i, t in enumerate(src_times)
    ]
    res = aligner.align_camera_from_anchors("cam_drift", "Cam Drift", anchors)

    assert res.status == "SYNCHRONIZED"
    assert abs(res.offset_sec - true_offset) < 0.05
    assert abs(res.rate_drift - true_rate) < 1e-5
    assert "Robust Linear Drift" in res.method_description


def test_cross_camera_anchor_math_two_non_reference_cameras_with_drift():
    """Verify exact mathematical transformation between two non-reference cameras with drift:
    T_ref(t_intermediate) = offset + rate * t_intermediate."""
    from src.visual_matcher import CandidateMatch

    cam2_offset = 12.3456
    cam2_rate = 1.000150
    alignments = {
        "cam_ref": CameraAlignmentResult("cam_ref", "Ref", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_2": CameraAlignmentResult("cam_2", "Cam 2", is_reference=False, offset_sec=cam2_offset, rate_drift=cam2_rate, status="SYNCHRONIZED"),
        "cam_3": CameraAlignmentResult("cam_3", "Cam 3", is_reference=False, offset_sec=0.0, rate_drift=1.0, status="UNALIGNED")
    }

    # Match: Cam 2 at 100.0s corresponds to Cam 3 at 95.0s
    t_cam2 = 100.0
    t_cam3 = 95.0
    expected_ref_time = round(cam2_offset + cam2_rate * t_cam2, 4)

    match = CandidateMatch(
        match_id="M-TRANS-DRIFT", camera_a_id="cam_2", camera_a_name="Cam 2", event_a_id="E2",
        frame_a=3000, time_a_sec=t_cam2, thumb_a=None, camera_b_id="cam_3", camera_b_name="Cam 3",
        event_b_id="E3", frame_b=2850, time_b_sec=t_cam3, thumb_b=None, similarity_score=0.92,
        phash_distance=4, color_similarity=0.9, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )

    anchor, msg = TimeAligner.resolve_anchor_from_match(match, "cam_ref", alignments)

    assert anchor is not None
    assert anchor.camera_id == "cam_3"
    assert anchor.source_time_sec == t_cam3
    assert abs(anchor.reference_time_sec - expected_ref_time) < 1e-4
    assert abs(anchor.reference_time_sec - 112.3606) < 1e-4


def test_cross_camera_anchor_both_uncalibrated_rejected():
    """Verify that attempting to anchor two uncalibrated non-reference cameras is safely rejected."""
    from src.visual_matcher import CandidateMatch

    alignments = {
        "cam_ref": CameraAlignmentResult("cam_ref", "Ref", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_x": CameraAlignmentResult("cam_x", "Cam X", is_reference=False, offset_sec=0.0, rate_drift=1.0, status="UNALIGNED"),
        "cam_y": CameraAlignmentResult("cam_y", "Cam Y", is_reference=False, offset_sec=0.0, rate_drift=1.0, status="UNALIGNED")
    }
    match = CandidateMatch(
        match_id="M-REJECT-UNCAL", camera_a_id="cam_x", camera_a_name="Cam X", event_a_id="EX",
        frame_a=100, time_a_sec=10.0, thumb_a=None, camera_b_id="cam_y", camera_b_name="Cam Y",
        event_b_id="EY", frame_b=200, time_b_sec=20.0, thumb_b=None, similarity_score=0.85,
        phash_distance=6, color_similarity=0.8, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )
    anchor, msg = TimeAligner.resolve_anchor_from_match(match, "cam_ref", alignments)

    assert anchor is None
    assert "Cannot establish anchor between non-reference cameras 'cam_x' and 'cam_y'" in msg
    assert "neither camera has an established calibration" in msg


def test_synchronized_status_criteria_enforcement():
    """Verify that status criteria definitions exist and are enforced across all operational regimes."""
    from src.time_aligner import FORENSIC_STATUS_CRITERIA

    assert "SYNCHRONIZED" in FORENSIC_STATUS_CRITERIA
    assert "APPROXIMATE" in FORENSIC_STATUS_CRITERIA
    assert "UNALIGNED" in FORENSIC_STATUS_CRITERIA
    assert "CONFLICTING" in FORENSIC_STATUS_CRITERIA
    assert "MANUAL_OVERRIDE" in FORENSIC_STATUS_CRITERIA

    aligner = TimeAligner(reference_camera_id="cam_ref")

    # 1. Zero anchors -> UNALIGNED
    res_zero = aligner.align_camera_from_anchors("cam_t", "Cam T", [])
    assert res_zero.status == "UNALIGNED"

    # 2. 1 anchor -> APPROXIMATE
    res_one = aligner.align_camera_from_anchors("cam_t", "Cam T", [TimeAnchorPoint("A1", "cam_t", 5.0, 10.0)])
    assert res_one.status == "APPROXIMATE"

    # 3. 2 anchors (DoF = 0) -> APPROXIMATE
    res_two = aligner.align_camera_from_anchors("cam_t", "Cam T", [
        TimeAnchorPoint("A1", "cam_t", 5.0, 10.0),
        TimeAnchorPoint("A2", "cam_t", 15.0, 20.0)
    ])
    assert res_two.status == "APPROXIMATE"

    # 4. 3+ well spread, 0 outliers -> SYNCHRONIZED
    res_sync = aligner.align_camera_from_anchors("cam_t", "Cam T", [
        TimeAnchorPoint("A1", "cam_t", 5.0, 10.0),
        TimeAnchorPoint("A2", "cam_t", 15.0, 20.0),
        TimeAnchorPoint("A3", "cam_t", 25.0, 30.0)
    ])
    assert res_sync.status == "SYNCHRONIZED"

    # 5. Manual override -> MANUAL_OVERRIDE
    res_man = aligner.align_camera_from_anchors("cam_t", "Cam T", [], manual_offset=3.5, manual_rate=1.0)
    assert res_man.status == "MANUAL_OVERRIDE"


def test_cross_camera_anchor_reverse_direction_transitive():
    """Verify transitive anchor mapping when Camera A is uncalibrated and Camera B is calibrated."""
    from src.visual_matcher import CandidateMatch

    cam2_offset = 12.3456
    cam2_rate = 1.000150
    alignments = {
        "cam_ref": CameraAlignmentResult("cam_ref", "Ref", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_2": CameraAlignmentResult("cam_2", "Cam 2", is_reference=False, offset_sec=cam2_offset, rate_drift=cam2_rate, status="SYNCHRONIZED"),
        "cam_3": CameraAlignmentResult("cam_3", "Cam 3", is_reference=False, offset_sec=0.0, rate_drift=1.0, status="UNALIGNED")
    }

    # Match where Camera A is uncalibrated (cam_3) and Camera B is calibrated (cam_2)
    t_cam3 = 95.0
    t_cam2 = 100.0
    expected_ref_time = round(cam2_offset + cam2_rate * t_cam2, 4)

    match = CandidateMatch(
        match_id="M-REVERSE-TRANS", camera_a_id="cam_3", camera_a_name="Cam 3", event_a_id="E3",
        frame_a=2850, time_a_sec=t_cam3, thumb_a=None, camera_b_id="cam_2", camera_b_name="Cam 2",
        event_b_id="E2", frame_b=3000, time_b_sec=t_cam2, thumb_b=None, similarity_score=0.91,
        phash_distance=4, color_similarity=0.9, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )

    anchor, msg = TimeAligner.resolve_anchor_from_match(match, "cam_ref", alignments)

    assert anchor is not None
    assert anchor.camera_id == "cam_3"
    assert anchor.source_time_sec == t_cam3
    assert abs(anchor.reference_time_sec - expected_ref_time) < 1e-4
    assert abs(anchor.reference_time_sec - 112.3606) < 1e-4
    assert "Transitive anchor confirmed for 'cam_3' via calibrated camera 'cam_2'" in msg


def test_candidate_confirmation_and_subsequent_rejection_lifecycle():
    """Verify that confirming a candidate creates an anchor, re-confirming deduplicates, and rejecting removes it."""
    from src.visual_matcher import CandidateMatch

    alignments = {
        "cam_ref": CameraAlignmentResult("cam_ref", "Ref", is_reference=True, offset_sec=0.0, rate_drift=1.0, status="SYNCHRONIZED"),
        "cam_2": CameraAlignmentResult("cam_2", "Cam 2", is_reference=False, offset_sec=0.0, rate_drift=1.0, status="UNALIGNED"),
    }

    match = CandidateMatch(
        match_id="M-CYCLE-01", camera_a_id="cam_ref", camera_a_name="Ref", event_a_id="ER",
        frame_a=150, time_a_sec=5.0, thumb_a=None, camera_b_id="cam_2", camera_b_name="Cam 2",
        event_b_id="E2", frame_b=180, time_b_sec=6.0, thumb_b=None, similarity_score=0.95,
        phash_distance=2, color_similarity=0.95, review_status="CANDIDATE_UNVERIFIED", is_anchor=False
    )

    confirmed_anchors = []

    # 1. Examiner confirms candidate
    anchor, msg = TimeAligner.resolve_anchor_from_match(match, "cam_ref", alignments)
    assert anchor is not None
    match.review_status = "CONFIRMED_BY_USER"
    match.is_anchor = True
    confirmed_anchors = [a for a in confirmed_anchors if a.match_id not in (match.match_id, f"{match.match_id}-transitive", anchor.match_id)]
    confirmed_anchors.append(anchor)
    assert len(confirmed_anchors) == 1
    assert confirmed_anchors[0].match_id == "M-CYCLE-01"

    # 2. Re-confirming should deduplicate and not duplicate
    confirmed_anchors = [a for a in confirmed_anchors if a.match_id not in (match.match_id, f"{match.match_id}-transitive", anchor.match_id)]
    confirmed_anchors.append(anchor)
    assert len(confirmed_anchors) == 1

    # 3. Examiner subsequently rejects candidate
    match.review_status = "REJECTED_BY_USER"
    match.is_anchor = False
    confirmed_anchors = [a for a in confirmed_anchors if a.match_id not in (match.match_id, f"{match.match_id}-transitive")]
    assert len(confirmed_anchors) == 0

    # Ensure alignment with empty anchors treats camera as UNALIGNED
    aligner = TimeAligner(reference_camera_id="cam_ref")
    res = aligner.align_camera_from_anchors("cam_2", "Cam 2", confirmed_anchors)
    assert res.status == "UNALIGNED"
