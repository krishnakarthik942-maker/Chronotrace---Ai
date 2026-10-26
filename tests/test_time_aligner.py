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
