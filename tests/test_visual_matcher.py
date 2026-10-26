"""Tests for perceptual hashing, visual similarity descriptors, and candidate generation."""

from pathlib import Path
import cv2
import numpy as np
import pytest
from src.event_detector import VisualEvent
from src.visual_matcher import CandidateMatch, VisualMatcher


def test_phash_and_dhash_on_identical_frames():
    # Create synthetic frame
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (0, 255, 255), -1)

    dhash1 = VisualMatcher.compute_dhash(img)
    dhash2 = VisualMatcher.compute_dhash(img)
    phash1 = VisualMatcher.compute_phash(img)
    phash2 = VisualMatcher.compute_phash(img)

    assert dhash1 == dhash2
    assert phash1 == phash2
    assert VisualMatcher.hamming_distance(phash1, phash2) == 0


def test_phash_similarity_robust_to_slight_lighting():
    img1 = np.full((120, 120, 3), (60, 60, 60), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), (0, 0, 255), -1)

    # Slightly brighter version
    img2 = cv2.convertScaleAbs(img1, alpha=1.1, beta=10)

    phash1 = VisualMatcher.compute_phash(img1)
    phash2 = VisualMatcher.compute_phash(img2)
    dist = VisualMatcher.hamming_distance(phash1, phash2)

    # Hamming distance should be very small (< 6 bits difference out of 64)
    assert dist <= 6


def test_distinct_images_have_low_similarity():
    # Image 1: Red circle
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img1, (50, 50), 40, (0, 0, 255), -1)

    # Image 2: Green stripes
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(0, 100, 20):
        img2[y:y+10, :] = (0, 255, 0)

    desc1 = VisualMatcher.compute_descriptor(img1)
    desc2 = VisualMatcher.compute_descriptor(img2)

    dist = VisualMatcher.hamming_distance(desc1["phash"], desc2["phash"])
    assert dist > 15  # Distinct structural patterns


def test_candidate_match_generation(tmp_path: Path):
    # Save 2 frames to disk
    matcher = VisualMatcher(min_similarity_threshold=0.6, max_hamming_dist=20)

    img_a = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(img_a, (60, 60), 30, (255, 0, 0), -1)
    p_a = tmp_path / "evt_a.jpg"
    cv2.imwrite(str(p_a), img_a)

    img_b = img_a.copy()
    p_b = tmp_path / "evt_b.jpg"
    cv2.imwrite(str(p_b), img_b)

    evt_a = VisualEvent(
        event_id="EVT-A1",
        camera_id="cam_1",
        camera_name="Cam 1",
        source_filename="cam1.mp4",
        frame_idx=30,
        source_timestamp_sec=1.0,
        event_type="visual_spike",
        score=0.9,
        thumbnail_path=str(p_a)
    )
    evt_b = VisualEvent(
        event_id="EVT-B1",
        camera_id="cam_2",
        camera_name="Cam 2",
        source_filename="cam2.mp4",
        frame_idx=100,
        source_timestamp_sec=4.5,
        event_type="visual_spike",
        score=0.9,
        thumbnail_path=str(p_b)
    )

    matches = matcher.match_events([evt_a], [evt_b])
    assert len(matches) == 1
    m = matches[0]
    assert m.similarity_score >= 0.9
    assert m.review_status == "CANDIDATE_UNVERIFIED"
    assert "heuristic candidate generator" in m.forensic_disclaimer
    assert m.raw_time_delta_sec == 3.5


def test_manual_anchor_creation():
    anchor = VisualMatcher.create_manual_anchor(
        camera_a_id="cam_1",
        camera_a_name="Cam 1",
        event_a_id="EVT-01",
        frame_a=45,
        time_a_sec=1.5,
        camera_b_id="cam_2",
        camera_b_name="Cam 2",
        event_b_id="EVT-02",
        frame_b=105,
        time_b_sec=4.5,
        notes="Impact flash confirmed on both camera views"
    )

    assert anchor.is_anchor is True
    assert anchor.review_status == "CONFIRMED_BY_USER"
    assert anchor.raw_time_delta_sec == 3.0
    assert anchor.similarity_score == 1.0
