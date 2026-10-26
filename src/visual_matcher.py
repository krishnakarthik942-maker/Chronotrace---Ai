"""Visual matching and multi-cue candidate generation for ChronoTrace AI.

Computes perceptual hashes (dHash, pHash) and color histogram descriptors.
Generates candidate cross-camera event correspondences with strict forensic disclaimers,
enforcing human-in-the-loop verification before clock alignment.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.event_detector import VisualEvent


@dataclass
class CandidateMatch:
    """A proposed visual correspondence between two cameras."""

    match_id: str
    camera_a_id: str
    camera_a_name: str
    event_a_id: str
    frame_a: int
    time_a_sec: float
    thumb_a: Optional[str]

    camera_b_id: str
    camera_b_name: str
    event_b_id: str
    frame_b: int
    time_b_sec: float
    thumb_b: Optional[str]

    similarity_score: float  # Combined score [0.0 - 1.0]
    phash_distance: int  # Hamming distance [0 - 64]
    color_similarity: float  # Histogram intersection [0.0 - 1.0]
    review_status: str = "CANDIDATE_UNVERIFIED"  # "CANDIDATE_UNVERIFIED", "CONFIRMED_BY_USER", "REJECTED_BY_USER"
    is_anchor: bool = False
    notes: str = ""
    forensic_disclaimer: str = (
        "Visual similarity is a heuristic candidate generator and NOT conclusive proof. "
        "Admissibility requires independent investigator corroboration."
    )

    @property
    def raw_time_delta_sec(self) -> float:
        """Raw difference in source timestamps (Cam B - Cam A)."""
        return round(self.time_b_sec - self.time_a_sec, 4)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["raw_time_delta_sec"] = self.raw_time_delta_sec
        return d


class VisualMatcher:
    """Generates cross-camera candidate event matches using perceptual hashing and color cues."""

    def __init__(self, min_similarity_threshold: float = 0.65, max_hamming_dist: int = 24):
        self.min_similarity = min_similarity_threshold
        self.max_hamming = max_hamming_dist

    @staticmethod
    def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
        """Compute difference hash (dHash) as a 64-bit integer."""
        resized = cv2.resize(image_bgr, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Compare adjacent pixels horizontally
        diff = gray[:, 1:] > gray[:, :-1]
        # Flatten boolean array into bit integer
        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    @staticmethod
    def compute_phash(image_bgr: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
        """Compute DCT-based perceptual hash (pHash) as a 64-bit integer."""
        img_size = hash_size * highfreq_factor
        resized = cv2.resize(image_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute 2D Discrete Cosine Transform
        dct = cv2.dct(gray)
        # Extract low frequencies (top-left block, excluding DC component at [0,0])
        dct_low = dct[0:hash_size, 0:hash_size]
        med = float(np.median(dct_low))
        diff = dct_low > med

        hash_val = 0
        for bit in diff.flatten():
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    @staticmethod
    def compute_color_histogram(image_bgr: np.ndarray, bins: Tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
        """Compute normalized 3D HSV color histogram."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Compute bitwise Hamming distance between two 64-bit integers."""
        return bin(hash1 ^ hash2).count("1")

    @classmethod
    def compute_descriptor(cls, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Compute unified descriptor dictionary for an image."""
        dhash = cls.compute_dhash(image_bgr)
        phash = cls.compute_phash(image_bgr)
        color_hist = cls.compute_color_histogram(image_bgr)
        return {
            "dhash": dhash,
            "phash": phash,
            "color_hist": color_hist.tolist()
        }

    def compute_event_descriptor(self, event: VisualEvent) -> None:
        """Ensure event has computed visual descriptors from its thumbnail or image."""
        if "phash" in event.descriptor:
            return

        if event.thumbnail_path and Path(event.thumbnail_path).is_file():
            img = cv2.imread(event.thumbnail_path)
            if img is not None:
                event.descriptor = self.compute_descriptor(img)

    def match_events(
        self,
        events_cam_a: List[VisualEvent],
        events_cam_b: List[VisualEvent],
        max_time_window_sec: Optional[float] = None
    ) -> List[CandidateMatch]:
        """Generate candidate visual matches between two camera event lists."""
        matches: List[CandidateMatch] = []
        if not events_cam_a or not events_cam_b:
            return matches

        # Ensure descriptors are populated
        for e in events_cam_a:
            self.compute_event_descriptor(e)
        for e in events_cam_b:
            self.compute_event_descriptor(e)

        match_count = 0
        for ea in events_cam_a:
            if "phash" not in ea.descriptor:
                continue
            phash_a = ea.descriptor["phash"]
            hist_a = np.array(ea.descriptor["color_hist"], dtype=np.float32)

            for eb in events_cam_b:
                if "phash" not in eb.descriptor:
                    continue

                # Optional time search window filter
                if max_time_window_sec is not None:
                    if abs(ea.source_timestamp_sec - eb.source_timestamp_sec) > max_time_window_sec:
                        continue

                phash_b = eb.descriptor["phash"]
                hist_b = np.array(eb.descriptor["color_hist"], dtype=np.float32)

                dist = self.hamming_distance(phash_a, phash_b)
                # Normalized phash similarity [0.0, 1.0] (64-bit max)
                phash_sim = max(0.0, 1.0 - (dist / 64.0))

                # Color histogram intersection
                color_sim = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_INTERSECT))
                color_sim = max(0.0, min(1.0, color_sim))

                # Fused similarity: 65% perceptual hash structure + 35% color distribution
                fused_sim = 0.65 * phash_sim + 0.35 * color_sim

                if dist <= self.max_hamming or fused_sim >= self.min_similarity:
                    match_count += 1
                    match_id = f"MATCH-{ea.camera_id[:4]}-{eb.camera_id[:4]}-{match_count:04d}"

                    candidate = CandidateMatch(
                        match_id=match_id,
                        camera_a_id=ea.camera_id,
                        camera_a_name=ea.camera_name,
                        event_a_id=ea.event_id,
                        frame_a=ea.frame_idx,
                        time_a_sec=ea.source_timestamp_sec,
                        thumb_a=ea.thumbnail_path,
                        camera_b_id=eb.camera_id,
                        camera_b_name=eb.camera_name,
                        event_b_id=eb.event_id,
                        frame_b=eb.frame_idx,
                        time_b_sec=eb.source_timestamp_sec,
                        thumb_b=eb.thumbnail_path,
                        similarity_score=round(fused_sim, 3),
                        phash_distance=dist,
                        color_similarity=round(color_sim, 3),
                        review_status="CANDIDATE_UNVERIFIED",
                        notes=f"Structural match (pHash dist={dist}/64, color={color_sim:.2f})"
                    )
                    matches.append(candidate)

        # Sort matches by fused similarity descending
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches

    @staticmethod
    def create_manual_anchor(
        camera_a_id: str,
        camera_a_name: str,
        event_a_id: str,
        frame_a: int,
        time_a_sec: float,
        camera_b_id: str,
        camera_b_name: str,
        event_b_id: str,
        frame_b: int,
        time_b_sec: float,
        notes: str = "Verified forensic anchor entered by investigator",
        thumb_a: Optional[str] = None,
        thumb_b: Optional[str] = None
    ) -> CandidateMatch:
        """Create a user-verified ground-truth anchor match."""
        match_id = f"ANCHOR-{camera_a_id[:4]}-{camera_b_id[:4]}-{int(time_a_sec)}"
        return CandidateMatch(
            match_id=match_id,
            camera_a_id=camera_a_id,
            camera_a_name=camera_a_name,
            event_a_id=event_a_id,
            frame_a=frame_a,
            time_a_sec=round(float(time_a_sec), 4),
            thumb_a=thumb_a,
            camera_b_id=camera_b_id,
            camera_b_name=camera_b_name,
            event_b_id=event_b_id,
            frame_b=frame_b,
            time_b_sec=round(float(time_b_sec), 4),
            thumb_b=thumb_b,
            similarity_score=1.0,
            phash_distance=0,
            color_similarity=1.0,
            review_status="CONFIRMED_BY_USER",
            is_anchor=True,
            notes=notes
        )
