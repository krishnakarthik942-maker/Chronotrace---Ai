"""Visual event and scene change detection for ChronoTrace AI.

Uses PySceneDetect ContentDetector and adaptive visual difference thresholding
to pinpoint scene cuts, motion spikes, and salient visual events across video streams.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video


@dataclass
class VisualEvent:
    """A detected salient visual occurrence in a video stream."""

    event_id: str
    camera_id: str
    camera_name: str
    source_filename: str
    frame_idx: int
    source_timestamp_sec: float
    event_type: str  # "scene_cut", "motion_spike", "visual_spike", "keyframe"
    score: float  # Salience or confidence metric (0.0 - 1.0)
    thumbnail_path: Optional[str] = None
    descriptor: Dict[str, Any] = field(default_factory=dict)
    review_status: str = "UNREVIEWED"  # "UNREVIEWED", "CONFIRMED", "REJECTED"
    is_anchor: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventDetector:
    """Detects scene boundaries and visual transition events in video streams."""

    def __init__(self, cache_dir: str | Path = "data/cache/events"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def detect_events(
        self,
        video_path: str | Path,
        camera_id: str,
        camera_name: str,
        threshold: float = 27.0,
        min_scene_len_sec: float = 1.0,
        detect_motion_spikes: bool = True,
        motion_sample_fps: float = 4.0,
        motion_sensitivity: float = 0.25
    ) -> List[VisualEvent]:
        """Detect scene cuts via PySceneDetect and visual/motion spikes via OpenCV."""
        path = Path(video_path)
        events: List[VisualEvent] = []
        if not path.is_file():
            return events

        source_filename = path.name

        # 1. PySceneDetect Scene Cut Detection
        scene_events = self._detect_pyscenedetect(
            path, camera_id, camera_name, source_filename, threshold, min_scene_len_sec
        )
        events.extend(scene_events)

        # 2. Significant Visual Motion / Intensity Spikes
        if detect_motion_spikes:
            motion_events = self._detect_motion_spikes(
                path, camera_id, camera_name, source_filename,
                sample_fps=motion_sample_fps,
                sensitivity=motion_sensitivity,
                existing_timestamps=[e.source_timestamp_sec for e in events]
            )
            events.extend(motion_events)

        # Sort events by source timestamp
        events.sort(key=lambda e: e.source_timestamp_sec)
        return events

    def _detect_pyscenedetect(
        self,
        video_path: Path,
        camera_id: str,
        camera_name: str,
        source_filename: str,
        threshold: float,
        min_scene_len_sec: float
    ) -> List[VisualEvent]:
        """Run PySceneDetect ContentDetector on the video."""
        events: List[VisualEvent] = []
        try:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(video.frame_rate * min_scene_len_sec)))
            scene_manager.detect_scenes(video=video)
            scene_list = scene_manager.get_scene_list()

            for i, scene in enumerate(scene_list):
                start_frame = getattr(scene[0], "frame_num", None)
                if start_frame is None:
                    start_frame = scene[0].get_frames()
                start_sec = getattr(scene[0], "seconds", None)
                if start_sec is None:
                    start_sec = scene[0].get_seconds()
                event_id = f"EVT-CUT-{camera_id}-{i+1:04d}"

                # Extract frame preview for the cut
                thumb_path = self._extract_and_save_thumb(video_path, start_frame, camera_id, event_id)

                event = VisualEvent(
                    event_id=event_id,
                    camera_id=camera_id,
                    camera_name=camera_name,
                    source_filename=source_filename,
                    frame_idx=start_frame,
                    source_timestamp_sec=round(start_sec, 4),
                    event_type="scene_cut",
                    score=0.85,
                    thumbnail_path=thumb_path,
                    review_status="UNREVIEWED",
                    notes="PySceneDetect Content cut detected"
                )
                events.append(event)
        except Exception as ex:
            # PySceneDetect fallback: continue gracefully
            pass

        return events

    def _detect_motion_spikes(
        self,
        video_path: Path,
        camera_id: str,
        camera_name: str,
        source_filename: str,
        sample_fps: float = 4.0,
        sensitivity: float = 0.25,
        existing_timestamps: Optional[List[float]] = None
    ) -> List[VisualEvent]:
        """Detect salient frame difference spikes using downsampled frame diffs."""
        events: List[VisualEvent] = []
        existing_ts = existing_timestamps or []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return events

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        step = max(1, int(round(fps / sample_fps)))
        frame_idx = 0
        prev_gray: Optional[np.ndarray] = None
        diffs: List[Tuple[int, float, float]] = []  # (frame_idx, timestamp_sec, diff_val)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts_sec = round(pos_msec / 1000.0 if pos_msec > 0 else frame_idx / fps, 4)

                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    # Normalized absolute difference
                    diff_val = float(np.mean(cv2.absdiff(gray, prev_gray)))
                    diffs.append((frame_idx, ts_sec, diff_val))

                prev_gray = gray

            frame_idx += 1

        cap.release()

        if not diffs:
            return events

        values = [d[2] for d in diffs]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        threshold = mean_val + (1.5 + (1.0 - sensitivity) * 2.0) * std_val

        # Find local peaks exceeding threshold
        for i in range(1, len(diffs) - 1):
            f_idx, ts_sec, val = diffs[i]
            if val > threshold and val > diffs[i - 1][2] and val >= diffs[i + 1][2]:
                # Ensure it's not within 1.0s of an existing scene cut
                if any(abs(ts_sec - ex_ts) < 1.0 for ex_ts in existing_ts):
                    continue

                event_id = f"EVT-SPIKE-{camera_id}-{len(events)+1:04d}"
                thumb_path = self._extract_and_save_thumb(video_path, f_idx, camera_id, event_id)

                # Normalize score to [0.5, 0.95]
                salience = min(0.95, max(0.5, float((val - mean_val) / (std_val * 4.0 + 1e-6))))

                event = VisualEvent(
                    event_id=event_id,
                    camera_id=camera_id,
                    camera_name=camera_name,
                    source_filename=source_filename,
                    frame_idx=f_idx,
                    source_timestamp_sec=ts_sec,
                    event_type="visual_spike",
                    score=round(salience, 3),
                    thumbnail_path=thumb_path,
                    review_status="UNREVIEWED",
                    notes=f"Visual difference spike: delta={val:.2f}"
                )
                events.append(event)
                existing_ts.append(ts_sec)

        return events

    def _extract_and_save_thumb(
        self,
        video_path: Path,
        frame_idx: int,
        camera_id: str,
        event_id: str
    ) -> Optional[str]:
        """Extract frame preview and save to event cache."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        h, w = frame.shape[:2]
        target_w = 320
        target_h = max(1, int(round(h * (target_w / w))))
        thumb = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        filename = f"{event_id}.jpg"
        dest = self.cache_dir / filename
        cv2.imwrite(str(dest), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return str(dest.resolve())
