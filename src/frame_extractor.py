"""Frame extraction and thumbnail sampling module for ChronoTrace AI.

Extracts frames at regular intervals or specific timestamps, preserves
frame indices, presentation timestamps (PTS), and provides cached frame previews.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SampledFrame:
    """Represents an extracted frame observation."""

    camera_id: str
    frame_idx: int
    source_timestamp_sec: float
    image_bgr: Optional[np.ndarray] = None
    thumbnail_path: Optional[str] = None

    def release_image(self) -> None:
        """Free memory of large raw image buffer if cached to disk."""
        self.image_bgr = None


class FrameExtractor:
    """Extracts frames from video files with robust timestamp handling."""

    def __init__(self, cache_dir: str | Path = "data/cache/frames"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def sample_frames(
        self,
        video_path: str | Path,
        camera_id: str,
        interval_sec: float = 1.0,
        max_frames: int = 1000,
        save_thumbnails: bool = True,
        thumbnail_width: int = 320
    ) -> List[SampledFrame]:
        """Sample frames uniformly at the specified time interval."""
        sampled: List[SampledFrame] = []
        path = Path(video_path)
        if not path.is_file():
            return sampled

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return sampled

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0:
            fps = 30.0

        frame_step = max(1, int(round(fps * interval_sec)))
        frame_idx = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_step == 0:
                # Get presentation timestamp in msec if supported, otherwise frame_idx / fps
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_msec > 0.0:
                    ts_sec = round(pos_msec / 1000.0, 4)
                else:
                    ts_sec = round(frame_idx / fps, 4)

                thumb_path = None
                if save_thumbnails:
                    thumb_path = self._save_thumbnail(
                        frame, camera_id, frame_idx, thumbnail_width
                    )

                obs = SampledFrame(
                    camera_id=camera_id,
                    frame_idx=frame_idx,
                    source_timestamp_sec=ts_sec,
                    image_bgr=frame.copy(),
                    thumbnail_path=thumb_path
                )
                sampled.append(obs)
                extracted_count += 1
                if extracted_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        return sampled

    def extract_frame_at_timestamp(
        self,
        video_path: str | Path,
        timestamp_sec: float,
        nominal_fps: float = 30.0
    ) -> Optional[np.ndarray]:
        """Extract a single frame closest to the specified source timestamp in seconds."""
        path = Path(video_path)
        if not path.is_file() or timestamp_sec < 0:
            return None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0:
            fps = nominal_fps if nominal_fps > 0 else 30.0

        target_msec = timestamp_sec * 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
        ret, frame = cap.read()

        # Fallback to frame index seek if POS_MSEC is unreliable
        if not ret or frame is None:
            target_frame_idx = int(round(timestamp_sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()

        cap.release()
        return frame if ret and frame is not None else None

    def extract_frame_at_index(
        self,
        video_path: str | Path,
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """Extract a single frame by exact zero-based frame index."""
        path = Path(video_path)
        if not path.is_file() or frame_idx < 0:
            return None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret and frame is not None else None

    def _save_thumbnail(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        frame_idx: int,
        target_width: int = 320
    ) -> str:
        """Resize and save thumbnail JPEG to cache directory."""
        h, w = frame_bgr.shape[:2]
        if w > 0:
            target_h = max(1, int(round(h * (target_width / w))))
            thumb = cv2.resize(frame_bgr, (target_width, target_h), interpolation=cv2.INTER_AREA)
        else:
            thumb = frame_bgr

        filename = f"thumb_{camera_id}_f{frame_idx:06d}.jpg"
        dest = self.cache_dir / filename
        cv2.imwrite(str(dest), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return str(dest.resolve())
