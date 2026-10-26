"""Synthetic forensic video generator for ChronoTrace AI testing and demonstration.

Generates multi-camera video datasets with known ground-truth clock offsets,
frame rates (15, 24, 30 fps), clock drift rates, dropped frames, and visual events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticGroundTruth:
    """Ground truth configuration for a synthetic camera video."""

    camera_id: str
    camera_name: str
    file_path: str
    nominal_fps: float
    true_offset_sec: float  # T_global = offset + rate * t_local
    true_rate_drift: float  # rate multiplier
    duration_sec: float
    global_event_times_sec: List[float]


class SyntheticVideoGenerator:
    """Generates synthetic test videos with controlled visual events and temporal offsets."""

    def __init__(self, output_dir: str | Path = "sample_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_forensic_hit_and_run_dataset(self) -> Dict[str, SyntheticGroundTruth]:
        """Generate a complete 3-camera forensic test dataset with known ground truth."""

        # Shared global events (True Universal Time in seconds):
        # Event 1: Vehicle entrance at T = 3.0s
        # Event 2: Hit-and-run impact flash at T = 6.0s
        # Event 3: Vehicle fleeing corner at T = 8.5s
        global_events = [3.0, 6.0, 8.5]
        dataset: Dict[str, SyntheticGroundTruth] = {}

        # Camera 1: Reference Traffic Pole (Ground Truth Clock)
        # offset = 0.0s, rate = 1.0, 30 fps, duration = 12s
        p1 = self.output_dir / "cam1_traffic_pole_reference.mp4"
        self.generate_camera_video(
            file_path=p1,
            camera_name="Camera 1 - Traffic Pole (Ref)",
            duration_sec=12.0,
            fps=30.0,
            true_offset_sec=0.0,
            true_rate_drift=1.0,
            global_event_times_sec=global_events,
            bg_color=(35, 40, 45),
            obj_color=(0, 0, 230)  # Bright Red
        )
        dataset["cam_1"] = SyntheticGroundTruth(
            camera_id="cam_1",
            camera_name="Camera 1 - Traffic Pole (Ref)",
            file_path=str(p1.resolve()),
            nominal_fps=30.0,
            true_offset_sec=0.0,
            true_rate_drift=1.0,
            duration_sec=12.0,
            global_event_times_sec=global_events
        )

        # Camera 2: Dashcam Vehicle (+3.5s clock offset, 24 fps, duration = 14s)
        # T_global = 3.5 + 1.0 * t_local -> t_local = (T_global - 3.5) / 1.0
        p2 = self.output_dir / "cam2_cruiser_dashcam.mp4"
        self.generate_camera_video(
            file_path=p2,
            camera_name="Camera 2 - Dashcam Patrol",
            duration_sec=14.0,
            fps=24.0,
            true_offset_sec=3.5,
            true_rate_drift=1.0,
            global_event_times_sec=global_events,
            bg_color=(40, 35, 30),
            obj_color=(0, 0, 230)
        )
        dataset["cam_2"] = SyntheticGroundTruth(
            camera_id="cam_2",
            camera_name="Camera 2 - Dashcam Patrol",
            file_path=str(p2.resolve()),
            nominal_fps=24.0,
            true_offset_sec=3.5,
            true_rate_drift=1.0,
            duration_sec=14.0,
            global_event_times_sec=global_events
        )

        # Camera 3: Storefront CCTV (-2.0s clock offset, slight clock drift 1.002, 15 fps, duration = 15s)
        # T_global = -2.0 + 1.002 * t_local
        p3 = self.output_dir / "cam3_storefront_cctv.mp4"
        self.generate_camera_video(
            file_path=p3,
            camera_name="Camera 3 - Storefront Security",
            duration_sec=15.0,
            fps=15.0,
            true_offset_sec=-2.0,
            true_rate_drift=1.002,
            global_event_times_sec=global_events,
            bg_color=(30, 45, 35),
            obj_color=(0, 0, 230)
        )
        dataset["cam_3"] = SyntheticGroundTruth(
            camera_id="cam_3",
            camera_name="Camera 3 - Storefront Security",
            file_path=str(p3.resolve()),
            nominal_fps=15.0,
            true_offset_sec=-2.0,
            true_rate_drift=1.002,
            duration_sec=15.0,
            global_event_times_sec=global_events
        )

        return dataset

    def generate_camera_video(
        self,
        file_path: str | Path,
        camera_name: str,
        duration_sec: float,
        fps: float,
        true_offset_sec: float,
        true_rate_drift: float,
        global_event_times_sec: List[float],
        bg_color: Tuple[int, int, int] = (40, 40, 40),
        obj_color: Tuple[int, int, int] = (0, 0, 220),
        width: int = 640,
        height: int = 360
    ) -> Path:
        """Render a synthetic video with burned timestamps and visual impact events."""
        out = Path(file_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
        if not writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))

        total_frames = int(round(duration_sec * fps))

        for frame_idx in range(total_frames):
            # Local camera timestamp
            t_local = frame_idx / fps
            # Corresponding global reference time
            t_global = true_offset_sec + true_rate_drift * t_local

            # Base frame background
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

            # Check if any global event is currently happening (window of 0.5s around event)
            is_event_active = False
            active_event_idx = -1
            for idx, ev_time in enumerate(global_event_times_sec):
                if 0 <= (t_global - ev_time) <= 0.6:
                    is_event_active = True
                    active_event_idx = idx
                    break

            if is_event_active:
                # Visual event rendering: Bright flash and salient colored vehicle marker
                if active_event_idx == 1:
                    # Flash impact event
                    overlay = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
                    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
                    # Impact starburst
                    cv2.circle(frame, (width // 2, height // 2), 65, (0, 255, 255), -1)
                    cv2.putText(frame, "IMPACT EVENT", (width // 2 - 90, height // 2 + 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                else:
                    # Vehicle passing
                    x_pos = int((width - 150) * min(1.0, max(0.0, (t_global - global_event_times_sec[active_event_idx]) / 0.6)))
                    cv2.rectangle(frame, (x_pos, height // 2 - 40), (x_pos + 120, height // 2 + 40), obj_color, -1)
                    cv2.putText(frame, f"TARGET CAR (EVT {active_event_idx+1})", (x_pos + 5, height // 2 + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Draw HUD forensic metadata
            # Grid lines
            cv2.line(frame, (0, height - 60), (width, height - 60), (70, 70, 70), 1)
            cv2.putText(frame, camera_name, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
            cv2.putText(frame, f"CAM LOCAL TIME: {t_local:06.3f}s | FRAME: {frame_idx:05d} | {fps:.1f} FPS",
                        (20, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"TRUE GLOBAL TIME: {t_global:06.3f}s (GT REF)",
                        (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 255), 1)

            writer.write(frame)

        writer.release()
        return out

    def generate_corrupt_video(self, filename: str = "corrupt_stream.mp4") -> Path:
        """Create an intentionally corrupted 0-byte video file."""
        out = self.output_dir / filename
        out.write_bytes(b"")
        return out

    def generate_garbage_header_video(self, filename: str = "invalid_header.mp4") -> Path:
        """Create an invalid header pseudo-video file."""
        out = self.output_dir / filename
        out.write_bytes(b"NON_VIDEO_RANDOM_BYTES_CORRUPTED_STREAM_42")
        return out
