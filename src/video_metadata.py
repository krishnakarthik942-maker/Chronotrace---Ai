"""Video metadata inspection and timestamp extraction for ChronoTrace AI.

Uses FFmpeg (bundled via imageio-ffmpeg or system) and OpenCV to extract
duration, nominal and calculated frame rates, frame counts, resolution,
codecs, and embedded container timestamps (creation_time/quicktime tags).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import imageio_ffmpeg


@dataclass
class VideoMetadata:
    """Forensic metadata profile for an inspected video file."""

    video_id: str
    camera_name: str
    file_path: str
    file_name: str
    file_size_bytes: int
    sha256_hash: str
    duration_sec: float = 0.0
    nominal_fps: float = 0.0
    total_frames: int = 0
    width: int = 0
    height: int = 0
    codec_name: str = "unknown"
    container_format: str = "unknown"
    creation_time_raw: Optional[str] = None
    creation_time_utc: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None
    extra_tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoMetadataExtractor:
    """Extracts forensic metadata from video containers using FFmpeg and OpenCV."""

    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_exe = ffmpeg_path or self._resolve_ffmpeg()

    @staticmethod
    def _resolve_ffmpeg() -> str:
        """Find FFmpeg executable from system PATH or imageio-ffmpeg bundle."""
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def extract(self, file_path: str | Path, camera_name: str, video_id: str,
                sha256_hash: str = "") -> VideoMetadata:
        """Inspect video file and return structured VideoMetadata."""
        path = Path(file_path)

        if not path.is_file():
            return VideoMetadata(
                video_id=video_id,
                camera_name=camera_name,
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=0,
                sha256_hash=sha256_hash,
                is_valid=False,
                error_message=f"File does not exist: {file_path}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            return VideoMetadata(
                video_id=video_id,
                camera_name=camera_name,
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=0,
                sha256_hash=sha256_hash,
                is_valid=False,
                error_message="Video file is empty (0 bytes)"
            )

        # Inspect using OpenCV
        cv_data, cv_valid, cv_err = self._inspect_opencv(path)
        if not cv_valid:
            return VideoMetadata(
                video_id=video_id,
                camera_name=camera_name,
                file_path=str(path),
                file_name=path.name,
                file_size_bytes=file_size,
                sha256_hash=sha256_hash,
                is_valid=False,
                error_message=f"OpenCV could not open stream: {cv_err}"
            )

        # Inspect container metadata via FFmpeg
        ff_data = self._inspect_ffmpeg(path)

        # Reconcile values
        nominal_fps = ff_data.get("fps") or cv_data.get("fps", 0.0)
        total_frames = cv_data.get("total_frames", 0)
        duration_sec = ff_data.get("duration") or cv_data.get("duration", 0.0)

        # If duration is 0 but we have frames and fps
        if duration_sec <= 0.0 and nominal_fps > 0 and total_frames > 0:
            duration_sec = total_frames / nominal_fps

        width = cv_data.get("width") or ff_data.get("width", 0)
        height = cv_data.get("height") or ff_data.get("height", 0)
        codec_name = ff_data.get("codec") or cv_data.get("fourcc", "unknown")
        container_format = path.suffix.lstrip(".").lower()

        creation_raw = ff_data.get("creation_time")
        creation_utc = self._parse_iso_datetime(creation_raw)

        return VideoMetadata(
            video_id=video_id,
            camera_name=camera_name,
            file_path=str(path.resolve()),
            file_name=path.name,
            file_size_bytes=file_size,
            sha256_hash=sha256_hash,
            duration_sec=round(float(duration_sec), 4),
            nominal_fps=round(float(nominal_fps), 4),
            total_frames=int(total_frames),
            width=int(width),
            height=int(height),
            codec_name=str(codec_name),
            container_format=str(container_format),
            creation_time_raw=creation_raw,
            creation_time_utc=creation_utc,
            is_valid=True,
            error_message=None,
            extra_tags=ff_data.get("tags", {})
        )

    def _inspect_opencv(self, path: Path) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """Extract frame parameters via cv2.VideoCapture."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {}, False, "Failed to initialize OpenCV VideoCapture"

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Read the first frame to confirm decodability
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {}, False, "Stream opened but failed to decode any frame (corrupt or unreadable stream)"

        # Safe duration estimate
        duration = 0.0
        if fps > 0 and total_frames > 0:
            duration = total_frames / fps

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "fourcc": fourcc.strip(),
            "duration": duration
        }, True, None

    def _inspect_ffmpeg(self, path: Path) -> Dict[str, Any]:
        """Parse stderr output of ffmpeg -i <path> to extract container tags and codec details."""
        data: Dict[str, Any] = {"tags": {}}
        if not self.ffmpeg_exe:
            return data

        try:
            cmd = [self.ffmpeg_exe, "-hide_banner", "-i", str(path)]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10
            )
            raw = proc.stderr

            # Extract Duration: 00:00:15.50, start: 0.000000, bitrate: 1234 kb/s
            dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", raw)
            if dur_match:
                h, m, s = dur_match.groups()
                data["duration"] = int(h) * 3600 + int(m) * 60 + float(s)

            # Extract Video Stream: Video: h264 (...), 1920x1080, 29.97 fps, 29.97 tbr
            video_match = re.search(r"Stream #\d+:\d+.*Video:\s*([^\s,]+).*, (\d+)x(\d+).*?,\s*([\d\.]+)\s*(?:fps|tbr)", raw)
            if video_match:
                data["codec"] = video_match.group(1)
                data["width"] = int(video_match.group(2))
                data["height"] = int(video_match.group(3))
                data["fps"] = float(video_match.group(4))

            # Extract creation_time tag
            creation_match = re.search(r"creation_time\s*:\s*([^\r\n]+)", raw, re.IGNORECASE)
            if creation_match:
                data["creation_time"] = creation_match.group(1).strip()

            # Extract additional tags
            for tag_match in re.finditer(r"^\s+([a-zA-Z_]+)\s*:\s*(.+)$", raw, re.MULTILINE):
                tag_name = tag_match.group(1).lower()
                tag_val = tag_match.group(2).strip()
                if tag_name not in ("duration", "stream", "handler_name"):
                    data["tags"][tag_name] = tag_val

        except Exception:
            pass

        return data

    @staticmethod
    def _parse_iso_datetime(dt_str: Optional[str]) -> Optional[str]:
        """Convert arbitrary container datetime strings to standard UTC ISO8601."""
        if not dt_str:
            return None
        dt_clean = dt_str.strip().replace("Z", "+00:00")
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(dt_clean, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
        return dt_str
