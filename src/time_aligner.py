"""Clock correction and synchronization engine for ChronoTrace AI.

Implements robust linear regression, RANSAC outlier rejection, drift estimation,
and comprehensive uncertainty quantification for multi-camera video timelines.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CameraAlignmentResult:
    """Forensic alignment parameters for a single camera stream."""

    camera_id: str
    camera_name: str
    is_reference: bool = False
    offset_sec: float = 0.0
    rate_drift: float = 1.0  # Drift multiplier (nominal is 1.000000)
    status: str = "UNALIGNED"  # "SYNCHRONIZED", "APPROXIMATE", "CONFLICTING", "UNALIGNED", "MANUAL_OVERRIDE"
    uncertainty_sec: float = float("inf")  # 95% confidence error margin in seconds
    rmse_residual_sec: float = 0.0
    drift_std_error: float = 0.0
    supporting_match_ids: List[str] = field(default_factory=list)
    rejected_match_ids: List[str] = field(default_factory=list)
    method_description: str = ""
    forensic_notes: str = ""
    derivation_label: str = "DERIVED_RESULT"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if math.isinf(d["uncertainty_sec"]):
            d["uncertainty_sec"] = None
        return d

    def to_corrected_time(self, source_time_sec: float) -> float:
        """Map source timestamp to corrected global reference time."""
        return round(self.offset_sec + self.rate_drift * source_time_sec, 4)

    def to_source_time(self, corrected_time_sec: float) -> float:
        """Map global reference time back to camera source timestamp."""
        if self.rate_drift == 0:
            return 0.0
        return round((corrected_time_sec - self.offset_sec) / self.rate_drift, 4)


@dataclass
class TimeAnchorPoint:
    """A correspondence point between camera source time and reference time."""

    match_id: str
    camera_id: str
    source_time_sec: float
    reference_time_sec: float
    weight: float = 1.0
    is_confirmed: bool = True


class TimeAligner:
    """Estimates time offsets, clock-rate drift, and uncertainty across camera networks."""

    def __init__(
        self,
        reference_camera_id: str,
        ransac_residual_threshold_sec: float = 0.35,
        ransac_max_iterations: int = 200,
        min_inlier_ratio: float = 0.5
    ):
        self.reference_camera_id = reference_camera_id
        self.residual_threshold = ransac_residual_threshold_sec
        self.max_iterations = ransac_max_iterations
        self.min_inlier_ratio = min_inlier_ratio

    def align_camera_from_anchors(
        self,
        camera_id: str,
        camera_name: str,
        anchors: List[TimeAnchorPoint],
        nominal_fps: float = 30.0,
        manual_offset: Optional[float] = None,
        manual_rate: Optional[float] = None
    ) -> CameraAlignmentResult:
        """Compute alignment model for a camera given reference anchor points."""

        # 1. Reference camera base case
        if camera_id == self.reference_camera_id:
            return CameraAlignmentResult(
                camera_id=camera_id,
                camera_name=camera_name,
                is_reference=True,
                offset_sec=0.0,
                rate_drift=1.0,
                status="SYNCHRONIZED",
                uncertainty_sec=0.0,
                rmse_residual_sec=0.0,
                drift_std_error=0.0,
                method_description="Designated ground-truth reference timeline",
                forensic_notes="All camera timelines are calibrated relative to this reference stream."
            )

        # 2. Manual Investigator Override
        if manual_offset is not None:
            rate = manual_rate if manual_rate is not None else 1.0
            return CameraAlignmentResult(
                camera_id=camera_id,
                camera_name=camera_name,
                is_reference=False,
                offset_sec=round(float(manual_offset), 4),
                rate_drift=round(float(rate), 6),
                status="MANUAL_OVERRIDE",
                uncertainty_sec=round(1.0 / max(1.0, nominal_fps), 4),
                rmse_residual_sec=0.0,
                drift_std_error=0.0,
                method_description="Manual investigator calibration override",
                forensic_notes="Offset and drift explicitly specified by forensic examiner."
            )

        # Filter confirmed anchors for this camera
        cam_anchors = [a for a in anchors if a.camera_id == camera_id and a.is_confirmed]

        # 3. Insufficient anchors (N = 0)
        if len(cam_anchors) == 0:
            return CameraAlignmentResult(
                camera_id=camera_id,
                camera_name=camera_name,
                is_reference=False,
                offset_sec=0.0,
                rate_drift=1.0,
                status="UNALIGNED",
                uncertainty_sec=float("inf"),
                method_description="Insufficient synchronization anchors",
                forensic_notes="No confirmed visual or manual anchors exist. Investigator review required."
            )

        # 4. Single anchor (N = 1) -> Constant offset, nominal rate = 1.0
        if len(cam_anchors) == 1:
            a0 = cam_anchors[0]
            offset = round(a0.reference_time_sec - a0.source_time_sec, 4)
            # Uncertainty based on frame sampling interval and detector resolution
            frame_period = 1.0 / max(1.0, nominal_fps)
            unc = round(max(0.1, frame_period * 2.0), 4)

            return CameraAlignmentResult(
                camera_id=camera_id,
                camera_name=camera_name,
                is_reference=False,
                offset_sec=offset,
                rate_drift=1.0,
                status="APPROXIMATE",
                uncertainty_sec=unc,
                rmse_residual_sec=0.0,
                drift_std_error=0.0,
                supporting_match_ids=[a0.match_id],
                method_description="Single-anchor constant offset estimation (drift assumed 1.000000)",
                forensic_notes=(
                    f"Calibrated from single confirmed match '{a0.match_id}'. "
                    f"Clock drift cannot be mathematically modeled without at least 2 temporal anchors."
                )
            )

        # 5. Multiple anchors (N >= 2) -> Robust Linear Regression / RANSAC
        t_src = np.array([a.source_time_sec for a in cam_anchors], dtype=np.float64)
        t_ref = np.array([a.reference_time_sec for a in cam_anchors], dtype=np.float64)
        match_ids = [a.match_id for a in cam_anchors]

        # Run RANSAC to detect inliers vs outliers
        best_offset, best_rate, inlier_indices, outlier_indices = self._fit_ransac(t_src, t_ref)

        supporting_ids = [match_ids[i] for i in inlier_indices]
        rejected_ids = [match_ids[i] for i in outlier_indices]

        # Plausibility check for clock drift rate (physically, rate must be within 0.85 to 1.15)
        is_rate_physically_plausible = (0.85 <= best_rate <= 1.15)
        inlier_ratio = len(inlier_indices) / len(cam_anchors)

        if not is_rate_physically_plausible or inlier_ratio < self.min_inlier_ratio:
            # Fallback to median constant offset if rate is nonsensical or severe conflict
            median_offset = float(np.median(t_ref - t_src))
            residuals = np.abs((t_ref - t_src) - median_offset)
            rmse = float(np.sqrt(np.mean(residuals**2)))

            return CameraAlignmentResult(
                camera_id=camera_id,
                camera_name=camera_name,
                is_reference=False,
                offset_sec=round(median_offset, 4),
                rate_drift=1.0,
                status="CONFLICTING",
                uncertainty_sec=round(rmse * 2.5, 4),
                rmse_residual_sec=round(rmse, 4),
                drift_std_error=0.0,
                supporting_match_ids=supporting_ids,
                rejected_match_ids=rejected_ids,
                method_description="Conflicting anchors detected - fallback to robust median offset",
                forensic_notes=(
                    f"Anchor regression produced suspect drift rate ({best_rate:.4f}) or low inlier ratio "
                    f"({inlier_ratio:.1%}). Flagged as CONFLICTING. Human verification required."
                )
            )

        # Compute uncertainties on inliers
        inlier_src = t_src[inlier_indices]
        inlier_ref = t_ref[inlier_indices]
        residuals = inlier_ref - (best_offset + best_rate * inlier_src)
        rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else 0.0

        # Linear regression standard errors
        n_inliers = len(inlier_src)
        s_xx = float(np.sum((inlier_src - np.mean(inlier_src))**2))

        if n_inliers > 2 and s_xx > 1e-4:
            se_rate = float(rmse / np.sqrt(s_xx))
            se_offset = float(rmse * np.sqrt(1.0 / n_inliers + (np.mean(inlier_src)**2) / s_xx))
            uncertainty_95 = max(round(1.96 * se_offset, 4), round(1.0 / max(1.0, nominal_fps), 4))
        else:
            se_rate = 0.0
            se_offset = rmse
            uncertainty_95 = max(round(rmse * 2.0, 4), 0.1)

        status = "SYNCHRONIZED" if (rmse <= self.residual_threshold and len(rejected_ids) == 0) else "APPROXIMATE"
        if len(rejected_ids) > 0:
            status = "APPROXIMATE"

        return CameraAlignmentResult(
            camera_id=camera_id,
            camera_name=camera_name,
            is_reference=False,
            offset_sec=round(best_offset, 4),
            rate_drift=round(best_rate, 6),
            status=status,
            uncertainty_sec=uncertainty_95,
            rmse_residual_sec=round(rmse, 4),
            drift_std_error=round(se_rate, 6),
            supporting_match_ids=supporting_ids,
            rejected_match_ids=rejected_ids,
            method_description=(
                f"RANSAC Robust Linear Drift Model: T_ref = {best_offset:.4f} + {best_rate:.6f} * t_src "
                f"({len(supporting_ids)} inliers, {len(rejected_ids)} outliers rejected)"
            ),
            forensic_notes=(
                f"Model calibrated against {len(supporting_ids)} verified anchor points. "
                f"Inlier residual RMSE: {rmse*1000.0:.1f}ms. Drift std err: {se_rate*1e6:.1f} ppm."
            )
        )

    def _fit_ransac(
        self,
        t_src: np.ndarray,
        t_ref: np.ndarray
    ) -> Tuple[float, float, List[int], List[int]]:
        """RANSAC estimator for linear parameters: t_ref = offset + rate * t_src."""
        n = len(t_src)
        if n == 2:
            dt_src = t_src[1] - t_src[0]
            if abs(dt_src) < 1e-6:
                # Same source time: use average offset and nominal rate
                avg_offset = float(np.mean(t_ref - t_src))
                return avg_offset, 1.0, [0, 1], []
            rate = float((t_ref[1] - t_ref[0]) / dt_src)
            offset = float(t_ref[0] - rate * t_src[0])
            return offset, rate, [0, 1], []

        best_inliers: List[int] = []
        best_offset = float(np.mean(t_ref - t_src))
        best_rate = 1.0
        best_cost = float("inf")

        # Deterministic sample generation for reproducibility
        rng = np.random.RandomState(42)

        for _ in range(self.max_iterations):
            idx1, idx2 = rng.choice(n, size=2, replace=False)
            dt_s = t_src[idx2] - t_src[idx1]
            if abs(dt_s) < 0.2:  # Points too close temporally for stable drift estimation
                continue

            cand_rate = float((t_ref[idx2] - t_ref[idx1]) / dt_s)
            cand_offset = float(t_ref[idx1] - cand_rate * t_src[idx1])

            # Filter unphysical candidate models immediately
            if not (0.80 <= cand_rate <= 1.20):
                continue

            residuals = np.abs(t_ref - (cand_offset + cand_rate * t_src))
            inliers = [i for i in range(n) if residuals[i] <= self.residual_threshold]

            cost = float(np.sum(residuals[inliers])) + (n - len(inliers)) * self.residual_threshold * 2.0

            if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and cost < best_cost):
                best_inliers = inliers
                best_cost = cost
                best_offset = cand_offset
                best_rate = cand_rate

        # If RANSAC found a good inlier set, re-estimate using ordinary least squares on all inliers
        if len(best_inliers) >= 2:
            in_src = t_src[best_inliers]
            in_ref = t_ref[best_inliers]
            A = np.vstack([in_src, np.ones_like(in_src)]).T
            # Solve [rate, offset]
            try:
                rate_ols, offset_ols = np.linalg.lstsq(A, in_ref, rcond=None)[0]
                if 0.85 <= rate_ols <= 1.15:
                    best_rate = float(rate_ols)
                    best_offset = float(offset_ols)
            except Exception:
                pass

        outliers = [i for i in range(n) if i not in best_inliers]
        return best_offset, best_rate, best_inliers, outliers

    @staticmethod
    def evaluate_timing_error(
        estimated_offsets: Dict[str, float],
        ground_truth_offsets: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute forensic timing error metrics (median, 95th-percentile, max) against ground truth."""
        errors: List[float] = []
        for cam_id, est_off in estimated_offsets.items():
            if cam_id in ground_truth_offsets:
                gt_off = ground_truth_offsets[cam_id]
                errors.append(abs(est_off - gt_off))

        if not errors:
            return {"median_error_sec": 0.0, "p95_error_sec": 0.0, "max_error_sec": 0.0, "count": 0}

        arr = np.array(errors)
        return {
            "median_error_sec": round(float(np.median(arr)), 5),
            "p95_error_sec": round(float(np.percentile(arr, 95)), 5),
            "max_error_sec": round(float(np.max(arr)), 5),
            "count": len(errors)
        }
