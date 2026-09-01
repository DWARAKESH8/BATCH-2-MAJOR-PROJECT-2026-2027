"""Ground-plane calibration.

Turning pixel positions into metres is the difference between a number that
means something ("4.6 persons per square metre, above Fruin's crush band") and
a number that means nothing ("4.6 blobs per screen area"). This module
implements a planar homography from the image plane to the ground plane and a
documented uniform-scale fallback for when no calibration is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import CalibrationConfig


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform for a 4-point planar homography.

    Implemented with numpy rather than cv2.findHomography so that calibration
    stays testable without an OpenCV install.
    """
    if src.shape[0] < 4:
        raise ValueError("At least 4 point correspondences are required")

    rows: List[List[float]] = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    a = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


@dataclass
class GroundPlane:
    """Result of a calibration: image -> ground-plane metres."""

    homography: Optional[np.ndarray]
    metres_per_pixel: float
    calibrated: bool
    frame_shape: Optional[Tuple[int, int]] = None
    reprojection_error_m: float = 0.0

    def to_world(self, points: Sequence[Tuple[float, float]]) -> np.ndarray:
        """Project image points (pixels) to ground-plane points (metres)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float64)

        if self.calibrated and self.homography is not None:
            ones = np.ones((pts.shape[0], 1), dtype=np.float64)
            hom = np.hstack([pts, ones]) @ self.homography.T
            w = hom[:, 2:3]
            # Guard against points projected at/behind the horizon.
            w = np.where(np.abs(w) < 1e-9, np.sign(w) * 1e-9 + 1e-9, w)
            return hom[:, :2] / w

        return pts * self.metres_per_pixel

    def scalar_to_world(self, pixel_length: float) -> float:
        """Approximate conversion of a pixel length to metres.

        Under a homography the scale varies across the image, so this returns
        the scale at the image centre and is only used for coarse reporting.
        """
        if not self.calibrated or self.homography is None or self.frame_shape is None:
            return pixel_length * self.metres_per_pixel
        h, w = self.frame_shape
        centre = np.array([[w / 2.0, h / 2.0], [w / 2.0 + pixel_length, h / 2.0]])
        world = self.to_world(centre)
        return float(np.linalg.norm(world[1] - world[0]))

    def describe(self) -> str:
        if self.calibrated:
            return (
                f"Homography calibrated (reprojection error "
                f"{self.reprojection_error_m:.3f} m)"
            )
        return (
            f"Uncalibrated: uniform scale {self.metres_per_pixel:.5f} m/px "
            f"derived from declared monitored area"
        )


class GroundPlaneCalibrator:
    """Builds a `GroundPlane` from a `CalibrationConfig` and a frame size."""

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()

    def build(self, frame_shape: Tuple[int, ...]) -> GroundPlane:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        fallback_mpp = float(np.sqrt(max(1e-6, self.config.fallback_area_m2) / max(1.0, h * w)))

        if not self.config.enabled:
            return GroundPlane(None, fallback_mpp, False, (h, w))

        img = np.asarray(self.config.image_points, dtype=np.float64)
        wrl = np.asarray(self.config.world_points, dtype=np.float64)
        if img.shape[0] < 4 or wrl.shape[0] < 4 or img.shape[0] != wrl.shape[0]:
            return GroundPlane(None, fallback_mpp, False, (h, w))

        try:
            homography = _solve_homography(img, wrl)
        except Exception:
            return GroundPlane(None, fallback_mpp, False, (h, w))

        plane = GroundPlane(homography, fallback_mpp, True, (h, w))
        projected = plane.to_world(img)
        error = float(np.mean(np.linalg.norm(projected - wrl, axis=1)))
        plane.reprojection_error_m = error

        # A wildly wrong calibration is more dangerous than none at all.
        if not np.isfinite(error) or error > 2.0:
            return GroundPlane(None, fallback_mpp, False, (h, w))
        return plane


def default_calibration_from_area(area_m2: float, frame_shape: Tuple[int, ...]) -> GroundPlane:
    """Convenience helper used when the operator only declares an area."""
    return GroundPlaneCalibrator(CalibrationConfig(enabled=False, fallback_area_m2=area_m2)).build(frame_shape)
