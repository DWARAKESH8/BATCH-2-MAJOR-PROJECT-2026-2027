"""Dense optical flow as an alternative source of crowd velocity.

Why the tracker is not enough overhead
--------------------------------------
Every movement-quality feature this system computes -- flow disorder, the
counter-flow index, Helbing crowd pressure, the oscillation index -- needs a
velocity per person. Those come from the tracker, which needs to associate the
same individual across frames.

That association degrades badly in exactly the deployment case. From a drone a
head is 10-25 pixels of nearly featureless blob; a hundred of them look alike,
they pass within a head's width of each other, and the camera itself is moving.
Nearest-centroid association under those conditions produces ID switches, and an
ID switch fabricates a velocity pointing at another person entirely -- which
registers as directional disorder, which the classifier reads as counter-flow.
The failure is not a slightly noisier number; it is a confident wrong answer.

Dense optical flow sidesteps association completely. Farneback estimates motion
per pixel from image gradients, so a velocity can be sampled at any point
without knowing who is who. It also keeps working when there are no individual
detections at all, which matters because a density-map counter (CSRNet class) --
the right estimator for very dense overhead crowds -- returns a count and a
field, never a list of people.

What it costs: flow measures apparent image motion, so a moving camera adds its
own motion to every vector. `estimate_camera_motion` removes the dominant global
component, which is the right correction for a drone holding station but not for
one translating quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FlowConfig:
    downscale: float = 0.5          # flow at half resolution is plenty and 4x faster
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 21
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    # Radius, in pixels of the original frame, over which to average flow when
    # sampling a person's velocity. Roughly one head.
    sample_radius_px: int = 6
    # Remove the dominant global motion, which for a drone is the platform
    # itself rather than the crowd.
    compensate_camera: bool = True
    # Flow below this magnitude (px/frame) is treated as stationary; it is
    # dominated by sensor noise and would otherwise fabricate disorder.
    min_magnitude_px: float = 0.35


class OpticalFlowField:
    """Dense Farneback flow between consecutive frames, sampled at points."""

    def __init__(self, config: Optional[FlowConfig] = None):
        self.config = config or FlowConfig()
        self._prev: Optional[np.ndarray] = None
        self.flow: Optional[np.ndarray] = None
        self.camera_motion: Tuple[float, float] = (0.0, 0.0)
        self.ready = False

    def _grey(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = self.config.downscale
        if s != 1.0:
            g = cv2.resize(g, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return g

    def update(self, frame: np.ndarray) -> bool:
        """Compute flow against the previous frame. False on the first call."""
        c = self.config
        grey = self._grey(frame)
        if self._prev is None or self._prev.shape != grey.shape:
            self._prev, self.ready = grey, False
            return False

        self.flow = cv2.calcOpticalFlowFarneback(
            self._prev, grey, None, c.pyr_scale, c.levels, c.winsize,
            c.iterations, c.poly_n, c.poly_sigma, 0)
        self._prev, self.ready = grey, True

        if c.compensate_camera and self.flow is not None:
            # The median is the right estimator of global motion here: a crowd
            # moving coherently would drag the mean with it, but it rarely
            # occupies more than half the frame, so the median tracks the
            # background -- that is, the platform.
            self.camera_motion = (float(np.median(self.flow[..., 0])),
                                  float(np.median(self.flow[..., 1])))
        else:
            self.camera_motion = (0.0, 0.0)
        return True

    # ------------------------------------------------------------------ #
    def sample(self, points: Sequence[Tuple[float, float]]) -> np.ndarray:
        """Velocity in original-frame pixels per frame, at each point."""
        n = len(points)
        if not self.ready or self.flow is None or n == 0:
            return np.zeros((n, 2), dtype=np.float64)

        c = self.config
        s = c.downscale
        h, w = self.flow.shape[:2]
        r = max(1, int(c.sample_radius_px * s))
        cam_x, cam_y = self.camera_motion

        out = np.zeros((n, 2), dtype=np.float64)
        for i, (px, py) in enumerate(points):
            x, y = int(px * s), int(py * s)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            patch = self.flow[y0:y1, x0:x1]
            vx = float(np.median(patch[..., 0])) - cam_x
            vy = float(np.median(patch[..., 1])) - cam_y
            # Back to original-frame pixels per frame.
            vx, vy = vx / s, vy / s
            if np.hypot(vx, vy) < c.min_magnitude_px:
                vx = vy = 0.0
            out[i] = (vx, vy)
        return out

    def velocity_vectors(self, tracks: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[float, float]]:
        """Drop-in replacement for `CentroidTracker.velocity_vectors`."""
        ids = list(tracks.keys())
        if not ids:
            return {}
        vels = self.sample([tracks[i] for i in ids])
        return {i: (float(v[0]), float(v[1])) for i, v in zip(ids, vels)}

    def net_displacements(self, tracks: Dict[int, Tuple[int, int]],
                          scale: float = 6.0) -> Dict[int, Tuple[float, float]]:
        """Approximate net travel, for the counter-flow index.

        The tracker measures this over a track's real history. Flow has no
        history, so this extrapolates the instantaneous vector over roughly the
        same interval. It is a weaker signal and is documented as such: it
        cannot distinguish a person who has genuinely walked backwards from one
        momentarily jostled backwards.
        """
        v = self.velocity_vectors(tracks)
        return {i: (vx * scale, vy * scale) for i, (vx, vy) in v.items()}

    # ------------------------------------------------------------------ #
    def field_summary(self) -> Dict[str, Any]:
        if not self.ready or self.flow is None:
            return {"ready": False}
        mag = np.linalg.norm(self.flow, axis=-1)
        return {
            "ready": True,
            "mean_magnitude_px": round(float(mag.mean()), 3),
            "p95_magnitude_px": round(float(np.percentile(mag, 95)), 3),
            "camera_motion_px": [round(v, 3) for v in self.camera_motion],
        }

    def render(self, frame: np.ndarray, step: int = 22) -> np.ndarray:
        """Flow field as arrows, for the dashboard."""
        out = frame.copy()
        if not self.ready or self.flow is None:
            return out
        s = self.config.downscale
        h, w = self.flow.shape[:2]
        cam = np.array(self.camera_motion)
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                v = self.flow[y, x] - cam
                if np.hypot(*v) < self.config.min_magnitude_px * s:
                    continue
                p0 = (int(x / s), int(y / s))
                p1 = (int((x + v[0] * 4) / s), int((y + v[1] * 4) / s))
                cv2.arrowedLine(out, p0, p1, (0, 200, 255), 1, cv2.LINE_AA, tipLength=0.35)
        return out
