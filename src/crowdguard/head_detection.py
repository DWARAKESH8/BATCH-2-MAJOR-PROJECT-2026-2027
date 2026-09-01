"""Top-down head detection, for drone and overhead camera views.

Why a separate detector is unavoidable
--------------------------------------
COCO's `person` class is overwhelmingly upright, side- or front-facing bodies
with visible torso and limbs. A camera looking straight down at a crowd sees
none of that: it sees compact, roughly circular head-and-shoulder blobs of
10-25 pixels. That is not a hard case for a person detector; it is outside its
distribution entirely. Measured on a synthetic overhead night crowd of 69
people, YOLOv8n detects **zero**, and lowering the confidence threshold to 0.03
does not change that -- the model is not producing weak detections, it is
producing none.

Which matters because overhead is the deployment view. Event security drones,
and most fixed cameras mounted high enough to see a whole concourse, look down.

Three strategies, in the order you should prefer them
----------------------------------------------------
1. **A trained head detector.** The correct answer. YOLOv8 fine-tuned on head
   boxes (CrowdHuman ships an `hbox` per person; SCUT-HEAD and CroHD/HT21 are
   purpose-built) or on VisDrone, which is drone-captured and includes night
   scenes. Drop a checkpoint in and this module uses it.

2. **A density-map counter** (CSRNet class) for very dense scenes where even
   heads overlap. Returns a count and a density field rather than individuals.

3. **Scale-space blob detection**, implemented here, which needs no training at
   all. It is a classical Laplacian-of-Gaussian detector tuned to head-sized
   circular blobs, and it works on overhead views today. It is a stopgap, not a
   research contribution: it will produce false positives on any round bright
   object, and it should be described that way rather than dressed up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HeadDetectorConfig:
    """Tuning for the untrained blob detector."""

    # Head size in pixels falls with camera altitude. At rally scale from a
    # drone a head can be 4 px across, so the sweep has to start low.
    min_radius_px: float = 4.0
    max_radius_px: float = 26.0
    scale_steps: int = 6
    # Response threshold as a percentile of the scale-space maximum. Higher is
    # stricter. 0.18 keeps recall high at the cost of some false positives,
    # which is the right trade for a safety system: an over-count is visible
    # and correctable, a missed dense pocket is not.
    # Fraction of the LOCAL peak response, not the global one. A single
    # specular highlight -- a phone screen, a bald patch, a stage light --
    # sets the global maximum far above every real head, and a global
    # threshold then rejects the entire crowd. That is precisely how this
    # detector returns zero on footage full of people.
    response_threshold: float = 0.12
    # Heads cannot be closer than roughly this fraction of their own radius.
    nms_distance_factor: float = 1.35
    polarity: str = "auto"          # bright | dark | auto
    # Aerial footage of a rally or festival routinely contains several
    # thousand heads. A cap of 1200 silently truncates exactly the scenes that
    # matter most, and the truncation looks like a plateau in the count rather
    # than an error.
    max_detections: int = 6000
    # Contrast-limited equalisation before detection. Night footage is
    # low-contrast by definition, and without this the response collapses.
    use_clahe: bool = True


class TopDownHeadDetector:
    """Detects heads from above without a trained model.

    Scale-space Laplacian-of-Gaussian: a blob of radius r produces its strongest
    normalised LoG response at sigma = r / sqrt(2). Sweeping sigma and taking
    local maxima across both image position and scale therefore finds circular
    structures of head size regardless of how far the camera is, which matters
    because a drone's altitude changes the head size in pixels continuously.
    """

    def __init__(self, config: Optional[HeadDetectorConfig] = None):
        self.config = config or HeadDetectorConfig()
        self.backend = "blob"
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ #
    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.config.use_clahe:
            gray = self._clahe.apply(gray)
        return gray.astype(np.float32) / 255.0

    def _scale_space(self, gray: np.ndarray, invert: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Return (max response over scales, radius at that maximum)."""
        c = self.config
        img = (1.0 - gray) if invert else gray
        radii = np.linspace(c.min_radius_px, c.max_radius_px, max(2, c.scale_steps))

        best = np.full(img.shape, -np.inf, dtype=np.float32)
        best_r = np.zeros(img.shape, dtype=np.float32)
        for r in radii:
            sigma = float(r) / np.sqrt(2.0)
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
            # Scale-normalised LoG. Without the sigma^2 factor the response
            # decays with scale and only the smallest blobs are ever found.
            log = (sigma ** 2) * cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
            response = -log            # bright blob on darker ground
            update = response > best
            best = np.where(update, response, best)
            best_r = np.where(update, np.float32(r), best_r)
        return best, best_r

    def _peaks(self, response: np.ndarray, radius_map: np.ndarray) -> List[Tuple[int, int, float, float]]:
        c = self.config
        if not np.isfinite(response).any():
            return []

        # LOCAL adaptive threshold. A global one fails on real footage in a way
        # that is easy to miss: a single bright highlight sets the maximum well
        # above any head, so `peak * fraction` lands above the whole crowd and
        # the detector reports zero people in a frame that is visibly packed.
        # Comparing each pixel against its own neighbourhood also absorbs the
        # illumination gradient every wide aerial shot has.
        blur = max(3, int(c.max_radius_px * 4) | 1)
        local_mean = cv2.blur(response, (blur, blur))
        local_sq = cv2.blur(response * response, (blur, blur))
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0.0))
        threshold = local_mean + local_std * (c.response_threshold * 8.0)

        # Local maxima via grey dilation: a pixel that equals the local maximum
        # of its neighbourhood is a peak.
        window = max(3, int(c.min_radius_px) | 1)
        dilated = cv2.dilate(response, np.ones((window, window), np.uint8))
        mask = (response >= dilated - 1e-6) & (response >= threshold) & (response > 0)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return []

        scores = response[ys, xs]
        order = np.argsort(scores)[::-1][: c.max_detections * 4]
        ys, xs, scores = ys[order], xs[order], scores[order]
        radii = radius_map[ys, xs]

        # Greedy non-maximum suppression by distance. Two heads cannot occupy
        # the same place; without this a single head yields a cluster of peaks.
        kept: List[Tuple[int, int, float, float]] = []
        taken = np.zeros((0, 2), dtype=np.float32)
        for x, y, s, r in zip(xs, ys, scores, radii):
            if taken.shape[0]:
                d = np.linalg.norm(taken - np.array([x, y], np.float32), axis=1)
                if (d < r * c.nms_distance_factor).any():
                    continue
            kept.append((int(x), int(y), float(s), float(r)))
            taken = np.vstack([taken, np.array([[x, y]], np.float32)])
            if len(kept) >= c.max_detections:
                break
        return kept

    # ------------------------------------------------------------------ #
    def diagnose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Why did this frame produce so few detections?

        A count of zero on a visibly crowded frame is the single most confusing
        failure this system can present, so it should explain itself rather than
        leave the operator guessing.
        """
        gray = self._prepare(frame)
        c = self.config
        found = {}
        for r in (3, 5, 8, 12, 18, 26):
            cfg = HeadDetectorConfig(min_radius_px=max(2.0, r * 0.7),
                                     max_radius_px=r * 1.4, scale_steps=3,
                                     response_threshold=c.response_threshold,
                                     polarity=c.polarity, max_detections=c.max_detections)
            found[r] = len(TopDownHeadDetector(cfg).detect(frame))
        best = max(found, key=found.get)
        return {
            "detections_by_head_radius_px": found,
            "best_radius_px": best,
            "current_range_px": [c.min_radius_px, c.max_radius_px],
            "hint": (f"Most heads look about {best} px across. Set "
                     f"min_radius_px around {max(2, int(best * 0.7))} and "
                     f"max_radius_px around {int(best * 1.5)}."
                     if found[best] > 0 else
                     "No head-like structure found at any scale. The crowd may be too "
                     "distant to resolve individuals at all, in which case a density-map "
                     "counter is the correct estimator rather than any detector."),
        }

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detections in the same shape the rest of the pipeline expects."""
        gray = self._prepare(frame)
        c = self.config

        candidates = []
        polarities = ([False, True] if c.polarity == "auto"
                      else [c.polarity == "dark"])
        for invert in polarities:
            response, radius_map = self._scale_space(gray, invert)
            peaks = self._peaks(response, radius_map)
            candidates.append((peaks, float(np.mean([p[2] for p in peaks])) if peaks else 0.0))

        # In "auto", keep whichever polarity produced the stronger mean response.
        # Overhead heads are bright against dark ground at night and dark hair
        # against bright ground by day, and a deployed camera sees both.
        peaks = max(candidates, key=lambda kv: (len(kv[0]) > 0, kv[1]))[0]
        if not peaks:
            return []

        top = max(p[2] for p in peaks)
        detections: List[Dict[str, Any]] = []
        for x, y, score, r in peaks:
            half = r * 1.15
            detections.append({
                "bbox": [float(x - half), float(y - half),
                         float(x + half), float(y + half)],
                "confidence": float(np.clip(score / max(top, 1e-6), 0.05, 0.99)),
                "class_id": 0,
                "label": "head",
                "radius_px": float(r),
            })
        return detections


# --------------------------------------------------------------------------- #
def heads_to_ground_points(detections: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Ground-plane reference point for an overhead head detection.

    For a side view the feet are the ground contact and the box bottom is
    correct. Looking straight down, the head IS directly above the feet, so the
    box centre is the right reference -- using the box bottom would displace
    every person by half a head, which at drone scale is a real metric error.
    """
    pts = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        pts.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
    return pts
