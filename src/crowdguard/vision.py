from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import VisionConfig
from .head_detection import HeadDetectorConfig, TopDownHeadDetector


class PersonDetector:
    """Person detector wrapper.

    Uses Ultralytics YOLO when available. If YOLO cannot be loaded, it falls back to
    OpenCV HOG person detection so the prototype can still run offline.
    """

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.model = None
        self.backend = "hog"
        self.hog = None
        self.load_error = ""
        self.head_model = None
        self.head_detector: Optional[TopDownHeadDetector] = None
        self._empty_streak = 0
        self.view_mode = self.config.mode
        self._load_model()
        if self.config.mode in {"head", "auto"}:
            self._load_head_backend()

    # ------------------------------------------------------------------ #
    def _load_head_backend(self) -> None:
        """Prefer a trained head model; fall back to the untrained detector."""
        path = Path(self.config.head_model_path)
        if path.exists():
            try:
                from ultralytics import YOLO

                self.head_model = YOLO(str(path))
                return
            except Exception as exc:
                self.load_error += f" | head model: {type(exc).__name__}: {exc}"
        self.head_detector = TopDownHeadDetector(HeadDetectorConfig())

    @property
    def head_backend(self) -> str:
        if self.head_model is not None:
            return f"trained:{Path(self.config.head_model_path).name}"
        return "blob (untrained)" if self.head_detector is not None else "none"

    def _detect_heads(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.head_model is not None:
            results = self.head_model.predict(frame, conf=self.config.confidence_threshold,
                                              verbose=False)
            out: List[Dict[str, Any]] = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                    out.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": float(box.conf[0].item()),
                                "class_id": int(box.cls[0].item()), "label": "head"})
            return out
        if self.head_detector is not None:
            return self.head_detector.detect(frame)
        return []

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.config.model_path)
            self.backend = "yolo"
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            if not self.config.use_hog_fallback:
                raise RuntimeError(f"Failed to load YOLO model {self.config.model_path}: {exc}") from exc
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.backend = "hog"

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.view_mode == "head":
            return self._detect_heads(frame)

        detections = self._detect_yolo(frame) if self.backend == "yolo" else self._detect_hog(frame)

        if self.view_mode == "auto":
            # Person detection returning nothing repeatedly is the signature of
            # an overhead view, not of an empty scene: an empty scene is
            # indistinguishable from one full of heads as far as a COCO person
            # model is concerned. Switching after a few consecutive empty
            # frames avoids flapping on a genuinely quiet moment.
            if detections:
                self._empty_streak = 0
            else:
                self._empty_streak += 1
                if self._empty_streak >= self.config.auto_switch_after:
                    heads = self._detect_heads(frame)
                    if heads:
                        self.view_mode = "head"
                        return heads
        return detections

    def _detect_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(frame, conf=self.config.confidence_threshold, verbose=False)
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls != self.config.person_class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "label": "person",
                }
            )
        return detections

    def _detect_hog(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.hog is None:
            return []
        resized = frame
        scale_factor = 1.0
        h, w = frame.shape[:2]
        if w > 960:
            scale_factor = 960.0 / w
            resized = cv2.resize(frame, (960, int(h * scale_factor)))

        boxes, weights = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections: List[Dict[str, Any]] = []
        for (x, y, bw, bh), weight in zip(boxes, weights):
            x1 = x / scale_factor
            y1 = y / scale_factor
            x2 = (x + bw) / scale_factor
            y2 = (y + bh) / scale_factor
            conf = float(weight) if np.isscalar(weight) else float(weight[0])
            if conf < 0.2:
                continue
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": min(1.0, max(0.0, conf)),
                    "class_id": 0,
                    "label": "person",
                }
            )
        return detections


def boxes_to_centroids(detections: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Ground-plane reference point for each detection.

    For a side view the *feet* -- bottom-centre of the box -- are the ground
    contact point. Under a homography the difference matters: projecting a
    torso centre places a distant person several metres behind their true
    position and corrupts every density estimate downstream.

    Looking straight down, that reverses. The head is directly above the feet,
    so the box CENTRE is the correct ground reference, and using the box bottom
    would displace every person by half a head -- a real metric error at drone
    altitude. Head detections are labelled, so the right rule is applied per
    detection rather than assumed for the whole frame.
    """
    centroids = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det.get("label") == "head":
            centroids.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        else:
            centroids.append((int((x1 + x2) / 2), int(y2)))
    return centroids


def box_centres(detections: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Geometric box centres, used for drawing only."""
    return [
        (int((d["bbox"][0] + d["bbox"][2]) / 2), int((d["bbox"][1] + d["bbox"][3]) / 2))
        for d in detections
    ]


# --------------------------------------------------------------------------- #
# Occlusion diagnostics
# --------------------------------------------------------------------------- #
def occlusion_estimate(
    detections: List[Dict[str, Any]],
    frame_shape: Tuple[int, ...],
    max_gain: float = 1.6,
) -> Dict[str, Any]:
    """Estimate how badly the detector is undercounting.

    Appearance-based detectors fail exactly where the risk is highest: in a
    dense crowd most bodies are mutually occluded, so YOLO's count falls below
    the true count precisely when an accurate count matters most. This is a
    real and well-documented limitation of detection-based counting, and the
    honest response is to *report* it rather than hide it.

    The occlusion ratio is 1 - (union area of all boxes / sum of box areas):
    zero when nobody overlaps anybody, approaching one when the crowd is a
    single mass of overlapping boxes. The corrected count is reported as a
    separate diagnostic and is NOT silently substituted into the density that
    drives alarms -- inflating a safety-critical measurement with an
    unvalidated correction factor would be worse than the undercount.
    """
    n = len(detections)
    if n == 0:
        return {
            "detected_count": 0,
            "occlusion_ratio": 0.0,
            "estimated_true_count": 0,
            "confidence_note": "no detections",
            "undercount_warning": False,
        }

    h, w = int(frame_shape[0]), int(frame_shape[1])
    scale = 4  # rasterise at quarter resolution; plenty for an area ratio
    mask = np.zeros((max(1, h // scale), max(1, w // scale)), dtype=np.uint8)

    sum_area = 0.0
    for det in detections:
        x1, y1, x2, y2 = [int(v / scale) for v in det["bbox"]]
        x1 = max(0, min(mask.shape[1] - 1, x1))
        x2 = max(0, min(mask.shape[1], x2))
        y1 = max(0, min(mask.shape[0] - 1, y1))
        y2 = max(0, min(mask.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue
        sum_area += (x2 - x1) * (y2 - y1)
        mask[y1:y2, x1:x2] = 1

    union_area = float(mask.sum())
    if sum_area <= 0:
        ratio = 0.0
    else:
        ratio = float(np.clip(1.0 - union_area / sum_area, 0.0, 0.95))

    gain = float(min(max_gain, 1.0 / max(0.25, 1.0 - ratio)))
    estimated = int(round(n * gain))

    if ratio < 0.15:
        note = "Low mutual occlusion -- detected count is reliable."
    elif ratio < 0.35:
        note = "Moderate occlusion -- expect a small undercount."
    else:
        note = (
            "Heavy occlusion -- the detected count is a LOWER BOUND. "
            "A density-map counter (CSRNet class) is the correct estimator in this regime."
        )

    return {
        "detected_count": n,
        "occlusion_ratio": round(ratio, 3),
        "estimated_true_count": estimated,
        "estimation_gain": round(gain, 3),
        "confidence_note": note,
        "undercount_warning": ratio >= 0.35,
    }
