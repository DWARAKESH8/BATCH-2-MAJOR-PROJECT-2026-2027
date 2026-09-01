from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial import distance as dist

from .config import TrackerConfig


class CentroidTracker:
    """Lightweight tracker for prototype use.

    It assigns IDs to detections using nearest-neighbor centroid matching.
    For publication-grade tracking, replace this with Deep-SORT.
    """

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.next_object_id = 0
        self.objects: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.history: Dict[int, Deque[Tuple[int, int]]] = {}

    def register(self, centroid: Tuple[int, int]) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.history[self.next_object_id] = deque(maxlen=self.config.history_size)
        self.history[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.history.pop(object_id, None)

    def update(self, input_centroids: Iterable[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        input_centroids = list(input_centroids)

        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.config.max_disappeared:
                    self.deregister(object_id)
            return dict(self.objects)

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.config.max_distance:
                continue
            object_id = object_ids[row]
            centroid = input_centroids[col]
            self.objects[object_id] = centroid
            self.disappeared[object_id] = 0
            self.history[object_id].append(centroid)
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])).difference(used_rows)
        unused_cols = set(range(D.shape[1])).difference(used_cols)

        if D.shape[0] >= D.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.config.max_disappeared:
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                self.register(input_centroids[col])

        return dict(self.objects)

    def velocity_vectors(self) -> Dict[int, Tuple[float, float]]:
        """Per-track velocity in pixels per processed frame.

        Averaged over `velocity_window` steps rather than taken from the last
        two points. A single-step difference is dominated by detector jitter:
        a box wobbling a few pixels between frames produces a spurious velocity
        whose direction is essentially random, and random directions inflate
        the flow-disorder metric -- which would make a calm crowd look like
        counter-flow. Smoothing costs a little responsiveness and buys a large
        reduction in false disorder.
        """
        vectors: Dict[int, Tuple[float, float]] = {}
        window = max(2, int(getattr(self.config, "velocity_window", 3)))
        for object_id, pts in self.history.items():
            if len(pts) < 2:
                vectors[object_id] = (0.0, 0.0)
                continue
            recent = list(pts)[-window:]
            steps = len(recent) - 1
            x0, y0 = recent[0]
            x1, y1 = recent[-1]
            vectors[object_id] = (float(x1 - x0) / steps, float(y1 - y0) / steps)
        return vectors

    def active_objects(self) -> Dict[int, Tuple[int, int]]:
        """Only tracks confirmed in the current frame.

        `objects` also holds tracks that have gone missing but are still being
        held open by `max_disappeared`. Those keep their last known position,
        so several of them pile up on the same coordinates -- and a density
        estimator fed those ghosts reports an enormous, entirely fictional
        local density. Anything spatial must use this method.
        """
        return {oid: c for oid, c in self.objects.items() if self.disappeared.get(oid, 0) == 0}

    def track_ages(self) -> Dict[int, int]:
        """How many observations each track has. Used to discount new tracks."""
        return {tid: len(pts) for tid, pts in self.history.items()}

    def net_displacements(self, min_points: int = 5) -> Dict[int, Tuple[float, float]]:
        """Each track's net displacement across its stored history, in pixels.

        Where a velocity vector is one noisy frame-to-frame difference, this is
        where the person has actually GOT to over a second or two. That is a
        far more reliable statement of which way somebody is travelling, and it
        is the right basis for deciding whether two crowds are walking into
        each other or merely converging on the same gate.
        """
        out: Dict[int, Tuple[float, float]] = {}
        for object_id, pts in self.history.items():
            if len(pts) < min_points:
                continue
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            out[object_id] = (float(x1 - x0), float(y1 - y0))
        return out

    def trails(self, length: int = 8) -> Dict[int, list]:
        """Recent path of each track, for the trajectory overlay."""
        return {tid: list(pts)[-length:] for tid, pts in self.history.items()}
