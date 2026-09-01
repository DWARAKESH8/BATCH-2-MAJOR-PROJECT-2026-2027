"""Loaders for real, externally annotated crowd data.

Everything else in this project is measured against the built-in simulator.
That is honest, and the scripted scenarios make the classifier falsifiable, but
it is not external validation: the same author wrote both the simulator and the
thing being measured.

This module is the bridge to data somebody else annotated. Drop files into
`data/` (see `data/README.md`) and `scripts/evaluate_dataset.py` will find them.

Supported:
    MOT20            per-frame ground-truth boxes on dense real crowds
    ShanghaiTech     point annotations on still crowd images
    counts.csv       your own per-frame counts on your own footage
    expert_ratings   independent human risk ratings, for Cohen's kappa
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# MOT20
# --------------------------------------------------------------------------- #
@dataclass
class MOT20Sequence:
    """One MOT20 sequence: JPEG frames plus ground-truth boxes.

    `gt/gt.txt` columns are:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

    Only `class == 1` (pedestrian) with `conf == 1` counts toward the ground
    truth. MOT20 also annotates distractor classes — vehicles, reflections,
    people behind glass — and including those would inflate the reference count
    and flatter the detector for the wrong reason.
    """

    name: str
    root: Path
    frame_paths: List[Path]
    counts: Dict[int, int]          # 1-based frame number -> ground-truth count
    boxes: Dict[int, np.ndarray]    # 1-based frame number -> (N, 4) xyxy
    fps: float = 25.0
    width: int = 0
    height: int = 0

    def __len__(self) -> int:
        return len(self.frame_paths)

    def frames(self, every: int = 1, limit: Optional[int] = None) -> Iterator[Tuple[int, Any, int]]:
        """Yield (frame_number, image, ground_truth_count)."""
        import cv2

        emitted = 0
        for i, path in enumerate(self.frame_paths):
            frame_no = i + 1
            if (frame_no - 1) % every:
                continue
            if limit is not None and emitted >= limit:
                return
            image = cv2.imread(str(path))
            if image is None:
                continue
            emitted += 1
            yield frame_no, image, self.counts.get(frame_no, 0)

    @classmethod
    def load(cls, seq_dir: Path) -> Optional["MOT20Sequence"]:
        img_dir, gt_file = seq_dir / "img1", seq_dir / "gt" / "gt.txt"
        if not img_dir.is_dir() or not gt_file.exists():
            return None

        frame_paths = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
        if not frame_paths:
            return None

        counts: Dict[int, int] = {}
        boxes: Dict[int, List[List[float]]] = {}
        for line in gt_file.read_text(errors="ignore").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                frame = int(float(parts[0]))
                x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                conf = float(parts[6])
                # Not `cls`: that name is the classmethod's own first parameter,
                # and shadowing it makes the constructor call below fail.
                obj_class = int(float(parts[7]))
            except ValueError:
                continue
            if obj_class != 1 or conf < 1:
                continue
            counts[frame] = counts.get(frame, 0) + 1
            boxes.setdefault(frame, []).append([x, y, x + w, y + h])

        fps, width, height = 25.0, 0, 0
        info = seq_dir / "seqinfo.ini"
        if info.exists():
            cp = configparser.ConfigParser()
            try:
                cp.read(info)
                fps = float(cp.get("Sequence", "frameRate", fallback=25))
                width = int(cp.get("Sequence", "imWidth", fallback=0))
                height = int(cp.get("Sequence", "imHeight", fallback=0))
            except Exception:
                pass

        return cls(
            name=seq_dir.name, root=seq_dir, frame_paths=frame_paths, counts=counts,
            boxes={k: np.asarray(v, dtype=np.float64) for k, v in boxes.items()},
            fps=fps, width=width, height=height,
        )


def find_mot20(root: Path) -> List[MOT20Sequence]:
    """Locate MOT20 sequences under `root`, however deeply the zip was nested."""
    if not root.exists():
        return []
    sequences: List[MOT20Sequence] = []
    for gt in sorted(root.rglob("gt/gt.txt")):
        seq = MOT20Sequence.load(gt.parent.parent)
        if seq is not None:
            sequences.append(seq)
    return sequences


# --------------------------------------------------------------------------- #
# ShanghaiTech
# --------------------------------------------------------------------------- #
@dataclass
class ShanghaiTechSet:
    name: str
    images: List[Path]
    counts: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)

    def items(self, limit: Optional[int] = None) -> Iterator[Tuple[Path, Any, int]]:
        import cv2

        for i, path in enumerate(self.images):
            if limit is not None and i >= limit:
                return
            image = cv2.imread(str(path))
            if image is None:
                continue
            yield path, image, self.counts.get(path.stem, 0)


def _mat_point_count(mat_path: Path) -> Optional[int]:
    """Extract the annotated head count from a ShanghaiTech .mat file.

    The nesting differs between releases and mirrors, so this walks the object
    tree for the first (N, 2) array of points rather than assuming one layout.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        return None
    try:
        mat = loadmat(str(mat_path))
    except Exception:
        return None

    stack = [v for k, v in mat.items() if not k.startswith("__")]
    seen = 0
    while stack and seen < 2000:
        seen += 1
        item = stack.pop()
        if isinstance(item, np.ndarray):
            if item.ndim == 2 and item.shape[1] == 2 and item.shape[0] > 0 \
                    and np.issubdtype(item.dtype, np.number):
                return int(item.shape[0])
            if item.dtype == object or item.dtype.names:
                stack.extend(list(item.flat))
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return None


def find_shanghaitech(root: Path, limit: Optional[int] = None) -> List[ShanghaiTechSet]:
    if not root.exists():
        return []
    sets: List[ShanghaiTechSet] = []
    for images_dir in sorted(root.rglob("images")):
        gt_dir = images_dir.parent / "ground_truth"
        images = sorted(images_dir.glob("*.jpg")) or sorted(images_dir.glob("*.png"))
        if not images:
            continue
        if limit:
            images = images[:limit]

        counts: Dict[str, int] = {}
        if gt_dir.is_dir():
            for img in images:
                for candidate in (gt_dir / f"GT_{img.stem}.mat", gt_dir / f"{img.stem}.mat"):
                    if candidate.exists():
                        n = _mat_point_count(candidate)
                        if n is not None:
                            counts[img.stem] = n
                        break
        if counts:
            label = str(images_dir.relative_to(root).parent) or images_dir.parent.name
            sets.append(ShanghaiTechSet(name=label, images=images, counts=counts))
    return sets


# --------------------------------------------------------------------------- #
# Your own annotations
# --------------------------------------------------------------------------- #
def load_counts_csv(path: Path) -> Dict[str, Dict[int, int]]:
    """`video,frame_id,true_count` -> {video: {frame_id: count}}."""
    if not path.exists():
        return {}
    import csv

    out: Dict[str, Dict[int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                video = row["video"].strip()
                out.setdefault(video, {})[int(row["frame_id"])] = int(row["true_count"])
            except (KeyError, ValueError):
                continue
    return out


def load_expert_ratings(path: Path) -> List[Dict[str, Any]]:
    """`video,start_sec,end_sec,rater,rating` -> list of rating records."""
    if not path.exists():
        return []
    import csv

    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append({
                    "video": row["video"].strip(),
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "rater": row["rater"].strip(),
                    "rating": row["rating"].strip().lower(),
                })
            except (KeyError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------- #
def discover(data_dir: Path) -> Dict[str, Any]:
    """Report what real data is actually present."""
    videos = [p for p in sorted((data_dir / "videos").glob("*"))
              if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    return {
        "videos": videos,
        "mot20": find_mot20(data_dir / "mot20"),
        "shanghaitech": find_shanghaitech(data_dir / "shanghaitech"),
        "counts_csv": load_counts_csv(data_dir / "annotations" / "counts.csv"),
        "expert_ratings": load_expert_ratings(data_dir / "annotations" / "expert_ratings.csv"),
        "calibrations": sorted((data_dir / "annotations").glob("calibration_*.json")),
    }
