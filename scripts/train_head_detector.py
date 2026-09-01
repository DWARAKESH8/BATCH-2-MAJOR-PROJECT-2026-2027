#!/usr/bin/env python3
"""Fine-tune YOLOv8 to detect heads from above.

The blob detector in `head_detection.py` needs no training and works today, but
it is a classical Laplacian-of-Gaussian detector: it will fire on any round
bright object -- a bald patch, a balloon, a light fitting, a white cap on dark
tarmac -- and it has no notion of what a person is. For anything beyond a
demonstration, train a real model.

Three dataset formats are supported, in order of usefulness for this task:

  crowdhuman  Every person carries a head box (`hbox`) alongside the full body
              box. Around 470k instances. The largest source of head labels
              there is, though the viewpoints are mostly oblique rather than
              straight down.
              https://www.crowdhuman.org/

  visdrone    Drone-captured, which is the actual deployment view, and includes
              night scenes. Classes `pedestrian` and `people` are merged into
              one `person` class here.
              https://github.com/VisDrone/VisDrone-Dataset

  scuthead    SCUT-HEAD, Pascal VOC XML, purpose-built head detection from
              elevated classroom and surveillance cameras.

Usage:
    python scripts/train_head_detector.py --format crowdhuman --root data/crowdhuman
    python scripts/train_head_detector.py --format visdrone   --root data/visdrone
    python scripts/train_head_detector.py --format visdrone --root data/visdrone --epochs 80

The result lands at models/yolov8n-head.pt, which `VisionConfig.head_model_path`
already points at -- the pipeline picks it up with no further configuration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMG_EXT = {".jpg", ".jpeg", ".png"}


def _yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float,
               w: int, h: int) -> Optional[str]:
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return (f"{cls} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} "
            f"{bw / w:.6f} {bh / h:.6f}")


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    import cv2

    img = cv2.imread(str(path))
    return (img.shape[1], img.shape[0]) if img is not None else None


# --------------------------------------------------------------------------- #
def convert_crowdhuman(root: Path, out: Path) -> Dict[str, int]:
    """CrowdHuman .odgt -> YOLO, keeping the HEAD box of each person."""
    stats = {"images": 0, "boxes": 0, "skipped": 0}
    for split, odgt in (("train", "annotation_train.odgt"), ("val", "annotation_val.odgt")):
        ann = next(iter(root.rglob(odgt)), None)
        if ann is None:
            print(f"  {odgt} not found, skipping {split}")
            continue
        img_dir = next((d for d in root.rglob("*") if d.is_dir() and split in d.name.lower()
                        and any(p.suffix.lower() in IMG_EXT for p in d.glob("*"))), None)
        if img_dir is None:
            img_dir = root
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for line in ann.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            src = next((p for p in [img_dir / f"{rec['ID']}.jpg",
                                    img_dir / f"{rec['ID']}.png"] if p.exists()), None)
            if src is None:
                stats["skipped"] += 1
                continue
            size = _image_size(src)
            if size is None:
                stats["skipped"] += 1
                continue
            w, h = size

            lines = []
            for box in rec.get("gtboxes", []):
                if box.get("tag") != "person":
                    continue
                # Ignore heavily occluded or explicitly ignored instances --
                # training on boxes a human could not label either just teaches
                # the model to hallucinate.
                if box.get("head_attr", {}).get("ignore", 0) == 1:
                    continue
                hb = box.get("hbox")
                if not hb or len(hb) != 4:
                    continue
                x, y, bw, bh = hb
                ln = _yolo_line(0, x, y, x + bw, y + bh, w, h)
                if ln:
                    lines.append(ln)
            if not lines:
                continue
            shutil.copy2(src, out / "images" / split / src.name)
            (out / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines))
            stats["images"] += 1
            stats["boxes"] += len(lines)
    return stats


def convert_visdrone(root: Path, out: Path) -> Dict[str, int]:
    """VisDrone -> YOLO. Classes 1 (pedestrian) and 2 (people) become `person`."""
    stats = {"images": 0, "boxes": 0, "skipped": 0}
    for ann_dir in sorted(root.rglob("annotations")):
        img_dir = ann_dir.parent / "images"
        if not img_dir.is_dir():
            continue
        split = "val" if "val" in ann_dir.parent.name.lower() else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for ann in sorted(ann_dir.glob("*.txt")):
            src = next((img_dir / f"{ann.stem}{e}" for e in (".jpg", ".png")
                        if (img_dir / f"{ann.stem}{e}").exists()), None)
            if src is None:
                stats["skipped"] += 1
                continue
            size = _image_size(src)
            if size is None:
                continue
            w, h = size
            lines = []
            for row in ann.read_text().splitlines():
                parts = row.strip().rstrip(",").split(",")
                if len(parts) < 8:
                    continue
                try:
                    x, y, bw, bh = (float(parts[0]), float(parts[1]),
                                    float(parts[2]), float(parts[3]))
                    score, category = int(parts[4]), int(parts[5])
                except ValueError:
                    continue
                if score == 0 or category not in (1, 2):
                    continue
                ln = _yolo_line(0, x, y, x + bw, y + bh, w, h)
                if ln:
                    lines.append(ln)
            if not lines:
                continue
            shutil.copy2(src, out / "images" / split / src.name)
            (out / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines))
            stats["images"] += 1
            stats["boxes"] += len(lines)
    return stats


def convert_scuthead(root: Path, out: Path) -> Dict[str, int]:
    """SCUT-HEAD (Pascal VOC XML) -> YOLO."""
    stats = {"images": 0, "boxes": 0, "skipped": 0}
    for ann_dir in sorted(root.rglob("Annotations")):
        img_dir = ann_dir.parent / "JPEGImages"
        if not img_dir.is_dir():
            continue
        split = "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for xml in sorted(ann_dir.glob("*.xml")):
            src = next((img_dir / f"{xml.stem}{e}" for e in (".jpg", ".png")
                        if (img_dir / f"{xml.stem}{e}").exists()), None)
            if src is None:
                stats["skipped"] += 1
                continue
            try:
                tree = ET.parse(xml)
            except Exception:
                continue
            size = tree.find("size")
            w = int(size.findtext("width", "0")); h = int(size.findtext("height", "0"))
            if w <= 0 or h <= 0:
                got = _image_size(src)
                if got is None:
                    continue
                w, h = got
            lines = []
            for obj in tree.findall("object"):
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                ln = _yolo_line(0, float(bb.findtext("xmin", "0")), float(bb.findtext("ymin", "0")),
                                float(bb.findtext("xmax", "0")), float(bb.findtext("ymax", "0")), w, h)
                if ln:
                    lines.append(ln)
            if not lines:
                continue
            shutil.copy2(src, out / "images" / split / src.name)
            (out / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines))
            stats["images"] += 1
            stats["boxes"] += len(lines)
    return stats


CONVERTERS = {"crowdhuman": convert_crowdhuman, "visdrone": convert_visdrone,
              "scuthead": convert_scuthead}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8 for top-down head detection")
    ap.add_argument("--format", required=True, choices=sorted(CONVERTERS))
    ap.add_argument("--root", required=True, help="Folder the dataset was unzipped into")
    ap.add_argument("--work", default=str(ROOT / "data" / "_head_yolo"),
                    help="Where the converted YOLO-format dataset is written")
    ap.add_argument("--base", default="yolov8n.pt", help="Starting weights")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="Heads are small; 960 or 1280 matters far more here than model size")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "models" / "yolov8n-head.pt"))
    ap.add_argument("--convert-only", action="store_true")
    args = ap.parse_args()

    root, work = Path(args.root), Path(args.work)
    if not root.exists():
        raise SystemExit(f"Dataset folder not found: {root}")

    print(f"Converting {args.format} from {root}")
    if work.exists():
        shutil.rmtree(work)
    stats = CONVERTERS[args.format](root, work)
    print(f"  images {stats['images']}   boxes {stats['boxes']}   skipped {stats['skipped']}")
    if stats["images"] == 0:
        raise SystemExit("Nothing converted. Check the folder layout against the docstring.")

    # If the dataset had no val split, carve one out. Ultralytics needs it, and
    # training without one gives no signal about overfitting.
    train_imgs = sorted((work / "images" / "train").glob("*"))
    val_dir = work / "images" / "val"
    if not val_dir.exists() or not any(val_dir.iterdir()):
        (work / "images" / "val").mkdir(parents=True, exist_ok=True)
        (work / "labels" / "val").mkdir(parents=True, exist_ok=True)
        for img in train_imgs[::10]:
            shutil.move(str(img), work / "images" / "val" / img.name)
            lab = work / "labels" / "train" / f"{img.stem}.txt"
            if lab.exists():
                shutil.move(str(lab), work / "labels" / "val" / lab.name)
        print(f"  carved out {len(train_imgs[::10])} validation images")

    yaml_path = work / "dataset.yaml"
    yaml_path.write_text(
        f"path: {work.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: head\n")
    print(f"  dataset.yaml written to {yaml_path}")
    if args.convert_only:
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics is required: pip install ultralytics")

    print(f"\nTraining {args.base} for {args.epochs} epochs at {args.imgsz}px")
    print("  Image size matters more than model size here: a head from a drone is "
          "10-25 px, and at the default 640 it is a handful of pixels after stride.")
    model = YOLO(args.base)
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=str(ROOT / "outputs" / "head_training"),
                name=args.format, exist_ok=True,
                # Heads have no meaningful vertical flip and little colour
                # signal; scale and mosaic matter far more, because altitude
                # varies continuously in flight.
                fliplr=0.5, flipud=0.0, scale=0.6, mosaic=1.0, degrees=10.0)

    best = ROOT / "outputs" / "head_training" / args.format / "weights" / "best.pt"
    if best.exists():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.out)
        print(f"\nSaved to {args.out}")
        print("The pipeline picks this up automatically. Verify with:")
        print("  python -m src.crowdguard.main --video data/videos/drone.mp4 --view head")
    else:
        print(f"\nTraining finished but {best} was not found.")


if __name__ == "__main__":
    main()
