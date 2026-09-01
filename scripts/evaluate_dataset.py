#!/usr/bin/env python3
"""Evaluate against REAL, externally annotated data.

`scripts/evaluate.py` measures the system against the built-in simulator. That
makes the failure-mode classifier falsifiable, but it is not external
validation, because the same author wrote both the simulator and the thing being
measured.

This script is the external check. It reads whatever real data is present in
`data/` and reports only on what it actually finds -- it never invents a number
for data that is not there.

    python scripts/evaluate_dataset.py                 # everything present
    python scripts/evaluate_dataset.py --only mot20
    python scripts/evaluate_dataset.py --max-frames 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crowdguard.config import VisionConfig  # noqa: E402
from src.crowdguard.datasets import discover  # noqa: E402
from src.crowdguard.vision import PersonDetector, occlusion_estimate  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def counting_stats(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    err = pred - true
    safe = np.maximum(true, 1)
    return {
        "n": int(pred.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / safe) * 100),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(pred, true)[0, 1]) if pred.size > 2 and np.std(true) > 0 else float("nan"),
        "true_mean": float(true.mean()),
        "true_max": float(true.max()),
    }


def report_counting(label: str, stats: Dict[str, float]) -> None:
    print(f"  {label}")
    print(f"    frames / images            {stats['n']}")
    print(f"    ground-truth count         mean {stats['true_mean']:.1f}, max {stats['true_max']:.0f}")
    print(f"    MAE                        {stats['mae']:.2f} people")
    print(f"    RMSE                       {stats['rmse']:.2f} people")
    print(f"    MAPE                       {stats['mape']:.1f} %")
    print(f"    bias (pred - true)         {stats['bias']:+.2f} people")
    print(f"    correlation r              {stats['r']:.3f}")


# --------------------------------------------------------------------------- #
def eval_mot20(sequences, detector, every: int, limit: int) -> Dict[str, Any]:
    rule("MOT20 — real dense crowds, per-frame ground-truth boxes")
    print("  Ground truth annotated by the MOTChallenge benchmark, not by this project.\n"
          "  Only class 1 (pedestrian) with conf 1 counts; MOT20's distractor classes are\n"
          "  excluded, because including them would inflate the reference count and\n"
          "  flatter the detector for the wrong reason.\n")

    out: Dict[str, Any] = {}
    all_pred, all_true, occl = [], [], []
    for seq in sequences:
        pred, true = [], []
        t0 = time.time()
        for frame_no, image, gt_count in seq.frames(every=every, limit=limit):
            dets = detector.detect(image)
            pred.append(len(dets))
            true.append(gt_count)
            occl.append(occlusion_estimate(dets, image.shape)["occlusion_ratio"])
        if not pred:
            continue
        p, t = np.asarray(pred, float), np.asarray(true, float)
        stats = counting_stats(p, t)
        stats["seconds"] = round(time.time() - t0, 1)
        out[seq.name] = stats
        report_counting(f"{seq.name}  ({seq.width}x{seq.height}, {len(seq)} frames total)", stats)
        print()
        all_pred.append(p)
        all_true.append(t)

    if all_pred:
        p, t = np.concatenate(all_pred), np.concatenate(all_true)
        overall = counting_stats(p, t)
        out["OVERALL"] = overall
        rule("MOT20 — overall")
        report_counting("all sequences pooled", overall)

        # The claim that matters: does the detector degrade where it counts?
        dense = t >= np.percentile(t, 75)
        if dense.sum() > 5:
            d = counting_stats(p[dense], t[dense])
            print(f"\n    In the densest quartile (ground truth >= {np.percentile(t, 75):.0f} people):")
            print(f"      MAPE                     {d['mape']:.1f} %")
            print(f"      bias                     {d['bias']:+.2f} people")
            if d["bias"] < overall["bias"]:
                print("      -> the detector UNDER-counts more as density rises, which is the\n"
                      "         documented occlusion limitation showing up on real data.")
            out["DENSEST_QUARTILE"] = d
        if occl:
            print(f"\n    mean measured occlusion ratio  {np.mean(occl):.3f}")
    return out


def eval_shanghaitech(sets, detector, limit: int) -> Dict[str, Any]:
    rule("ShanghaiTech — real crowd images, point-annotated head counts")
    out: Dict[str, Any] = {}
    for s in sets:
        pred, true = [], []
        for path, image, gt_count in s.items(limit=limit):
            pred.append(len(detector.detect(image)))
            true.append(gt_count)
        if not pred:
            continue
        stats = counting_stats(np.asarray(pred, float), np.asarray(true, float))
        out[s.name] = stats
        report_counting(f"{s.name}  ({len(s)} images)", stats)
        print()
    if out:
        print("  Note: ShanghaiTech annotates every head, including ones a detector cannot\n"
              "  resolve at all. A large negative bias here is expected and is a statement\n"
              "  about detection-based counting, not a bug.")
    return out


def eval_own_counts(counts, videos, detector, every: int) -> Dict[str, Any]:
    rule("Your own annotated footage — data/annotations/counts.csv")
    import cv2

    out: Dict[str, Any] = {}
    by_name = {p.name: p for p in videos}
    for video_name, frame_counts in counts.items():
        path = by_name.get(video_name) or (ROOT / "data" / "videos" / video_name)
        if not Path(path).exists():
            print(f"  {video_name}: annotated but the video is not in data/videos/ — skipped")
            continue
        cap = cv2.VideoCapture(str(path))
        pred, true = [], []
        wanted = set(frame_counts)
        frame_id = -1
        while cap.isOpened() and len(pred) < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id not in wanted:
                continue
            pred.append(len(detector.detect(frame)))
            true.append(frame_counts[frame_id])
        cap.release()
        if not pred:
            continue
        stats = counting_stats(np.asarray(pred, float), np.asarray(true, float))
        out[video_name] = stats
        report_counting(video_name, stats)
        print()
    return out


def eval_expert_ratings(ratings: List[Dict[str, Any]]) -> Dict[str, Any]:
    rule("Expert agreement — data/annotations/expert_ratings.csv")
    print("  No labelled dataset of true crowd risk exists, so the closest honest check on\n"
          "  the fused risk score is whether qualified people agree with it, and with each\n"
          "  other. Inter-rater agreement is reported first: if the humans do not agree with\n"
          "  one another, agreement with the system means very little.\n")

    from itertools import combinations

    from sklearn.metrics import cohen_kappa_score

    by_clip: Dict[tuple, Dict[str, str]] = {}
    for r in ratings:
        by_clip.setdefault((r["video"], r["start_sec"], r["end_sec"]), {})[r["rater"]] = r["rating"]

    raters = sorted({r["rater"] for r in ratings})
    print(f"  clips {len(by_clip)}   raters {len(raters)}: {', '.join(raters)}\n")

    pairs: List[float] = []
    for a, b in combinations(raters, 2):
        shared = [(v[a], v[b]) for v in by_clip.values() if a in v and b in v]
        if len(shared) < 3:
            continue
        k = cohen_kappa_score([x for x, _ in shared], [y for _, y in shared])
        pairs.append(k)
        print(f"    {a} vs {b:12} kappa {k:+.3f}   (n={len(shared)})")

    out: Dict[str, Any] = {"clips": len(by_clip), "raters": len(raters)}
    if pairs:
        out["mean_inter_rater_kappa"] = float(np.mean(pairs))
        print(f"\n  mean inter-rater kappa       {np.mean(pairs):+.3f}")
        print("  (0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect)")
    else:
        print("  Not enough overlapping clips to compute pairwise agreement.")

    print("\n  To compare these against the system, run the pipeline over the same clips and\n"
          "  take the modal risk level for each interval; `outputs/risk_log.jsonl` has the\n"
          "  per-frame levels with timestamps.")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CrowdGuard against real annotated data")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--every", type=int, default=10, help="Use every Nth annotated frame")
    ap.add_argument("--max-frames", type=int, default=150, help="Cap per sequence, for speed")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["mot20", "shanghaitech", "counts", "experts"])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dataset_evaluation.json"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    found = discover(data_dir)

    print("CrowdGuard-RAG — evaluation against real annotated data")
    print(f"data directory: {data_dir}\n")
    print(f"  videos in data/videos/       {len(found['videos'])}")
    print(f"  MOT20 sequences              {len(found['mot20'])}")
    print(f"  ShanghaiTech splits          {len(found['shanghaitech'])}")
    print(f"  annotated videos in CSV      {len(found['counts_csv'])}")
    print(f"  expert rating records        {len(found['expert_ratings'])}")
    print(f"  calibration files            {len(found['calibrations'])}")

    nothing = not any([found["mot20"], found["shanghaitech"],
                       found["counts_csv"], found["expert_ratings"]])
    if nothing:
        rule("NO REAL ANNOTATED DATA FOUND")
        print("""  Nothing to evaluate. Every number this project currently reports comes from
  the built-in simulator, which is stated plainly in the README but is NOT
  external validation.

  To change that, put data in place and run this again -- see data/README.md:

    data/videos/          any real crowd video (fixes the demo: the bundled
                          sample_crowd.mp4 contains drawn shapes and YOLO
                          detects zero people in it)

    data/mot20/train/     MOT20 -- the highest-value addition. Real dense
                          crowds with per-frame ground-truth boxes, giving a
                          counting MAE measured against annotations you did
                          not create.  https://motchallenge.net/data/MOT20/

    data/shanghaitech/    smaller alternative (~300 MB) if 5 GB is too much

    data/annotations/counts.csv          your own per-frame counts
    data/annotations/expert_ratings.csv  independent human risk ratings""")
        return

    want = set(args.only) if args.only else {"mot20", "shanghaitech", "counts", "experts"}
    detector = None
    if want & {"mot20", "shanghaitech", "counts"}:
        detector = PersonDetector(VisionConfig(model_path=args.model))
        print(f"\n  detector backend             {detector.backend}")
        if detector.backend != "yolo":
            print(f"  WARNING: YOLO unavailable ({detector.load_error}); using the HOG fallback,\n"
                  f"  which counts far less accurately. These numbers will understate the system.")

    report: Dict[str, Any] = {"detector": detector.backend if detector else None}
    if "mot20" in want and found["mot20"]:
        report["mot20"] = eval_mot20(found["mot20"], detector, args.every, args.max_frames)
    if "shanghaitech" in want and found["shanghaitech"]:
        report["shanghaitech"] = eval_shanghaitech(found["shanghaitech"], detector, args.max_frames)
    if "counts" in want and found["counts_csv"]:
        report["own_counts"] = eval_own_counts(found["counts_csv"], found["videos"],
                                               detector, args.every)
    if "experts" in want and found["expert_ratings"]:
        report["expert_agreement"] = eval_expert_ratings(found["expert_ratings"])

    rule("HOW TO REPORT THIS")
    print("""  These are the numbers to put in the report, because the annotations came from
  outside the project. Quote them alongside the simulator results, and say which
  is which -- a counting MAE on MOT20 and a counting MAE against your own
  simulator are not the same kind of claim, and a panel will know the difference.

  If the bias is strongly negative in the densest frames, do not hide it. That is
  the documented occlusion limitation appearing on real data exactly where the
  README predicted it would, which is a better result than a number with no
  explanation behind it.""")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
