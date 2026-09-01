#!/usr/bin/env python3
"""Survey a folder of videos and report which failure modes each one exercises.

Downloading crowd footage is easy; knowing which clip demonstrates which case is
not. This runs the real pipeline briefly over every video in a folder and reports
what each one actually triggers, so a folder of thirty downloads can be triaged
into "this is the counter-flow demo, this is the bottleneck demo" in one pass.

It also flags the two failure conditions that waste the most time:
  * clips where the detector finds nobody (wrong footage, or too low resolution)
  * clips too sparse for any density-based mode to fire

    python scripts/survey_videos.py
    python scripts/survey_videos.py --dir data/videos --frames 150
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crowdguard.config import (  # noqa: E402
    CalibrationConfig, ForecastConfig, LLMConfig, RAGConfig, RiskConfig,
    VisionConfig, ZoneConfig,
)
from src.crowdguard.pipeline import CrowdGuardPipeline, OverlayOptions  # noqa: E402
from src.crowdguard.risk_taxonomy import RISK_TYPES  # noqa: E402
from src.crowdguard.simulator import SCENARIOS  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def survey(path: Path, pipeline: CrowdGuardPipeline, frames: int,
           every: int) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"video": path.name, "error": "could not open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline.reset()
    counts, densities, modes, states = [], [], collections.Counter(), collections.Counter()
    frame_id, processed = -1, 0
    overlays = OverlayOptions(boxes=False, ids=False, vectors=False, zones=False, hud=False)

    while processed < frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % every:
            continue
        processed += 1
        with redirect_stdout(io.StringIO()):
            r = pipeline.process(frame, frame_id, frame_id / fps, every / fps,
                                 overlays=overlays, want_annotated=False,
                                 dispatch_alerts=False)
        counts.append(r.features.person_count)
        densities.append(r.features.local_density_peak)
        modes[r.features.primary_risk_type] += 1
        states[r.status.state] += 1
    cap.release()

    if not counts:
        return {"video": path.name, "error": "no frames read"}

    c, d = np.asarray(counts, float), np.asarray(densities, float)
    useful = [m for m, n in modes.most_common() if m != "NORMAL_FLOW" and n >= 3]

    if c.max() == 0:
        verdict = "UNUSABLE - detector finds nobody. Wrong footage, too low-res, or too far away."
    elif c.mean() < 5:
        verdict = "TOO SPARSE - under 5 people on average; no density-based mode can fire."
    elif d.max() < 1.5:
        verdict = "LOW DENSITY - good for Normal Flow only. Check the --area-m2 estimate."
    elif useful:
        verdict = "USEFUL - demonstrates: " + ", ".join(RISK_TYPES[m].label for m in useful[:3])
    else:
        verdict = "NORMAL ONLY - no failure mode sustained. Try a busier segment."

    return {
        "video": path.name,
        "resolution": f"{w}x{h}", "fps": round(fps, 1),
        "duration_sec": round(total / fps, 1) if total else None,
        "frames_analysed": processed,
        "count_mean": round(float(c.mean()), 1), "count_max": int(c.max()),
        "density_mean": round(float(d.mean()), 2), "density_max": round(float(d.max()), 2),
        "modes": {RISK_TYPES[m].label: n for m, n in modes.most_common()},
        "states": dict(states),
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Report which failure modes each video exercises")
    ap.add_argument("--dir", default=str(ROOT / "data" / "videos"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--area-m2", type=float, default=150.0,
                    help="Rough ground area each camera covers; affects density only")
    ap.add_argument("--frames", type=int, default=120, help="Frames to analyse per video")
    ap.add_argument("--every", type=int, default=5, help="Use every Nth frame")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "video_survey.json"))
    args = ap.parse_args()

    folder = Path(args.dir)
    videos = sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXT)

    print(f"Surveying {folder}")
    if not videos:
        print(f"""
  No videos found.

  Put clips in {folder} and run this again. See data/README.md for sources.
  One 30-second clip of a real crowd is enough to make detection demonstrable —
  the bundled sample_crowd.mp4 is drawn shapes and YOLO detects zero people in it.
""")
        return

    pipeline = CrowdGuardPipeline(
        vision=VisionConfig(model_path=args.model),
        risk=RiskConfig(camera_area_m2=args.area_m2),
        calibration=CalibrationConfig(enabled=False, fallback_area_m2=args.area_m2),
        zones=ZoneConfig(zones=ZoneConfig.default_layout()),
        forecast=ForecastConfig(), rag=RAGConfig(), llm=LLMConfig(),
        load_detector=True, load_rag=False,
    )
    print(f"detector: {pipeline.backends['detector']}   videos: {len(videos)}\n")

    results = []
    for v in videos:
        r = survey(v, pipeline, args.frames, args.every)
        results.append(r)
        if r.get("error"):
            print(f"  {v.name:34} ERROR: {r['error']}")
            continue
        print(f"  {r['video']}")
        print(f"    {r['resolution']} @ {r['fps']}fps"
              + (f", {r['duration_sec']}s" if r['duration_sec'] else "")
              + f"   people {r['count_mean']} avg / {r['count_max']} peak"
              + f"   density {r['density_mean']} / {r['density_max']} peak")
        print(f"    modes: " + ", ".join(f"{k} {v}" for k, v in list(r["modes"].items())[:4]))
        print(f"    {r['verdict']}\n")

    # Which of the eight cases does this collection still not cover?
    covered = set()
    for r in results:
        for label, n in r.get("modes", {}).items():
            if n >= 3:
                covered.add(label)
    missing = [s.label for c, s in RISK_TYPES.items() if s.label not in covered]

    print("=" * 74)
    print("COVERAGE ACROSS ALL VIDEOS IN THIS FOLDER")
    print("=" * 74)
    for code, spec in RISK_TYPES.items():
        mark = "covered" if spec.label in covered else "NOT COVERED"
        print(f"  {spec.label:26} {mark}")
    if missing:
        print(f"""
  {len(missing)} case(s) not covered by real footage.

  For the ones real footage cannot reasonably supply -- crush and turbulent
  surge in particular, where public video essentially does not exist and
  filming it is not an option -- use the built-in scenarios instead:

""" + "\n".join(f"    python -m src.crowdguard.main --simulate {k}"
                for k, s in SCENARIOS.items()
                if s.expected_type in [c for c, sp in RISK_TYPES.items() if sp.label in missing]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
