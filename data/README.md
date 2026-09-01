# Where to put your data

Nothing in this folder is tracked or bundled — it is where **you** put real footage and real
annotations. Until you do, every number the project reports comes from the built-in simulator,
which is honest but is not external validation.

Drop files in, then run:

```bash
python scripts/evaluate_dataset.py
```

It auto-detects whatever is present and evaluates only that. Nothing else needs configuring.

---

## 1. `data/videos/` — real crowd video (do this first)

**Put here:** any `.mp4`, `.avi`, `.mov` or `.mkv` of a real crowd. Festival footage, a station
concourse, a market, public CCTV, a clip you filmed yourself. No annotations needed.

**Why it matters most:** the bundled `sample_crowd.mp4` is drawn shapes — YOLO detects **zero**
people in it. Until a real video is here, you cannot demonstrate detection working at all. One
30-second clip fixes that.

**Then run:**

```bash
python -m src.crowdguard.main --video data/videos/YOURFILE.mp4 --area-m2 150 --sample-every 3
```

Or select **Upload video** in the dashboard sidebar.

> Set `--area-m2` to your honest estimate of the ground area the camera covers. Better still,
> calibrate it — see §4.

---

## 1b. Where to actually get crowd video, per failure mode

The blunt fact first: **public footage does not exist evenly across the eight cases.**
Normal flow and bottlenecks are everywhere; crush and turbulent surge essentially are not,
because the only real recordings are of people dying and they are not distributed as
research datasets. That asymmetry is the honest justification for the simulator — say it
in the viva rather than being asked about it.

### A note on MOT20 — you are right, it is not .mp4

MOT20 ships as **numbered JPEG frames**, not video files:

```
data/mot20/train/MOT20-01/img1/000001.jpg, 000002.jpg, ...
```

They are still video sequences — just stored frame-by-frame, which is how tracking
benchmarks are distributed so annotations line up with exact frame numbers. The loader in
`src/crowdguard/datasets.py` reads them directly, so nothing needs converting for evaluation.

If you want an actual `.mp4` for the dashboard or a slide:

```bash
ffmpeg -framerate 25 -i data/mot20/train/MOT20-02/img1/%06d.jpg -c:v libx264 -pix_fmt yuv420p data/videos/mot20-02.mp4
```

Datasets that **do** ship real video files: PETS2009, UMN Crowd Anomaly, CUHK Crowd,
Grand Central Station, VisDrone-VID, UCSD Anomaly, Motion Emotion Dataset.

### Annotated research datasets — for measured accuracy

| Dataset | What it is | Get it |
|---|---|---|
| **MOT20** | 4 train sequences of genuinely dense real crowds, per-frame ground-truth boxes. **The one to get.** | [motchallenge.net/data/MOT20](https://motchallenge.net/data/MOT20/) |
| **CroHD / HT21** | Head tracking in dense crowds — heads stay visible where bodies do not, so it counts better than box detection at high density | motchallenge.net, "Head Tracking 21" |
| **Grand Central Station** | 33 minutes of a real station concourse with ~50,000 trajectories. Excellent for bottleneck and counter-flow | search "Grand Central Station dataset CUHK Bolei Zhou" |
| **WorldExpo'10** | 1,132 annotated video sequences from many cameras | search "WorldExpo10 crowd counting dataset" |
| **ShanghaiTech A/B** | Still images, point annotations. Small download (~300 MB) | search "ShanghaiTech crowd counting" (Kaggle mirrors) |
| **UCF-QNRF / NWPU-Crowd / JHU-CROWD++** | Very dense still images, large annotation counts | search by name |

### Behaviour and anomaly datasets — closest to your failure modes

| Dataset | Why it matters here |
|---|---|
| **UMN Crowd Anomaly** | Staged escape scenarios: people walk normally, then suddenly scatter. **This is your Panic Dispersal case, and it is one of the few honest sources for it.** 3 scenes, 11 sequences |
| **PETS2009** | Multi-camera, with explicitly scripted crowd events — walking, running, **flow splitting and merging**, evacuation. S3 is the flow-analysis set. Good for Counter-Flow |
| **Motion Emotion Dataset (MED)** | Crowd clips labelled panic / fight / congestion / obstacle / neutral — labels that map almost directly onto your taxonomy |
| **CUHK Crowd Dataset** | 474 videos across 215 scenes with crowd-attribute labels; wide variety in one download |
| **UCSD Anomaly / CUHK Avenue / ShanghaiTech Campus** | Standard anomaly benchmarks. Sparse crowds, so useful for tracking robustness rather than density |

### Overhead and drone footage — for the deployment view

COCO person detection returns **nothing** from a camera looking straight down. Measured on an
overhead night scene of 69 people: **0 detections at confidence 0.30, and still 0 at 0.03.**
That is not a threshold problem, it is a distribution problem — the model was trained on
upright bodies with visible torso and limbs, and overhead it sees circular head blobs.

Since a security drone is the eventual deployment, these datasets matter more than the
ground-level ones:

| Dataset | Why it matters | Get it |
|---|---|---|
| **VisDrone** | Drone-captured, **includes night scenes**, `pedestrian` and `people` classes. VisDrone-DET for images, VisDrone-VID for video. The closest public data to your actual use case | [github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset) |
| **CrowdHuman** | Every person carries a **head box** (`hbox`) alongside the body box, ~470k instances. The largest source of head labels available | [crowdhuman.org](https://www.crowdhuman.org/) |
| **CroHD / HT21** | Head tracking in dense crowds. Heads stay visible where bodies do not, so it counts where box detection fails | motchallenge.net, "Head Tracking 21" |
| **SCUT-HEAD** | Purpose-built head detection from elevated classroom and surveillance cameras, Pascal VOC format | search "SCUT-HEAD dataset" |
| **UAVDT** | UAV detection and tracking benchmark, varied altitude and weather | search "UAVDT benchmark" |
| **Okutama-Action** | Aerial human activity from a drone at 10–45 m | search "Okutama-Action dataset" |

Train a real head detector from any of the first three:

```bash
python scripts/train_head_detector.py --format crowdhuman --root data/crowdhuman
```

```bash
python scripts/train_head_detector.py --format visdrone --root data/visdrone --imgsz 1280
```

The result lands at `models/yolov8n-head.pt`, which the pipeline picks up automatically.
Until then an untrained blob detector handles overhead views — see §1c.

### Free footage with no annotations — for demonstration

Fastest way to make YOLO detection demonstrable at all:

- **Pexels** and **Pixabay** — free stock video, no account. Search *crowd walking*, *busy street*,
  *train station*, *festival crowd*, *pedestrian crossing*. Download 1080p, 20–40 seconds.
- **Public live webcams** — Skyline Webcams, EarthCam, and many city/transport authority streams
  cover squares, promenades and station concourses. The dashboard accepts a stream URL directly.
- **YouTube** — the dashboard resolves YouTube URLs via `yt-dlp`. Useful search terms per case are
  in the table below. Check the licence before putting a frame in your report.

### Mapping each failure mode to a realistic source

| Failure mode | Realistic real-world source | Difficulty |
|---|---|---|
| **Normal Flow** | Any street or concourse footage; MOT20-01; Pexels | trivial |
| **Rapid Influx** | Stadium or venue gates opening; PETS2009 S1; festival entry | easy |
| **Bottleneck Congestion** | Station ticket gates, escalator foot, narrow bridge; MOT20-02; Grand Central | easy |
| **Counter-Flow Conflict** | Station concourse at changeover; PETS2009 S3 splitting flows; Grand Central | moderate |
| **Static Blockage** | Queue footage where the head stops moving; hard to find deliberately | hard |
| **Panic Dispersal** | **UMN Crowd Anomaly** staged escape scenarios; fire-drill footage | moderate |
| **Turbulent Surge** | Front-of-stage concert footage; Hajj/Mina documentary analysis | very hard |
| **Progressive Crowd Crush** | Effectively unavailable, and using real disaster footage raises obvious ethical problems | not available |

**For the last two, use the simulator and say so plainly.** That is the argument:

> "Turbulent surge and progressive crush have no public dataset, because the only real
> recordings are of people dying. That is precisely why I built a physics-based simulator
> with ground truth — it lets me demonstrate and *measure* the two most dangerous cases
> without pretending to have footage I could not ethically obtain."

## 1c. If your footage is top-down (drone, high mast, night)

Run it with the overhead detector:

```bash
python -m src.crowdguard.main --video data/videos/drone.mp4 --view head --velocity flow
```

`--view auto` is the default and switches by itself: it runs person detection, and after five
consecutive frames with zero detections it concludes the view is overhead and switches. An
empty scene and a scene full of heads look identical to a COCO person model, so the streak is
the only reliable signal.

Two things change automatically in head mode:

- **Ground reference point.** For a side view the feet are the ground contact, so the box
  bottom is right. Straight down, the head is directly above the feet, so the box *centre* is
  right — using the bottom would displace every person by half a head, a real metric error at
  drone altitude.
- **Velocity source switches to dense optical flow.** Associating 10–25 px head blobs between
  frames produces ID switches, and an ID switch fabricates a velocity pointing at another
  person — which reads downstream as counter-flow. Optical flow needs no association at all,
  and it removes the drone's own motion by taking the median flow as the platform component.

**Honesty about the untrained blob detector.** With no trained model present, head mode uses a
scale-space Laplacian-of-Gaussian detector. On a synthetic overhead night crowd it reaches
recall 1.00 and precision 0.99 where YOLO reaches zero — but that scene contains clean circular
blobs. Real heads carry hats, hoods, umbrellas, backpacks and shadows, and the blob detector
will fire on anything round and bright. **Treat it as a working stopgap that proves the
architecture, not as a result.** Train a real model before quoting any number from it.

### Triage whatever you download

Drop everything into `data/videos/` and run:

```bash
python scripts/survey_videos.py
```

It reports, per clip: people detected, density range, which failure modes fired, and a verdict
(useful / too sparse / detector finds nobody). At the end it prints which of the eight cases
your collection still does not cover, and the simulator command for each gap.

---

## 2. `data/mot20/` — MOT20 (the highest-value addition)

MOT20 is the standard dense-crowd benchmark: real crowds, per-frame ground-truth boxes for every
person. It gives you a **real counting MAE** — the single most credible number you can add to the
report, because it is measured against annotations you did not create.

**Download:** <https://motchallenge.net/data/MOT20/> (~5 GB, free, registration-free)

**Unzip so it looks like this:**

```
data/mot20/
└── train/
    ├── MOT20-01/
    │   ├── img1/            000001.jpg, 000002.jpg, ...
    │   ├── gt/gt.txt        the ground-truth annotations
    │   └── seqinfo.ini
    ├── MOT20-02/
    ├── MOT20-03/
    └── MOT20-05/
```

The loader reads `gt/gt.txt` directly. Only the `train` split has ground truth, which is the
split you want.

---

## 3. `data/shanghaitech/` — ShanghaiTech (smaller alternative)

If 5 GB is too much, ShanghaiTech Part B is ~300 MB and gives you real counting error on still
images rather than video.

**Download:** search "ShanghaiTech Crowd Counting Dataset" (Kaggle mirrors are easiest)

**Unzip so it looks like this:**

```
data/shanghaitech/
└── part_B_final/
    └── test_data/
        ├── images/          IMG_1.jpg, IMG_2.jpg, ...
        └── ground_truth/    GT_IMG_1.mat, GT_IMG_2.mat, ...
```

Part A also works. The loader reads the point annotations out of the `.mat` files and counts them.

---

## 4. `data/annotations/` — your own ground truth

For any video you annotate yourself, or for camera calibration.

### Counting ground truth — `counts.csv`

One row per annotated frame. You do not need every frame; twenty or thirty spread across the
video is enough for a meaningful error estimate.

```csv
video,frame_id,true_count
gate_camera.mp4,0,34
gate_camera.mp4,150,51
gate_camera.mp4,300,78
```

### Expert risk ratings — `expert_ratings.csv`

Ask three to five people to independently watch short clips and rate them. Comparing their
ratings with the system using Cohen's kappa is the closest thing to validating the fused risk
score that exists, since no labelled crowd-risk dataset does.

```csv
video,start_sec,end_sec,rater,rating
gate_camera.mp4,0,10,rater1,low
gate_camera.mp4,0,10,rater2,low
gate_camera.mp4,60,70,rater1,high
gate_camera.mp4,60,70,rater2,moderate
```

### Camera calibration — `calibration_<name>.json`

Four points of a known ground rectangle. Without this, persons/m² is not a physical quantity and
cannot be compared against the published safety thresholds.

```json
{
  "image_points": [[120, 700], [1150, 700], [830, 300], [420, 300]],
  "world_points": [[0, 0], [12, 0], [12, 20], [0, 20]]
}
```

`image_points` are pixel coordinates of the four corners in the frame; `world_points` are the
same four corners in metres on the ground. Use paving slabs, pitch markings, barrier spacing,
parking bays — anything whose real size you can measure. Full procedure is in
`knowledge_base/10_camera_zones_and_calibration.txt`.

Then:

```bash
python -m src.crowdguard.main --video data/videos/gate.mp4 --calibration data/annotations/calibration_gate.json
```

---

## What each addition buys you

| You add | You gain | Effort |
|---|---|---|
| One real crowd video | A demo where YOLO visibly detects people | 5 minutes |
| MOT20 train split | **Real counting MAE against annotations you did not make** | 30 minutes |
| ShanghaiTech Part B | Real counting error, smaller download | 15 minutes |
| `counts.csv` on your own video | Counting error on footage from your actual venue | 1 hour |
| `expert_ratings.csv` | Cohen's kappa — the only honest check on the fused risk score | 2 hours |
| A calibration file | Densities become physically meaningful persons/m² | 20 minutes |
