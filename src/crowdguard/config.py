"""Configuration dataclasses for the CrowdGuard-RAG system.

Every tunable constant lives here so that the viva/report can point at a single
source of truth, and so thresholds can be justified against published crowd
science rather than being hidden magic numbers in the code.

Threshold provenance
--------------------
* Density bands follow Fruin's pedestrian Level-of-Service work (1971, 1993):
  movement becomes constrained around 3-4 persons/m2, involuntary contact and
  shock-wave propagation appear above ~5 persons/m2.
* Crowd pressure uses the DEFINITION from Helbing, Johansson & Al-Abideen
  (2007), "Dynamics of crowd disasters: An empirical study", Phys. Rev. E 75,
  046109: crowd pressure = local density x local velocity variance, and its
  rise is the published precursor of "crowd turbulence".

  The THRESHOLD, however, is not transferable. Helbing reports turbulence onset
  near 0.02 s^-2 for a velocity field estimated by optical flow over a smoothed
  spatial grid. This system estimates velocity from discrete tracked centroids
  at a much lower sample rate, which yields a variance -- and therefore a
  pressure -- on a completely different numerical scale. Quoting 0.02 here
  would be citation theatre: the right number with the wrong estimator.

  So the defaults below are calibrated against this estimator's own observed
  distribution, and `scripts/evaluate.py --calibrate-pressure` reports the
  percentiles needed to re-calibrate them for any new site or frame rate.
  The claim this system makes is that the *quantity* is Helbing's; the
  threshold is empirical and site-specific, as it must be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Vision / tracking
# --------------------------------------------------------------------------- #
@dataclass
class VisionConfig:
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.30
    person_class_id: int = 0
    use_hog_fallback: bool = True

    # View mode. COCO person detection assumes an upright, side-on body and
    # returns literally nothing from a camera looking straight down -- measured
    # at 0 detections out of 69 on an overhead night scene, unchanged at
    # confidence 0.03. Overhead is the deployment view for a drone, so it needs
    # its own detector rather than a lower threshold.
    #   person : COCO person detection (side/oblique views)
    #   head   : top-down head detection
    #   auto   : person detection, switching to head detection when it finds
    #            nothing for several consecutive frames
    mode: str = "auto"
    # Path to a fine-tuned head/aerial model (CrowdHuman hbox, SCUT-HEAD,
    # CroHD, or VisDrone). When present it is used instead of the untrained
    # blob detector, and it will be markedly better on real footage.
    head_model_path: str = "models/yolov8n-head.pt"
    # Consecutive empty person-detection frames before `auto` switches over.
    auto_switch_after: int = 5
    # Occlusion compensation: YOLO systematically undercounts in dense crowds.
    # A multiplicative correction estimated from the detection overlap ratio.
    occlusion_compensation: bool = True
    occlusion_max_gain: float = 1.6


@dataclass
class TrackerConfig:
    max_disappeared: int = 30
    max_distance: float = 80.0
    history_size: int = 20
    # Number of history points used for velocity; >2 smooths detector jitter,
    # which otherwise inflates flow-disorder through ID noise.
    velocity_window: int = 3


# --------------------------------------------------------------------------- #
# Ground-plane calibration
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationConfig:
    """Maps image pixels to ground-plane metres via a planar homography.

    Without this, "persons per square metre" is not physically meaningful: a
    person 40 m from the camera occupies a fraction of the pixels of a person
    5 m away, so uniform pixel->metre scaling under-counts far-field density.

    Supply four image points (pixels) and the four corresponding real-world
    points (metres) of any known ground rectangle -- a penalty box, a paving
    grid, a barrier spacing, a marked walkway.
    """

    enabled: bool = False
    image_points: List[Tuple[float, float]] = field(default_factory=list)
    world_points: List[Tuple[float, float]] = field(default_factory=list)
    # Used when calibration is disabled: uniform scale derived from the
    # operator-declared monitored area. Documented as an approximation.
    fallback_area_m2: float = 120.0


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #
@dataclass
class ZoneConfig:
    """Named ground zones. Risk is reported per zone, not per whole frame.

    Each zone is {"name", "kind", "polygon" (normalised 0-1 image coords),
    "capacity_pm2"}. `kind` drives which SOPs get retrieved.
    """

    zones: List[Dict] = field(default_factory=list)
    enabled: bool = True

    @staticmethod
    def default_layout() -> List[Dict]:
        return [
            {
                "name": "Left Gate",
                "kind": "gate",
                "polygon": [[0.00, 0.15], [0.32, 0.15], [0.32, 1.00], [0.00, 1.00]],
                "capacity_pm2": 4.0,
                "alternate": "Right Gate", "meter_at": "Left Gate approach",
                "staging": "Left Gate outer barrier",
            },
            {
                "name": "Central Corridor",
                "kind": "corridor",
                "polygon": [[0.32, 0.15], [0.68, 0.15], [0.68, 1.00], [0.32, 1.00]],
                "capacity_pm2": 5.0,
                "alternate": "Left Gate", "meter_at": "Central Corridor entry",
                "staging": "Corridor midpoint",
            },
            {
                "name": "Right Gate",
                "kind": "gate",
                "polygon": [[0.68, 0.15], [1.00, 0.15], [1.00, 1.00], [0.68, 1.00]],
                "capacity_pm2": 4.0,
                "alternate": "Left Gate", "meter_at": "Right Gate approach",
                "staging": "Right Gate outer barrier",
            },
        ]


# --------------------------------------------------------------------------- #
# Risk scoring
# --------------------------------------------------------------------------- #
@dataclass
class RiskWeights:
    """Weights of the fused risk score. Sum is normalised at use time.

    Local density and crowd pressure dominate because the crowd-safety
    literature identifies them as the causal variables; speed alone is the
    weakest predictor and is weighted accordingly.
    """

    local_density: float = 0.30
    crowd_pressure: float = 0.24
    flow_disorder: float = 0.18
    bottleneck: float = 0.12
    oscillation: float = 0.10
    speed: float = 0.06

    def as_dict(self) -> Dict[str, float]:
        return {
            "local_density": self.local_density,
            "crowd_pressure": self.crowd_pressure,
            "flow_disorder": self.flow_disorder,
            "bottleneck": self.bottleneck,
            "oscillation": self.oscillation,
            "speed": self.speed,
        }


@dataclass
class RiskConfig:
    # -- legacy / global ----------------------------------------------------
    camera_area_m2: float = 120.0
    low_threshold: float = 0.35
    high_threshold: float = 0.65
    alert_threshold: float = 0.55

    # -- Fruin density bands (persons per square metre) ---------------------
    density_soft_limit: float = 3.0
    density_hard_limit: float = 5.0

    # -- legacy soft limits kept for backwards compatibility ---------------
    speed_soft_limit: float = 0.20
    disorder_soft_limit: float = 0.45
    bottleneck_soft_limit: float = 0.55

    # -- local density estimation ------------------------------------------
    # k-nearest-neighbour density estimator: rho_i = k / (pi * r_k^2)
    knn_k: int = 4
    # Robust peak instead of the true max, so one detector artefact cannot
    # drive the whole system into alarm.
    density_percentile: float = 90.0

    # -- Helbing crowd pressure --------------------------------------------
    neighbourhood_radius_m: float = 1.5
    # Calibrated against the tracked-centroid estimator (see module docstring).
    # Re-derive per site with: scripts/evaluate.py --calibrate-pressure
    pressure_warning: float = 1.20
    pressure_critical: float = 2.60

    # -- temporal analysis --------------------------------------------------
    # Tracks younger than this contribute no velocity statistics.
    min_track_age: int = 3
    # Samples retained for trend and oscillation analysis. Three seconds is far
    # too short to establish that a crowd is filling: the slope is then
    # dominated by detector churn rather than by people arriving.
    temporal_window: int = 60
    oscillation_soft_limit: float = 0.30
    # Relative amplitude at which a stop-go swing counts as fully developed.
    oscillation_amplitude_scale: float = 0.18
    speed_surge_ratio: float = 1.8     # panic-dispersal trigger vs baseline

    weights: RiskWeights = field(default_factory=RiskWeights)


# --------------------------------------------------------------------------- #
# Escalation state machine
# --------------------------------------------------------------------------- #
@dataclass
class EscalationConfig:
    """Hysteresis + dwell time, so the system does not flap.

    Alarm flapping is the single most common reason operators mute a safety
    system, so raising a state requires the score to be sustained, and
    dropping a state requires a lower score sustained for longer.
    """

    watch_raise: float = 0.35
    alert_raise: float = 0.55
    critical_raise: float = 0.72

    # Hysteresis margin: de-escalation happens this far below the raise level.
    hysteresis: float = 0.08

    # Seconds the condition must hold before escalating into each state.
    dwell_watch_sec: float = 5.0
    dwell_alert_sec: float = 8.0
    dwell_critical_sec: float = 4.0     # critical escalates fastest by design

    # Seconds the lower condition must hold before de-escalating.
    dwell_deescalate_sec: float = 20.0

    # Re-notify interval while a state persists unacknowledged.
    renotify_sec: float = 60.0

    # Suppress re-notification once an operator acknowledges, for this long.
    ack_suppression_sec: float = 300.0


# --------------------------------------------------------------------------- #
# Alerting / notification
# --------------------------------------------------------------------------- #
@dataclass
class AlertConfig:
    """Outbound notification. Everything is OFF by default.

    No message leaves the machine unless the operator explicitly enables a
    sink and supplies its credentials at runtime.
    """

    console: bool = True
    jsonl_path: Optional[str] = "outputs/alerts.jsonl"

    webhook_enabled: bool = False
    webhook_url: str = ""

    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    min_state: str = "ALERT"           # NORMAL | WATCH | ALERT | CRITICAL
    attach_frame: bool = True
    timeout_sec: float = 6.0
    ack_log_path: str = "outputs/acknowledgements.jsonl"


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
@dataclass
class ForecastConfig:
    """Predict risk at t+H instead of only classifying the present frame."""

    enabled: bool = True
    horizon_sec: float = 30.0
    # "auto" uses the trained checkpoint when one exists and falls back to the
    # trend heuristic otherwise. Measured on unseen seeds, the trained model
    # beats a persistence baseline by 32% while the trend heuristic is 27%
    # WORSE than persistence, so the choice matters and should not be a guess.
    backend: str = "auto"              # auto | trend | transformer
    min_history: int = 6
    window: int = 30
    checkpoint: str = "outputs/risk_forecaster.pt"
    # Confidence band width in residual standard deviations.
    band_sigma: float = 1.5


# --------------------------------------------------------------------------- #
# Retrieval / generation
# --------------------------------------------------------------------------- #
@dataclass
class RAGConfig:
    knowledge_base_dir: str = "knowledge_base"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_words: int = 140
    chunk_overlap_words: int = 25
    top_k: int = 4
    use_faiss: bool = True


@dataclass
class LLMConfig:
    provider: str = "fallback"         # fallback | openai | local_hf
    model_name: str = "gpt-4o-mini"
    max_context_words: int = 450
    temperature: float = 0.2
    openai_api_key_env: str = "OPENAI_API_KEY"
    local_model_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Master config
# --------------------------------------------------------------------------- #
@dataclass
class SystemConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
