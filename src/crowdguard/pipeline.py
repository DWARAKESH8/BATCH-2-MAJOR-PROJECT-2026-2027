"""The single processing pipeline shared by the CLI, the simulator and the UI.

Everything that turns a frame into a decision lives here exactly once. The
dashboard used to reimplement the loop separately from the CLI, which meant the
demo and the reported results could silently diverge -- a real problem when the
numbers end up in a report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .advisor import Advisory, CrowdSafetyAdvisor
from .alerting import AckStore, AlertDispatcher
from .config import (
    AlertConfig,
    CalibrationConfig,
    EscalationConfig,
    ForecastConfig,
    LLMConfig,
    RAGConfig,
    RiskConfig,
    TrackerConfig,
    VisionConfig,
    ZoneConfig,
)
from .escalation import EscalationController, EscalationEvent, EscalationStatus
from .flow import FlowConfig, OpticalFlowField
from .forecast import Forecast, RiskForecaster
from .rag_engine import RAGIndex, RetrievedChunk
from .risk_engine import RiskEngine, SceneFeatures
from .risk_taxonomy import RISK_TYPES
from .tracker import CentroidTracker
from .utils import (
    draw_overlay,
    draw_trajectories_and_flow,
    draw_zones,
    frame_to_png_bytes,
    generate_density_heatmap,
)
from .vision import PersonDetector, boxes_to_centroids, occlusion_estimate
from .zones import ZoneManager


@dataclass
class PipelineResult:
    """Everything produced for one processed frame."""

    features: SceneFeatures
    forecast: Forecast
    status: EscalationStatus
    occlusion: Dict[str, Any]
    detections: List[Dict[str, Any]]
    tracks: Dict[int, Tuple[int, int]]
    velocities: Dict[int, Tuple[float, float]]
    chunks: List[RetrievedChunk] = field(default_factory=list)
    advisory: Optional[Advisory] = None
    annotated: Optional[np.ndarray] = None
    stage_ms: Dict[str, float] = field(default_factory=dict)
    ground_truth: Optional[Dict[str, Any]] = None

    @property
    def events(self) -> List[EscalationEvent]:
        return self.status.events

    def log_row(self, rag_backend: str = "") -> Dict[str, Any]:
        row = self.features.to_dict()
        row.update(
            {
                "escalation_state": self.status.state,
                "escalation": self.status.to_dict(),
                "forecast": self.forecast.to_dict(),
                "occlusion": self.occlusion,
                "rag_backend": rag_backend,
                "retrieved_sources": [
                    {"source": c.source, "score": round(c.score, 4), "text": c.text[:240]}
                    for c in self.chunks
                ],
                "advisory": self.advisory.to_dict() if self.advisory else None,
                "stage_ms": self.stage_ms,
            }
        )
        if self.ground_truth:
            row["ground_truth"] = self.ground_truth
        return row


@dataclass
class OverlayOptions:
    boxes: bool = True
    ids: bool = True
    heatmap: bool = False
    vectors: bool = True
    zones: bool = True
    hud: bool = True


class CrowdGuardPipeline:
    """Detection -> tracking -> risk -> typology -> forecast -> escalation -> advice."""

    def __init__(
        self,
        vision: Optional[VisionConfig] = None,
        tracker: Optional[TrackerConfig] = None,
        risk: Optional[RiskConfig] = None,
        calibration: Optional[CalibrationConfig] = None,
        zones: Optional[ZoneConfig] = None,
        escalation: Optional[EscalationConfig] = None,
        forecast: Optional[ForecastConfig] = None,
        alerts: Optional[AlertConfig] = None,
        rag: Optional[RAGConfig] = None,
        llm: Optional[LLMConfig] = None,
        load_detector: bool = True,
        load_rag: bool = True,
        velocity_source: str = "auto",
    ):
        self.risk_config = risk or RiskConfig()
        self.escalation_config = escalation or EscalationConfig()
        self.alert_config = alerts or AlertConfig()

        self.detector = PersonDetector(vision or VisionConfig()) if load_detector else None
        self.tracker = CentroidTracker(tracker or TrackerConfig())
        # Velocity source. Centroid association is unreliable for 10-25 px
        # overhead heads -- an ID switch fabricates a velocity pointing at
        # another person, which reads downstream as counter-flow. Dense optical
        # flow needs no association, so `auto` switches to it whenever the
        # detector is working in head mode.
        self.velocity_source = velocity_source
        self.flow = OpticalFlowField(FlowConfig())
        self.zone_manager = ZoneManager(zones or ZoneConfig(zones=ZoneConfig.default_layout()))
        self.risk_engine = RiskEngine(
            self.risk_config,
            calibration or CalibrationConfig(enabled=False, fallback_area_m2=self.risk_config.camera_area_m2),
            self.zone_manager,
        )
        self.escalation = EscalationController(self.escalation_config)
        self.forecaster = RiskForecaster(
            forecast or ForecastConfig(),
            alert_threshold=self.escalation_config.alert_raise,
            critical_threshold=self.escalation_config.critical_raise,
            low_threshold=self.escalation_config.watch_raise,
        )
        self.dispatcher = AlertDispatcher(self.alert_config)
        self.acks = AckStore(self.alert_config.ack_log_path)

        self.rag: Optional[RAGIndex] = None
        self.rag_error = ""
        if load_rag:
            rag_config = rag or RAGConfig()
            try:
                self.rag = RAGIndex(rag_config)
                self.rag.load_documents(rag_config.knowledge_base_dir)
                self.rag.build()
            except Exception as exc:
                self.rag = None
                self.rag_error = f"{type(exc).__name__}: {exc}"

        self.advisor = CrowdSafetyAdvisor(llm or LLMConfig())
        self.stage_totals: Dict[str, float] = {
            "detect": 0.0, "track": 0.0, "risk": 0.0, "rag": 0.0, "advise": 0.0, "draw": 0.0
        }
        self.processed = 0

    # ------------------------------------------------------------------ #
    @property
    def backends(self) -> Dict[str, str]:
        return {
            "detector": (f"{self.detector.backend}/{self.detector.view_mode}"
                         + (f" [{self.detector.head_backend}]"
                            if self.detector.view_mode == "head" else "")
                         if self.detector else "supplied"),
            "rag": self.rag.backend if self.rag else f"unavailable ({self.rag_error})",
            "advisor": self.advisor.config.provider,
            "forecast": self.forecaster.backend,
            "calibration": "homography" if (self.risk_engine.ground and self.risk_engine.ground.calibrated) else "uniform-scale",
            "velocity": ("optical-flow"
                         if (self.velocity_source == "flow"
                             or (self.velocity_source == "auto" and self.detector
                                 and self.detector.view_mode == "head"))
                         else "centroid-tracking"),
        }

    def reset(self) -> None:
        self.risk_engine.reset_history()
        self.forecaster.reset()
        self.escalation = EscalationController(self.escalation_config)
        self.tracker = CentroidTracker(self.tracker.config)
        self.processed = 0
        for k in self.stage_totals:
            self.stage_totals[k] = 0.0

    # ------------------------------------------------------------------ #
    def process(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_sec: float,
        dt_sec: float,
        detections: Optional[List[Dict[str, Any]]] = None,
        overlays: Optional[OverlayOptions] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        want_annotated: bool = True,
        dispatch_alerts: bool = True,
    ) -> PipelineResult:
        overlays = overlays or OverlayOptions()
        stage: Dict[str, float] = {}

        # --- detect -------------------------------------------------------
        t0 = time.time()
        if detections is None:
            if self.detector is None:
                raise RuntimeError("No detector loaded and no detections supplied")
            detections = self.detector.detect(frame)
        stage["detect"] = (time.time() - t0) * 1000

        # --- track --------------------------------------------------------
        t0 = time.time()
        centroids = boxes_to_centroids(detections)
        tracks = self.tracker.update(centroids)
        active = self.tracker.active_objects()

        overhead = bool(self.detector and self.detector.view_mode == "head")
        use_flow = self.velocity_source == "flow" or (
            self.velocity_source == "auto" and overhead)

        if use_flow:
            self.flow.update(frame)
            velocities = self.flow.velocity_vectors(active)
            displacements = self.flow.net_displacements(active)
        else:
            velocities = self.tracker.velocity_vectors()
            displacements = self.tracker.net_displacements()
        stage["track"] = (time.time() - t0) * 1000

        # --- measure ------------------------------------------------------
        t0 = time.time()
        features = self.risk_engine.compute_features(
            frame_id, timestamp_sec, frame.shape, detections, tracks, velocities,
            dt_sec=dt_sec, active_tracks=active,
            # Flow velocities are valid the moment they exist, so the
            # track-maturity gate that protects against association noise is
            # not needed -- and would suppress most of the crowd overhead,
            # where tracks are short-lived by nature.
            track_ages=(None if use_flow else self.tracker.track_ages()),
            displacements=displacements,
        )
        occlusion = occlusion_estimate(detections, frame.shape)
        forecast = self.forecaster.update(features, timestamp_sec)
        stage["risk"] = (time.time() - t0) * 1000

        status = self.escalation.update(features, timestamp_sec)

        # --- retrieve + advise -------------------------------------------
        chunks: List[RetrievedChunk] = []
        advisory: Optional[Advisory] = None
        if status.state != "NORMAL" or features.risk_level != "low":
            t0 = time.time()
            if self.rag is not None:
                query = self.risk_engine.scene_query(
                    features, RISK_TYPES[features.primary_risk_type].sop_terms
                )
                try:
                    chunks = self.rag.search(query)
                except Exception:
                    chunks = []
            stage["rag"] = (time.time() - t0) * 1000

            t0 = time.time()
            advisory = self.advisor.generate(features, chunks, status.state, forecast)
            stage["advise"] = (time.time() - t0) * 1000

        # --- alert --------------------------------------------------------
        if dispatch_alerts:
            for event in status.events:
                event.advisory = advisory.to_markdown() if advisory else ""
                png = frame_to_png_bytes(frame) if self.dispatcher.config.attach_frame else None
                self.dispatcher.dispatch(
                    event,
                    action=(advisory.actions[0] if advisory and advisory.actions else ""),
                    forecast_text=forecast.headline(),
                    advisory=advisory.to_markdown() if advisory else "",
                    frame_png=png,
                    extra={"occlusion": occlusion, "forecast": forecast.to_dict()},
                )

        # --- draw ---------------------------------------------------------
        annotated = None
        if want_annotated:
            t0 = time.time()
            annotated = frame
            if overlays.heatmap:
                annotated = generate_density_heatmap(annotated, centroids)
            if overlays.zones:
                annotated = draw_zones(annotated, self.zone_manager, features.zones)
            if overlays.vectors:
                annotated = draw_trajectories_and_flow(annotated, tracks, velocities, self.tracker.trails())
            if overlays.hud:
                annotated = draw_overlay(
                    annotated, detections, tracks, features, status.state, forecast,
                    show_boxes=overlays.boxes, show_ids=overlays.ids,
                )
            elif overlays.boxes or overlays.ids:
                annotated = draw_overlay(
                    annotated, detections, tracks, features, status.state, forecast,
                    show_boxes=overlays.boxes, show_ids=overlays.ids,
                )
            stage["draw"] = (time.time() - t0) * 1000

        for k, v in stage.items():
            self.stage_totals[k] = self.stage_totals.get(k, 0.0) + v
        self.processed += 1

        return PipelineResult(
            features=features,
            forecast=forecast,
            status=status,
            occlusion=occlusion,
            detections=detections,
            tracks=tracks,
            velocities=velocities,
            chunks=chunks,
            advisory=advisory,
            annotated=annotated,
            stage_ms={k: round(v, 2) for k, v in stage.items()},
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------ #
    def acknowledge(
        self,
        event_id: str,
        operator: str,
        decision: str,
        action_taken: str,
        note: str,
        timestamp_sec: float,
    ):
        from .alerting import Acknowledgement

        event = self.escalation.acknowledge(event_id, operator, action_taken, timestamp_sec)
        ack = Acknowledgement(
            event_id=event_id,
            operator=operator,
            decision=decision,
            action_taken=action_taken,
            note=note,
            timestamp_sec=timestamp_sec,
            risk_score=event.risk_score if event else 0.0,
            risk_type=event.risk_type if event else "",
            zone=event.zone if event else "",
        )
        return self.acks.add(ack)

    def runtime_report(self) -> Dict[str, Any]:
        n = max(1, self.processed)
        per_frame = {k: round(v / n, 2) for k, v in self.stage_totals.items()}
        total_ms = sum(per_frame.values())
        return {
            "frames": self.processed,
            "stage_ms_per_frame": per_frame,
            "total_ms_per_frame": round(total_ms, 2),
            "throughput_fps": round(1000.0 / total_ms, 2) if total_ms > 0 else None,
            "backends": self.backends,
        }
