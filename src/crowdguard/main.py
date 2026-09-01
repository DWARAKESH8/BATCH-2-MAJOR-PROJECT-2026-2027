"""CLI entry point: video or simulated feed in, measured risk and advisories out."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .config import (
    AlertConfig,
    CalibrationConfig,
    ForecastConfig,
    LLMConfig,
    RAGConfig,
    RiskConfig,
    VisionConfig,
    ZoneConfig,
)
from .pipeline import CrowdGuardPipeline, OverlayOptions
from .utils import append_jsonl, generate_incident_report, open_video_writer


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_zone_layout(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Zone layout not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("zones", data) if isinstance(data, dict) else data


def load_calibration(path: Optional[str], area_m2: float) -> CalibrationConfig:
    if not path:
        return CalibrationConfig(enabled=False, fallback_area_m2=area_m2)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return CalibrationConfig(
        enabled=True,
        image_points=[tuple(pt) for pt in data["image_points"]],
        world_points=[tuple(pt) for pt in data["world_points"]],
        fallback_area_m2=area_m2,
    )


def build_pipeline(
    model_path: str = "yolov8n.pt",
    area_m2: float = 120.0,
    knowledge_base_dir: str = "knowledge_base",
    llm_provider: str = "fallback",
    zone_layout: Optional[List[Dict[str, Any]]] = None,
    calibration: Optional[CalibrationConfig] = None,
    forecast_horizon: float = 30.0,
    forecast_backend: str = "auto",
    alert_config: Optional[AlertConfig] = None,
    load_detector: bool = True,
    view_mode: str = "auto",
    velocity_source: str = "auto",
) -> CrowdGuardPipeline:
    return CrowdGuardPipeline(
        vision=VisionConfig(model_path=model_path, mode=view_mode),
        risk=RiskConfig(camera_area_m2=area_m2),
        calibration=calibration or CalibrationConfig(enabled=False, fallback_area_m2=area_m2),
        zones=ZoneConfig(zones=zone_layout or ZoneConfig.default_layout()),
        forecast=ForecastConfig(horizon_sec=forecast_horizon, backend=forecast_backend),
        alerts=alert_config or AlertConfig(),
        rag=RAGConfig(knowledge_base_dir=knowledge_base_dir),
        llm=LLMConfig(provider=llm_provider),
        load_detector=load_detector,
        velocity_source=velocity_source,
    )


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def _finish(
    pipeline: CrowdGuardPipeline,
    rows: List[Dict[str, Any]],
    last_advisory,
    elapsed: float,
    outputs: Dict[str, Optional[str]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "processed_frames": pipeline.processed,
        "elapsed_sec": round(elapsed, 2),
        "fps_processed": round(pipeline.processed / max(1e-6, elapsed), 2),
        "runtime": pipeline.runtime_report(),
        "escalation": pipeline.escalation.summary(),
        "forecast_accuracy": pipeline.forecaster.accuracy_report(),
        "alert_sinks": pipeline.dispatcher.sink_names,
        "alert_failures": pipeline.dispatcher.failures(),
        **{k: v for k, v in outputs.items() if v},
    }
    result.update(extra or {})
    result.update(pipeline.backends)

    report_path = outputs.get("output_report")
    if report_path:
        report = generate_incident_report(
            rows,
            advisory=last_advisory,
            escalation_summary=pipeline.escalation.summary(),
            events=[e.to_dict() for e in pipeline.escalation.events],
            forecast_report=pipeline.forecaster.accuracy_report(),
            ack_stats=pipeline.acks.stats(),
            run_meta={
                "detector_backend": pipeline.backends["detector"],
                "rag_backend": pipeline.backends["rag"],
                "llm_provider": pipeline.backends["advisor"],
                "forecast_backend": pipeline.backends["forecast"],
            },
        )
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(report, encoding="utf-8")
    return result


def process_video(
    video_path: Any,
    model_path: str = "yolov8n.pt",
    area_m2: float = 120.0,
    output_video: Optional[str] = "outputs/annotated_crowdguard.mp4",
    output_json: Optional[str] = "outputs/risk_log.jsonl",
    output_events: Optional[str] = "outputs/escalation_events.jsonl",
    output_report: Optional[str] = "outputs/incident_report.md",
    knowledge_base_dir: str = "knowledge_base",
    sample_every: int = 1,
    llm_provider: str = "fallback",
    max_frames: Optional[int] = None,
    zone_layout: Optional[List[Dict[str, Any]]] = None,
    calibration: Optional[CalibrationConfig] = None,
    forecast_horizon: float = 30.0,
    forecast_backend: str = "auto",
    alert_config: Optional[AlertConfig] = None,
    show_heatmap: bool = False,
    verbose: bool = True,
    view_mode: str = "auto",
    velocity_source: str = "auto",
) -> Dict[str, Any]:
    pipeline = build_pipeline(
        model_path, area_m2, knowledge_base_dir, llm_provider, zone_layout,
        calibration, forecast_horizon, forecast_backend, alert_config, load_detector=True,
        view_mode=view_mode, velocity_source=velocity_source,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, sample_every)
    dt_sec = step / max(1e-6, fps)

    writer = open_video_writer(output_video, fps / step, width, height) if output_video else None
    for path in (output_json, output_events):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("", encoding="utf-8")

    overlays = OverlayOptions(heatmap=show_heatmap)
    rows: List[Dict[str, Any]] = []
    last_advisory = None
    frame_id = -1
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % step != 0:
            continue
        if max_frames is not None and pipeline.processed >= max_frames:
            break

        result = pipeline.process(
            frame, frame_id, frame_id / fps, dt_sec,
            overlays=overlays, want_annotated=writer is not None,
        )
        if result.advisory:
            last_advisory = result.advisory

        row = result.log_row(pipeline.backends["rag"])
        rows.append(row)
        if output_json:
            append_jsonl(output_json, row)
        for event in result.events:
            if output_events:
                append_jsonl(output_events, event.to_dict())

        if writer and result.annotated is not None:
            writer.write(result.annotated)

        if verbose and pipeline.processed % 25 == 0:
            f = result.features
            print(f"  frame {frame_id:6d} t={f.timestamp_sec:7.1f}s  {result.status.state:8s} "
                  f"risk={f.risk_score:.2f}  {f.primary_risk_label:24s} "
                  f"rho={f.local_density_peak:.2f}/m2  n={f.person_count}", flush=True)

    cap.release()
    if writer:
        writer.release()

    return _finish(
        pipeline, rows, last_advisory, time.time() - start,
        {"output_video": output_video, "output_json": output_json,
         "output_events": output_events, "output_report": output_report},
        {"source": str(video_path), "simulated": False},
    )


def process_simulation(
    scenario: str = "escalating_crush",
    fps: float = 10.0,
    seed: int = 7,
    max_frames: Optional[int] = None,
    output_video: Optional[str] = "outputs/simulated_crowdguard.mp4",
    output_json: Optional[str] = "outputs/risk_log.jsonl",
    output_events: Optional[str] = "outputs/escalation_events.jsonl",
    output_report: Optional[str] = "outputs/incident_report.md",
    knowledge_base_dir: str = "knowledge_base",
    llm_provider: str = "fallback",
    forecast_horizon: float = 30.0,
    forecast_backend: str = "auto",
    alert_config: Optional[AlertConfig] = None,
    calibrated: bool = True,
    show_heatmap: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the pipeline against the built-in crowd simulator.

    Detection is bypassed: the simulator supplies detections directly. That is
    the point -- it isolates the measurement, classification, forecasting and
    escalation layers and tests them against known ground truth, with no
    detector error confounding the result.
    """
    from .simulator import SCENARIOS, CrowdSimulator, WORLD_D, WORLD_W

    sim = CrowdSimulator(scenario, fps=fps, seed=seed)
    calib_payload = CrowdSimulator.calibration_payload()
    calibration = (
        CalibrationConfig(
            enabled=True,
            image_points=[tuple(p) for p in calib_payload["image_points"]],
            world_points=[tuple(p) for p in calib_payload["world_points"]],
            fallback_area_m2=WORLD_W * WORLD_D,
        )
        if calibrated
        else CalibrationConfig(enabled=False, fallback_area_m2=WORLD_W * WORLD_D)
    )

    pipeline = build_pipeline(
        area_m2=WORLD_W * WORLD_D,
        knowledge_base_dir=knowledge_base_dir,
        llm_provider=llm_provider,
        zone_layout=CrowdSimulator.zone_layout(),
        calibration=calibration,
        forecast_horizon=forecast_horizon,
        forecast_backend=forecast_backend,
        alert_config=alert_config,
        load_detector=False,
    )

    for path in (output_json, output_events):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("", encoding="utf-8")

    writer = None
    overlays = OverlayOptions(heatmap=show_heatmap)
    rows: List[Dict[str, Any]] = []
    last_advisory = None
    total = max_frames or sim.total_frames
    start = time.time()

    for sf in sim.run(total):
        if writer is None and output_video:
            writer = open_video_writer(output_video, fps, sf.frame.shape[1], sf.frame.shape[0])

        result = pipeline.process(
            sf.frame, sf.frame_id, sf.timestamp_sec, 1.0 / fps,
            detections=sf.detections, overlays=overlays,
            ground_truth=sf.ground_truth(), want_annotated=writer is not None,
        )
        if result.advisory:
            last_advisory = result.advisory

        row = result.log_row(pipeline.backends["rag"])
        rows.append(row)
        if output_json:
            append_jsonl(output_json, row)
        for event in result.events:
            if output_events:
                append_jsonl(output_events, event.to_dict())

        if writer and result.annotated is not None:
            writer.write(result.annotated)

        if verbose and pipeline.processed % 40 == 0:
            f = result.features
            print(f"  t={f.timestamp_sec:6.1f}s [{sf.phase:11s}] {result.status.state:8s} "
                  f"risk={f.risk_score:.2f} {f.primary_risk_label:24s} "
                  f"rho={f.local_density_peak:.2f} (true {sf.true_density_peak:.2f}) "
                  f"n={f.person_count} (true {sf.true_count})", flush=True)

    if writer:
        writer.release()

    return _finish(
        pipeline, rows, last_advisory, time.time() - start,
        {"output_video": output_video, "output_json": output_json,
         "output_events": output_events, "output_report": output_report},
        {
            "source": f"simulation:{scenario}",
            "simulated": True,
            "scenario": SCENARIOS[scenario].label,
            "expected_risk_type": SCENARIOS[scenario].expected_type,
        },
    )


# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    from .simulator import SCENARIOS

    p = argparse.ArgumentParser(description="CrowdGuard-RAG processing pipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Input video path, camera index, or stream URL")
    src.add_argument("--simulate", choices=sorted(SCENARIOS), help="Run a built-in crowd scenario")

    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights path")
    p.add_argument("--view", default="auto", choices=["auto", "person", "head"],
                   help="Camera view. 'head' for top-down drone or overhead footage, where "
                        "COCO person detection returns nothing at all. 'auto' starts with "
                        "person detection and switches after several empty frames.")
    p.add_argument("--velocity", default="auto", choices=["auto", "tracker", "flow"],
                   help="Velocity source. 'flow' uses dense optical flow, which needs no "
                        "track association and is the right choice overhead.")
    p.add_argument("--area-m2", type=float, default=120.0, help="Monitored area in square metres")
    p.add_argument("--output-video", default="outputs/annotated_crowdguard.mp4")
    p.add_argument("--output-json", default="outputs/risk_log.jsonl")
    p.add_argument("--output-events", default="outputs/escalation_events.jsonl")
    p.add_argument("--output-report", default="outputs/incident_report.md")
    p.add_argument("--knowledge-base", default="knowledge_base")
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--llm-provider", default="fallback", choices=["fallback", "openai", "local_hf"])
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--zones", default=None, help="JSON file with the zone layout")
    p.add_argument("--calibration", default=None, help="JSON file with image_points and world_points")
    p.add_argument("--forecast-horizon", type=float, default=30.0)
    p.add_argument("--forecast-backend", default="auto", choices=["auto", "trend", "transformer"])
    p.add_argument("--sim-fps", type=float, default=10.0, help="Simulation step rate")
    p.add_argument("--sim-seed", type=int, default=7)
    p.add_argument("--heatmap", action="store_true", help="Burn the density heatmap into the output video")
    p.add_argument("--alert-min-state", default="ALERT", choices=["WATCH", "ALERT", "CRITICAL"])
    p.add_argument("--webhook-url", default="", help="POST alerts to this URL (opt-in)")
    p.add_argument("--telegram-token", default="", help="Telegram bot token (opt-in)")
    p.add_argument("--telegram-chat", default="", help="Telegram chat id (opt-in)")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    alerts = AlertConfig(
        min_state=args.alert_min_state,
        webhook_enabled=bool(args.webhook_url),
        webhook_url=args.webhook_url,
        telegram_enabled=bool(args.telegram_token and args.telegram_chat),
        telegram_bot_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat,
    )

    if args.simulate:
        result = process_simulation(
            scenario=args.simulate,
            fps=args.sim_fps,
            seed=args.sim_seed,
            max_frames=args.max_frames,
            output_video=args.output_video.replace("annotated_", "simulated_"),
            output_json=args.output_json,
            output_events=args.output_events,
            output_report=args.output_report,
            knowledge_base_dir=args.knowledge_base,
            llm_provider=args.llm_provider,
            forecast_horizon=args.forecast_horizon,
            forecast_backend=args.forecast_backend,
            alert_config=alerts,
            show_heatmap=args.heatmap,
            verbose=not args.quiet,
        )
    else:
        video: Any = int(args.video) if str(args.video).isdigit() else args.video
        result = process_video(
            video_path=video,
            model_path=args.model,
            area_m2=args.area_m2,
            output_video=args.output_video,
            output_json=args.output_json,
            output_events=args.output_events,
            output_report=args.output_report,
            knowledge_base_dir=args.knowledge_base,
            sample_every=args.sample_every,
            llm_provider=args.llm_provider,
            max_frames=args.max_frames,
            zone_layout=load_zone_layout(args.zones),
            calibration=load_calibration(args.calibration, args.area_m2),
            forecast_horizon=args.forecast_horizon,
            forecast_backend=args.forecast_backend,
            alert_config=alerts,
            show_heatmap=args.heatmap,
            verbose=not args.quiet,
            view_mode=args.view,
            velocity_source=args.velocity,
        )

    print("\nCrowdGuard-RAG completed")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
