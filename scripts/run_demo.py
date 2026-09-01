#!/usr/bin/env python3
"""Demo the retrieval and advisory layers without any video.

Runs a scene through the RAG retriever and the advisor for each failure mode,
so the grounding behaviour can be inspected on its own -- useful when the
question is "where did that recommendation come from?" rather than "what is in
the video?".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crowdguard.advisor import CrowdSafetyAdvisor
from src.crowdguard.config import LLMConfig, RAGConfig, RiskConfig
from src.crowdguard.forecast import Forecast
from src.crowdguard.rag_engine import RAGIndex
from src.crowdguard.risk_engine import RiskEngine, SceneFeatures
from src.crowdguard.risk_taxonomy import RISK_TYPES, RiskTypeClassifier

# Representative measurements for each failure mode, in the units the engine uses.
SCENES = {
    "crush": dict(person_count=168, local_density_peak=6.4, local_density_mean=4.8,
                  avg_speed_ms=0.09, flow_disorder=0.71, counterflow_index=0.12,
                  bottleneck_ratio=0.94, crowd_pressure=4.10, flux_efficiency=0.18,
                  oscillation_index=0.19, density_rate_per_min=0.6, count_rate_per_min=22,
                  speed_surge_ratio=0.4, sustained_density_sec=48.0,
                  bottleneck_concentration=1.9, lateral_spread=0.42),
    "counterflow": dict(person_count=74, local_density_peak=2.6, local_density_mean=1.9,
                        avg_speed_ms=0.82, flow_disorder=0.78, counterflow_index=0.66,
                        bottleneck_ratio=0.31, crowd_pressure=0.90, flux_efficiency=0.62,
                        oscillation_index=0.18, density_rate_per_min=0.2, count_rate_per_min=8,
                        speed_surge_ratio=1.0, sustained_density_sec=0.0,
                        bottleneck_concentration=1.2, lateral_spread=0.75),
    "bottleneck": dict(person_count=96, local_density_peak=3.9, local_density_mean=2.7,
                       avg_speed_ms=0.55, flow_disorder=0.52, counterflow_index=0.14,
                       bottleneck_ratio=0.88, crowd_pressure=1.60, flux_efficiency=0.44,
                       oscillation_index=0.21, density_rate_per_min=0.9, count_rate_per_min=34,
                       speed_surge_ratio=0.8, sustained_density_sec=2.0,
                       bottleneck_concentration=2.4, lateral_spread=0.33),
    "panic": dict(person_count=58, local_density_peak=1.7, local_density_mean=1.1,
                  avg_speed_ms=2.55, flow_disorder=0.74, counterflow_index=0.21,
                  bottleneck_ratio=0.22, crowd_pressure=2.90, flux_efficiency=0.95,
                  oscillation_index=0.44, density_rate_per_min=-0.9, count_rate_per_min=-30,
                  speed_surge_ratio=2.6, sustained_density_sec=0.0,
                  bottleneck_concentration=0.9, lateral_spread=0.95),
    "blockage": dict(person_count=112, local_density_peak=4.3, local_density_mean=3.1,
                     avg_speed_ms=0.16, flow_disorder=0.55, counterflow_index=0.19,
                     bottleneck_ratio=0.79, crowd_pressure=1.10, flux_efficiency=0.08,
                     oscillation_index=0.12, density_rate_per_min=0.4, count_rate_per_min=14,
                     speed_surge_ratio=0.3, sustained_density_sec=26.0,
                     bottleneck_concentration=1.1, lateral_spread=0.86),
}


def make_features(values: dict, zone: str) -> SceneFeatures:
    cfg = RiskConfig()
    typology = RiskTypeClassifier(cfg.pressure_warning, cfg.pressure_critical).classify(
        {**values, "density": values["local_density_mean"]}
    )
    density_norm = min(1.0, max(0.0, (values["local_density_peak"] - 0.5) / (cfg.density_hard_limit - 0.5)))
    score = min(1.0, max(density_norm * 0.6 + typology.hazard_index * 0.5, typology.hazard_index))
    return SceneFeatures(
        frame_id=0, timestamp_sec=0.0,
        person_count=values["person_count"],
        density=values["local_density_mean"],
        local_density_peak=values["local_density_peak"],
        local_density_mean=values["local_density_mean"],
        avg_speed=values["avg_speed_ms"] / 3.0,
        avg_speed_ms=values["avg_speed_ms"],
        flow_disorder=values["flow_disorder"],
        counterflow_index=values["counterflow_index"],
        bottleneck_ratio=values["bottleneck_ratio"],
        bottleneck_concentration=values["bottleneck_concentration"],
        lateral_spread=values["lateral_spread"],
        crowd_pressure=values["crowd_pressure"],
        flux=values["local_density_mean"] * values["avg_speed_ms"],
        flux_efficiency=values["flux_efficiency"],
        oscillation_index=values["oscillation_index"],
        density_rate_per_min=values["density_rate_per_min"],
        count_rate_per_min=values["count_rate_per_min"],
        speed_surge_ratio=values["speed_surge_ratio"],
        sustained_density_sec=values["sustained_density_sec"],
        risk_score=score,
        risk_level="high" if score >= cfg.high_threshold else "moderate" if score >= cfg.low_threshold else "low",
        factors=[e for t in typology.ranked[:1] for e in t.evidence] or ["nominal"],
        primary_risk_type=typology.primary.code,
        primary_risk_label=typology.primary.label,
        primary_risk_score=typology.primary.score,
        hazard_index=typology.hazard_index,
        risk_type_ranking=[t.to_dict() for t in typology.ranked],
        type_evidence=typology.primary.evidence,
        hotspot_zone=zone,
        calibrated=True,
        calibration_note="demo scene, assumed calibrated",
    )


def demo_forecast(score: float) -> Forecast:
    return Forecast(horizon_sec=30.0, predicted_score=min(1.0, score + 0.09),
                    lower=max(0.0, score + 0.01), upper=min(1.0, score + 0.17),
                    predicted_level="high", trend_per_min=0.18,
                    time_to_alert_sec=None, time_to_critical_sec=62.0,
                    confidence=0.71, backend="trend", ready=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo the RAG and advisor layers without video")
    ap.add_argument("--scene", default="crush", choices=sorted(SCENES) + ["all"])
    ap.add_argument("--query", default=None, help="Retrieve for a raw query and stop")
    ap.add_argument("--zone", default="Gate 3 throat")
    ap.add_argument("--knowledge-base", default=str(ROOT / "knowledge_base"))
    ap.add_argument("--llm-provider", default="fallback", choices=["fallback", "openai", "local_hf"])
    args = ap.parse_args()

    rag = RAGIndex(RAGConfig(knowledge_base_dir=args.knowledge_base))
    rag.load_documents(args.knowledge_base)
    rag.build()
    print(f"Retrieval backend: {rag.backend}  |  {len(rag.chunks)} chunks "
          f"from {len(set(c[0] for c in rag.chunks))} documents\n")

    if args.query:
        print(f'Query: "{args.query}"\n')
        for c in rag.search(args.query):
            print(f"  {c.source:42} score={c.score:.3f}")
            print(f"      {c.text[:150]}...\n")
        return

    engine = RiskEngine(RiskConfig())
    advisor = CrowdSafetyAdvisor(LLMConfig(provider=args.llm_provider))
    scenes = sorted(SCENES) if args.scene == "all" else [args.scene]

    for name in scenes:
        features = make_features(SCENES[name], args.zone)
        query = engine.scene_query(features, RISK_TYPES[features.primary_risk_type].sop_terms)
        chunks = rag.search(query)
        forecast = demo_forecast(features.risk_score)

        print("=" * 78)
        print(f"SCENE: {name}")
        print("=" * 78)
        print(f"Detected failure mode : {features.primary_risk_label} "
              f"(confidence {features.primary_risk_score:.2f})")
        print(f"Risk score            : {features.risk_score:.2f} ({features.risk_level})")
        print(f"Forecast              : {forecast.headline()}")
        print("\nRetrieved context:")
        for c in chunks:
            print(f"  - {c.source:42} score={c.score:.3f}")
        print("\n" + advisor.generate(features, chunks, "ALERT", forecast).to_markdown())
        print()


if __name__ == "__main__":
    main()
