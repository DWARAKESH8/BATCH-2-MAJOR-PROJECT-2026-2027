"""Generation of the operator advisory.

The output is deliberately not free prose. It is a fixed eight-part structure,
because in a control room under time pressure an operator must be able to find
the action line without reading a paragraph. The structure also makes the
system auditable: every advisory carries the measurements that produced it and
the SOP documents it was drawn from.

Two sections are worth calling out as design decisions:

* **DO NOT** -- the counter-indication. Crowd interventions are not symmetric.
  Narrowing exits calms a bottleneck and kills people during a dispersal;
  reversing a crowd relieves a queue and kills people inside a jam. A system
  that only ever says what to do will eventually tell somebody to do the
  lethal version of the right idea.

* **TIME BUDGET** -- how long the operator has before the forecast crosses the
  next threshold, set against how long the recommended action takes to have a
  physical effect. This is the difference between advice and a decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import LLMConfig
from .rag_engine import RetrievedChunk
from .risk_engine import SceneFeatures
from .risk_taxonomy import RISK_TYPES
from .utils import compress_words

# Rough time-to-physical-effect for each intervention, used to check whether
# the recommended action can still land inside the forecast window.
ACTION_LATENCY_SEC: Dict[str, int] = {
    "meter_inflow": 60,
    "open_route": 120,
    "deploy_marshals": 180,
    "pa_broadcast": 30,
    "halt_inflow": 45,
    "emergency": 300,
}


@dataclass
class Advisory:
    """Structured, explainable, citable safety advice."""

    risk_level: str
    escalation_state: str
    situation: str
    failure_mode: str
    mechanism: str
    root_cause: str
    actions: List[str] = field(default_factory=list)
    do_not: str = ""
    broadcast: str = ""
    time_budget: str = ""
    evidence: List[str] = field(default_factory=list)
    zone: str = ""
    notify_role: str = ""
    generator: str = "fallback"
    raw_llm: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "escalation_state": self.escalation_state,
            "situation": self.situation,
            "failure_mode": self.failure_mode,
            "mechanism": self.mechanism,
            "root_cause": self.root_cause,
            "actions": self.actions,
            "do_not": self.do_not,
            "broadcast": self.broadcast,
            "time_budget": self.time_budget,
            "evidence": self.evidence,
            "zone": self.zone,
            "notify_role": self.notify_role,
            "generator": self.generator,
        }

    def to_markdown(self) -> str:
        action_block = "\n".join(f"   {i}. {a}" for i, a in enumerate(self.actions, 1)) or "   1. Continue routine monitoring."
        parts = [
            f"**1. SITUATION**  \n{self.situation}",
            f"**2. FAILURE MODE — {self.failure_mode}**  \n{self.mechanism}",
            f"**3. ROOT CAUSE**  \n{self.root_cause}",
            f"**4. IMMEDIATE ACTION**\n{action_block}",
        ]
        if self.do_not:
            parts.append(f"**5. DO NOT**  \n{self.do_not}")
        parts.append(f"**6. OPERATOR BROADCAST**  \n> {self.broadcast}")
        if self.time_budget:
            parts.append(f"**7. TIME BUDGET**  \n{self.time_budget}")
        parts.append("**8. EVIDENCE**  \n" + ("; ".join(self.evidence) if self.evidence else "No retrieved context."))
        if self.notify_role:
            parts.append(f"**9. NOTIFY**  \n{self.notify_role}")
        return "\n\n".join(parts)

    def __str__(self) -> str:
        return self.to_markdown()


class CrowdSafetyAdvisor:
    """Turns measurements + retrieved SOPs into an actionable advisory."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    # ------------------------------------------------------------------ #
    def _context_block(self, chunks: List[RetrievedChunk]) -> str:
        items = []
        for i, chunk in enumerate(chunks, start=1):
            text = compress_words(chunk.text, self.config.max_context_words // max(1, len(chunks)))
            items.append(f"[{i}] Source: {chunk.source} | Score: {chunk.score:.3f}\n{text}")
        return "\n\n".join(items) if items else "No SOP context retrieved."

    def build_prompt(
        self,
        features: SceneFeatures,
        chunks: List[RetrievedChunk],
        escalation_state: str = "NORMAL",
        forecast: Optional[Any] = None,
    ) -> str:
        spec = RISK_TYPES.get(features.primary_risk_type)
        forecast_line = forecast.headline() if forecast is not None else "not available"
        zone_line = features.hotspot_zone or "whole monitored area"

        return f"""
You are CrowdGuard-RAG, a decision-support assistant for a crowd-safety control room.
Use ONLY the scene measurements and the retrieved safety context below. Never invent a
procedure that is not supported by the retrieved context. If the context does not cover
the situation, say so explicitly rather than guessing.

ESCALATION STATE: {escalation_state}
FORECAST: {forecast_line}
HOTSPOT ZONE: {zone_line}

Detected failure mode: {features.primary_risk_label} (confidence {features.primary_risk_score:.2f})
Known mechanism: {spec.mechanism if spec else "n/a"}

Scene measurements:
- Risk score: {features.risk_score:.2f} ({features.risk_level})
- Person count: {features.person_count}
- Peak local density: {features.local_density_peak:.2f} persons/m2
- Mean speed: {features.avg_speed_ms:.2f} m/s
- Flow disorder: {features.flow_disorder:.2f}
- Bottleneck concentration: {features.bottleneck_ratio:.2f}
- Crowd pressure (Helbing): {features.crowd_pressure:.4f} s^-2
- Stop-and-go oscillation: {features.oscillation_index:.2f}
- Flux efficiency: {features.flux_efficiency:.2f}
- Density trend: {features.density_rate_per_min:+.2f} persons/m2 per minute
- Calibration: {features.calibration_note}
- Observed factors: {', '.join(features.factors)}

Retrieved SOP / incident context:
{self._context_block(chunks)}

Answer in exactly this structure, one short paragraph or list per heading:
1. SITUATION
2. FAILURE MODE
3. ROOT CAUSE
4. IMMEDIATE ACTION (numbered, most urgent first, each naming who does it)
5. DO NOT (the intervention that would make this specific failure mode worse)
6. OPERATOR BROADCAST (one sentence, calm, addressed to the crowd)
7. TIME BUDGET
8. EVIDENCE (cite the retrieved source filenames you used)
""".strip()

    # ------------------------------------------------------------------ #
    def generate(
        self,
        features: SceneFeatures,
        chunks: List[RetrievedChunk],
        escalation_state: str = "NORMAL",
        forecast: Optional[Any] = None,
    ) -> Advisory:
        if self.config.provider == "openai":
            return self._generate_openai(features, chunks, escalation_state, forecast)
        if self.config.provider == "local_hf":
            return self._generate_local_hf(features, chunks, escalation_state, forecast)
        return self.build_fallback(features, chunks, escalation_state, forecast)

    def _generate_openai(self, features, chunks, escalation_state, forecast) -> Advisory:
        api_key = os.getenv(self.config.openai_api_key_env)
        if not api_key:
            adv = self.build_fallback(features, chunks, escalation_state, forecast)
            adv.generator = "fallback (no API key)"
            return adv
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a crowd safety decision-support assistant. "
                                                  "Ground every recommendation in the retrieved SOP text."},
                    {"role": "user", "content": self.build_prompt(features, chunks, escalation_state, forecast)},
                ],
                temperature=self.config.temperature,
            )
            text = response.choices[0].message.content.strip()
            adv = self.build_fallback(features, chunks, escalation_state, forecast)
            adv.generator = f"openai:{self.config.model_name}"
            adv.raw_llm = text
            return self._merge_llm(adv, text)
        except Exception as exc:
            adv = self.build_fallback(features, chunks, escalation_state, forecast)
            adv.generator = f"fallback (OpenAI failed: {type(exc).__name__})"
            return adv

    def _generate_local_hf(self, features, chunks, escalation_state, forecast) -> Advisory:
        try:
            from transformers import pipeline

            model_name = self.config.local_model_name or "google/flan-t5-base"
            pipe = pipeline("text2text-generation", model=model_name)
            out = pipe(self.build_prompt(features, chunks, escalation_state, forecast),
                       max_new_tokens=320, do_sample=False)
            text = out[0]["generated_text"].strip()
            adv = self.build_fallback(features, chunks, escalation_state, forecast)
            adv.generator = f"local_hf:{model_name}"
            adv.raw_llm = text
            return self._merge_llm(adv, text)
        except Exception as exc:
            adv = self.build_fallback(features, chunks, escalation_state, forecast)
            adv.generator = f"fallback (local model failed: {type(exc).__name__})"
            return adv

    @staticmethod
    def _merge_llm(advisory: Advisory, text: str) -> Advisory:
        """Overlay LLM prose onto the deterministic skeleton.

        The rule-derived fields stay as the floor: if the model returns
        something unparseable, the operator still gets a correct advisory
        rather than an empty one. This is what "grounded" means in practice.
        """
        import re

        sections = re.split(r"\n\s*(?=\d\.\s*[A-Z])", text)
        mapping = {
            "SITUATION": "situation",
            "FAILURE MODE": "failure_mode_text",
            "ROOT CAUSE": "root_cause",
            "IMMEDIATE ACTION": "actions_text",
            "DO NOT": "do_not",
            "OPERATOR BROADCAST": "broadcast",
            "TIME BUDGET": "time_budget",
        }
        for section in sections:
            head = section.strip()[:40].upper()
            for key, attr in mapping.items():
                if key in head:
                    body = section.split("\n", 1)[1].strip() if "\n" in section else ""
                    body = body or re.sub(r"^\d\.\s*[A-Z ]+[:\-]?\s*", "", section.strip())
                    if not body:
                        continue
                    if attr == "actions_text":
                        lines = [re.sub(r"^[\d\.\)\-\*\s]+", "", ln).strip()
                                 for ln in body.splitlines() if ln.strip()]
                        lines = [ln for ln in lines if ln]
                        if lines:
                            advisory.actions = lines
                    elif attr == "failure_mode_text":
                        advisory.mechanism = body
                    else:
                        setattr(advisory, attr, body)
        return advisory

    # ------------------------------------------------------------------ #
    def build_fallback(
        self,
        features: SceneFeatures,
        chunks: List[RetrievedChunk],
        escalation_state: str = "NORMAL",
        forecast: Optional[Any] = None,
    ) -> Advisory:
        spec = RISK_TYPES.get(features.primary_risk_type, RISK_TYPES["NORMAL_FLOW"])
        zone = features.hotspot_zone or "the monitored area"

        situation = (
            f"{escalation_state} at {features.timestamp_sec:.0f}s in {zone}. "
            f"{features.person_count} people tracked; peak local density "
            f"{features.local_density_peak:.2f} persons/m2; "
            f"mean speed {features.avg_speed_ms:.2f} m/s; "
            f"crowd pressure {features.crowd_pressure:.3f} s^-2. "
            f"Fused risk {features.risk_score:.2f} ({features.risk_level})."
        )
        # The trend line is what a human watching the same screen cannot
        # produce, so it goes in the situation rather than buried in evidence.
        if getattr(features, "trend_summary", ""):
            situation += " " + features.trend_summary

        evidence_items = features.type_evidence or features.factors
        root_cause = (
            f"{spec.label} detected at confidence {features.primary_risk_score:.2f}. "
            + ("Measured evidence: " + "; ".join(evidence_items) + "." if evidence_items else "")
        )

        actions = self._actions_for(features, spec, zone, escalation_state)
        broadcast = self._broadcast_for(features, spec, zone)
        time_budget = self._time_budget(features, forecast)

        evidence = [f"{c.source} (relevance {c.score:.2f})" for c in chunks[:3]]
        if not features.calibrated:
            evidence.append(
                "NOTE: camera not homography-calibrated -- densities are approximate"
            )

        return Advisory(
            risk_level=features.risk_level,
            escalation_state=escalation_state,
            situation=situation,
            failure_mode=spec.label,
            mechanism=spec.mechanism,
            root_cause=root_cause,
            actions=actions,
            do_not=spec.counter_indication,
            broadcast=broadcast,
            time_budget=time_budget,
            evidence=evidence,
            zone=zone,
            notify_role={
                "CRITICAL": "Safety officer, medical lead, and all marshals",
                "ALERT": "Zone supervisor and marshals on the ground",
                "WATCH": "Control room and zone supervisor",
            }.get(escalation_state, "Control room"),
            generator="fallback",
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _actions_for(features: SceneFeatures, spec, zone: str, state: str) -> List[str]:
        code = features.primary_risk_type
        a: List[str] = []

        # Routing comes from the venue's own zone definition. An advisory that
        # says "open an alternate route" is a suggestion; one that says "divert
        # to Gate 4, meter at the north approach, stage marshals at the outer
        # barrier" is a dispatch. Only the venue knows its topology, so these
        # are read from the zone config rather than invented here.
        alt = features.hotspot_alternate
        meter = features.hotspot_meter_at
        stage = features.hotspot_staging
        to_alt = f" Divert to {alt}." if alt else ""
        at_meter = f" at {meter}" if meter else " at the nearest upstream decision point"
        at_stage = f" Assemble at {stage}." if stage else ""

        if code == "PROGRESSIVE_CRUSH":
            a = [
                f"SAFETY OFFICER: declare a major incident for {zone} and halt all inflow{at_meter} now.{to_alt}",
                f"MARSHALS: release barriers at the leading edge of {zone} -- relief must come from the front, never the rear.{at_stage}",
                "PA: instruct the crowd to stop pushing forward and to raise arms to the chest to protect the ribcage.",
                "MEDICAL: stage at the dense edge and prepare for compressive asphyxia casualties.",
                "CONTROL: hold this camera on the main wall and log the time of the first alarm.",
            ]
        elif code == "TURBULENT_SURGE":
            a = [
                f"SAFETY OFFICER: stop all inflow to {zone} immediately{at_meter}; the crowd is already in the turbulent regime.",
                f"MARSHALS: enter from the DOWNSTREAM side and break the wave laterally in sections.{at_stage}",
                f"CONTROL: open every available egress route to bleed pressure out of the block.{to_alt}",
                "PA: give one clear, calm, repeated instruction -- conflicting messages amplify the wave.",
            ]
        elif code == "STATIC_BLOCKAGE":
            a = [
                f"CONTROL: stop inflow to {zone}{at_meter} -- the crowd is jammed and pressure is accumulating from behind.",
                f"MARSHALS: identify and clear the obstruction at the head of the jam; that is the only real fix.{at_stage}",
                "CONTROL: open a release path in the crowd's existing direction of travel.",
                "SUPERVISOR: if the jam does not move within 60 seconds, escalate to a crush response.",
            ]
        elif code == "COUNTERFLOW_CONFLICT":
            a = [
                f"MARSHALS: impose one-way movement through {zone} and physically separate the two streams with line barriers.{at_stage}",
                f"CONTROL: divert the minority stream rather than trying to interleave them.{to_alt}",
                "PA: announce the one-way direction before density rises -- this is cheap now and impossible later.",
            ]
        elif code == "BOTTLENECK_CONGESTION":
            a = [
                f"CONTROL: open additional discharge capacity at {zone}; throughput is on the congested branch.{to_alt}",
                f"MARSHALS: hold upstream flow in controlled batches{at_meter} until the queue drains.{at_stage}",
                "SUPERVISOR: check for a physical obstruction narrowing the chokepoint -- congestion often has a cause you can remove.",
            ]
        elif code == "PANIC_DISPERSAL":
            a = [
                "SAFETY OFFICER: identify and neutralise the trigger -- the crowd is escaping something.",
                f"CONTROL: open every exit fully, including {alt if alt else 'all reserved routes'}. "
                f"Do NOT meter or narrow flow during a dispersal.",
                f"MARSHALS: clear trip hazards and pull fallen people out of the flow line immediately.{at_stage}",
                "PA: calm, factual announcement naming the safe direction -- silence increases panic.",
            ]
        elif code == "RAPID_INFLUX":
            a = [
                f"CONTROL: begin metering inflow{at_meter} now, while density is still sub-critical.",
                f"SUPERVISOR: pre-position marshals at the constriction before it becomes one.{at_stage}",
                f"CONTROL: prepare the diversion route so it can be opened without a decision delay.{to_alt}",
            ]
        else:
            a = [
                "CONTROL: continue routine monitoring.",
                "MARSHALS: keep entry and egress routes unobstructed.",
            ]

        if state == "WATCH" and code != "NORMAL_FLOW":
            a.append("SUPERVISOR: acknowledge this alert so the control room knows it is being handled.")
        return a

    @staticmethod
    def _broadcast_for(features: SceneFeatures, spec, zone: str) -> str:
        code = features.primary_risk_type
        alt = features.hotspot_alternate
        route = alt if alt else "the route signposted by staff"
        return {
            "PROGRESSIVE_CRUSH": "Everyone please STOP pushing forward. Keep your arms up at chest height. "
                                 "Staff are opening the barriers ahead of you. Do not turn around.",
            "TURBULENT_SURGE": "Please stop moving forward and hold your position. Staff are opening additional "
                               "exits now. Move only when a marshal directs you.",
            "STATIC_BLOCKAGE": "The route ahead is temporarily blocked. Please hold your position and do not push. "
                               "Staff are clearing it now.",
            "COUNTERFLOW_CONFLICT": f"{zone} is now one-way only. If you are heading the other way, "
                                    f"please use {route}.",
            "BOTTLENECK_CONGESTION": f"{zone} is congested. {route.capitalize()} is now open -- please "
                                     f"follow marshal directions and keep moving steadily.",
            "PANIC_DISPERSAL": "Please walk, do not run. All exits are open and there is room for everyone. "
                               "If someone falls, help them up.",
            "RAPID_INFLUX": "Entry is being managed to keep the area safe. Please move steadily and avoid stopping "
                            "near the gates.",
        }.get(code, "Crowd movement is normal. Please continue to follow route signage.")

    @staticmethod
    def _time_budget(features: SceneFeatures, forecast: Optional[Any]) -> str:
        if forecast is None or not getattr(forecast, "ready", False):
            return ""
        ttc = getattr(forecast, "time_to_critical_sec", None)
        tta = getattr(forecast, "time_to_alert_sec", None)
        code = features.primary_risk_type

        latency = {
            "PROGRESSIVE_CRUSH": ACTION_LATENCY_SEC["emergency"],
            "TURBULENT_SURGE": ACTION_LATENCY_SEC["halt_inflow"],
            "STATIC_BLOCKAGE": ACTION_LATENCY_SEC["halt_inflow"],
            "COUNTERFLOW_CONFLICT": ACTION_LATENCY_SEC["deploy_marshals"],
            "BOTTLENECK_CONGESTION": ACTION_LATENCY_SEC["open_route"],
            "PANIC_DISPERSAL": ACTION_LATENCY_SEC["pa_broadcast"],
            "RAPID_INFLUX": ACTION_LATENCY_SEC["meter_inflow"],
        }.get(code, ACTION_LATENCY_SEC["meter_inflow"])

        if ttc is not None:
            margin = ttc - latency
            verdict = (
                f"Recommended action needs about {latency}s to take physical effect, leaving "
                f"{margin:.0f}s of margin."
                if margin > 0
                else f"WARNING: the recommended action needs about {latency}s to take effect, which is "
                     f"{abs(margin):.0f}s LONGER than the time available. Escalate one level immediately "
                     f"and act on the faster intervention."
            )
            return f"On the current trend, CRITICAL is reached in about {ttc:.0f}s. {verdict}"
        if tta is not None:
            return (
                f"On the current trend, ALERT level is reached in about {tta:.0f}s. "
                f"The recommended action needs about {latency}s to take effect -- act now."
            )
        return f"{forecast.headline()}. No threshold crossing projected inside the forecast window."
