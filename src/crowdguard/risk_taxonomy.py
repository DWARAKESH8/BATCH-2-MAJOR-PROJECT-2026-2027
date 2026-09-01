"""Crowd risk typology -- *what kind* of risk, not just how much.

A single scalar "risk = 0.78" tells an operator nothing about what to do.
A crush and a panic dispersal both score high and require opposite responses:
one needs inflow stopped and pressure released, the other needs exits opened
and obstacles cleared. This module classifies the failure mode.

Each risk type is expressed as:
    score = gate(necessary condition) x weighted sum(supporting evidence)

The `gate` encodes the physically necessary precondition -- there is no crush
without density, no counter-flow conflict without directional disorder -- so a
type can never fire on circumstantial evidence alone. This keeps the classifier
explainable, which is a hard requirement in a safety-critical setting.

Literature anchors
------------------
* Fruin (1993), "The Causes and Prevention of Crowd Disasters": density bands,
  the distinction between crowd *crush* and crowd *panic*.
* Helbing, Johansson & Al-Abideen (2007), Phys. Rev. E 75, 046109: crowd
  turbulence / stop-and-go waves as the precursor to the Mina disasters.
* Fundamental diagram of pedestrian flow: flux J = rho * v rises with density,
  peaks, then collapses on the congested branch -- used here to separate
  free-flow crowding from genuine bottleneck failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Fuzzy helpers
# --------------------------------------------------------------------------- #
def ramp(x: float, lo: float, hi: float) -> float:
    """Soft membership: 0 at/below `lo`, 1 at/above `hi`, linear between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def inv_ramp(x: float, lo: float, hi: float) -> float:
    """1 at/below `lo`, 0 at/above `hi`."""
    return 1.0 - ramp(x, lo, hi)


# --------------------------------------------------------------------------- #
# Type registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskTypeSpec:
    code: str
    label: str
    severity: float                 # intrinsic lethality multiplier, 0-1
    icon: str
    color: str
    description: str
    mechanism: str                  # the physics, for the report / viva
    primary_action: str
    sop_terms: str                  # injected into the RAG query
    counter_indication: str = ""    # what NOT to do -- matters a lot here


RISK_TYPES: Dict[str, RiskTypeSpec] = {
    "NORMAL_FLOW": RiskTypeSpec(
        code="NORMAL_FLOW",
        label="Normal Flow",
        severity=0.05,
        icon="✓",
        color="#2ecc71",
        description="Crowd is moving coherently at a density well inside the safe band.",
        mechanism="Density below Fruin LOS C, coherent direction, stable flux.",
        primary_action="Routine monitoring. Keep entry and egress routes clear.",
        sop_terms="routine monitoring normal pedestrian flow patrol",
    ),
    "RAPID_INFLUX": RiskTypeSpec(
        code="RAPID_INFLUX",
        label="Rapid Influx",
        severity=0.35,
        icon="↑",
        color="#4aa8ff",
        description="Occupancy is climbing fast. Not yet dangerous, but the trend is.",
        mechanism="d(rho)/dt strongly positive while density is still sub-critical. "
                  "This is the only window in which metering inflow is cheap.",
        primary_action="Meter inflow now at the upstream gates before density reaches the crush band.",
        sop_terms="gate metering inflow control queue holding upstream entry rate",
        counter_indication="Do NOT hold people where there is no room to hold them. Meter at the "
                           "furthest upstream point that still has holding space, never at the "
                           "constriction itself.",
    ),
    "BOTTLENECK_CONGESTION": RiskTypeSpec(
        code="BOTTLENECK_CONGESTION",
        label="Bottleneck Congestion",
        severity=0.55,
        icon="⧉",
        color="#f5a623",
        description="People are accumulating faster than the chokepoint can discharge them.",
        mechanism="Crowd concentrated in a constriction while flux J = rho*v falls "
                  "despite rising density -- the congested branch of the fundamental diagram.",
        primary_action="Open additional discharge routes and hold upstream flow until the queue drains.",
        sop_terms="gate bottleneck chokepoint queue discharge alternate exit throughput",
        counter_indication="Do NOT close or narrow the gate the crowd is already pressing toward. "
                           "Closing the discharge point of a loaded queue converts congestion into "
                           "compression within seconds -- reduce the flow upstream instead and let "
                           "the queue drain forward.",
    ),
    "COUNTERFLOW_CONFLICT": RiskTypeSpec(
        code="COUNTERFLOW_CONFLICT",
        label="Counter-Flow Conflict",
        severity=0.60,
        icon="⇄",
        color="#e67e22",
        description="Two opposing streams are colliding and cancelling each other's throughput.",
        mechanism="High directional disorder: the vector sum of velocities collapses "
                  "relative to the scalar sum, so people move but the crowd does not.",
        primary_action="Impose one-way movement. Physically separate the two streams before density rises.",
        sop_terms="counter flow opposing streams one way lane separation directional control",
    ),
    "STATIC_BLOCKAGE": RiskTypeSpec(
        code="STATIC_BLOCKAGE",
        label="Static Blockage",
        severity=0.72,
        icon="■",
        color="#ff7043",
        description="A dense crowd has stopped moving. This is the state immediately before a crush.",
        mechanism="High density with near-zero speed and low disorder: the crowd is "
                  "jammed, not flowing. Pressure now accumulates from behind with no release path.",
        primary_action="STOP all inflow immediately and create a release path in the direction of travel.",
        sop_terms="crowd stopped standstill jam blockage stop inflow release pressure",
        counter_indication="Do NOT broadcast messages that cause people to turn around -- "
                           "reversing a jammed crowd creates counter-flow inside the jam.",
    ),
    "TURBULENT_SURGE": RiskTypeSpec(
        code="TURBULENT_SURGE",
        label="Turbulent Surge",
        severity=0.88,
        icon="≈",
        color="#ff4757",
        description="Stop-and-go shock waves are propagating through a dense crowd.",
        mechanism="Crowd turbulence: local density x velocity variance exceeds the "
                  "Helbing critical pressure, producing involuntary lateral displacement. "
                  "This is the mechanism of the Mina and Love Parade disasters.",
        primary_action="Declare an emergency. Stop inflow, open every egress, and "
                       "insert marshals to break the wave laterally.",
        sop_terms="crowd turbulence shock wave surge pressure emergency egress crush",
        counter_indication="Do NOT push responders against the flow -- enter from the "
                           "downstream side of the wave.",
    ),
    "PROGRESSIVE_CRUSH": RiskTypeSpec(
        code="PROGRESSIVE_CRUSH",
        label="Progressive Crowd Crush",
        severity=1.00,
        icon="⚠",
        color="#ff1744",
        description="Compressive load is building in a dense, slow-moving crowd. Life-threatening.",
        mechanism="Density above the Fruin crush band with movement suppressed: "
                  "individuals lose the ability to control their own position and "
                  "compressive asphyxia becomes possible within roughly 30 seconds.",
        primary_action="MAJOR INCIDENT. Halt all inflow, open every barrier, extract "
                       "from the front of the crowd, and dispatch medical to the dense edge.",
        sop_terms="crowd crush compressive asphyxia emergency evacuation barrier release medical",
        counter_indication="Never attempt to extract from the rear of a crush -- "
                           "relief must come from the leading edge.",
    ),
    "PANIC_DISPERSAL": RiskTypeSpec(
        code="PANIC_DISPERSAL",
        label="Panic Dispersal",
        severity=0.80,
        icon="✹",
        color="#c644fc",
        description="A sudden speed surge with scattering directions -- the crowd is fleeing something.",
        mechanism="Mean speed spikes far above its running baseline while directional "
                  "disorder rises and density starts to fall: escape behaviour, not congestion. "
                  "Trampling and falls dominate the injury profile here, not compression.",
        primary_action="Identify and remove the source. Open all exits, clear trip hazards, "
                       "and do not obstruct the direction of travel.",
        sop_terms="panic stampede evacuation trampling exit clearance emergency announcement",
        counter_indication="Do NOT close or narrow exits to control the flow during a "
                           "dispersal -- that converts a stampede into a crush.",
    ),
}

ORDERED_CODES: List[str] = [
    "PROGRESSIVE_CRUSH",
    "TURBULENT_SURGE",
    "PANIC_DISPERSAL",
    "STATIC_BLOCKAGE",
    "COUNTERFLOW_CONFLICT",
    "BOTTLENECK_CONGESTION",
    "RAPID_INFLUX",
    "NORMAL_FLOW",
]


# --------------------------------------------------------------------------- #
# Classification result
# --------------------------------------------------------------------------- #
@dataclass
class RiskTypeScore:
    code: str
    label: str
    score: float
    severity: float
    icon: str
    color: str
    evidence: List[str] = field(default_factory=list)

    @property
    def weighted(self) -> float:
        return self.score * self.severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "score": round(self.score, 4),
            "severity": self.severity,
            "weighted": round(self.weighted, 4),
            "evidence": self.evidence,
        }


@dataclass
class TypologyResult:
    primary: RiskTypeScore
    ranked: List[RiskTypeScore]
    hazard_index: float           # severity-weighted aggregate, 0-1

    @property
    def spec(self) -> RiskTypeSpec:
        return RISK_TYPES[self.primary.code]

    def secondary(self) -> Optional[RiskTypeScore]:
        for t in self.ranked[1:]:
            if t.score > 0.25 and t.code != "NORMAL_FLOW":
                return t
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_type": self.primary.code,
            "primary_label": self.primary.label,
            "primary_score": round(self.primary.score, 4),
            "hazard_index": round(self.hazard_index, 4),
            "ranked_types": [t.to_dict() for t in self.ranked],
        }

    def sop_terms(self) -> str:
        terms = [RISK_TYPES[self.primary.code].sop_terms]
        sec = self.secondary()
        if sec:
            terms.append(RISK_TYPES[sec.code].sop_terms)
        return " ".join(terms)


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class RiskTypeClassifier:
    """Rule-based, fully explainable failure-mode classifier.

    Deliberately not a black box: in a safety context an operator must be able
    to ask "why did you say crush?" and get the exact measurement back.
    """

    def __init__(self, pressure_warning: float = 0.010, pressure_critical: float = 0.020):
        self.p_warn = pressure_warning
        self.p_crit = pressure_critical

    def classify(self, f: Dict[str, float]) -> TypologyResult:
        density = float(f.get("local_density_peak", f.get("density", 0.0)))
        mean_density = float(f.get("density", 0.0))
        speed = float(f.get("avg_speed_ms", 0.0))
        disorder = float(f.get("flow_disorder", 0.0))
        bottleneck = float(f.get("bottleneck_ratio", 0.0))
        pressure = float(f.get("crowd_pressure", 0.0))
        oscillation = float(f.get("oscillation_index", 0.0))
        flux_eff = float(f.get("flux_efficiency", 1.0))
        density_rate = float(f.get("density_rate_per_min", 0.0))
        speed_surge = float(f.get("speed_surge_ratio", 1.0))
        count_rate = float(f.get("count_rate_per_min", 0.0))
        counterflow = float(f.get("counterflow_index", 0.0))
        sustained = float(f.get("sustained_density_sec", 0.0))
        constriction = float(f.get("bottleneck_concentration", 1.0))
        spread = float(f.get("lateral_spread", 1.0))

        scores: List[RiskTypeScore] = []

        # -- Progressive crowd crush ---------------------------------------
        gate = ramp(density, 3.5, 6.0)
        slow = inv_ramp(speed, 0.25, 0.90)
        press = ramp(pressure, self.p_warn * 0.6, self.p_crit * 1.5)
        # "Progressive" is the operative word: a crush is built by compression
        # HELD over time, which is exactly what separates it from a surge that
        # touches the same density for a second and releases.
        held = ramp(sustained, 4.0, 25.0)
        stalled = ramp(1.0 - flux_eff, 0.25, 0.65)
        # A chokepoint that is loaded but still discharging people is a
        # bottleneck. It only becomes a crush when the crowd stops draining and
        # the pressure has nowhere to go.
        s = gate * max(slow, stalled) * (0.34 + 0.22 * slow + 0.16 * press + 0.28 * held)
        ev = []
        if gate > 0:
            ev.append(f"peak local density {density:.2f}/m2 in the Fruin crush band")
        if held > 0.1:
            ev.append(f"compression sustained for {sustained:.0f}s above the safe limit")
        if slow > 0.4:
            ev.append(f"movement suppressed ({speed:.2f} m/s)")
        if press > 0.3:
            ev.append(f"crowd pressure {pressure:.2f} above the calibrated warning level")
        scores.append(self._mk("PROGRESSIVE_CRUSH", min(1.0, s), ev))

        # -- Turbulent surge / shock waves ---------------------------------
        gate = ramp(density, 3.0, 5.5)
        turb = max(ramp(oscillation, 0.20, 0.65), ramp(pressure, self.p_warn, self.p_crit * 2.0))
        # Turbulence is motion. A dense crowd at a dead stop is a blockage or a
        # crush, and calling it a surge would send responders the wrong advice.
        motion = ramp(speed, 0.30, 0.80)
        # A surge is transient. Once compression has been held for a long time
        # the situation has stopped being a wave and become a crush, and the
        # advice diverges completely, so decay the surge score as it persists.
        transient = 1.0 - 0.75 * ramp(sustained, 8.0, 30.0)
        s = gate * turb * motion * transient * (0.70 + 0.30 * ramp(oscillation, 0.15, 0.55))
        ev = []
        if oscillation > 0.2:
            ev.append(f"stop-and-go oscillation index {oscillation:.2f}")
        if pressure > self.p_warn:
            ev.append(f"pressure {pressure:.4f} above Helbing warning level {self.p_warn:.3f}")
        if gate > 0:
            ev.append(f"supporting density {density:.2f}/m2")
        scores.append(self._mk("TURBULENT_SURGE", min(1.0, s), ev))

        # -- Panic dispersal -----------------------------------------------
        # Two ways in: a relative surge above the site baseline, or an absolute
        # speed that simply is not walking. A crowd averaging over 2 m/s is
        # running, and a running crowd is abnormal regardless of history.
        gate = max(ramp(speed_surge, 1.5, 2.6), ramp(speed, 1.9, 2.8))
        s = gate * (0.45 + 0.35 * ramp(disorder, 0.30, 0.75) + 0.20 * ramp(-density_rate, 0.2, 1.2))
        ev = []
        if speed_surge > 1.5:
            ev.append(f"mean speed {speed_surge:.1f}x the site baseline")
        if speed > 1.9:
            ev.append(f"crowd is running, not walking ({speed:.1f} m/s)")
        if disorder > 0.3:
            ev.append(f"scattering directions (disorder {disorder:.2f})")
        if density_rate < -0.2:
            ev.append("density falling while speed rises -- escape behaviour")
        scores.append(self._mk("PANIC_DISPERSAL", min(1.0, s), ev))

        # -- Static blockage -----------------------------------------------
        # Deliberately banded: above the crush density this is no longer a
        # blockage but a crush, and the response escalates accordingly.
        gate = ramp(density, 2.2, 4.0) * inv_ramp(density, 5.5, 7.5) * inv_ramp(speed, 0.35, 1.05)
        s = gate * (
            0.34
            + 0.20 * ramp(1.0 - flux_eff, 0.2, 0.7)
            + 0.20 * ramp(sustained, 3.0, 20.0)
            + 0.26 * ramp(spread, 0.55, 0.90)      # stopped across a broad front
        )
        ev = []
        if gate > 0:
            ev.append(f"dense crowd at a standstill ({density:.2f}/m2, {speed:.2f} m/s)")
        if spread > 0.6:
            ev.append(f"stopped across a broad front, not funnelled ({spread * 100:.0f}% of the corridor width)")
        if flux_eff < 0.7:
            ev.append(f"flux at {flux_eff * 100:.0f}% of observed peak -- not discharging")
        scores.append(self._mk("STATIC_BLOCKAGE", min(1.0, s), ev))

        # -- Counter-flow conflict -----------------------------------------
        # Two necessary conditions, not one. Directional coherence collapses in
        # a jam as well, because a stalled crowd shuffles in random micro-
        # directions -- so disorder alone over-fires on every dense scene.
        # Genuine counter-flow additionally requires people to still be MOVING:
        # opposing streams cancel each other's progress while individuals keep
        # walking. A jam is disordered and slow; counter-flow is disordered and
        # fast.
        # Gated on the circular bimodality index, which fires only when the
        # motion actually splits into two opposing axes. Raw disorder cannot do
        # this job: a jammed crowd is disordered too, and telling marshals to
        # separate two streams inside a jam is the wrong instruction.
        # Counter-flow is a free-flow phenomenon: two streams passing THROUGH
        # each other at walking pace. Above the jam density nobody is streaming
        # anywhere, so whatever backward motion remains is crush pressure, not
        # counter-flow, and telling marshals to separate streams inside a jam
        # would be actively dangerous.
        gate = (
            ramp(counterflow, 0.18, 0.55)
            * ramp(speed, 0.45, 0.95)
            * inv_ramp(density, 4.0, 6.5)
        )
        s = gate * (0.55 + 0.25 * ramp(density, 1.5, 4.0) + 0.20 * ramp(disorder, 0.3, 0.7))
        ev = []
        if gate > 0:
            ev.append(f"two opposing movement axes detected (counter-flow index {counterflow:.2f}) "
                      f"with the crowd still moving at {speed:.2f} m/s")
        if density > 1.5:
            ev.append(f"under load at {density:.2f}/m2")
        scores.append(self._mk("COUNTERFLOW_CONFLICT", min(1.0, s), ev))

        # -- Bottleneck congestion -----------------------------------------
        gate = (
            ramp(bottleneck, 0.35, 0.80)
            * ramp(constriction, 1.15, 1.90)
            * inv_ramp(spread, 0.55, 0.90)
        )
        s = gate * (0.45 + 0.30 * ramp(1.0 - flux_eff, 0.20, 0.70) + 0.25 * ramp(density, 2.0, 4.5))
        ev = []
        if gate > 0:
            ev.append(f"chokepoint at {bottleneck * 100:.0f}% of design capacity, "
                      f"holding {constriction:.1f}x its share of the crowd")
        if flux_eff < 0.8:
            ev.append(f"throughput on the congested branch (flux efficiency {flux_eff:.2f})")
        scores.append(self._mk("BOTTLENECK_CONGESTION", min(1.0, s), ev))

        # -- Rapid influx ---------------------------------------------------
        gate = max(ramp(density_rate, 0.90, 3.00), ramp(count_rate, 45.0, 160.0))
        # Only meaningful once there is a crowd to speak of, and only while the
        # density is still sub-critical -- past that it is no longer an influx
        # problem, it is a density problem.
        gate *= ramp(density, 0.35, 1.20)
        s = gate * (0.55 + 0.45 * ramp(density, 1.0, 3.5)) * inv_ramp(density, 4.0, 6.0)
        ev = []
        if density_rate > 0.25:
            ev.append(f"density climbing {density_rate:.2f}/m2 per minute")
        if count_rate > 12:
            ev.append(f"occupancy climbing {count_rate:.0f} people per minute")
        scores.append(self._mk("RAPID_INFLUX", min(1.0, s), ev))

        # -- Normal flow -----------------------------------------------------
        worst = max((t.score for t in scores), default=0.0)
        normal_ev = []
        if mean_density < 2.0:
            normal_ev.append(f"mean density {mean_density:.2f}/m2 inside the free-flow band")
        if disorder < 0.3:
            normal_ev.append("coherent directional movement")
        scores.append(self._mk("NORMAL_FLOW", float(max(0.0, 1.0 - worst)), normal_ev))

        order = {c: i for i, c in enumerate(ORDERED_CODES)}
        ranked = sorted(scores, key=lambda t: (-t.weighted, order.get(t.code, 99)))
        primary = ranked[0]

        # Hazard index: severity-weighted aggregate over the non-normal types,
        # so two co-occurring moderate failure modes are not treated as benign.
        hazardous = [t.weighted for t in scores if t.code != "NORMAL_FLOW"]
        hazardous.sort(reverse=True)
        hazard = 0.0
        for i, v in enumerate(hazardous[:3]):
            hazard += v * (0.6 ** i)
        hazard = float(min(1.0, hazard))

        return TypologyResult(primary=primary, ranked=ranked, hazard_index=hazard)

    @staticmethod
    def _mk(code: str, score: float, evidence: List[str]) -> RiskTypeScore:
        spec = RISK_TYPES[code]
        return RiskTypeScore(
            code=spec.code,
            label=spec.label,
            score=float(max(0.0, min(1.0, score))),
            severity=spec.severity,
            icon=spec.icon,
            color=spec.color,
            evidence=evidence,
        )
