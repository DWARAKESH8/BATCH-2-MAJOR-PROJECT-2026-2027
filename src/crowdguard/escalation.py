"""Escalation state machine.

Why this exists
---------------
A per-frame classifier that fires the instant a score crosses a line produces
alarm *flapping*: dozens of alerts a minute as the score jitters around the
threshold. Operators respond to flapping by muting the system, which is how
safety systems fail in the real world -- not by being wrong, but by being
ignored.

Three mechanisms prevent that:

* **Dwell time** -- the condition must be sustained before the state rises.
* **Hysteresis** -- de-escalation happens at a lower score than escalation, so
  a score hovering on a boundary cannot oscillate between states.
* **Acknowledgement suppression** -- once an operator has accepted an alert,
  the system stops re-notifying for a configured period.

CRITICAL is deliberately given the *shortest* dwell time: when the crowd is
already in the crush band, a slow alarm is a useless alarm.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import EscalationConfig

STATES: List[str] = ["NORMAL", "WATCH", "ALERT", "CRITICAL"]
STATE_RANK: Dict[str, int] = {s: i for i, s in enumerate(STATES)}

STATE_META: Dict[str, Dict[str, str]] = {
    "NORMAL": {
        "color": "#2ecc71",
        "icon": "●",
        "posture": "Routine monitoring",
        "who": "Control room only",
    },
    "WATCH": {
        "color": "#f1c40f",
        "icon": "◐",
        "posture": "Heightened vigilance -- prepare contingency routes",
        "who": "Control room + zone supervisor",
    },
    "ALERT": {
        "color": "#f5a623",
        "icon": "▲",
        "posture": "Active intervention -- meter inflow, deploy marshals",
        "who": "Zone supervisor + marshals on the ground",
    },
    "CRITICAL": {
        "color": "#ff1744",
        "icon": "⬤",
        "posture": "Emergency response -- halt inflow, open all egress",
        "who": "Safety officer, medical, and all marshals",
    },
}


@dataclass
class EscalationEvent:
    """Something an operator needs to see."""

    event_id: str
    kind: str                      # ESCALATE | DEESCALATE | RENOTIFY
    from_state: str
    to_state: str
    timestamp_sec: float
    risk_score: float
    risk_type: str
    risk_type_label: str
    zone: str
    density: float
    reason: str
    notify_role: str = ""          # who this alert is addressed to
    trend: str = ""                # what changed over the last minute
    acknowledged: bool = False
    ack_operator: str = ""
    ack_action: str = ""
    ack_timestamp_sec: Optional[float] = None
    advisory: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "risk_score": round(self.risk_score, 4),
            "risk_type": self.risk_type,
            "risk_type_label": self.risk_type_label,
            "zone": self.zone,
            "density": round(self.density, 3),
            "reason": self.reason,
            "notify_role": self.notify_role,
            "trend": self.trend,
            "acknowledged": self.acknowledged,
            "ack_operator": self.ack_operator,
            "ack_action": self.ack_action,
            "ack_timestamp_sec": self.ack_timestamp_sec,
        }


@dataclass
class EscalationStatus:
    state: str
    previous_state: str
    since_sec: float
    dwell_sec: float
    candidate_state: str
    candidate_progress: float      # 0-1 toward the pending transition
    acknowledged: bool
    events: List[EscalationEvent] = field(default_factory=list)

    @property
    def meta(self) -> Dict[str, str]:
        return STATE_META[self.state]

    @property
    def rank(self) -> int:
        return STATE_RANK[self.state]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "previous_state": self.previous_state,
            "dwell_sec": round(self.dwell_sec, 1),
            "candidate_state": self.candidate_state,
            "candidate_progress": round(self.candidate_progress, 3),
            "acknowledged": self.acknowledged,
        }


class EscalationController:
    """Turns a stream of risk scores into a small number of meaningful events."""

    # Time below the threshold drains the dwell accumulator at this fraction
    # of the rate at which time above it fills the accumulator.
    LEAK_RATE = 0.5

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config or EscalationConfig()
        self.state: str = "NORMAL"
        self.previous_state: str = "NORMAL"
        self.state_since: float = 0.0
        self._candidate: Optional[str] = None
        self._candidate_dir: int = 0
        self._dwell_accum: float = 0.0
        self._last_update: Optional[float] = None
        self._last_notify: Optional[float] = None
        self._acknowledged_until: Optional[float] = None
        self._acknowledged: bool = False
        self.events: List[EscalationEvent] = []
        self._started: bool = False

    # ------------------------------------------------------------------ #
    def _instant_state(self, score: float) -> str:
        c = self.config
        # Escalation uses the raise thresholds; de-escalation uses the raise
        # threshold minus the hysteresis margin. Which set applies depends on
        # where we currently are, which is what makes this hysteretic.
        rank = STATE_RANK[self.state]
        h = c.hysteresis

        crit = c.critical_raise - (h if rank >= 3 else 0.0)
        alert = c.alert_raise - (h if rank >= 2 else 0.0)
        watch = c.watch_raise - (h if rank >= 1 else 0.0)

        if score >= crit:
            return "CRITICAL"
        if score >= alert:
            return "ALERT"
        if score >= watch:
            return "WATCH"
        return "NORMAL"

    def _dwell_for(self, target: str) -> float:
        c = self.config
        if STATE_RANK[target] < STATE_RANK[self.state]:
            return c.dwell_deescalate_sec
        return {
            "WATCH": c.dwell_watch_sec,
            "ALERT": c.dwell_alert_sec,
            "CRITICAL": c.dwell_critical_sec,
        }.get(target, c.dwell_watch_sec)

    # ------------------------------------------------------------------ #
    def update(self, features: Any, timestamp_sec: float, advisory: str = "") -> EscalationStatus:
        score = float(getattr(features, "risk_score", 0.0))
        if not self._started:
            self.state_since = timestamp_sec
            self._last_update = timestamp_sec
            self._started = True

        dt = max(0.0, timestamp_sec - (self._last_update if self._last_update is not None else timestamp_sec))
        self._last_update = timestamp_sec

        target = self._instant_state(score)
        emitted: List[EscalationEvent] = []

        # ---- leaky dwell accumulator -------------------------------------
        # A plain "how long has the condition held continuously" clock does not
        # survive contact with real data. A crowd sitting on a threshold flips
        # above and below it every few frames; every dip resets the clock, and
        # the alert never fires at all -- the system goes quiet exactly while
        # the situation is at its worst. Measured on a bottleneck run holding
        # risk 0.79: one escalation to WATCH, then silence for fifty seconds.
        #
        # So the condition must be PREDOMINANTLY sustained rather than
        # continuously sustained. Time above the threshold accumulates; time
        # below it drains at half the rate. Brief dips slow the escalation
        # without cancelling it, and a genuine return to calm still drains the
        # accumulator to zero. This is the standard leaky-bucket approach used
        # in industrial alarm systems, for exactly this reason.
        if target == self.state:
            self._dwell_accum = max(0.0, self._dwell_accum - dt * self.LEAK_RATE)
            if self._dwell_accum <= 0.0:
                self._candidate, self._candidate_dir = None, 0
        else:
            direction = 1 if STATE_RANK[target] > STATE_RANK[self.state] else -1
            if self._candidate_dir != direction:
                self._dwell_accum, self._candidate_dir, self._candidate = dt, direction, target
            else:
                self._dwell_accum += dt
                extreme = max if direction > 0 else min
                self._candidate = extreme(self._candidate or target, target,
                                          key=lambda x: STATE_RANK[x])

        candidate = self._candidate or self.state
        required = self._dwell_for(candidate) if candidate != self.state else 1.0
        progress = min(1.0, self._dwell_accum / max(1e-6, required)) if candidate != self.state else 0.0

        if candidate != self.state and self._dwell_accum >= required:
            # Move one rung at a time. Jumping NORMAL -> CRITICAL skips the
            # WATCH and ALERT notifications, and those early warnings are the
            # entire point: they reach someone while action is still cheap.
            # Stepping costs no delay -- the next frame steps again -- but it
            # produces a complete, ordered record of who was told what, when.
            rank_now, rank_target = STATE_RANK[self.state], STATE_RANK[candidate]
            stepped = STATES[rank_now + (1 if rank_target > rank_now else -1)]
            kind = "ESCALATE" if rank_target > rank_now else "DEESCALATE"
            reason = (f"risk {score:.2f} sustained for {self._dwell_accum:.0f}s "
                      f"(threshold dwell {required:.0f}s)")
            event = self._make_event(kind, self.state, stepped, timestamp_sec,
                                     features, reason, advisory)
            self.previous_state, self.state = self.state, stepped
            self.state_since = timestamp_sec
            self._dwell_accum = 0.0
            self._candidate, self._candidate_dir = None, 0
            progress = 0.0
            if kind == "ESCALATE":
                # A fresh escalation invalidates any prior acknowledgement.
                self._acknowledged, self._acknowledged_until = False, None
                self._last_notify = timestamp_sec
            emitted.append(event)

        # ---- periodic re-notification while unacknowledged ----------------
        if STATE_RANK[self.state] >= STATE_RANK["ALERT"] and not emitted:
            suppressed = (self._acknowledged_until is not None
                          and timestamp_sec < self._acknowledged_until)
            due = (self._last_notify is None
                   or (timestamp_sec - self._last_notify) >= self.config.renotify_sec)
            if due and not suppressed:
                reason = (f"{self.state} unresolved for "
                          f"{timestamp_sec - self.state_since:.0f}s at risk {score:.2f}")
                emitted.append(self._make_event("RENOTIFY", self.state, self.state,
                                                timestamp_sec, features, reason, advisory))
                self._last_notify = timestamp_sec

        if self._acknowledged_until is not None and timestamp_sec >= self._acknowledged_until:
            self._acknowledged, self._acknowledged_until = False, None

        self.events.extend(emitted)
        return EscalationStatus(
            state=self.state, previous_state=self.previous_state, since_sec=self.state_since,
            dwell_sec=timestamp_sec - self.state_since,
            candidate_state=self._candidate or self.state,
            candidate_progress=progress, acknowledged=self._acknowledged, events=emitted,
        )

    # ------------------------------------------------------------------ #
    def _make_event(
        self,
        kind: str,
        from_state: str,
        to_state: str,
        timestamp_sec: float,
        features: Any,
        reason: str,
        advisory: str,
    ) -> EscalationEvent:
        return EscalationEvent(
            event_id=uuid.uuid4().hex[:12],
            kind=kind,
            from_state=from_state,
            to_state=to_state,
            timestamp_sec=float(timestamp_sec),
            risk_score=float(getattr(features, "risk_score", 0.0)),
            risk_type=str(getattr(features, "primary_risk_type", "")),
            risk_type_label=str(getattr(features, "primary_risk_label", "")),
            zone=str(getattr(features, "hotspot_zone", "")),
            density=float(getattr(features, "local_density_peak", 0.0)),
            reason=reason,
            # An alert with no addressee is a log line, not a notification.
            notify_role=STATE_META.get(to_state, {}).get("who", ""),
            trend=str(getattr(features, "trend_summary", "") or ""),
            advisory=advisory,
        )

    # ------------------------------------------------------------------ #
    def acknowledge(
        self,
        event_id: str,
        operator: str,
        action: str,
        timestamp_sec: float,
    ) -> Optional[EscalationEvent]:
        """Operator accepts an alert. Suppresses re-notification and records a label."""
        for event in reversed(self.events):
            if event.event_id == event_id:
                event.acknowledged = True
                event.ack_operator = operator
                event.ack_action = action
                event.ack_timestamp_sec = float(timestamp_sec)
                self._acknowledged = True
                self._acknowledged_until = timestamp_sec + self.config.ack_suppression_sec
                self._last_notify = timestamp_sec
                return event
        return None

    def open_events(self) -> List[EscalationEvent]:
        return [e for e in self.events if not e.acknowledged and e.kind != "DEESCALATE"]

    def summary(self) -> Dict[str, Any]:
        return {
            "current_state": self.state,
            "total_events": len(self.events),
            "escalations": sum(1 for e in self.events if e.kind == "ESCALATE"),
            "renotifications": sum(1 for e in self.events if e.kind == "RENOTIFY"),
            "acknowledged": sum(1 for e in self.events if e.acknowledged),
            "unacknowledged": len(self.open_events()),
        }
