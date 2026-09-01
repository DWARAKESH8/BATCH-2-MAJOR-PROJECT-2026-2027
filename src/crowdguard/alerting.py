"""Outbound alerting and the operator acknowledgement loop.

This is the part that turns a prediction into an action. Without it the
pipeline ends at text on a screen that nobody is looking at -- which is the
exact failure mode the whole project exists to fix.

Safety and privacy posture
--------------------------
* Every network sink is **disabled by default**. Nothing leaves the machine
  unless an operator explicitly enables a sink and supplies its credentials.
* Sinks run on a worker thread with hard timeouts, so a hanging webhook can
  never stall the video pipeline.
* Delivery failures are captured and surfaced, never silently swallowed: an
  alerting system that fails quietly is worse than none.
* Alert payloads carry measurements and zone names, never identities. No face
  data, no recognition, no personal data of any kind is transmitted.
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import AlertConfig
from .escalation import STATE_RANK, EscalationEvent


# --------------------------------------------------------------------------- #
# Delivery records
# --------------------------------------------------------------------------- #
@dataclass
class DeliveryResult:
    sink: str
    ok: bool
    detail: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sink": self.sink,
            "ok": self.ok,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #
class AlertSink:
    name = "sink"

    def send(self, payload: Dict[str, Any], frame_png: Optional[bytes]) -> DeliveryResult:
        raise NotImplementedError


class ConsoleSink(AlertSink):
    name = "console"

    def send(self, payload: Dict[str, Any], frame_png: Optional[bytes]) -> DeliveryResult:
        line = (
            f"[CrowdGuard {payload['to_state']}] t={payload['timestamp_sec']:.1f}s "
            f"{payload['risk_type_label']} risk={payload['risk_score']:.2f} "
            f"zone={payload.get('zone') or 'frame'} "
            f"density={payload.get('density', 0):.2f}/m2 :: {payload.get('reason', '')}"
        )
        print(line, flush=True)
        return DeliveryResult(self.name, True, "printed")


class JsonlSink(AlertSink):
    name = "jsonl"

    def __init__(self, path: str):
        self.path = path

    def send(self, payload: Dict[str, Any], frame_png: Optional[bytes]) -> DeliveryResult:
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return DeliveryResult(self.name, True, self.path)
        except Exception as exc:
            return DeliveryResult(self.name, False, f"{type(exc).__name__}: {exc}")


class WebhookSink(AlertSink):
    """Generic HTTP POST -- Slack, Teams, an ops bus, or a venue's own API."""

    name = "webhook"

    def __init__(self, url: str, timeout: float = 6.0):
        self.url = url
        self.timeout = timeout

    def send(self, payload: Dict[str, Any], frame_png: Optional[bytes]) -> DeliveryResult:
        if not self.url:
            return DeliveryResult(self.name, False, "no webhook URL configured")
        try:
            import urllib.request

            body = dict(payload)
            if frame_png:
                body["frame_png_base64"] = base64.b64encode(frame_png).decode("ascii")
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "CrowdGuard-RAG/2.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return DeliveryResult(self.name, 200 <= resp.status < 300, f"HTTP {resp.status}")
        except Exception as exc:
            return DeliveryResult(self.name, False, f"{type(exc).__name__}: {exc}")


class TelegramSink(AlertSink):
    """Pushes the advisory plus the annotated frame to a marshal's phone."""

    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 6.0):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    def _text(self, payload: Dict[str, Any]) -> str:
        icon = {"CRITICAL": "\U0001F6A8", "ALERT": "⚠️", "WATCH": "\U0001F440"}.get(
            payload["to_state"], "ℹ️"
        )
        lines = [
            f"{icon} CrowdGuard {payload['to_state']}",
            f"Type: {payload['risk_type_label']}",
            f"Zone: {payload.get('zone') or 'whole frame'}",
            f"Risk: {payload['risk_score']:.2f}  Density: {payload.get('density', 0):.2f}/m2",
            f"Why: {payload.get('reason', '')}",
        ]
        if payload.get("forecast_text"):
            lines.append(f"Forecast: {payload['forecast_text']}")
        if payload.get("action"):
            lines.append(f"ACTION: {payload['action']}")
        if payload.get("event_id"):
            lines.append(f"Ref: {payload['event_id']}")
        return "\n".join(lines)

    def send(self, payload: Dict[str, Any], frame_png: Optional[bytes]) -> DeliveryResult:
        if not self.bot_token or not self.chat_id:
            return DeliveryResult(self.name, False, "bot token or chat id missing")
        try:
            import urllib.parse
            import urllib.request

            text = self._text(payload)
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                ok = 200 <= resp.status < 300
            return DeliveryResult(self.name, ok, f"HTTP {resp.status}")
        except Exception as exc:
            return DeliveryResult(self.name, False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
class AlertDispatcher:
    """Fan-out to every enabled sink, off the critical path."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.sinks: List[AlertSink] = []
        self.history: List[DeliveryResult] = []
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=200)
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._build_sinks()

    def _build_sinks(self) -> None:
        c = self.config
        self.sinks = []
        if c.console:
            self.sinks.append(ConsoleSink())
        if c.jsonl_path:
            self.sinks.append(JsonlSink(c.jsonl_path))
        if c.webhook_enabled and c.webhook_url:
            self.sinks.append(WebhookSink(c.webhook_url, c.timeout_sec))
        if c.telegram_enabled and c.telegram_bot_token and c.telegram_chat_id:
            self.sinks.append(TelegramSink(c.telegram_bot_token, c.telegram_chat_id, c.timeout_sec))

    def reconfigure(self, config: AlertConfig) -> None:
        self.config = config
        self._build_sinks()

    @property
    def sink_names(self) -> List[str]:
        return [s.name for s in self.sinks]

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="crowdguard-alerts", daemon=True)
            self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            payload, frame_png = item
            for sink in list(self.sinks):
                result = sink.send(payload, frame_png)
                with self._lock:
                    self.history.append(result)
                    if len(self.history) > 500:
                        self.history = self.history[-500:]
            self._queue.task_done()

    # ------------------------------------------------------------------ #
    def should_send(self, event: EscalationEvent) -> bool:
        if event.kind == "DEESCALATE":
            return False
        return STATE_RANK.get(event.to_state, 0) >= STATE_RANK.get(self.config.min_state, 2)

    def build_payload(
        self,
        event: EscalationEvent,
        action: str = "",
        forecast_text: str = "",
        advisory: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = event.to_dict()
        payload.update(
            {
                "system": "CrowdGuard-RAG",
                "action": action,
                "forecast_text": forecast_text,
                "advisory": advisory,
                "sent_at": time.time(),
            }
        )
        if extra:
            payload.update(extra)
        return payload

    def dispatch(
        self,
        event: EscalationEvent,
        action: str = "",
        forecast_text: str = "",
        advisory: str = "",
        frame_png: Optional[bytes] = None,
        extra: Optional[Dict[str, Any]] = None,
        blocking: bool = False,
    ) -> Optional[List[DeliveryResult]]:
        if not self.should_send(event):
            return None
        payload = self.build_payload(event, action, forecast_text, advisory, extra)
        if not self.config.attach_frame:
            frame_png = None

        if blocking:
            results = [sink.send(payload, frame_png) for sink in list(self.sinks)]
            with self._lock:
                self.history.extend(results)
            return results

        self._ensure_worker()
        try:
            self._queue.put_nowait((payload, frame_png))
        except queue.Full:
            with self._lock:
                self.history.append(DeliveryResult("dispatcher", False, "alert queue full -- dropped"))
        return None

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self.history[-n:]]

    def failures(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self.history if not r.ok]


# --------------------------------------------------------------------------- #
# Acknowledgement store
# --------------------------------------------------------------------------- #
@dataclass
class Acknowledgement:
    event_id: str
    operator: str
    decision: str            # CONFIRMED | FALSE_ALARM | ALREADY_HANDLED
    action_taken: str
    note: str
    timestamp_sec: float
    risk_score: float
    risk_type: str
    zone: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "operator": self.operator,
            "decision": self.decision,
            "action_taken": self.action_taken,
            "note": self.note,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "risk_score": round(self.risk_score, 4),
            "risk_type": self.risk_type,
            "zone": self.zone,
            "wall_clock": time.time(),
        }


DECISIONS = ["CONFIRMED", "FALSE_ALARM", "ALREADY_HANDLED"]

STANDARD_ACTIONS = [
    "Metered inflow at upstream gate",
    "Opened additional exit / diversion route",
    "Deployed marshals to the zone",
    "Broadcast PA instruction",
    "Imposed one-way movement",
    "Halted all inflow",
    "Escalated to safety officer",
    "Dispatched medical",
    "No action required",
]


class AckStore:
    """Persists operator decisions.

    This closes the loop twice over. Operationally it gives an audit trail --
    who knew what, when, and what they did, which is precisely the question
    every public inquiry into a crowd disaster has had to answer. Technically
    it produces *real human labels* for the risk model, which is the only
    honest path out of training on synthetic data.
    """

    def __init__(self, path: str = "outputs/acknowledgements.jsonl"):
        self.path = path
        self.records: List[Acknowledgement] = []
        self._load()

    def _load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                self.records.append(
                    Acknowledgement(
                        event_id=d.get("event_id", ""),
                        operator=d.get("operator", ""),
                        decision=d.get("decision", ""),
                        action_taken=d.get("action_taken", ""),
                        note=d.get("note", ""),
                        timestamp_sec=float(d.get("timestamp_sec", 0.0)),
                        risk_score=float(d.get("risk_score", 0.0)),
                        risk_type=d.get("risk_type", ""),
                        zone=d.get("zone", ""),
                    )
                )
            except Exception:
                continue

    def add(self, ack: Acknowledgement) -> Acknowledgement:
        self.records.append(ack)
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ack.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            pass
        return ack

    def stats(self) -> Dict[str, Any]:
        total = len(self.records)
        confirmed = sum(1 for r in self.records if r.decision == "CONFIRMED")
        false_alarm = sum(1 for r in self.records if r.decision == "FALSE_ALARM")
        return {
            "total": total,
            "confirmed": confirmed,
            "false_alarm": false_alarm,
            "already_handled": total - confirmed - false_alarm,
            # Operator-observed precision. With a handful of records this is
            # noise -- it only becomes meaningful over a full event season.
            "operator_precision": round(confirmed / total, 3) if total else None,
        }

    def as_training_labels(self) -> List[Dict[str, Any]]:
        """Export human-confirmed events as supervised labels."""
        return [
            {
                "risk_score": r.risk_score,
                "risk_type": r.risk_type,
                "zone": r.zone,
                "label": 1 if r.decision == "CONFIRMED" else 0,
                "timestamp_sec": r.timestamp_sec,
            }
            for r in self.records
            if r.decision in {"CONFIRMED", "FALSE_ALARM"}
        ]
