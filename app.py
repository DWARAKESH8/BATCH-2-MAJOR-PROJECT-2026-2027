"""CrowdGuard-RAG — control-room dashboard.

The interface answers four questions without the operator reading a paragraph:

    What is happening?      the failure mode, named
    Where?                  the zone, named
    How long have I got?    the forecast countdown
    What do I do?           ranked actions, and what not to do

Everything on the page is subordinate to those four.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.crowdguard.alerting import DECISIONS, STANDARD_ACTIONS, AlertConfig
from src.crowdguard.config import (
    CalibrationConfig,
    EscalationConfig,
    ForecastConfig,
    LLMConfig,
    RAGConfig,
    RiskConfig,
    VisionConfig,
    ZoneConfig,
)
from src.crowdguard.escalation import STATE_META, STATES
from src.crowdguard.head_detection import HeadDetectorConfig, TopDownHeadDetector
from src.crowdguard.pipeline import CrowdGuardPipeline, OverlayOptions
from src.crowdguard.risk_taxonomy import ORDERED_CODES, RISK_TYPES
from src.crowdguard.simulator import SCENARIOS, WORLD_D, WORLD_W, CrowdSimulator
from src.crowdguard.utils import density_field_image, generate_incident_report, resolve_youtube_url
from src.crowdguard.vision import boxes_to_centroids

st.set_page_config(
    page_title="CrowdGuard-RAG",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_STORED_FRAMES = 240
REPLAY_WIDTH = 720          # stored replay frames are downscaled to this width


def _pack(frame: Optional[np.ndarray]) -> Optional[bytes]:
    """JPEG-encode a frame for replay storage.

    Three panels per frame at full resolution would put well over a gigabyte of
    raw arrays into session state for a single run. JPEG at 80% brings that to
    tens of megabytes with no visible loss at review size.
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > REPLAY_WIDTH:
        frame = cv2.resize(frame, (REPLAY_WIDTH, int(h * REPLAY_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


def _unpack(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def _show(slot: Any, frame: Optional[np.ndarray], caption: str) -> None:
    if frame is not None:
        slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                   use_container_width=True, caption=caption)

# Muted, print-legible palette. Saturated enough to read on white, restrained
# enough that four states on one screen do not fight each other.
C_OK, C_WATCH, C_ALERT, C_CRIT = "#1a7f52", "#a67908", "#c2650d", "#b32436"
C_ACCENT, C_INK, C_MUTED, C_DIM = "#1f5fbf", "#14161a", "#5b6472", "#8a929e"
STATE_HEX = {"NORMAL": C_OK, "WATCH": C_WATCH, "ALERT": C_ALERT, "CRITICAL": C_CRIT}


# =========================================================================== #
# Styling
# =========================================================================== #
CSS = f"""
<style>
:root{{
  --panel:#ffffff; --panel-2:#f7f8fa;
  --line:#e3e6ec; --line-2:#cbd1da;
  --ink:{C_INK}; --muted:{C_MUTED}; --dim:{C_DIM};
  --accent:{C_ACCENT};
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
[data-testid="stAppViewContainer"]{{background:#ffffff;}}
[data-testid="stHeader"]{{background:transparent;}}
[data-testid="stToolbar"]{{display:none;}}
[data-testid="stSidebar"]{{background:#fafbfc;border-right:1px solid var(--line);}}
.block-container{{padding-top:1.4rem;padding-bottom:3rem;max-width:1500px;}}
body,p,span,div,label,li,td,th{{color:var(--ink);}}
#MainMenu,footer{{visibility:hidden;}}
h1,h2,h3,h4{{color:var(--ink);letter-spacing:-.015em;}}

/* ---------- header ---------- */
.cg-head{{display:flex;align-items:flex-end;gap:20px;flex-wrap:wrap;
         padding-bottom:14px;margin-bottom:18px;border-bottom:2px solid var(--ink);}}
.cg-title{{font-size:1.42rem;font-weight:700;letter-spacing:-.025em;line-height:1.1;}}
.cg-tag{{color:var(--muted);font-size:.75rem;letter-spacing:.09em;
        text-transform:uppercase;margin-top:3px;}}
.cg-chips{{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;}}
.cg-chip{{font-family:var(--mono);font-size:.66rem;padding:3px 9px;border-radius:3px;
         border:1px solid var(--line-2);background:var(--panel-2);color:var(--muted);
         white-space:nowrap;}}
.cg-chip b{{color:var(--ink);font-weight:600;}}

/* ---------- status banner ---------- */
.cg-status{{border:1px solid var(--line);border-left:5px solid var(--sc);border-radius:4px;
           padding:16px 20px;margin-bottom:14px;background:var(--panel-2);}}
.cg-status-row{{display:flex;align-items:center;gap:30px;flex-wrap:wrap;}}
.cg-lab{{font-size:.6rem;letter-spacing:.16em;color:var(--dim);font-weight:700;
        text-transform:uppercase;margin-bottom:5px;}}
.cg-state{{font-size:1.72rem;font-weight:700;letter-spacing:-.03em;line-height:1;color:var(--sc);}}
.cg-mode{{font-size:1.14rem;font-weight:650;letter-spacing:-.015em;line-height:1.25;}}
.cg-mech{{color:var(--muted);font-size:.8rem;margin-top:5px;max-width:64ch;line-height:1.5;}}
.cg-budget{{margin-left:auto;text-align:right;}}
.cg-budget .v{{font-family:var(--mono);font-size:1.5rem;font-weight:700;line-height:1;color:var(--sc);}}
.cg-budget .s{{font-size:.66rem;color:var(--muted);margin-top:4px;}}

/* ---------- metric cards ---------- */
.cg-kpi{{border:1px solid var(--line);border-radius:4px;background:var(--panel);
        padding:11px 13px;height:100%;}}
.cg-kpi .k{{font-size:.6rem;letter-spacing:.11em;color:var(--dim);
           text-transform:uppercase;font-weight:700;}}
.cg-kpi .v{{font-family:var(--mono);font-size:1.34rem;font-weight:700;
           margin:6px 0 2px;letter-spacing:-.02em;color:var(--ink);}}
.cg-kpi .u{{font-size:.68rem;color:var(--muted);font-weight:500;}}
.cg-kpi .bar{{height:3px;border-radius:2px;background:#eceef2;margin-top:8px;overflow:hidden;}}
.cg-kpi .bar i{{display:block;height:100%;border-radius:2px;}}
.cg-kpi .n{{font-size:.62rem;color:var(--dim);margin-top:6px;line-height:1.35;}}

/* ---------- panels ---------- */
.cg-card{{border:1px solid var(--line);border-radius:4px;background:var(--panel);
         padding:15px 17px;margin-bottom:12px;}}
.cg-h{{font-size:.62rem;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);
      font-weight:700;margin-bottom:11px;}}
.cg-note{{font-size:.74rem;color:var(--muted);line-height:1.55;}}

/* ---------- ranked bars ---------- */
.cg-row{{display:flex;align-items:center;gap:9px;margin-bottom:7px;}}
.cg-row .sw{{width:9px;height:9px;border-radius:2px;flex:0 0 9px;}}
.cg-row .nm{{font-size:.775rem;flex:0 0 150px;color:var(--ink);}}
.cg-row .tr{{flex:1;height:6px;border-radius:3px;background:#eceef2;overflow:hidden;}}
.cg-row .tr i{{display:block;height:100%;border-radius:3px;}}
.cg-row .pc{{font-family:var(--mono);font-size:.7rem;color:var(--muted);width:36px;text-align:right;}}
.cg-row.top .nm{{font-weight:700;}}

/* ---------- ladder ---------- */
.cg-ladder{{display:flex;gap:8px;}}
.cg-rung{{flex:1;border:1px solid var(--line);border-radius:4px;padding:10px 12px;
         background:var(--panel);position:relative;}}
.cg-rung.act{{border-color:var(--rc);border-width:2px;background:#fbfcfd;}}
.cg-rung.past{{background:var(--panel-2);}}
.cg-rung .n{{font-size:.74rem;font-weight:700;letter-spacing:.04em;}}
.cg-rung .d{{font-size:.66rem;color:var(--muted);margin-top:4px;line-height:1.4;}}
.cg-rung .w{{font-size:.61rem;color:var(--dim);margin-top:6px;}}
.cg-rung .prog{{position:absolute;left:0;bottom:0;height:3px;background:var(--rc);}}

/* ---------- zones ---------- */
.cg-zone{{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #eef0f4;}}
.cg-zone:last-child{{border-bottom:none;}}
.cg-zone .sw{{width:9px;height:9px;border-radius:2px;flex:0 0 9px;}}
.cg-zone .nm{{font-size:.79rem;flex:1;}}
.cg-zone .kd{{font-size:.6rem;color:var(--dim);letter-spacing:.07em;text-transform:uppercase;}}
.cg-zone .mt{{font-family:var(--mono);font-size:.73rem;color:var(--muted);
             text-align:right;min-width:100px;}}
.cg-zone .mt b{{color:var(--ink);}}

/* ---------- advisory ---------- */
.cg-adv{{border:1px solid var(--line);border-left:4px solid var(--sc);border-radius:4px;
        background:var(--panel);padding:16px 19px;}}
.cg-adv h4{{font-size:.61rem;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);
           margin:0 0 6px;font-weight:700;}}
.cg-adv p{{font-size:.85rem;line-height:1.62;margin:0 0 15px;color:var(--ink);}}
.cg-adv ol{{margin:0 0 15px;padding-left:19px;}}
.cg-adv li{{font-size:.85rem;line-height:1.6;margin-bottom:7px;}}
.cg-adv .who{{font-family:var(--mono);font-size:.68rem;color:var(--accent);font-weight:600;}}
.cg-box{{border-radius:4px;padding:11px 14px;margin-bottom:15px;}}
.cg-box .l{{font-size:.61rem;letter-spacing:.14em;font-weight:700;margin-bottom:5px;}}
.cg-box p{{font-size:.83rem;margin:0;line-height:1.58;}}
.cg-no{{border:1px solid #e9c3c9;background:#fdf4f5;}}
.cg-no .l{{color:{C_CRIT};}} .cg-no p{{color:#7d2231;}}
.cg-bc{{border:1px solid #c9d8ef;background:#f5f8fd;}}
.cg-bc .l{{color:{C_ACCENT};}} .cg-bc p{{color:#1b3f78;font-style:italic;}}
.cg-tb{{border:1px solid #e6d6a8;background:#fdfaf1;}}
.cg-tb .l{{color:{C_WATCH};}} .cg-tb p{{color:#6b5307;}}
.cg-cite{{font-family:var(--mono);font-size:.66rem;color:var(--dim);line-height:1.6;}}

/* ---------- events ---------- */
.cg-ev{{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #eef0f4;}}
.cg-ev .t{{font-family:var(--mono);font-size:.7rem;color:var(--dim);min-width:54px;}}
.cg-ev .b{{flex:1;}}
.cg-ev .h{{font-size:.79rem;font-weight:600;}}
.cg-ev .r{{font-size:.71rem;color:var(--muted);margin-top:2px;}}
.cg-ev .s{{font-family:var(--mono);font-size:.61rem;padding:2px 8px;border-radius:3px;
          border:1px solid var(--line-2);white-space:nowrap;}}
.cg-ev .s.ok{{border-color:#b6ddc9;background:#f2faf6;color:{C_OK};}}
.cg-ev .s.open{{border-color:#e9c3c9;background:#fdf4f5;color:{C_CRIT};}}

.cg-flag{{border:1px solid #e6d6a8;background:#fdfaf1;border-radius:4px;padding:10px 13px;
         font-size:.76rem;color:#6b5307;margin-bottom:12px;line-height:1.55;}}

/* ---------- widgets ---------- */
.stButton>button{{border-radius:4px;border:1px solid var(--line-2);background:var(--panel);
                 color:var(--ink);font-size:.83rem;font-weight:600;}}
.stButton>button:hover{{border-color:var(--accent);color:var(--accent);}}
.stTabs [data-baseweb="tab"]{{font-size:.83rem;font-weight:600;}}
[data-testid="stMetricValue"]{{font-family:var(--mono);color:var(--ink);}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================================== #
# Render helpers
# =========================================================================== #
def esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def header(backends: Dict[str, str], simulated: bool, calibrated: bool) -> str:
    def chip(k: str, v: str) -> str:
        return f'<div class="cg-chip">{esc(k)} <b>{esc(v)}</b></div>'

    chips = [
        chip("Source", "Simulation" if simulated else "Camera"),
        chip("Detector", backends.get("detector", "—")),
        chip("Retrieval", backends.get("rag", "—")[:16]),
        chip("Advisor", backends.get("advisor", "—")),
        chip("Forecast", backends.get("forecast", "—")),
        chip("Calibration", "homography" if calibrated else "uncalibrated"),
    ]
    return f"""<div class="cg-head">
  <div><div class="cg-title">CrowdGuard-RAG</div>
  <div class="cg-tag">Crowd risk prediction and safety decision support</div></div>
  <div class="cg-chips">{''.join(chips)}</div>
</div>"""


def status_banner(f: Any, state: str, forecast: Any) -> str:
    spec = RISK_TYPES.get(f.primary_risk_type, RISK_TYPES["NORMAL_FLOW"])
    sc = STATE_HEX.get(state, C_OK)

    ttc = getattr(forecast, "time_to_critical_sec", None)
    tta = getattr(forecast, "time_to_alert_sec", None)
    if not getattr(forecast, "ready", False):
        head, sub = "—", "collecting history"
    elif ttc is not None and ttc >= 1:
        head, sub = f"{ttc:.0f}s", "until CRITICAL on current trend"
    elif ttc is not None:
        head, sub = "NOW", "already at critical risk"
    elif tta is not None and tta >= 1:
        head, sub = f"{tta:.0f}s", "until ALERT on current trend"
    else:
        trend = forecast.trend_per_min
        head = "STABLE" if abs(trend) < 0.02 else ("RISING" if trend > 0 else "FALLING")
        sub = f"projected {forecast.predicted_score:.2f} at +{forecast.horizon_sec:.0f}s"

    where = (f'<span style="color:{C_ACCENT};font-weight:600"> · {esc(f.hotspot_zone)}</span>'
             if f.hotspot_zone else "")
    return f"""<div class="cg-status" style="--sc:{sc}">
  <div class="cg-status-row">
    <div><div class="cg-lab">State</div><div class="cg-state">{esc(state)}</div></div>
    <div style="flex:1;min-width:300px">
      <div class="cg-lab">Failure mode</div>
      <div class="cg-mode">{esc(spec.label)}
        <span style="font-family:var(--mono);font-size:.8rem;color:var(--muted);font-weight:500">
        {f.primary_risk_score:.2f}</span>{where}</div>
      <div class="cg-mech">{esc(spec.mechanism)}</div>
    </div>
    <div class="cg-budget"><div class="cg-lab">Time budget</div>
      <div class="v">{esc(head)}</div><div class="s">{esc(sub)}</div></div>
  </div>
</div>"""


def kpi(label: str, value: str, unit: str, frac: float, color: str, note: str = "") -> str:
    frac = max(0.0, min(1.0, frac))
    return f"""<div class="cg-kpi">
  <div class="k">{esc(label)}</div>
  <div class="v">{esc(value)}<span class="u"> {esc(unit)}</span></div>
  <div class="bar"><i style="width:{frac*100:.0f}%;background:{color}"></i></div>
  {f'<div class="n">{esc(note)}</div>' if note else ''}
</div>"""


def gauge(score: float, level: str, predicted: float) -> str:
    color = {"high": C_CRIT, "moderate": C_ALERT}.get(level, C_OK)
    r = 60.0
    circ = 2 * np.pi * r * 0.75
    return f"""<div class="cg-card" style="text-align:center">
  <div class="cg-h">Fused risk score</div>
  <svg viewBox="0 0 160 132" style="width:100%;max-width:196px">
    <g transform="rotate(135 80 72)">
      <circle cx="80" cy="72" r="{r}" fill="none" stroke="#eceef2" stroke-width="11"
              stroke-dasharray="{circ} 999" stroke-linecap="round"/>
      <circle cx="80" cy="72" r="{r}" fill="none" stroke="{color}" stroke-opacity=".25"
              stroke-width="11" stroke-dasharray="{circ*max(0,min(1,predicted)):.1f} 999"
              stroke-linecap="round"/>
      <circle cx="80" cy="72" r="{r}" fill="none" stroke="{color}" stroke-width="11"
              stroke-dasharray="{circ*max(0,min(1,score)):.1f} 999" stroke-linecap="round"/>
    </g>
    <text x="80" y="70" text-anchor="middle" fill="{color}"
          style="font:700 29px ui-monospace,monospace">{score:.2f}</text>
    <text x="80" y="87" text-anchor="middle" fill="{C_MUTED}"
          style="font:600 9px system-ui;letter-spacing:.15em">{esc(level.upper())}</text>
    <text x="80" y="116" text-anchor="middle" fill="{C_DIM}"
          style="font:500 8.5px system-ui">pale arc = forecast {predicted:.2f}</text>
  </svg>
</div>"""


def ranking_panel(ranking: List[Dict[str, Any]], evidence: List[str]) -> str:
    by_code = {r["code"]: r for r in ranking}
    top_code = ranking[0]["code"] if ranking else None
    rows = []
    for code in ORDERED_CODES:
        r = by_code.get(code)
        if not r:
            continue
        spec = RISK_TYPES[code]
        rows.append(
            f'<div class="cg-row {"top" if code == top_code else ""}">'
            f'<div class="sw" style="background:{spec.color}"></div>'
            f'<div class="nm">{esc(spec.label)}</div>'
            f'<div class="tr"><i style="width:{r["score"]*100:.0f}%;background:{spec.color};'
            f'opacity:{0.4 + 0.6*r["score"]:.2f}"></i></div>'
            f'<div class="pc">{r["score"]*100:.0f}%</div></div>'
        )
    ev = ""
    if evidence:
        items = "".join(f"<li>{esc(e)}</li>" for e in evidence[:4])
        ev = ('<div style="margin-top:12px;padding-top:11px;border-top:1px solid #eef0f4">'
              '<div class="cg-h" style="margin-bottom:7px">Measured evidence</div>'
              f'<ul style="margin:0;padding-left:16px;font-size:.74rem;color:{C_MUTED};'
              f'line-height:1.6">{items}</ul></div>')
    return f'<div class="cg-card"><div class="cg-h">Failure-mode ranking</div>{"".join(rows)}{ev}</div>'


def zones_panel(zones: List[Dict[str, Any]]) -> str:
    if not zones:
        return ('<div class="cg-card"><div class="cg-h">Zones</div>'
                '<div class="cg-note">No zones configured.</div></div>')
    rows = []
    for z in sorted(zones, key=lambda x: -x["risk_score"]):
        c = {"high": C_CRIT, "moderate": C_ALERT}.get(z["risk_level"], C_OK)
        rows.append(
            f'<div class="cg-zone"><div class="sw" style="background:{c}"></div>'
            f'<div class="nm">{esc(z["name"])}<div class="kd">{esc(z["kind"])}</div></div>'
            f'<div class="mt"><b>{z["density_pm2"]:.1f}</b> /m² · n={z["count"]}<br>'
            f'<span style="color:{c}">risk {z["risk_score"]:.2f}</span></div></div>'
        )
    return f'<div class="cg-card"><div class="cg-h">Zone status</div>{"".join(rows)}</div>'


def ladder(state: str, candidate: str, progress: float) -> str:
    cur = STATES.index(state)
    rungs = []
    for i, s in enumerate(STATES):
        meta, c = STATE_META[s], STATE_HEX[s]
        cls = "act" if i == cur else ("past" if i < cur else "")
        prog = (f'<div class="prog" style="width:{progress*100:.0f}%"></div>'
                if s == candidate and s != state and progress > 0 else "")
        rungs.append(
            f'<div class="cg-rung {cls}" style="--rc:{c}">'
            f'<div class="n" style="color:{c if i <= cur else C_DIM}">{s}</div>'
            f'<div class="d">{esc(meta["posture"])}</div>'
            f'<div class="w">Notifies: {esc(meta["who"])}</div>{prog}</div>'
        )
    return (f'<div class="cg-card"><div class="cg-h">Escalation ladder</div>'
            f'<div class="cg-ladder">{"".join(rungs)}</div></div>')


def advisory_panel(adv: Any, state: str) -> str:
    if adv is None:
        return ('<div class="cg-card"><div class="cg-h">Safety advisory</div>'
                '<div class="cg-note">Nothing to advise — the crowd is inside normal '
                'parameters. Retrieval and generation are skipped while the state is NORMAL, '
                'so an advisory appears only when it would change a decision.</div></div>')
    sc = STATE_HEX.get(state, C_OK)
    acts = "".join(
        f'<li>{("<span class=who>" + esc(a.split(":")[0]) + "</span> " + esc(a.split(":", 1)[1])) if ":" in a[:22] else esc(a)}</li>'
        for a in adv.actions
    )
    parts = [f'<div class="cg-adv" style="--sc:{sc}">',
             f'<h4>1 · Situation</h4><p>{esc(adv.situation)}</p>',
             f'<h4>2 · Failure mode — {esc(adv.failure_mode)}</h4><p>{esc(adv.mechanism)}</p>',
             f'<h4>3 · Root cause</h4><p>{esc(adv.root_cause)}</p>',
             f'<h4>4 · Immediate action</h4><ol>{acts}</ol>']
    if adv.do_not:
        parts.append(f'<div class="cg-box cg-no"><div class="l">5 · DO NOT</div>'
                     f'<p>{esc(adv.do_not)}</p></div>')
    parts.append(f'<div class="cg-box cg-bc"><div class="l">6 · Operator broadcast</div>'
                 f'<p>“{esc(adv.broadcast)}”</p></div>')
    if adv.time_budget:
        parts.append(f'<div class="cg-box cg-tb"><div class="l">7 · Time budget</div>'
                     f'<p>{esc(adv.time_budget)}</p></div>')
    cites = " · ".join(esc(e) for e in adv.evidence) or "No retrieved context."
    parts.append(f'<h4>8 · Evidence</h4><div class="cg-cite">{cites}<br>'
                 f'<span style="color:{C_DIM}">generator: {esc(adv.generator)}</span></div>')
    if getattr(adv, "notify_role", ""):
        parts.append(f'<h4 style="margin-top:15px">9 · Notify</h4>'
                     f'<p style="margin-bottom:0">{esc(adv.notify_role)}</p>')
    parts.append("</div>")
    return "".join(parts)


PLOT_LAYOUT = dict(
    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fcfcfd",
    font=dict(color=C_INK, size=11), margin=dict(l=8, r=8, t=8, b=8),
)


def timeline_figure(df: pd.DataFrame, cfg: EscalationConfig) -> go.Figure:
    fig = go.Figure()
    if "forecast_upper" in df:
        fig.add_trace(go.Scatter(x=df.t, y=df.forecast_upper, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df.t, y=df.forecast_lower, line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(31,95,191,.10)", name="forecast band",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df.t, y=df.forecast, name="forecast",
                                 line=dict(color=C_ACCENT, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=df.t, y=df.risk, name="risk score",
                             line=dict(color=C_CRIT, width=2.4)))
    for level, colour, label in ((cfg.watch_raise, C_WATCH, "WATCH"),
                                 (cfg.alert_raise, C_ALERT, "ALERT"),
                                 (cfg.critical_raise, C_CRIT, "CRITICAL")):
        fig.add_hline(y=level, line=dict(color=colour, width=1, dash="dash"), opacity=.55,
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(size=9, color=colour))
    fig.update_layout(height=320, hovermode="x unified",
                      legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10)),
                      xaxis=dict(title="time (s)", gridcolor="#eef0f4"),
                      yaxis=dict(title="score", range=[0, 1.02], gridcolor="#eef0f4"),
                      **PLOT_LAYOUT)
    return fig


# =========================================================================== #
# Sidebar
# =========================================================================== #
def sidebar() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    with st.sidebar:
        st.markdown('<div style="font-size:1.02rem;font-weight:700;letter-spacing:-.02em">'
                    'Configuration</div>'
                    '<div class="cg-tag" style="margin-bottom:14px">Operator settings</div>',
                    unsafe_allow_html=True)

        with st.expander("Input source", expanded=True):
            cfg["source"] = st.radio("Input", ["Simulated scenario", "Upload video",
                                               "Webcam", "Stream URL"],
                                     index=0, label_visibility="collapsed")
            if cfg["source"] == "Simulated scenario":
                keys = list(SCENARIOS)
                cfg["scenario"] = st.selectbox("Scenario", keys,
                                               index=keys.index("escalating_crush"),
                                               format_func=lambda k: SCENARIOS[k].label)
                st.caption(SCENARIOS[cfg["scenario"]].description)
                cfg["sim_fps"], cfg["sim_seed"] = 8, 7
            elif cfg["source"] == "Upload video":
                cfg["upload"] = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
            elif cfg["source"] == "Webcam":
                cfg["camera_index"] = st.number_input("Camera index", 0, 8, 0)
            else:
                cfg["stream_url"] = st.text_input("RTSP / HTTP / YouTube URL", "")

        with st.expander("Detection", expanded=True):
            cfg["model_path"] = st.text_input("YOLO weights", "yolov8n.pt")
            cfg["view_mode"] = st.selectbox(
                "Camera view", ["auto", "person", "head"], index=0,
                format_func=lambda v: {"auto": "Auto-detect",
                                       "person": "Ground level (side view)",
                                       "head": "Overhead / drone (top-down)"}[v])
            st.caption("COCO person detection returns **nothing** from a top-down view — "
                       "measured 0 detections out of 69 on an overhead scene, unchanged at "
                       "confidence 0.03. Pick *Overhead* for drone or high-mast footage.")
            if cfg["view_mode"] in ("auto", "head"):
                cfg["head_radius"] = st.slider(
                    "Head size in pixels", 2, 40, (4, 20),
                    help="Head diameter falls with altitude. Use the diagnostic shown after "
                         "a run with zero detections to find the right range for your footage.")
            else:
                cfg["head_radius"] = (4, 20)
            cfg["velocity_source"] = st.selectbox(
                "Velocity source", ["auto", "tracker", "flow"], index=0,
                format_func=lambda v: {"auto": "Auto (flow when overhead)",
                                       "tracker": "Centroid tracking",
                                       "flow": "Dense optical flow"}[v])
            cfg["sample_every"] = st.slider("Process every Nth frame", 1, 15, 3)
            cfg["max_frames"] = st.slider("Frame budget", 40, 2000, 400, step=20)

        with st.expander("Calibration and zones"):
            cfg["area_m2"] = st.number_input("Monitored area (m²)", 10.0, 5000.0, 120.0, 10.0)
            cfg["use_calibration"] = st.checkbox("Use a homography calibration file")
            cfg["calibration_file"] = (
                st.text_input("Calibration JSON", "outputs/calibration_simulator.json")
                if cfg["use_calibration"] else "")
            cfg["zone_file"] = st.text_input("Zone layout JSON", "",
                                             help="Blank uses the default three-zone layout.")
            st.caption("Without calibration, persons/m² is a trend indicator rather than an "
                       "absolute value comparable to published safety thresholds.")

        with st.expander("Thresholds"):
            cfg["density_hard"] = st.slider("Crush density limit (persons/m²)", 3.0, 8.0, 5.0, 0.1)
            cfg["pressure_critical"] = st.slider("Crowd-pressure critical", 0.5, 6.0, 2.6, 0.1)
            cfg["watch"] = st.slider("WATCH at", 0.10, 0.60, 0.35, 0.01)
            cfg["alert"] = st.slider("ALERT at", 0.30, 0.80, 0.55, 0.01)
            cfg["critical"] = st.slider("CRITICAL at", 0.50, 0.95, 0.72, 0.01)
            cfg["hysteresis"] = st.slider("Hysteresis margin", 0.0, 0.25, 0.08, 0.01)
            cfg["dwell_alert"] = st.slider("Dwell before ALERT (s)", 0, 30, 8)
            st.caption("Dwell and hysteresis prevent alarm flapping, which is the usual reason "
                       "operators mute a safety system.")

        with st.expander("Forecast and advisor"):
            cfg["horizon"] = st.slider("Forecast horizon (s)", 10, 90, 30, 5)
            ckpt = Path("outputs/risk_forecaster.pt")
            cfg["forecast_backend"] = st.selectbox(
                "Forecast backend",
                ["auto", "transformer", "trend"] if ckpt.exists() else ["trend"])
            st.caption("On held-out seeds the trained model beats a persistence baseline by 32%; "
                       "the untrained trend heuristic is 27% worse than it.")
            cfg["kb_dir"] = st.text_input("Knowledge base folder", "knowledge_base")
            cfg["llm_provider"] = st.selectbox("Advisor backend", ["fallback", "openai", "local_hf"])
            if cfg["llm_provider"] == "openai":
                key = st.text_input("OpenAI API key", type="password")
                if key:
                    os.environ["OPENAI_API_KEY"] = key

        with st.expander("Outbound alerts"):
            st.caption("All network sinks are off by default. Nothing leaves this machine "
                       "unless enabled here.")
            cfg["alert_min_state"] = st.selectbox("Notify from", ["WATCH", "ALERT", "CRITICAL"], 1)
            cfg["webhook_enabled"] = st.checkbox("HTTP webhook")
            cfg["webhook_url"] = st.text_input("Webhook URL", "") if cfg["webhook_enabled"] else ""
            cfg["telegram_enabled"] = st.checkbox("Telegram")
            cfg["tg_token"] = st.text_input("Bot token", type="password") if cfg["telegram_enabled"] else ""
            cfg["tg_chat"] = st.text_input("Chat id", "") if cfg["telegram_enabled"] else ""

        with st.expander("Display"):
            cfg["ov_boxes"] = st.checkbox("Detection boxes", True)
            cfg["ov_ids"] = st.checkbox("Track IDs", False)
            cfg["ov_vectors"] = st.checkbox("Velocity arrows and trails", True)
            cfg["ov_zones"] = st.checkbox("Zone overlay", True)
            cfg["show_heatmap"] = st.checkbox("Density heatmap panel", True)
    return cfg


# =========================================================================== #
# Pipeline assembly
# =========================================================================== #
def build_pipeline(cfg: Dict[str, Any], simulated: bool) -> CrowdGuardPipeline:
    if simulated:
        payload = CrowdSimulator.calibration_payload()
        calibration = CalibrationConfig(
            enabled=True,
            image_points=[tuple(p) for p in payload["image_points"]],
            world_points=[tuple(p) for p in payload["world_points"]],
            fallback_area_m2=WORLD_W * WORLD_D)
        zones, area = ZoneConfig(zones=CrowdSimulator.zone_layout()), WORLD_W * WORLD_D
    else:
        area = cfg["area_m2"]
        calibration = CalibrationConfig(enabled=False, fallback_area_m2=area)
        if cfg.get("use_calibration") and cfg.get("calibration_file"):
            p = Path(cfg["calibration_file"])
            if p.exists():
                d = json.loads(p.read_text())
                calibration = CalibrationConfig(
                    enabled=True,
                    image_points=[tuple(x) for x in d["image_points"]],
                    world_points=[tuple(x) for x in d["world_points"]],
                    fallback_area_m2=area)
            else:
                st.warning(f"Calibration file not found: {p} — continuing uncalibrated.")
        layout = None
        if cfg.get("zone_file"):
            p = Path(cfg["zone_file"])
            if p.exists():
                d = json.loads(p.read_text())
                layout = d.get("zones", d) if isinstance(d, dict) else d
            else:
                st.warning(f"Zone file not found: {p} — using the default layout.")
        zones = ZoneConfig(zones=layout or ZoneConfig.default_layout())

    return CrowdGuardPipeline(
        vision=VisionConfig(model_path=cfg["model_path"], mode=cfg.get("view_mode", "auto")),
        risk=RiskConfig(camera_area_m2=area, density_hard_limit=cfg["density_hard"],
                        pressure_critical=cfg["pressure_critical"],
                        pressure_warning=cfg["pressure_critical"] * 0.46),
        calibration=calibration, zones=zones,
        escalation=EscalationConfig(
            watch_raise=cfg["watch"], alert_raise=cfg["alert"], critical_raise=cfg["critical"],
            hysteresis=cfg["hysteresis"], dwell_alert_sec=float(cfg["dwell_alert"])),
        forecast=ForecastConfig(horizon_sec=float(cfg["horizon"]),
                                backend=cfg["forecast_backend"]),
        alerts=AlertConfig(
            min_state=cfg["alert_min_state"],
            webhook_enabled=cfg["webhook_enabled"], webhook_url=cfg["webhook_url"],
            telegram_enabled=cfg["telegram_enabled"],
            telegram_bot_token=cfg.get("tg_token", ""), telegram_chat_id=cfg.get("tg_chat", "")),
        rag=RAGConfig(knowledge_base_dir=cfg["kb_dir"]),
        llm=LLMConfig(provider=cfg["llm_provider"]),
        load_detector=not simulated,
        velocity_source=cfg.get("velocity_source", "auto"),
    )


def frame_iterator(cfg: Dict[str, Any], simulated: bool):
    if simulated:
        sim = CrowdSimulator(cfg["scenario"], fps=float(cfg["sim_fps"]), seed=int(cfg["sim_seed"]))
        for sf in sim.run(min(cfg["max_frames"], sim.total_frames)):
            yield sf.frame, sf.frame_id, sf.timestamp_sec, 1.0 / sim.fps, sf.detections, sf.ground_truth()
        return

    if cfg["source"] == "Upload video":
        up = cfg.get("upload")
        if up is None:
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
            tmp.write(up.read())
            source: Any = tmp.name
    elif cfg["source"] == "Webcam":
        source = int(cfg["camera_index"])
    else:
        url = cfg.get("stream_url", "").strip()
        if not url:
            return
        source = resolve_youtube_url(url) if "youtu" in url.lower() else url

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Could not open the source: {source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, cfg["sample_every"])
    frame_id, processed = -1, 0
    try:
        while processed < cfg["max_frames"]:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % step:
                continue
            processed += 1
            yield frame, frame_id, frame_id / fps, step / fps, None, None
    finally:
        cap.release()


def heatmap_for(result: Any, pipeline: CrowdGuardPipeline, hard_limit: float) -> np.ndarray:
    shape = result.annotated.shape if result.annotated is not None else (540, 960, 3)
    return density_field_image(
        shape, boxes_to_centroids(result.detections),
        pipeline.zone_manager, result.features.zones,
        peak_density=result.features.local_density_peak, hard_limit=hard_limit)


# =========================================================================== #
# Live run
# =========================================================================== #
def run_analysis(cfg: Dict[str, Any], simulated: bool) -> None:
    pipeline = build_pipeline(cfg, simulated)
    overlays = OverlayOptions(boxes=cfg["ov_boxes"], ids=cfg["ov_ids"], heatmap=False,
                              vectors=cfg["ov_vectors"], zones=cfg["ov_zones"])
    # Apply the operator's head-size range to the untrained blob detector.
    lo, hi = cfg.get("head_radius", (4, 20))
    if pipeline.detector is not None and pipeline.detector.head_detector is not None:
        pipeline.detector.head_detector.config.min_radius_px = float(lo)
        pipeline.detector.head_detector.config.max_radius_px = float(hi)

    calibrated = simulated or bool(pipeline.risk_engine.ground
                                   and pipeline.risk_engine.ground.calibrated)

    st.markdown(header(pipeline.backends, simulated, calibrated), unsafe_allow_html=True)
    if simulated:
        st.markdown('<div class="cg-flag"><b>Simulation.</b> Detections come from an '
                    'agent-based crowd model, not a camera. Everything downstream — '
                    'measurement, classification, forecasting, escalation, retrieval and '
                    'advice — is the production code, and the simulator supplies ground truth '
                    'so those layers can be scored against a known correct answer.</div>',
                    unsafe_allow_html=True)
    if not calibrated:
        st.markdown('<div class="cg-flag"><b>Uncalibrated camera.</b> Densities are '
                    'approximate and should be read as trends, not compared against published '
                    'safety thresholds.</div>', unsafe_allow_html=True)

    banner, kpis = st.empty(), st.empty()
    col_l, col_r = st.columns([1.6, 1], gap="medium")

    # Source video large on top; detection and density beneath it, side by side.
    # The raw frame is what an operator recognises the scene from, so it gets
    # the space; the two derived views are read as instruments against it.
    video_slot = col_l.empty()
    caption_slot = col_l.empty()
    sub_l, sub_r = col_l.columns(2, gap="small")
    detect_slot, heat_slot = sub_l.empty(), sub_r.empty()

    gauge_slot, rank_slot, zone_slot = col_r.empty(), col_r.empty(), col_r.empty()
    ladder_slot, chart_slot, adv_slot = st.empty(), st.empty(), st.empty()
    progress = st.progress(0.0)

    rows: List[Dict[str, Any]] = []
    series: List[Dict[str, float]] = []
    frames: List[Any] = []
    last_advisory = None
    last_frame = None
    esc_cfg = pipeline.escalation_config
    started = time.time()

    for frame, fid, ts, dt, dets, gt in frame_iterator(cfg, simulated):
        last_frame = frame
        result = pipeline.process(frame, fid, ts, dt, detections=dets, overlays=overlays,
                                  ground_truth=gt, want_annotated=True)
        f, fc, status = result.features, result.forecast, result.status
        if result.advisory:
            last_advisory = result.advisory

        rows.append(result.log_row(pipeline.backends["rag"]))
        series.append({"t": f.timestamp_sec, "risk": f.risk_score,
                       "density": f.local_density_peak, "forecast": fc.predicted_score,
                       "forecast_lower": fc.lower, "forecast_upper": fc.upper,
                       "horizon": fc.horizon_sec, "count": f.person_count})

        heat = heatmap_for(result, pipeline, cfg["density_hard"]) if cfg["show_heatmap"] else None
        if len(frames) < MAX_STORED_FRAMES:
            frames.append((f.timestamp_sec, _pack(frame), _pack(result.annotated), _pack(heat)))

        banner.markdown(status_banner(f, status.state, fc), unsafe_allow_html=True)

        occ = result.occlusion
        with kpis.container():
            cols = st.columns(6)
            cards = [
                ("Peak local density", f"{f.local_density_peak:.2f}", "p/m²",
                 f.local_density_peak / max(0.1, cfg["density_hard"]),
                 C_CRIT if f.local_density_peak >= cfg["density_hard"]
                 else C_ALERT if f.local_density_peak >= 3.0 else C_OK,
                 f"crush band at {cfg['density_hard']:.1f}"),
                ("People tracked", f"{f.person_count}", "", min(1.0, f.person_count / 200),
                 C_ACCENT,
                 "lower bound — heavy occlusion" if occ.get("undercount_warning")
                 else f"occlusion {occ.get('occlusion_ratio', 0):.2f}"),
                ("Crowd pressure", f"{f.crowd_pressure:.2f}", "s⁻²",
                 f.crowd_pressure / max(0.1, cfg["pressure_critical"]),
                 C_CRIT if f.crowd_pressure >= cfg["pressure_critical"] else C_WATCH,
                 "density × velocity variance"),
                ("Mean speed", f"{f.avg_speed_ms:.2f}", "m/s", min(1.0, f.avg_speed_ms / 2.5),
                 C_OK, f"{f.speed_surge_ratio:.1f}× site baseline"),
                ("Counter-flow", f"{f.counterflow_index:.2f}", "", f.counterflow_index,
                 C_ALERT if f.counterflow_index > 0.35 else C_MUTED, "opposing streams"),
                ("Chokepoint load", f"{f.bottleneck_ratio*100:.0f}", "%", f.bottleneck_ratio,
                 C_CRIT if f.bottleneck_ratio > 0.8 else C_WATCH,
                 f"throughput {f.flux_efficiency*100:.0f}% of peak"),
            ]
            for c, args in zip(cols, cards):
                c.markdown(kpi(*args), unsafe_allow_html=True)

        video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        caption_slot.markdown(
            f'<div class="cg-note"><b>Source feed</b> · frame {fid} · '
            f't = {f.timestamp_sec:.1f}s · '
            f'{"simulated" if simulated else "camera"}</div>',
            unsafe_allow_html=True)
        _show(detect_slot, result.annotated,
              f"Detection — {f.person_count} tracked, zones and motion")
        _show(heat_slot, heat, "Density field — persons/m² on the ground plane")

        gauge_slot.markdown(gauge(f.risk_score, f.risk_level, fc.predicted_score),
                            unsafe_allow_html=True)
        rank_slot.markdown(ranking_panel(f.risk_type_ranking, f.type_evidence),
                           unsafe_allow_html=True)
        zone_slot.markdown(zones_panel(f.zones), unsafe_allow_html=True)
        ladder_slot.markdown(ladder(status.state, status.candidate_state,
                                    status.candidate_progress), unsafe_allow_html=True)
        if len(series) % 6 == 0 or len(series) < 4:
            chart_slot.plotly_chart(timeline_figure(pd.DataFrame(series), esc_cfg),
                                    use_container_width=True, key=f"tl{len(series)}")
        adv_slot.markdown(advisory_panel(last_advisory, status.state), unsafe_allow_html=True)
        progress.progress(min(1.0, len(series) / max(1, cfg["max_frames"])))

    progress.empty()
    if not rows:
        st.info("No frames were processed. Check the source and try again.")
        return

    # A count of zero on visibly crowded footage is the most confusing failure
    # this system can present, so it explains itself instead of leaving the
    # operator to guess.
    if max(r["person_count"] for r in rows) == 0 and not simulated:
        st.markdown('<div class="cg-flag"><b>Nothing detected in any frame.</b> '
                    'Running a scale sweep to find out why.</div>', unsafe_allow_html=True)
        if last_frame is not None:
            det = TopDownHeadDetector(HeadDetectorConfig(min_radius_px=float(lo),
                                                         max_radius_px=float(hi)))
            diag = det.diagnose(last_frame)
            rows_html = "".join(
                f'<div class="cg-row"><div class="sw" style="background:{C_ACCENT}"></div>'
                f'<div class="nm">head ≈ {r} px across</div>'
                f'<div class="tr"><i style="width:{min(100, n / max(1, max(diag["detections_by_head_radius_px"].values())) * 100):.0f}%;'
                f'background:{C_ACCENT}"></i></div><div class="pc">{n}</div></div>'
                for r, n in diag["detections_by_head_radius_px"].items())
            st.markdown(f'<div class="cg-card"><div class="cg-h">Detections at each head size'
                        f'</div>{rows_html}<div class="cg-note" style="margin-top:10px">'
                        f'{esc(diag["hint"])}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="cg-note">If the view is top-down, set <b>Camera view</b> to '
                    '<b>Overhead / drone</b> in the sidebar rather than leaving it on Auto — '
                    'Auto only switches after five consecutive empty frames, and it cannot '
                    'switch at all if the blob detector also finds nothing at the configured '
                    'head size.</div>', unsafe_allow_html=True)

    st.session_state["cg"] = {
        "rows": rows, "series": series, "frames": frames, "advisory": last_advisory,
        "events": [e.to_dict() for e in pipeline.escalation.events],
        "summary": pipeline.escalation.summary(),
        "forecast_accuracy": pipeline.forecaster.accuracy_report(),
        "runtime": pipeline.runtime_report(), "backends": pipeline.backends,
        "simulated": simulated, "elapsed": time.time() - started,
        "alert_failures": pipeline.dispatcher.failures(),
        "alert_sinks": pipeline.dispatcher.sink_names,
    }
    st.session_state["cg_pipeline"] = pipeline
    st.rerun()


# =========================================================================== #
# Post-run review
# =========================================================================== #
def review(cfg: Dict[str, Any]) -> None:
    data = st.session_state["cg"]
    rows, series, frames = data["rows"], data["series"], data["frames"]
    pipeline: Optional[CrowdGuardPipeline] = st.session_state.get("cg_pipeline")
    df = pd.DataFrame(series)
    calibrated = bool(rows[-1].get("calibrated"))

    st.markdown(header(data["backends"], data["simulated"], calibrated), unsafe_allow_html=True)

    peak_i = int(np.argmax(df.risk))
    peak_row = rows[peak_i]
    cells = [("Frames", str(len(rows))), ("Duration", f"{df.t.iloc[-1]:.0f} s"),
             ("Peak risk", f"{df.risk.max():.2f}"),
             ("Peak density", f"{df.density.max():.2f} /m²"),
             ("Peak count", str(int(df["count"].max()))),
             ("Alerts raised", str(len(data["events"]))),
             ("Throughput", f"{data['runtime'].get('throughput_fps', '—')} fps")]
    st.markdown('<div class="cg-card"><div class="cg-h">Run summary</div>'
                '<div style="display:flex;gap:38px;flex-wrap:wrap">'
                + "".join(f'<div><div class="cg-lab">{esc(k)}</div>'
                          f'<div style="font-family:var(--mono);font-size:1.02rem;'
                          f'font-weight:700">{esc(v)}</div></div>' for k, v in cells)
                + "</div></div>", unsafe_allow_html=True)

    tabs = st.tabs(["Replay", "Alerts", "Timeline", "Failure modes",
                    "Forecast quality", "Report"])

    # ------------------------------------------------------------- replay
    with tabs[0]:
        if frames:
            idx = st.slider("Frame", 0, len(frames) - 1, min(peak_i, len(frames) - 1),
                            help="Step through the run to show any moment to a reviewer.")
            ts, raw_b, det_b, heat_b = frames[idx]
            r = rows[min(idx, len(rows) - 1)]
            c1, c2 = st.columns([1.6, 1], gap="medium")
            _show(c1, _unpack(raw_b),
                  f"Source feed · t = {ts:.1f}s · {r['escalation_state']} · "
                  f"{r['primary_risk_label']} · risk {r['risk_score']:.2f}")
            s1, s2 = c1.columns(2, gap="small")
            _show(s1, _unpack(det_b), f"Detection — {r['person_count']} tracked")
            _show(s2, _unpack(heat_b),
                  f"Density field — peak {r['local_density_peak']:.1f}/m²")
            if r.get("ground_truth"):
                gt = r["ground_truth"]
                c1.markdown(f'<div class="cg-note">Ground truth at this frame: '
                            f'{gt["true_count"]} people, {gt["true_density_peak"]:.2f} /m² — '
                            f'measured {r["person_count"]} people, '
                            f'{r["local_density_peak"]:.2f} /m².</div>', unsafe_allow_html=True)
            c2.markdown(gauge(r["risk_score"], r["risk_level"],
                              r["forecast"]["predicted_score"]), unsafe_allow_html=True)
            c2.markdown(ranking_panel(r["risk_type_ranking"], r["type_evidence"]),
                        unsafe_allow_html=True)
            c2.markdown(zones_panel(r["zones"]), unsafe_allow_html=True)
        else:
            st.info("No frames were retained for replay.")

    # ------------------------------------------------------------- alerts
    with tabs[1]:
        events = data["events"]
        failures = data["alert_failures"]
        st.markdown(f'<div class="cg-card"><div class="cg-h">Escalation summary</div>'
                    f'<div style="font-family:var(--mono);font-size:.8rem;color:{C_MUTED}">'
                    f'{esc(json.dumps(data["summary"]))}</div>'
                    f'<div class="cg-note" style="margin-top:8px">Sinks active: '
                    f'{esc(", ".join(data["alert_sinks"]))}. '
                    f'{"Delivery failures: " + esc(json.dumps(failures)) if failures else "No delivery failures."}'
                    f'</div></div>', unsafe_allow_html=True)

        if not events:
            st.markdown('<div class="cg-card"><div class="cg-note">No escalation events were '
                        'raised. With dwell times and hysteresis in force this is the correct '
                        'outcome for a crowd that never sustained a threshold.</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="cg-note" style="margin-bottom:10px">Acknowledging an alert '
                        'suppresses re-notification, records who accepted it and what they did, '
                        'and stores a human label for future training.</div>',
                        unsafe_allow_html=True)
            for ev in events:
                cls = "ok" if ev["acknowledged"] else "open"
                st.markdown(f'<div class="cg-ev"><div class="t">{ev["timestamp_sec"]:.1f}s</div>'
                            f'<div class="b"><div class="h">{esc(ev["from_state"])} &rarr; '
                            f'{esc(ev["to_state"])} — {esc(ev["risk_type_label"])}</div>'
                            f'<div class="r">{esc(ev["zone"] or "whole frame")} · '
                            f'risk {ev["risk_score"]:.2f} · density {ev["density"]:.2f} /m² · '
                            f'{esc(ev["reason"])}</div></div>'
                            f'<div class="s {cls}">'
                            f'{"ACKNOWLEDGED" if ev["acknowledged"] else "OPEN"}</div></div>',
                            unsafe_allow_html=True)
                if not ev["acknowledged"] and pipeline is not None:
                    with st.expander(f"Acknowledge {ev['event_id']}"):
                        with st.form(f"ack_{ev['event_id']}"):
                            c1, c2 = st.columns(2)
                            operator = c1.text_input("Operator", "control-room-1")
                            decision = c2.selectbox("Decision", DECISIONS)
                            action = st.selectbox("Action taken", STANDARD_ACTIONS)
                            note = st.text_area("Note", "", height=68)
                            if st.form_submit_button("Record acknowledgement"):
                                pipeline.acknowledge(ev["event_id"], operator, decision,
                                                     action, note, ev["timestamp_sec"])
                                ev["acknowledged"] = True
                                st.rerun()

    # ------------------------------------------------------------- timeline
    with tabs[2]:
        st.plotly_chart(timeline_figure(df, EscalationConfig(
            watch_raise=cfg["watch"], alert_raise=cfg["alert"],
            critical_raise=cfg["critical"])), use_container_width=True, key="tl_final")

        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.t, y=df.density, name="measured",
                                 line=dict(color=C_ALERT, width=2)))
        if rows[0].get("ground_truth"):
            fig.add_trace(go.Scatter(x=df.t,
                                     y=[r["ground_truth"]["true_density_peak"] for r in rows],
                                     name="ground truth",
                                     line=dict(color=C_OK, width=1.6, dash="dot")))
        fig.add_hline(y=cfg["density_hard"], line=dict(color=C_CRIT, dash="dash", width=1),
                      annotation_text="crush band", annotation_font=dict(size=9, color=C_CRIT))
        fig.update_layout(height=280, title=dict(text="Peak local density", font=dict(size=12)),
                          xaxis=dict(title="time (s)", gridcolor="#eef0f4"),
                          yaxis=dict(title="persons/m²", gridcolor="#eef0f4"),
                          legend=dict(orientation="h", y=1.18, font=dict(size=10)), **PLOT_LAYOUT)
        c1.plotly_chart(fig, use_container_width=True, key="dens")

        fig2 = go.Figure()
        for name, colour, label in (("flow_disorder", C_ALERT, "flow disorder"),
                                    ("counterflow_index", C_CRIT, "counter-flow"),
                                    ("oscillation_index", C_ACCENT, "oscillation"),
                                    ("flux_efficiency", C_OK, "flux efficiency")):
            fig2.add_trace(go.Scatter(x=df.t, y=[r[name] for r in rows], name=label,
                                      line=dict(width=1.5, color=colour)))
        fig2.update_layout(height=280,
                           title=dict(text="Movement-quality signals", font=dict(size=12)),
                           xaxis=dict(title="time (s)", gridcolor="#eef0f4"),
                           yaxis=dict(range=[0, 1.02], gridcolor="#eef0f4"),
                           legend=dict(orientation="h", y=1.18, font=dict(size=9)), **PLOT_LAYOUT)
        c2.plotly_chart(fig2, use_container_width=True, key="mq")

    # ------------------------------------------------------------- failure modes
    with tabs[3]:
        counts = pd.Series([r["primary_risk_label"] for r in rows]).value_counts()
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            bars = []
            for label, n in counts.items():
                code = next((c for c, s in RISK_TYPES.items() if s.label == label), "NORMAL_FLOW")
                spec, share = RISK_TYPES[code], n / len(rows)
                bars.append(f'<div class="cg-row">'
                            f'<div class="sw" style="background:{spec.color}"></div>'
                            f'<div class="nm">{esc(label)}</div>'
                            f'<div class="tr"><i style="width:{share*100:.0f}%;'
                            f'background:{spec.color}"></i></div>'
                            f'<div class="pc">{share*100:.0f}%</div></div>')
            st.markdown('<div class="cg-card"><div class="cg-h">Time spent in each failure '
                        f'mode</div>{"".join(bars)}</div>', unsafe_allow_html=True)

            contrib = peak_row.get("contributions", {})
            total = sum(contrib.values()) or 1
            cbars = "".join(
                f'<div class="cg-row"><div class="sw" style="background:{C_ACCENT}"></div>'
                f'<div class="nm">{esc(k.replace("_", " "))}</div>'
                f'<div class="tr"><i style="width:{v/total*100:.0f}%;background:{C_ACCENT}"></i>'
                f'</div><div class="pc">{v/total*100:.0f}%</div></div>'
                for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]))
            st.markdown('<div class="cg-card"><div class="cg-h">Risk-score contributions at '
                        f'peak</div>{cbars}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(ranking_panel(peak_row["risk_type_ranking"],
                                      peak_row["type_evidence"]), unsafe_allow_html=True)
            st.markdown(f'<div class="cg-note">Ranking shown at the peak-risk moment, '
                        f't = {peak_row["timestamp_sec"]:.1f}s.</div>', unsafe_allow_html=True)
            st.markdown(advisory_panel(data["advisory"],
                                       data["summary"].get("current_state", "NORMAL")),
                        unsafe_allow_html=True)

    # ------------------------------------------------------------- forecast
    with tabs[4]:
        st.markdown(f'<div class="cg-card"><div class="cg-h">Forecast self-evaluation</div>'
                    f'<div class="cg-note">Every forecast is compared against the risk score '
                    f'actually observed one horizon later. These are out-of-sample errors: the '
                    f'forecaster never sees the values it is scored against.</div>'
                    f'<div style="font-family:var(--mono);font-size:.8rem;margin-top:10px">'
                    f'{esc(json.dumps(data["forecast_accuracy"]))}</div></div>',
                    unsafe_allow_html=True)

        matured, times = [], df.t.values
        for i in range(len(rows)):
            fc = rows[i]["forecast"]
            if not fc["ready"]:
                continue
            j = int(np.searchsorted(times, times[i] + fc["horizon_sec"]))
            if j < len(rows):
                matured.append({"t": times[i], "predicted": fc["predicted_score"],
                                "actual": rows[j]["risk_score"],
                                "persistence": rows[i]["risk_score"]})
        if matured:
            m = pd.DataFrame(matured)
            mae = float((m.predicted - m.actual).abs().mean())
            base = float((m.persistence - m.actual).abs().mean())
            skill = 1 - mae / max(1e-9, base)
            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast MAE", f"{mae:.4f}")
            c2.metric("Persistence baseline MAE", f"{base:.4f}")
            c3.metric("Skill vs persistence", f"{skill:+.1%}",
                      delta=("beats baseline" if skill >= 0.05 else
                             "no better than baseline" if skill > -0.05 else "worse than baseline"),
                      delta_color="normal" if skill >= 0.05 else "off")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m.t, y=m.actual, name="actual risk at t+H",
                                     line=dict(color=C_CRIT, width=2)))
            fig.add_trace(go.Scatter(x=m.t, y=m.predicted, name="forecast made at t",
                                     line=dict(color=C_ACCENT, width=1.6, dash="dot")))
            fig.add_trace(go.Scatter(x=m.t, y=m.persistence, name="persistence baseline",
                                     line=dict(color=C_MUTED, width=1, dash="dash")))
            fig.update_layout(height=300,
                              xaxis=dict(title="time the forecast was made (s)",
                                         gridcolor="#eef0f4"),
                              yaxis=dict(range=[0, 1.02], gridcolor="#eef0f4"),
                              legend=dict(orientation="h", y=1.16, font=dict(size=10)),
                              **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True, key="fcq")
            if skill < 0.05:
                st.markdown('<div class="cg-flag"><b>On this run the forecaster does not '
                            'meaningfully beat the persistence baseline</b>, and that is reported '
                            'rather than hidden. Over a short horizon, assuming the risk stays '
                            'where it is remains a strong predictor, and a scenario that goes '
                            'from an empty walkway to a crush inside a minute asks the forecaster '
                            'to predict a regime change from a scene containing almost no '
                            'evidence of it.</div>', unsafe_allow_html=True)
        else:
            st.info("No forecasts matured within this run — extend the frame budget.")

    # ------------------------------------------------------------- report
    with tabs[5]:
        report = generate_incident_report(
            rows, advisory=data["advisory"], escalation_summary=data["summary"],
            events=data["events"], forecast_report=data["forecast_accuracy"],
            ack_stats=pipeline.acks.stats() if pipeline else {},
            run_meta={"detector_backend": data["backends"]["detector"],
                      "rag_backend": data["backends"]["rag"],
                      "llm_provider": data["backends"]["advisor"],
                      "forecast_backend": data["backends"]["forecast"]})
        flat = pd.DataFrame([{k: v for k, v in r.items()
                              if not isinstance(v, (dict, list))} for r in rows])
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Risk log (CSV)", flat.to_csv(index=False).encode(),
                           f"crowdguard_risk_{stamp}.csv", "text/csv", use_container_width=True)
        c2.download_button("Full log (JSONL)",
                           "\n".join(json.dumps(r, default=str) for r in rows).encode(),
                           f"crowdguard_log_{stamp}.jsonl", "application/x-ndjson",
                           use_container_width=True)
        c3.download_button("Incident report (Markdown)", report.encode(),
                           f"crowdguard_report_{stamp}.md", "text/markdown",
                           use_container_width=True)
        with st.expander("Preview the incident report"):
            st.markdown(report)
        with st.expander("Runtime profile"):
            st.json(data["runtime"])

    if st.button("New run", use_container_width=True):
        st.session_state.pop("cg", None)
        st.session_state.pop("cg_pipeline", None)
        st.rerun()


# =========================================================================== #
def landing() -> str:
    rows = "".join(
        f'<tr><td style="padding:8px 14px 8px 0;white-space:nowrap;vertical-align:top">'
        f'<span style="display:inline-block;width:9px;height:9px;border-radius:2px;'
        f'background:{s.color};margin-right:8px"></span>'
        f'<b style="font-size:.83rem">{esc(s.label)}</b></td>'
        f'<td style="padding:8px 0;font-size:.79rem;color:{C_MUTED};line-height:1.5">'
        f'{esc(s.description)}</td></tr>'
        for code, s in RISK_TYPES.items() if code != "NORMAL_FLOW")
    return f"""<div class="cg-card">
  <div class="cg-h">What the system classifies</div>
  <div class="cg-note" style="margin-bottom:14px;max-width:82ch">
    A single risk number tells an operator nothing about what to do. A crush and a panic
    dispersal both score high and need opposite responses — one requires inflow stopped and
    pressure released, the other requires every exit opened. The system therefore names the
    failure mode, names the zone, forecasts the time budget, and retrieves the venue's own
    procedure for that specific hazard.
  </div>
  <table style="width:100%;border-collapse:collapse">{rows}</table>
</div>"""


def main() -> None:
    cfg = sidebar()
    simulated = cfg["source"] == "Simulated scenario"

    if "cg" in st.session_state:
        review(cfg)
        return

    st.markdown(header({"detector": "supplied" if simulated else "ready", "rag": "ready",
                        "advisor": cfg["llm_provider"], "forecast": cfg["forecast_backend"]},
                       simulated, simulated or cfg.get("use_calibration", False)),
                unsafe_allow_html=True)

    ready = True
    if cfg["source"] == "Upload video" and cfg.get("upload") is None:
        st.info("Upload a video file in the sidebar to begin.")
        ready = False
    elif cfg["source"] == "Stream URL" and not cfg.get("stream_url", "").strip():
        st.info("Enter a stream URL in the sidebar to begin.")
        ready = False

    label = (f"Run scenario — {SCENARIOS[cfg['scenario']].label}" if simulated
             else "Start monitoring")
    if ready and st.button(label, type="primary", use_container_width=True):
        run_analysis(cfg, simulated)
        return

    st.markdown(landing(), unsafe_allow_html=True)
    if simulated:
        st.markdown('<div class="cg-card"><div class="cg-h">Why a simulated scenario</div>'
                    '<div class="cg-note" style="max-width:82ch">Real crowd-disaster footage is '
                    'scarce and impossible to obtain on demand, so every failure mode above can '
                    'be produced here on command. The simulator also knows the true position of '
                    'every person it placed, which lets the measurement, classification and '
                    'forecasting layers be scored against a known correct answer rather than '
                    'merely demonstrated. Detections are synthetic and labelled as such; '
                    'everything downstream is the production code. To run on real footage, '
                    'choose <b>Upload video</b> in the sidebar.</div></div>',
                    unsafe_allow_html=True)


main()
