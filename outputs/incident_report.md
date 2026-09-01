# CrowdGuard-RAG — Post-Event Safety Report

**Generated:** 2026-08-27 13:36:54  
**Monitoring duration:** 4.7 s across 60 analysed samples  
**Detector:** yolo · **Retrieval:** faiss · **Advisor:** fallback · **Forecast:** trend  
**Calibration:** Uncalibrated: uniform scale 0.01976 m/px derived from declared monitored area

---

## 1. Executive summary

| Metric | Peak | Mean |
|---|---|---|
| Fused risk score | 0.000 | 0.000 |
| Peak local density (persons/m²) | 0.00 | 0.00 |
| Crowd pressure (s⁻²) | 0.00000 | 0.00000 |
| Tracked person count | 0 | 0.0 |

- Samples at HIGH risk: **0** (0.0%)
- Samples above the Fruin soft density limit (3.0/m²): **0**
- Samples above the Helbing turbulence pressure (0.02 s⁻²): **0**

## 2. Failure modes observed

| Risk type | Samples | Share |
|---|---|---|
| Normal Flow | 60 | 100.0% |

## 3. Zone hotspots

| Zone | Peak density (persons/m²) |
|---|---|
| Left Gate | 0.00 |
| Central Corridor | 0.00 |
| Right Gate | 0.00 |

## 4. Escalation history

_No escalation events were raised._

**Summary:** {"current_state": "NORMAL", "total_events": 0, "escalations": 0, "renotifications": 0, "acknowledged": 0, "unacknowledged": 0}

**Operator acknowledgements:** {"total": 0, "confirmed": 0, "false_alarm": 0, "already_handled": 0, "operator_precision": null}

## 5. Forecast performance

{"samples": 0, "mae": null, "note": "no matured forecasts yet"}

_Self-evaluated: each forecast is compared against the risk score actually observed one horizon later._

## 6. Final advisory

None generated.

## 7. Method and known limitations

**Method.** Persons are detected per frame, tracked by centroid association, and projected
to the ground plane. Local density is estimated per person with a k-nearest-neighbour
estimator; crowd pressure follows Helbing's definition (local density × local velocity
variance). Six normalised features are fused into a single risk score, and a rule-based
classifier assigns the failure mode. An escalation state machine with dwell times and
hysteresis converts the score series into discrete operator events.

**Limitations, stated plainly:**

1. **Detection undercounts under occlusion.** Appearance-based detection degrades exactly
   where density is highest. The occlusion ratio is reported per frame; where it is high,
   the count should be read as a lower bound. A density-map counter (CSRNet class) is the
   correct estimator in that regime.
2. **Densities are approximate without calibration.** Absolute persons/m² and the
   comparison against published thresholds are only physically meaningful once the camera
   has been homography-calibrated to the ground plane.
3. **The risk weights are literature-informed, not learned.** They encode published crowd
   science rather than being fitted to labelled incident data, because no such labelled
   dataset was available for this deployment.
4. **Centroid tracking produces ID switches** in dense scenes. Velocity is smoothed over
   several frames to limit the effect on the disorder metric, but a Deep-SORT class
   tracker would reduce it further.
5. **No validated ground truth for "risk" exists here.** Counting, retrieval, forecasting
   and runtime are each evaluated against measurable ground truth (see `scripts/evaluate.py`);
   the fused risk score is validated only by expert agreement, and is reported as such.

---

*CrowdGuard-RAG — decision support, not a substitute for a trained safety officer.*