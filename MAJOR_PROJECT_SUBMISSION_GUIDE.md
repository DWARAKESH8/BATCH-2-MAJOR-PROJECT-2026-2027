# 🎓 Major Project — Submission & Defence Guide

**CrowdGuard-RAG: Real-Time Crowd Risk Prediction and Safety Decision Support**

This document is for the viva. It contains the one-paragraph problem statement, the answers to
the questions you will actually be asked, the numbers you are allowed to quote, and — most
importantly — the weaknesses you should raise *before* the panel does.

---

## 1. The problem statement (memorise this paragraph)

> Large-scale events generate continuously changing crowd conditions across many zones that
> security personnel cannot monitor or quantify manually. The failure is not that people cannot
> see a crowded video; it is that visual observation produces no measurement, no threshold, no
> prediction, and no auditable record. Crowd disasters are therefore not a *detection* problem
> but a *decision-latency* problem: every physical intervention — closing a gate, opening a
> diversion route, moving marshals — takes one to three minutes to take effect, while a crowd can
> pass from crowded to critical in under a minute. This project develops a real-time system that
> measures the physical precursors of crowd crush, classifies the specific failure mode, forecasts
> escalation before it becomes visually obvious, retrieves the venue's own safety procedure for
> that hazard using Retrieval-Augmented Generation, and converts the result into an acknowledged,
> logged, actionable instruction for a named zone.

**The four gaps** — memorise this table, it structures your entire defence:

| Gap | What fails today | Which module closes it |
|---|---|---|
| Observation | 300 cameras, 6 operators, ~20-min attention span | Continuous detection + tracking on every feed |
| Quantification | An impression, not a number | Multi-modal risk engine (6 measured features) |
| Interpretation | A density figure means nothing without context | Failure-mode taxonomy + literature-anchored bands |
| Action | The SOP is a 200-page binder | RAG over the venue's own documents + escalation |

---

## 2. The question your colleague asked

> **"By seeing the video itself we can say whether it is risky — so why is this useful?"**

**Answer, ~40 seconds:**

> "A human can, for the one feed they happen to be watching. A stadium has around 300 cameras and
> six operators, and surveillance research consistently finds operator detection performance
> degrades after roughly twenty minutes. More importantly, what actually kills people is not
> density — it is crowd turbulence, local density multiplied by the variance of local velocities.
> Helbing's analysis of the 2006 Mina disaster established this. A human looking at a frame sees
> 'crowded'; they cannot compute directional coherence across ninety moving people. My system
> computes it every frame. And an impression cannot cross a threshold, page a marshal, or be
> produced as evidence afterwards — a timestamped risk score can. Itaewon, Astroworld,
> Kanjuruhan, Hillsborough: every one of those venues had CCTV with people watching it. The
> missing layer was never video. It was measurement, threshold, prediction, and a grounded
> instruction."

**If pressed further, the closing line:**

> "The system is decision support, not a replacement for the safety officer. AI measures
> continuously; the human decides. That combination is stronger than either alone."

---

## 3. Numbers you are allowed to quote

All reproducible with `python scripts/evaluate.py`.

| Claim | Value | Basis |
|---|---|---|
| Count MAE | 4.78 people (MAPE 3.0 %) | 12,429 samples vs simulator ground truth |
| Density MAE / correlation | 0.99 persons/m² / r = 0.926 | same |
| **Density bias in crush band** | **−1.09 persons/m²** | *state this yourself — see §6* |
| Failure-mode classification | 7/8 scenarios, 72.9 % frame-level | 8 scenarios × 3 seeds |
| RAG retrieval | P@1 0.800, MRR 0.831, nDCG 0.842 | 35 hand-written queries |
| Forecast skill vs persistence | **+32.1 %** (MAE 0.0895 vs 0.1318) | unseen seeds, trained transformer |
| Trend heuristic, same test | **−26.5 %** — worse than persistence | *state this yourself — see §6* |
| Alarm flapping | **0.0 %** over 24 runs | dwell + hysteresis design |
| Baseline classifier | 0.74 on unseen scenarios, ROC-AUC 0.918 | labels from Fruin bands, not our scorer |
| Throughput | 91 fps simulated / ~41 fps with YOLOv8n on CPU | per-stage profiling |

### ⚠️ Say where the data comes from, unprompted

**Every number above is measured against the built-in simulator, not real footage.** Say that
yourself, in this form:

> "These are simulation results. The simulator knows the true position of every agent, so the
> counting and density errors are real errors rather than self-consistency checks, and the
> scenarios were scripted with fixed correct answers before the classifier saw them — so the
> classification result is falsifiable. But I wrote both the simulator and the system, so it is
> not external validation. The loaders for MOT20 and ShanghaiTech are written and tested;
> `scripts/evaluate_dataset.py` produces the same metrics against annotations I did not create,
> and that is the next thing to run."

If you have had time to download MOT20, quote that MAE **first** and the simulator numbers second.
An externally annotated number is worth more than three internal ones.

### ⚠️ The number you must NOT quote

The old README claimed **~96 % Random Forest accuracy**. That figure was **circular**: the
synthetic labels were generated by essentially the same weighted formula the risk engine uses to
score, so the model was rediscovering its own labels. It has been removed.

If anyone raises it, say so directly — *"that number was circular, I found it, and I removed it"*
is a far stronger answer than defending it. `--source synthetic` still reproduces the old
behaviour and prints a circularity warning, so you can demonstrate the difference on the spot.

---

## 4. Live demo script (5 minutes)

```bash
streamlit run app.py
```

1. **Landing page** — point at the eight failure modes. *"A single risk number tells an operator
   nothing about what to do. A crush and a panic dispersal both score high and need opposite
   responses."*
2. **Run `Escalating Crush (full arc)`** — 400 frames, ~30 seconds. Narrate the arc as it plays:
   normal flow → rapid influx → bottleneck → jamming → crush.
3. **Point at the threat banner** as it changes: the failure mode is *named*, the zone is *named*,
   and the countdown gives the *time budget*.
4. **Escalation ladder** — NORMAL → WATCH → ALERT → CRITICAL, one step at a time, each with dwell
   progress. *"It never skips a rung, so the early warning always reaches someone."*
5. **Alerts tab** — acknowledge one alert. *"This is the audit trail every crowd-disaster inquiry
   has had to reconstruct afterwards and could not. It also produces a human label that feeds
   back into training."*
6. **Forecast quality tab** — show the self-evaluation. *"Every forecast is scored against what
   actually happened one horizon later. It reports honestly when it fails to beat the baseline."*
7. **Report & exports** — download the incident report, scroll to the limitations section.

**Then, if there is time:** switch scenario to `counterflow` and `panic_dispersal` to show the
classifier picking different failure modes and the advisory changing its **DO NOT** line
accordingly. That contrast is the single most persuasive thing in the demo.

---

## 5. Viva Q&A

### Q1 — What is the core objective?
> Real-time crowd-crush risk **prediction** with explainable, SOP-grounded operator advisories.
> Not detection, not counting — those are inputs. The output is a decision.

### Q2 — How does YOLOv8 detect persons?
> CSP-Darknet backbone with C2f feature blocks; single forward pass per frame; bounding boxes for
> class 0 (`person`). I take the **bottom-centre** of each box, not the centre, because that is
> where the person stands on the ground plane — projecting a torso centre through the homography
> places a distant person several metres behind their true position.

### Q3 — How is crowd density calculated?
> Not `count / area`. That gives the same answer for 60 people spread evenly and 60 packed into a
> corner, and only the second one kills anybody. I estimate density **per person** with a
> k-nearest-neighbour estimator, `ρᵢ = k / (π·r_k²)`, computed on the **ground plane** after a
> 4-point homography, and report a robust 90th percentile.

### Q4 — Why does the homography matter?
> Because persons per square metre from raw pixels is not a physical quantity. Under perspective,
> a person 40 m away occupies a fraction of the pixels of one 5 m away, so uniform pixel-to-metre
> scaling systematically under-reports far-field density. Without calibration the system says so
> explicitly and treats densities as trend indicators only.

### Q5 — What is crowd pressure?
> `P = ρ_local × Var(v_local)` — Helbing's definition, the published precursor of crowd
> turbulence. I use his **quantity** but not his **threshold**: he reports onset near 0.02 s⁻² for
> optical flow over a smoothed grid, whereas I estimate velocity from discrete tracked centroids
> at a lower sample rate, which puts the value on a different numerical scale. Quoting 0.02 would
> be the right number with the wrong estimator. The threshold is calibrated empirically per site,
> and `scripts/evaluate.py --calibrate-pressure` does that.

### Q6 — How do you tell counter-flow from a jam?
> This is the hardest measurement in the project. Directional disorder cannot do it: a stalled
> crowd shuffles in scattered directions and scores just as disordered as two streams walking into
> each other — and they need opposite interventions. Instantaneous velocity cannot do it either,
> because at a funnel people fan inward from both sides while still walking forward, which looks
> identical to counter-flow. So I use each person's **net displacement over their track history**
> and two circular resultants: `R₁` (one dominant direction) and `R₂` (one dominant axis). One
> stream has both high; a jam has both low; two opposing streams share an axis but cancel in
> direction. `R₂ · (1 − R₁)` isolates exactly that case.

### Q7 — Does it actually predict, or just classify the current frame?
> It predicts. A 30-sample window feeds a temporal transformer that outputs risk at t+30 s plus a
> **time-to-threshold** countdown — "CRITICAL in ~42 s". It self-scores: every forecast is compared
> against the risk actually observed one horizon later. On seeds held out of training, MAE 0.0895
> against a persistence baseline of 0.1318 — a skill score of +32.1 %, with 74.8 % of forecasts
> within 0.10 of the truth.

### Q8 — Why compare against a "persistence baseline"?
> Because over a 30-second horizon, "assume the risk stays where it is" is a genuinely strong
> predictor, and a learned forecaster only earns deployment by beating it. My untrained trend
> heuristic scored **worse** than persistence — −52 % skill. Shrinking the extrapolated slope by
> the fit's own R² improved it to −27 %, but it is still worse than doing nothing, and I report
> that. It is exactly why the learned model is the default: on the same held-out seeds it scores
> +32 %. A project that only reports the method that worked has not actually measured anything.

### Q9 — Why RAG instead of just prompting an LLM?
> In a safety-critical domain, unattributable advice is legally unusable. RAG forces every
> recommendation to come from *this venue's* approved SOP, with the source filename attached as
> evidence. A plain LLM would invent plausible generic advice with no provenance. The advisor also
> falls back to a deterministic rule engine, so the system degrades rather than hallucinating when
> the model is unavailable.

### Q10 — What happens *after* the risk level is predicted?
> Six things. (1) An escalation state machine with dwell times and hysteresis converts the score
> series into a small number of discrete events — zero flapping across 24 runs. (2) The event names
> the zone, not the frame. (3) It fans out to console, JSONL, webhook and Telegram sinks, all off
> by default. (4) The advisory gives ranked actions with an owner each, plus a **DO NOT** line and
> a **TIME BUDGET** that compares the forecast against how long the recommended action takes to
> have physical effect. (5) The operator acknowledges with a decision and an action taken. (6) That
> acknowledgement suppresses re-notification, creates the audit trail, and becomes a human label
> for future training.

### Q11 — What is the "DO NOT" section for?
> Crowd interventions are not symmetric. Narrowing exits calms a bottleneck and kills people during
> a dispersal. Reversing a crowd relieves a queue and kills people inside a jam. A system that only
> ever says what to do will eventually recommend the lethal version of the right idea, so every
> failure mode carries an explicit counter-indication.

### Q12 — What fallback mechanisms exist?
> Vision: OpenCV HOG if YOLO is unavailable. Retrieval: TF-IDF if FAISS or sentence-transformers
> are missing. Advisor: a deterministic rule engine if no LLM is reachable. Forecast: the trend
> backend if no trained checkpoint exists. The system is fully functional offline with no API key,
> which matters for rural festivals and temple crowds where connectivity is poor.

### Q13 — Why did you build a crowd simulator?
> Three reasons. Real crowd-disaster footage is scarce and ethically fraught, so the simulator lets
> me produce every failure mode on demand for this demonstration. It knows the true position of
> every agent, which gives me real counting and density errors instead of self-consistency checks.
> And scripted scenarios with fixed correct answers make the classifier falsifiable — that is where
> the 7/8 result *and the one failure* come from. The physics is a reduced social-force model
> (Helbing & Molnár 1995). Simulated detections are labelled as such everywhere; everything
> downstream is production code.

### Q13b — Where is your dataset?
> There is no bundled real-world dataset, and I will not pretend otherwise. All the results I
> quoted are measured against the agent-based simulator I built, which supplies ground truth so
> those errors are real rather than self-consistency checks — but I wrote both sides, so it is not
> external validation. `data/` is set up for MOT20, ShanghaiTech and my own annotations, the
> loaders are written and tested, and `scripts/evaluate_dataset.py` reruns the same counting
> metrics against annotations I did not create. That is the single highest-value thing left to do.

### Q14 — What is your accuracy?
> That depends which module, and I deliberately do not quote one number for the fused risk score,
> because no labelled ground truth for "crowd risk" exists. Counting MAE 4.78 people; density
> correlation 0.926; failure-mode classification 7/8 scenarios; retrieval P@1 0.80 and MRR 0.831;
> forecasting +30.9 % skill over persistence. Any single headline accuracy figure for the fused
> score would have to be produced by scoring the model against labels it generated itself.

### Q15 — What is the weakest part of your system?
> *(Answer this one confidently — see §6.)*

---

## 6. Raise these weaknesses before the panel does

Volunteering a limitation with a measurement attached reads as rigour. Being caught hiding one
reads as carelessness. Lead with these:

0. **No real-world dataset.** All results are simulation-based. Raise this first — see §3.
1. **Detection undercounts under occlusion.** Measured bias **−1.09 persons/m² in the crush band**
   — worst exactly where accuracy matters most, because bodies occlude one another. The system
   reports the occlusion ratio per frame and flags the count as a lower bound. A density-map
   counter (CSRNet class) is the correct estimator in that regime and is the clearest next step.
2. **Rapid Influx classification fails** at 38 % recall, confused with Normal Flow. The inflow-rate
   threshold is not yet separable from ordinary churn.
3. **Risk weights are literature-informed, not learned** — no labelled incident dataset exists to
   fit them to.
4. **The trend forecast heuristic is worse than a persistence baseline** (−26.5 % skill). It ships
   only as a fallback for when no trained checkpoint exists, and the default backend is `auto`.
5. **Forecast lead time is short** in these scenarios (median 8 s before CRITICAL) because the
   scripted arcs escalate abruptly. A 30-second horizon earns its keep on longer, steadier footage.
6. **The transformer has seen these scenario types in training**, so its advantage on a genuinely
   novel venue is unproven — only its advantage on unseen random seeds is established.
7. **Centroid tracking produces ID switches** in dense scenes. Mitigated by smoothing velocity over
   several frames and excluding young tracks from kinematic statistics; Deep-SORT would do better.

### The three things needed to validate this properly

Say this — it shows you know what a real validation would require:

1. Counting error against an annotated public dataset (MOT20, JHU-CROWD++).
2. Expert agreement — several qualified people independently rating recorded clips, compared with
   the system using Cohen's κ.
3. Operator acknowledgements collected over a real event season. The acknowledgement store already
   records them in exactly the form needed for supervised training.

---

## 7. Submission checklist

- [ ] `pip install -r requirements.txt` on a clean machine
- [ ] `streamlit run app.py` → run `escalating_crush` end to end
- [ ] `python scripts/evaluate.py` → regenerate every simulator number in README §4
- [ ] **Put a real crowd video in `data/videos/`** → so YOLO detection is demonstrable at all
- [ ] *(high value)* Download MOT20 into `data/mot20/` → `python scripts/evaluate_dataset.py`
- [ ] `python scripts/train_forecaster.py` → `outputs/risk_forecaster.pt` exists
- [ ] `python scripts/train_risk_baseline.py` → note the honest 0.74, not 96 %
- [ ] `python scripts/run_demo.py --scene all` → screenshots of each advisory
- [ ] Export one incident report as a report appendix
- [ ] Cite Fruin (1971, 1993) and Helbing, Johansson & Al-Abideen (2007) in the references
- [ ] Include the confusion matrix and the ablation table as figures
- [ ] Include §6 verbatim as a "Limitations and Future Work" chapter

---

## 8. References to cite

- J. J. Fruin, *Pedestrian Planning and Design*, 1971 — level-of-service density bands.
- J. J. Fruin, "The Causes and Prevention of Crowd Disasters", 1993 — crush versus panic.
- D. Helbing, A. Johansson, H. Z. Al-Abideen, "Dynamics of crowd disasters: An empirical study",
  *Physical Review E* 75, 046109, 2007 — crowd turbulence and crowd pressure.
- D. Helbing, P. Molnár, "Social force model for pedestrian dynamics", *Physical Review E* 51,
  4282, 1995 — the simulator's physics.
- U. Weidmann, *Transporttechnik der Fussgänger*, 1993 — the fundamental diagram of pedestrian flow.
- G. K. Still, *Introduction to Crowd Science*, 2014 — crowd risk analysis practice.

---

*Decision support, not a substitute for a trained safety officer.*
