#!/usr/bin/env python3
"""Measured evaluation of CrowdGuard-RAG.

Design principle: **evaluate each module against ground truth that actually
exists, and refuse to invent a number for the one that has none.**

There is no labelled dataset of "true crowd risk", so any headline accuracy
figure for the fused risk score would be either circular or fabricated. What
*can* be measured honestly is measured here:

  counting      -- detected count versus the simulator's known agent count
  density       -- estimated persons/m2 versus the simulator's known density
  typology      -- failure-mode classification against scripted scenarios
  retrieval     -- Precision@K, Recall@K, MRR and nDCG on a hand-written query
                   set with declared relevant documents
  forecasting   -- every forecast scored against the risk actually observed one
                   horizon later, on data the forecaster never fitted
  escalation    -- alert count, flapping rate and warning lead time
  runtime       -- per-stage latency and end-to-end throughput
  ablation      -- contribution of each feature to the fused score
  calibration   -- observed crowd-pressure percentiles, for threshold setting

Usage:
    python scripts/evaluate.py                    # everything
    python scripts/evaluate.py --only retrieval
    python scripts/evaluate.py --calibrate-pressure
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crowdguard.config import RAGConfig, RiskConfig  # noqa: E402
from src.crowdguard.rag_engine import RAGIndex  # noqa: E402
from src.crowdguard.risk_taxonomy import ORDERED_CODES  # noqa: E402
from src.crowdguard.simulator import SCENARIOS  # noqa: E402


# --------------------------------------------------------------------------- #
def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _run_scenarios(fps: float, out_dir: Path, seeds: List[int],
                   backend: str = "auto") -> Dict[str, List[List[Dict[str, Any]]]]:
    """Run every scenario at every seed; return {scenario: [rows_per_seed]}."""
    from src.crowdguard.main import process_simulation

    out_dir.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, List[List[Dict[str, Any]]]] = {}
    for key in SCENARIOS:
        per_seed = []
        for seed in seeds:
            path = out_dir / f"eval_{key}_{seed}_{backend}.jsonl"
            with redirect_stdout(io.StringIO()):
                process_simulation(
                    scenario=key, fps=fps, seed=seed, output_video=None,
                    output_json=str(path), output_events=None, output_report=None,
                    forecast_backend=backend, verbose=False,
                )
            per_seed.append([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
        runs[key] = per_seed
    return runs


# --------------------------------------------------------------------------- #
def eval_counting_and_density(runs) -> Dict[str, Any]:
    _rule("1. COUNTING AND DENSITY  (against simulator ground truth)")
    print("The simulator knows where it put every agent, so these are true errors,\n"
          "not self-consistency checks. Detections carry positional noise and\n"
          "occlusion dropout, so the counting error here is a real detector-style error.\n")

    rows = [r for per_seed in runs.values() for rs in per_seed for r in rs
            if r.get("ground_truth") and r["ground_truth"]["true_count"] >= 5]
    if not rows:
        return {}

    pred_n = np.array([r["person_count"] for r in rows], float)
    true_n = np.array([r["ground_truth"]["true_count"] for r in rows], float)
    pred_d = np.array([r["local_density_peak"] for r in rows], float)
    true_d = np.array([r["ground_truth"]["true_density_peak"] for r in rows], float)

    cnt_mae = float(np.mean(np.abs(pred_n - true_n)))
    cnt_rmse = float(np.sqrt(np.mean((pred_n - true_n) ** 2)))
    cnt_mape = float(np.mean(np.abs(pred_n - true_n) / np.maximum(true_n, 1)) * 100)
    den_mae = float(np.mean(np.abs(pred_d - true_d)))
    den_bias = float(np.mean(pred_d - true_d))
    corr = float(np.corrcoef(pred_d, true_d)[0, 1])

    print(f"  samples                      {len(rows)}")
    print(f"  count MAE                    {cnt_mae:.2f} people")
    print(f"  count RMSE                   {cnt_rmse:.2f} people")
    print(f"  count MAPE                   {cnt_mape:.1f} %")
    print(f"  density MAE                  {den_mae:.2f} persons/m2")
    print(f"  density bias (pred - true)   {den_bias:+.2f} persons/m2")
    print(f"  density correlation r        {corr:.3f}")

    dense = true_d >= 4.0
    if dense.sum() > 10:
        print(f"\n  In the crush band (true density >= 4/m2), n={int(dense.sum())}:")
        print(f"    count MAPE                 {np.mean(np.abs(pred_n[dense] - true_n[dense]) / true_n[dense]) * 100:.1f} %")
        print(f"    density MAE                {np.mean(np.abs(pred_d[dense] - true_d[dense])):.2f} persons/m2")
        print(f"    density bias               {np.mean(pred_d[dense] - true_d[dense]):+.2f} persons/m2"
              "   <- negative means the system UNDER-reports exactly where it matters most")
    return {"count_mae": cnt_mae, "count_rmse": cnt_rmse, "count_mape": cnt_mape,
            "density_mae": den_mae, "density_bias": den_bias, "density_r": corr,
            "samples": len(rows)}


# --------------------------------------------------------------------------- #
def eval_typology(runs) -> Dict[str, Any]:
    _rule("2. FAILURE-MODE CLASSIFICATION  (against scripted scenarios)")
    print("Each scenario is physically constructed to produce one specific failure\n"
          "mode. The classifier is scored on the developed second half of each run,\n"
          "over multiple random seeds.\n")

    confusion: Dict[str, Dict[str, int]] = {}
    per_scenario, correct = {}, 0
    for key, per_seed in runs.items():
        expected = SCENARIOS[key].expected_type
        votes: Dict[str, int] = {}
        for rows in per_seed:
            half = rows[len(rows) // 2:]
            for r in half:
                got = r["primary_risk_type"]
                votes[got] = votes.get(got, 0) + 1
                confusion.setdefault(expected, {}).setdefault(got, 0)
                confusion[expected][got] += 1
        top = max(votes, key=votes.get)
        share = votes[top] / sum(votes.values())
        ok = top == expected
        correct += ok
        per_scenario[key] = {"expected": expected, "predicted": top,
                             "share": round(share, 3), "correct": ok}
        print(f"  {key:18} expected {expected:22} got {top:22} {share * 100:5.1f}%  {'PASS' if ok else 'FAIL'}")

    total_frames = sum(sum(v.values()) for v in confusion.values())
    frame_correct = sum(v.get(k, 0) for k, v in confusion.items())
    print(f"\n  scenario-level accuracy      {correct}/{len(runs)}")
    print(f"  frame-level accuracy         {frame_correct}/{total_frames} = {frame_correct / max(1, total_frames) * 100:.1f} %")

    print("\n  Confusion (rows = expected, columns = predicted, frame-level):")
    codes = [c for c in ORDERED_CODES if c in confusion or any(c in v for v in confusion.values())]
    print("    " + "".join(f"{c[:9]:>10}" for c in codes))
    for exp in codes:
        if exp not in confusion:
            continue
        row = confusion[exp]
        tot = sum(row.values())
        print(f"    {exp[:9]:>9} " + "".join(f"{row.get(c, 0) / tot * 100:9.0f}%" for c in codes))
    return {"scenario_accuracy": correct / len(runs),
            "frame_accuracy": frame_correct / max(1, total_frames),
            "per_scenario": per_scenario}


# --------------------------------------------------------------------------- #
def eval_retrieval(kb_dir: str, query_file: Path, ks=(1, 3, 5)) -> Dict[str, Any]:
    _rule("3. RAG RETRIEVAL  (hand-written query set with declared relevant documents)")
    if not query_file.exists():
        print(f"  query set not found: {query_file}")
        return {}
    spec = json.loads(query_file.read_text())
    queries = spec["queries"]

    rag = RAGIndex(RAGConfig(knowledge_base_dir=kb_dir, top_k=max(ks)))
    rag.load_documents(kb_dir)
    rag.build()
    print(f"  backend                      {rag.backend}")
    print(f"  indexed chunks               {len(rag.chunks)} from {len(set(c[0] for c in rag.chunks))} documents")
    print(f"  queries                      {len(queries)}\n")

    metrics = {k: {"p": [], "r": []} for k in ks}
    rr, ndcg, latencies, failures = [], [], [], []

    for item in queries:
        relevant = set(item["relevant"])
        t0 = time.time()
        hits = rag.search(item["query"], top_k=max(ks))
        latencies.append((time.time() - t0) * 1000)

        # Collapse chunks to documents, keeping best rank per document.
        ranked: List[str] = []
        for h in hits:
            if h.source not in ranked:
                ranked.append(h.source)

        for k in ks:
            top = ranked[:k]
            tp = len(set(top) & relevant)
            metrics[k]["p"].append(tp / max(1, len(top)))
            metrics[k]["r"].append(tp / max(1, len(relevant)))

        first = next((i + 1 for i, s in enumerate(ranked) if s in relevant), None)
        rr.append(1.0 / first if first else 0.0)
        if first is None or first > 1:
            failures.append((item["query"], ranked[:3], sorted(relevant)))

        gains = [1.0 if s in relevant else 0.0 for s in ranked[:max(ks)]]
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), max(ks))))
        ndcg.append(dcg / idcg if idcg else 0.0)

    out = {}
    for k in ks:
        p, r = float(np.mean(metrics[k]["p"])), float(np.mean(metrics[k]["r"]))
        out[f"precision@{k}"], out[f"recall@{k}"] = p, r
        print(f"  Precision@{k}                 {p:.3f}        Recall@{k}  {r:.3f}")
    out["mrr"] = float(np.mean(rr))
    out["ndcg"] = float(np.mean(ndcg))
    out["latency_ms"] = float(np.mean(latencies))
    out["backend"] = rag.backend
    print(f"  MRR                          {out['mrr']:.3f}")
    print(f"  nDCG@{max(ks)}                       {out['ndcg']:.3f}")
    print(f"  mean query latency           {out['latency_ms']:.1f} ms")

    if failures:
        print(f"\n  Queries whose top hit was not a declared relevant document ({len(failures)}):")
        for q, got, want in failures[:8]:
            print(f"    \"{q[:58]}\"\n       got  {got}\n       want {want}")
    return out


# --------------------------------------------------------------------------- #
def eval_forecast(runs, horizon: float = 30.0) -> Dict[str, Any]:
    _rule("4. FORECASTING  (each prediction scored against what actually happened)")
    print(f"  Every forecast made at time t is compared with the risk score observed\n"
          f"  at t+{horizon:.0f}s in the same run. Nothing is fitted to this data.\n")

    errs, base_errs, lead_times = [], [], []
    for key, per_seed in runs.items():
        for rows in per_seed:
            times = np.array([r["timestamp_sec"] for r in rows])
            risk = np.array([r["risk_score"] for r in rows])
            pred = np.array([r["forecast"]["predicted_score"] for r in rows])
            ready = np.array([r["forecast"]["ready"] for r in rows])

            for i in range(len(rows)):
                if not ready[i]:
                    continue
                j = int(np.searchsorted(times, times[i] + horizon))
                if j >= len(rows):
                    continue
                errs.append(abs(pred[i] - risk[j]))
                base_errs.append(abs(risk[i] - risk[j]))   # persistence baseline

            # Lead time: first moment the forecast warned of CRITICAL, versus the
            # first moment risk actually reached it.
            crit = np.where(risk >= 0.72)[0]
            if crit.size:
                warned = [i for i in range(len(rows))
                          if ready[i] and rows[i]["forecast"].get("time_to_critical_sec") is not None
                          and i < crit[0]]
                if warned:
                    lead_times.append(float(times[crit[0]] - times[warned[0]]))

    if not errs:
        print("  no matured forecasts")
        return {}
    e, b = np.array(errs), np.array(base_errs)
    print(f"  matured forecasts            {len(e)}")
    print(f"  forecast MAE                 {e.mean():.4f}")
    print(f"  persistence-baseline MAE     {b.mean():.4f}   (assume risk stays where it is)")
    skill = 1 - e.mean() / max(1e-9, b.mean())
    print(f"  skill score vs persistence   {skill:+.3f}   ({'better' if skill > 0 else 'WORSE'} than doing nothing)")
    print(f"  within 0.10                  {(e <= 0.10).mean() * 100:.1f} %")
    print(f"  within 0.20                  {(e <= 0.20).mean() * 100:.1f} %")
    if lead_times:
        print(f"  median warning lead time     {np.median(lead_times):.0f} s before risk actually reached CRITICAL")
        print(f"                               (n={len(lead_times)} runs that reached CRITICAL)")
    return {"mae": float(e.mean()), "baseline_mae": float(b.mean()), "skill": float(skill),
            "within_0.1": float((e <= 0.10).mean()), "n": int(e.size),
            "median_lead_sec": float(np.median(lead_times)) if lead_times else None}


# --------------------------------------------------------------------------- #
def eval_escalation(runs) -> Dict[str, Any]:
    _rule("5. ESCALATION BEHAVIOUR  (alert volume and flapping)")
    print("  A safety system that cries wolf gets muted, so alert volume is itself a\n"
          "  quality metric. Flapping counts state changes that reverse within 30 s.\n")

    total_alerts, flaps, minutes = 0, 0, 0.0
    for key, per_seed in runs.items():
        for rows in per_seed:
            states = [r["escalation_state"] for r in rows]
            times = [r["timestamp_sec"] for r in rows]
            minutes += (times[-1] - times[0]) / 60.0
            changes = [(times[i], states[i - 1], states[i])
                       for i in range(1, len(states)) if states[i] != states[i - 1]]
            total_alerts += len(changes)
            for a in range(len(changes) - 1):
                t0, f0, t0_to = changes[a]
                t1, f1, t1_to = changes[a + 1]
                if t1 - t0 < 30.0 and t1_to == f0:
                    flaps += 1
    print(f"  total state changes          {total_alerts}")
    print(f"  monitored duration           {minutes:.1f} minutes across {sum(len(v) for v in runs.values())} runs")
    print(f"  state changes per minute     {total_alerts / max(1e-6, minutes):.2f}")
    print(f"  flapping transitions         {flaps}  ({flaps / max(1, total_alerts) * 100:.1f} % of all changes)")
    return {"changes": total_alerts, "per_minute": total_alerts / max(1e-6, minutes),
            "flap_rate": flaps / max(1, total_alerts)}


# --------------------------------------------------------------------------- #
def eval_ablation(runs) -> Dict[str, Any]:
    _rule("6. FEATURE ABLATION  (what each term actually contributes)")
    print("  Recomputes the fused score with each feature's weight zeroed, and reports\n"
          "  how much the score moves and how often the risk band changes.\n")

    rows = [r for per_seed in runs.values() for rs in per_seed for r in rs if r.get("contributions")]
    if not rows:
        return {}
    keys = list(rows[0]["contributions"].keys())
    base = np.array([sum(r["contributions"].values()) for r in rows])
    cfg = RiskConfig()

    def band(x):
        return np.where(x >= cfg.high_threshold, 2, np.where(x >= cfg.low_threshold, 1, 0))

    out = {}
    print(f"  {'feature':18} {'mean share':>11} {'MAE if removed':>16} {'band changes':>14}")
    for k in keys:
        contrib = np.array([r["contributions"][k] for r in rows])
        without = base - contrib
        changed = float((band(base) != band(without)).mean())
        out[k] = {"share": float(contrib.mean()), "delta": float(np.abs(contrib).mean()),
                  "band_change_rate": changed}
        print(f"  {k:18} {contrib.mean():11.4f} {np.abs(contrib).mean():16.4f} {changed * 100:13.1f}%")
    return out


# --------------------------------------------------------------------------- #
def eval_runtime(fps: float) -> Dict[str, Any]:
    _rule("7. RUNTIME  (per-stage latency)")
    from src.crowdguard.main import process_simulation

    with redirect_stdout(io.StringIO()):
        res = process_simulation(scenario="escalating_crush", fps=fps, output_video=None,
                                 output_json=None, output_events=None, output_report=None,
                                 verbose=False)
    rt = res["runtime"]
    print(f"  frames processed             {rt['frames']}")
    for stage, ms in rt["stage_ms_per_frame"].items():
        print(f"    {stage:12}               {ms:7.2f} ms/frame")
    print(f"  total                        {rt['total_ms_per_frame']:7.2f} ms/frame")
    print(f"  throughput                   {rt['throughput_fps']} fps (detection bypassed in simulation)")
    print(f"  backends                     {rt['backends']}")
    return rt


# --------------------------------------------------------------------------- #
def calibrate_pressure(runs) -> Dict[str, Any]:
    _rule("8. CROWD-PRESSURE CALIBRATION")
    print("  Crowd pressure uses Helbing's definition (local density x local velocity\n"
          "  variance), but its absolute scale depends on the velocity estimator and the\n"
          "  sample rate, so the published 0.02 threshold does not transfer to a\n"
          "  tracked-centroid estimator. Set thresholds from the observed distribution:\n"
          "  the warning level at roughly the 85th percentile of a normal crowd, and the\n"
          "  critical level where genuinely dangerous scenarios separate from safe ones.\n")

    safe, dangerous = [], []
    for key, per_seed in runs.items():
        target = "safe" if SCENARIOS[key].expected_type in {"NORMAL_FLOW", "RAPID_INFLUX"} else "danger"
        for rows in per_seed:
            for r in rows:
                (safe if target == "safe" else dangerous).append(r["crowd_pressure"])
    s, d = np.array(safe), np.array(dangerous)
    print(f"  {'percentile':>12} {'safe scenarios':>16} {'hazardous scenarios':>22}")
    for q in (50, 75, 85, 90, 95, 99):
        print(f"  {q:>11}% {np.percentile(s, q):16.3f} {np.percentile(d, q):22.3f}")
    warn, crit = float(np.percentile(s, 85)), float(np.percentile(s, 97))
    print(f"\n  suggested pressure_warning   {warn:.2f}")
    print(f"  suggested pressure_critical  {crit:.2f}")
    print(f"  currently configured         warning {RiskConfig().pressure_warning:.2f}  "
          f"critical {RiskConfig().pressure_critical:.2f}")
    return {"suggested_warning": warn, "suggested_critical": crit}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Measured evaluation of CrowdGuard-RAG")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["counting", "typology", "retrieval", "forecast",
                             "escalation", "ablation", "runtime", "pressure"])
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 21, 99])
    ap.add_argument("--knowledge-base", default=str(ROOT / "knowledge_base"))
    ap.add_argument("--queries", default=str(ROOT / "evaluation" / "retrieval_queries.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "evaluation_report.json"))
    ap.add_argument("--calibrate-pressure", action="store_true")
    ap.add_argument("--forecast-backend", default="auto", choices=["auto", "trend", "transformer"])
    ap.add_argument("--compare-forecasts", action="store_true",
                    help="Run both forecast backends and report each against persistence")
    args = ap.parse_args()

    want = set(args.only) if args.only else {"counting", "typology", "retrieval", "forecast",
                                             "escalation", "ablation", "runtime"}
    if args.calibrate_pressure:
        want = {"pressure"}

    print("CrowdGuard-RAG evaluation")
    print(f"seeds={args.seeds}  fps={args.fps}  scenarios={len(SCENARIOS)}")

    report: Dict[str, Any] = {"seeds": args.seeds, "fps": args.fps}
    needs_runs = want & {"counting", "typology", "forecast", "escalation", "ablation", "pressure"}
    runs = {}
    if needs_runs:
        print("\nrunning simulations...", flush=True)
        runs = _run_scenarios(args.fps, ROOT / "outputs" / "eval_runs", args.seeds,
                              args.forecast_backend)
        print(f"  {sum(len(v) for v in runs.values())} runs, "
              f"{sum(len(r) for v in runs.values() for r in v)} frames")

    if "counting" in want:
        report["counting"] = eval_counting_and_density(runs)
    if "typology" in want:
        report["typology"] = eval_typology(runs)
    if "retrieval" in want:
        report["retrieval"] = eval_retrieval(args.knowledge_base, Path(args.queries))
    if "forecast" in want:
        report["forecast"] = eval_forecast(runs)
        if args.compare_forecasts and args.forecast_backend != "transformer":
            print("\n  -- same evaluation with the trained transformer backend --")
            tr = _run_scenarios(args.fps, ROOT / "outputs" / "eval_runs", args.seeds, "transformer")
            report["forecast_transformer"] = eval_forecast(tr)
    if "escalation" in want:
        report["escalation"] = eval_escalation(runs)
    if "ablation" in want:
        report["ablation"] = eval_ablation(runs)
    if "runtime" in want:
        report["runtime"] = eval_runtime(args.fps)
    if "pressure" in want:
        report["pressure_calibration"] = calibrate_pressure(runs)

    _rule("WHAT THIS EVALUATION DELIBERATELY DOES NOT CLAIM")
    print("""  There is no headline "risk prediction accuracy" figure here, and that is a
  deliberate choice rather than an omission.

  No labelled dataset of true crowd risk exists for this deployment. Any single
  accuracy number for the fused risk score would have to be produced by scoring
  the model against labels the model itself generated, which measures nothing.
  The scenario results above are the honest version of that claim: the failure-mode
  classifier is scored against physically scripted situations whose correct answer
  was fixed by construction before the classifier saw them.

  To validate the fused score properly, three things are needed, and the code to
  collect all three is already in place:
    1. Counting error against an annotated public dataset (MOT20, JHU-CROWD++).
    2. Expert agreement: several qualified people independently rating recorded
       clips, compared against the system with Cohen's kappa.
    3. Operator acknowledgements gathered over a real event season, which the
       acknowledgement store already records in the form needed for training.""")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
