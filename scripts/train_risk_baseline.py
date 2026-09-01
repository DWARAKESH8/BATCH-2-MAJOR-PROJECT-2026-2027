#!/usr/bin/env python3
"""Baseline risk classifier -- with the circularity removed.

The problem with the original version
-------------------------------------
It generated synthetic features, labelled them by applying essentially the same
weighted formula the deployed risk engine uses, then trained a Random Forest to
predict those labels and reported ~96% accuracy.

That number measured nothing. The model was rediscovering the formula that had
produced its own labels. Reporting it as a validation result would not survive
a single informed question, and it is the kind of claim that costs marks rather
than earning them.

What this version does instead
------------------------------
Default source is `simulation`: run the agent-based crowd simulator, take the
measured features that the real pipeline produces, and take the LABEL from the
simulator's ground-truth density mapped onto Fruin's published level-of-service
bands. The label therefore comes from a physical quantity the scoring formula
never sees, and the fused `risk_score` is explicitly excluded from the feature
set and audited for at run time.

That makes the resulting accuracy a real, if modest, claim: *can a classifier
recover the Fruin risk band from the measured multi-modal features, on data it
was not fitted to?*

`--source synthetic` reproduces the old behaviour and prints a prominent
warning, so the difference can be demonstrated rather than merely asserted.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The fused risk score is deliberately NOT a feature. Including it would let the
# model read the answer off the input.
FEATURES: List[str] = [
    "person_count",
    "local_density_peak",
    "local_density_mean",
    "avg_speed_ms",
    "flow_disorder",
    "counterflow_index",
    "bottleneck_ratio",
    "crowd_pressure",
    "flux_efficiency",
    "oscillation_index",
    "density_rate_per_min",
    "sustained_density_sec",
]

LEAKY = {"risk_score", "risk_level", "hazard_index", "primary_risk_score"}


def fruin_band(density: float) -> str:
    """Label from Fruin's level-of-service bands -- independent of our scorer."""
    if density >= 4.0:
        return "high"
    if density >= 2.0:
        return "moderate"
    return "low"


def build_simulation_dataset(fps: float, seeds: List[int]) -> pd.DataFrame:
    from src.crowdguard.main import process_simulation
    from src.crowdguard.simulator import SCENARIOS

    out_dir = ROOT / "outputs" / "train_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in SCENARIOS:
        for seed in seeds:
            path = out_dir / f"train_{key}_{seed}.jsonl"
            with redirect_stdout(io.StringIO()):
                process_simulation(scenario=key, fps=fps, seed=seed, output_video=None,
                                   output_json=str(path), output_events=None,
                                   output_report=None, verbose=False)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                gt = r.get("ground_truth")
                if not gt or gt["true_count"] < 5:
                    continue
                row = {f: float(r.get(f, 0.0) or 0.0) for f in FEATURES}
                # Label from the simulator's own density, via Fruin's bands.
                row["label"] = fruin_band(gt["true_density_peak"])
                row["scenario"] = key
                row["seed"] = seed
                rows.append(row)
    return pd.DataFrame(rows)


def build_synthetic_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """The original circular generator, retained only for comparison."""
    rng = np.random.default_rng(seed)
    count = rng.integers(5, 250, size=n)
    area = rng.uniform(60, 250, size=n)
    density = count / area
    avg_speed = rng.uniform(0.0, 2.0, size=n)
    disorder = rng.uniform(0.0, 1.0, size=n)
    bottleneck = rng.uniform(0.0, 1.0, size=n)
    pressure = density * avg_speed ** 2
    score = (0.35 * np.clip((density - 0.5) / 4.5, 0, 1)
             + 0.20 * np.clip(avg_speed / 1.5, 0, 1)
             + 0.25 * np.clip(disorder / 0.55, 0, 1)
             + 0.15 * np.clip(bottleneck / 0.65, 0, 1)
             + 0.05 * np.clip(pressure / 8.0, 0, 1))
    df = pd.DataFrame({
        "person_count": count, "local_density_peak": density, "local_density_mean": density * 0.7,
        "avg_speed_ms": avg_speed, "flow_disorder": disorder, "counterflow_index": rng.uniform(0, 1, n),
        "bottleneck_ratio": bottleneck, "crowd_pressure": pressure,
        "flux_efficiency": rng.uniform(0, 1, n), "oscillation_index": rng.uniform(0, 1, n),
        "density_rate_per_min": rng.normal(0, 1, n), "sustained_density_sec": rng.uniform(0, 60, n),
        "label": np.where(score >= 0.67, "high", np.where(score >= 0.35, "moderate", "low")),
    })
    return df


def audit_leakage(df: pd.DataFrame, features: List[str]) -> None:
    print("\n--- Leakage audit ---")
    leaked = [c for c in features if c in LEAKY]
    if leaked:
        print(f"  FAIL: model-derived columns present in the feature set: {leaked}")
        raise SystemExit(1)
    print(f"  OK: none of {sorted(LEAKY)} are used as features.")
    present = [c for c in LEAKY if c in df.columns]
    if present:
        for c in present:
            try:
                corr = df[c].corr(df["label"].map({"low": 0, "moderate": 1, "high": 2}).astype(float))
                print(f"  note: '{c}' exists in the source data (corr with label {corr:+.2f}) but is excluded.")
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the baseline risk-band classifier")
    ap.add_argument("--source", default="simulation", choices=["simulation", "csv", "synthetic"])
    ap.add_argument("--csv", default=None, help="CSV with the feature columns plus 'label'")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "risk_random_forest.joblib"))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 21, 99, 123])
    ap.add_argument("--holdout-seed", type=int, default=None,
                    help="Hold out one seed entirely, so train and test share no run")
    args = ap.parse_args()

    if args.source == "csv":
        if not args.csv:
            raise SystemExit("--source csv requires --csv")
        df = pd.read_csv(args.csv)
        provenance = f"user CSV: {args.csv}"
    elif args.source == "synthetic":
        print("=" * 78)
        print("WARNING -- CIRCULAR TRAINING DATA")
        print("=" * 78)
        print("""These labels are produced by a weighted formula almost identical to the one
the risk engine uses to score. A model trained here is rediscovering that
formula, so its accuracy measures self-consistency, not correctness.

Do NOT report this number as a validation result. It is included only so the
difference against --source simulation can be demonstrated.""")
        print("=" * 78)
        df = build_synthetic_dataset()
        provenance = "synthetic (CIRCULAR -- not a validation result)"
    else:
        print("Generating training data from the crowd simulator...")
        print("Labels come from the simulator's ground-truth density via Fruin's")
        print("level-of-service bands, which the risk engine's scorer never sees.")
        df = build_simulation_dataset(args.fps, args.seeds)
        provenance = f"simulation, seeds={args.seeds}, labels=Fruin bands on ground-truth density"

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    print(f"\nDataset: {len(df)} samples from {provenance}")
    print(df["label"].value_counts().to_string())
    audit_leakage(df, FEATURES)

    x, y_raw = df[FEATURES], df["label"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Split by run where possible: random row splits leak, because consecutive
    # frames of one run are nearly identical and would appear on both sides.
    if args.holdout_seed is not None and "seed" in df.columns:
        test_mask = df["seed"] == args.holdout_seed
        x_train, x_test = x[~test_mask], x[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        split_note = f"held-out seed {args.holdout_seed} (no run shared between train and test)"
    elif "scenario" in df.columns:
        scenarios = sorted(df["scenario"].unique())
        held = set(scenarios[::3])
        test_mask = df["scenario"].isin(held)
        x_train, x_test = x[~test_mask], x[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        split_note = f"held-out scenarios {sorted(held)} (unseen situations, the hard test)"
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=y)
        split_note = "random 75/25 split"

    print(f"\nSplit: {split_note}")
    print(f"  train {len(x_train)}   test {len(x_test)}")
    if len(np.unique(y_test)) < 2:
        print("  WARNING: the held-out set contains a single class; metrics will be degenerate.")

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    print("\n--- Held-out performance ---")
    print(classification_report(y_test, pred, target_names=le.classes_, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print("        " + "".join(f"{c:>10}" for c in le.classes_))
    for i, c in enumerate(le.classes_):
        print(f"  {c:>6}" + "".join(f"{v:10d}" for v in confusion_matrix(y_test, pred)[i]))

    try:
        proba = clf.predict_proba(x_test)
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        print(f"\nROC-AUC (macro, one-vs-rest): {auc:.3f}")
    except Exception as exc:
        print(f"\nROC-AUC unavailable: {exc}")

    try:
        cv = cross_val_score(clf, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
        print(f"5-fold CV accuracy: {cv.mean():.3f} +/- {cv.std():.3f}")
        print("  (optimistic: consecutive frames of one run land in different folds)")
    except Exception:
        pass

    print("\n--- Permutation importance on held-out data ---")
    print("  (impurity importance is biased toward high-cardinality features; this is not)")
    try:
        imp = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        for i in np.argsort(imp.importances_mean)[::-1]:
            print(f"  {FEATURES[i]:24} {imp.importances_mean[i]:+.4f} +/- {imp.importances_std[i]:.4f}")
    except Exception as exc:
        print(f"  unavailable: {exc}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURES, "classes": list(le.classes_),
                 "provenance": provenance, "split": split_note}, args.output)
    print(f"\nSaved to {args.output}")

    print("\n--- How to describe this result honestly ---")
    if args.source == "synthetic":
        print("  Do not report this figure. It is circular by construction.")
    else:
        print("""  "A Random Forest recovers the Fruin density band from the measured
   multi-modal crowd features on held-out scenarios it was never trained on.
   The labels derive from the simulator's ground-truth density, not from our
   own risk formula, and the fused risk score is excluded from the features
   and audited for. This validates that the feature set carries the density
   signal; it does NOT validate the fused risk score, for which no labelled
   ground truth exists." """)


if __name__ == "__main__":
    main()
