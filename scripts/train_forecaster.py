#!/usr/bin/env python3
"""Train the temporal risk forecaster.

This is the model behind the claim that the system *predicts* rather than
merely classifies. It consumes a window of measured crowd features and predicts
the risk score H seconds into the future.

Two honesty requirements shape the whole script:

1. **Split by run, never by row.** Consecutive frames of one video are nearly
   identical, so a random row split puts near-duplicates on both sides and
   produces a beautiful, meaningless validation score. Entire runs are held out
   instead.

2. **Report the persistence baseline.** "Assume the risk stays exactly where it
   is" is a genuinely strong predictor over 30 seconds. A learned forecaster is
   only worth deploying if it beats that, so the skill score against persistence
   is printed alongside the raw error, and the script says plainly when the
   model has failed to beat it.

Data sources:
    --source simulation  (default) run the scenarios and train on the result
    --source logs        train on risk_log.jsonl files from real video runs
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crowdguard.forecast import FEATURE_ORDER, build_sequences  # noqa: E402


def collect_simulation_runs(fps: float, seeds: List[int]) -> Dict[str, List[dict]]:
    from src.crowdguard.main import process_simulation
    from src.crowdguard.simulator import SCENARIOS

    out_dir = ROOT / "outputs" / "forecast_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, List[dict]] = {}
    for key in SCENARIOS:
        for seed in seeds:
            path = out_dir / f"fc_{key}_{seed}.jsonl"
            if not path.exists():
                with redirect_stdout(io.StringIO()):
                    process_simulation(scenario=key, fps=fps, seed=seed, output_video=None,
                                       output_json=str(path), output_events=None,
                                       output_report=None, verbose=False)
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            if len(rows) > 60:
                runs[f"{key}_{seed}"] = rows
    return runs


def collect_log_runs(paths: List[str]) -> Dict[str, List[dict]]:
    runs = {}
    for p in paths:
        rows = [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
        if len(rows) > 60:
            runs[Path(p).stem] = rows
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the temporal risk forecaster")
    ap.add_argument("--source", default="simulation", choices=["simulation", "logs"])
    ap.add_argument("--logs", nargs="*", default=[], help="risk_log.jsonl paths for --source logs")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 21, 99, 123, 404])
    ap.add_argument("--window", type=int, default=30, help="input window in samples")
    ap.add_argument("--horizon-sec", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", default=str(ROOT / "outputs" / "risk_forecaster.pt"))
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit("PyTorch is required: pip install torch")

    from src.crowdguard.transformer_model import TemporalRiskForecaster

    horizon_steps = max(1, int(round(args.horizon_sec * args.fps)))
    print(f"Horizon {args.horizon_sec:.0f}s = {horizon_steps} samples at {args.fps} fps")
    print(f"Input window {args.window} samples ({args.window / args.fps:.1f}s of history)")

    runs = (collect_log_runs(args.logs) if args.source == "logs"
            else collect_simulation_runs(args.fps, args.seeds))
    if len(runs) < 4:
        raise SystemExit(f"Need at least 4 usable runs, found {len(runs)}")

    names = sorted(runs)
    # Hold out whole runs. A random row split would put frame t in train and
    # frame t+1 in validation, and those are the same picture.
    val_names = set(names[::4])
    train_names = [n for n in names if n not in val_names]
    print(f"\nRuns: {len(names)} total -> {len(train_names)} train, {len(val_names)} validation")
    print(f"  validation runs: {sorted(val_names)}")

    def assemble(subset):
        xs, ys, cs = [], [], []
        for n in subset:
            x, y, c = build_sequences(runs[n], window=args.window, horizon_steps=horizon_steps)
            if x.shape[0]:
                xs.append(x); ys.append(y); cs.append(c)
        if not xs:
            return None
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)

    train = assemble(train_names)
    val = assemble(sorted(val_names))
    if train is None or val is None:
        raise SystemExit("Not enough contiguous data to build sequences")

    xtr, ytr, ctr = train
    xva, yva, cva = val
    print(f"  training sequences   {xtr.shape}")
    print(f"  validation sequences {xva.shape}")
    print(f"  features: {FEATURE_ORDER}")

    # Persistence baseline: the last observed risk, carried forward.
    risk_idx = FEATURE_ORDER.index("risk_score")
    persistence = xva[:, -1, risk_idx]          # already scaled by 1.0 for risk
    base_mae = float(np.mean(np.abs(persistence - yva)))
    print(f"\nPersistence baseline MAE on validation: {base_mae:.4f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalRiskForecaster(feature_dim=len(FEATURE_ORDER)).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber, ce = nn.SmoothL1Loss(beta=0.1), nn.CrossEntropyLoss()

    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    ctr_t = torch.from_numpy(ctr).to(device)
    xva_t = torch.from_numpy(xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    best, best_state, patience = float("inf"), None, 0
    print(f"\n{'epoch':>6} {'train':>9} {'val MAE':>9} {'vs base':>9}")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(xtr_t.shape[0], device=device)
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            out = model(xtr_t[idx])
            # The classification head regularises the regression: the high-risk
            # tail is rare, and a pure regression loss simply predicts the mean.
            loss = huber(out["risk"], ytr_t[idx]) + 0.3 * ce(out["logits"], ctr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_mae = float(torch.mean(torch.abs(model(xva_t)["risk"] - yva_t)))
        skill = 1 - val_mae / max(1e-9, base_mae)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"{epoch:6d} {total / len(perm):9.4f} {val_mae:9.4f} {skill:+9.3f}")

        if val_mae < best - 1e-5:
            best, patience = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 12:
                print(f"  early stop at epoch {epoch}")
                break

    skill = 1 - best / max(1e-9, base_mae)
    print(f"\n--- Result ---")
    print(f"  best validation MAE        {best:.4f}")
    print(f"  persistence baseline MAE   {base_mae:.4f}")
    print(f"  skill score                {skill:+.3f}")

    if skill <= 0.02:
        print("""
  The learned model does NOT meaningfully beat persistence on this data, and
  that is the result -- not a reason to keep tuning until a better number
  appears. Over a 30 s horizon, "the risk stays roughly where it is" is a
  strong predictor, and the simulated runs are smooth. Say so, keep the
  `trend` backend as the deployed default, and note that the transformer
  needs many hours of real, varied footage before it can be expected to win.""")
    else:
        print(f"""
  The learned model beats persistence by {skill * 100:.1f}%. Deploy it with:
      --forecast-backend transformer""")

    if best_state:
        model.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_dim": len(FEATURE_ORDER),
        "hidden_dim": 64, "num_heads": 4, "num_layers": 2,
        "window": args.window, "horizon_sec": args.horizon_sec, "fps": args.fps,
        "val_mae": best, "baseline_mae": base_mae, "skill": skill,
        "features": FEATURE_ORDER,
        "train_runs": train_names, "val_runs": sorted(val_names),
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
