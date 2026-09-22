#!/usr/bin/env python
"""
tune.py — search alpha/beta, sub-signal weights, and the gate on the DEV split.

The architecture requires every tunable value to be searched on development data
and its final setting reported; the TEST split stays untouched until run_all.py.
The objective below (faithfulness - hallucination_rate on dev) is a simple,
transparent composite — adjust it to your reporting target. Results are dev-only
and must be labelled as such.

Usage: python tune.py --config config.yaml --trials 50
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import pipeline as pl           # noqa: E402
from run_all import build_everything, load_cfg   # noqa: E402


def objective_factory(pipe, sf, cfg):
    dev, qrels = sf.dev_claims, sf.dev_qrels

    def objective(trial):
        import optuna  # noqa: F401
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        # weights on the simplex: sample 3 positives, normalise
        raw = [trial.suggest_float(f"w{i}", 0.01, 1.0) for i in (1, 2, 3)]
        s = sum(raw)
        w1, w2, w3 = (r / s for r in raw)
        tau_f = trial.suggest_float("tau_f", 0.3, 0.7)
        lam = trial.suggest_float("lam", 0.0, 0.5)
        gate = trial.suggest_categorical("gate", ["soft", "hard", "off"])

        pipe.cfg.update({"alpha": alpha, "beta": 1 - alpha, "tau_f": tau_f,
                         "lam": lam, "gate": gate})
        pipe.faith.w = {"w1": w1, "w2": w2, "w3": w3, "w4": 0.0}

        e = pl.evaluate(pipe, dev, qrels, "S3", cfg["faith_entail_threshold"])
        faith = 0.0 if np.isnan(e["faithfulness"]) else e["faithfulness"]
        halluc = 0.0 if np.isnan(e["hallucination_rate"]) else e["hallucination_rate"]
        return faith - halluc          # maximise

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    import optuna
    cfg = load_cfg(args.config)
    sf, pipe = build_everything(cfg)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_factory(pipe, sf, cfg), n_trials=args.trials)

    print("[tune] best dev objective:", study.best_value)
    print("[tune] best params (DEV-selected):", json.dumps(study.best_params, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/tuned_params.json", "w") as f:
        json.dump({"best_value_dev": study.best_value, "params": study.best_params}, f, indent=2)
    print("[tune] wrote results/tuned_params.json — copy these into config.yaml, "
          "then run run_all.py on TEST.")


if __name__ == "__main__":
    main()
