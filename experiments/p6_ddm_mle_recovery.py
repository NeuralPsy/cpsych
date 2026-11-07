"""P6 Step 3a: MLE recovery validation on synthetic DDM data.

Generates data from known parameters, fits with MLE, checks recovery.
This script was lost and restored from specs/p6_validation.md and
results/p6_ddm_mle_recovery.json on 2026-08-05. See decisions.md D48.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpsych.fitting.ddm_mle import fit_ddm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_POINTS = 50
N_TRIALS = 500
SEED = 42


def generate_ddm_trials(kappa, a, Ter, n_trials, rng):
    """Simulate DDM trials with known parameters."""
    dt = 0.001
    sigma = 1.0
    # Single-condition: v = kappa (coherence absorbed into kappa)
    v = kappa

    rts = []
    corrects = []
    for _ in range(n_trials):
        x = 0.0
        t = 0.0
        while True:
            x += v * dt + sigma * np.sqrt(dt) * rng.standard_normal()
            t += dt
            if x >= a / 2:
                rts.append(t + Ter)
                corrects.append(1.0)
                break
            elif x <= -a / 2:
                rts.append(t + Ter)
                corrects.append(0.0)
                break
            if t > 30.0:
                rts.append(t + Ter)
                corrects.append(float(rng.random() > 0.5))
                break

    return np.array(rts), np.array(corrects)


def main():
    rng = np.random.default_rng(SEED)

    # Random parameter sets
    kappas_true = rng.uniform(0.5, 5.0, N_POINTS)
    a_true = rng.uniform(0.5, 3.0, N_POINTS)
    Ter_true = rng.uniform(0.1, 0.4, N_POINTS)

    kappas_hat = np.zeros(N_POINTS)
    a_hat = np.zeros(N_POINTS)
    Ter_hat = np.zeros(N_POINTS)

    t0 = time.perf_counter()
    for i in range(N_POINTS):
        rt, correct = generate_ddm_trials(
            kappas_true[i], a_true[i], Ter_true[i], N_TRIALS, rng)
        res = fit_ddm(rt, correct, coherence=None,
                      init=(2.0, 1.5, 0.15))
        kappas_hat[i] = res["kappa"]
        a_hat[i] = res["a"]
        Ter_hat[i] = res["Ter"]

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_POINTS}")

    wall = time.perf_counter() - t0

    r_kappa = float(np.corrcoef(kappas_true, kappas_hat)[0, 1])
    r_a = float(np.corrcoef(a_true, a_hat)[0, 1])
    r_Ter = float(np.corrcoef(Ter_true, Ter_hat)[0, 1])

    print(f"\nRecovery (N_TRIALS={N_TRIALS}, N_POINTS={N_POINTS}):")
    print(f"  r(kappa) = {r_kappa:.4f}")
    print(f"  r(a)     = {r_a:.4f}")
    print(f"  r(Ter)   = {r_Ter:.4f}")
    print(f"  Wall time: {wall:.1f}s ({wall/N_POINTS:.2f}s per fit)")

    result = {
        "r_kappa": r_kappa,
        "r_a": r_a,
        "r_Ter": r_Ter,
        "wall_s": wall,
        "per_fit_s": wall / N_POINTS,
        "N_TRIALS": N_TRIALS,
        "N_POINTS": N_POINTS,
    }

    out_path = os.path.join(RESULTS_DIR, "p6_ddm_mle_recovery.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
