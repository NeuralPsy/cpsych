"""P20: does a wrongly fixed affect term bias the other estimates? Ground truth.

Why this exists. In the participants, the published structure's learning rate
for rewarded outcomes had a median of 0.72 with the affect coupling off and
0.31 with it at its default (P16). P19 showed that this parameter is poorly
recovered one participant at a time, so the shift in the participants cannot
by itself say whether the fixed term biases the estimate or the estimate is
merely unstable. With synthetic agents the answer is known.

What it does. It takes the synthetic data sets of P19, which were generated
with the affect coupling OFF from known values, and fits each structure to its
own data with the coupling ON at the default (lambda = 0.49), using the same
likelihood, bounds, optimiser and number of starts. The fits with the coupling
off are read from P19, so both fits see identical data.

SELF-CHECK. The data are regenerated from P19's seeds. Before anything is
fitted, the BIC of P19's stored estimates is recomputed on the regenerated
data and must equal the BIC stored by P19, which proves the data are the same.

READINGS, fixed before this script was first run on the full sample:
  1. The fixed term "biases" a parameter if its median estimate with the
     coupling on differs from its median estimate with the coupling off by
     more than 0.05 for a learning rate, or by more than 20 per cent for
     inverse temperature and persistence.
  2. The direction is "the same as in the participants" for the published
     structure's rewarded-outcome rate if that median is lower with the
     coupling on.
  3. Likelihood "detects" the wrong fixed term if the summed negative
     log-likelihood is lower with the coupling off for a structure in every
     replication.

UNREGISTERED addition, made after the second round of independent review.

Rule 1: every number lands in results/p20_fixed_term_bias.json.

Usage:  python experiments/p20_fixed_term_bias.py [--workers N] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import p10_model_comparison as p10              # noqa: E402
import p13_bounds_sensitivity as p13            # noqa: E402
import p16_split_and_affect_sensitivity as p16  # noqa: E402
import p19_recovery_ml as p19                   # noqa: E402

RESULTS_DIR = os.path.join(HERE, "..", "results")
LAM_ON = float(p10.LAM)
READ = {"alpha_shift": 0.05, "relative_shift_other": 0.20}
MODELS = list(p10.ALL_MODELS)


def fit_on(task):
    key, choice, reward, model, seed = task
    rng = np.random.default_rng(seed)
    bnds = p13.bounds_new(model)
    best = None
    for x0 in p10.starts(model, rng, n=p19.N_STARTS):
        x0 = [min(max(v, lo), hi) for v, (lo, hi) in zip(x0, bnds)]
        try:
            res = minimize(p16.nll, x0, args=(choice, reward, model, LAM_ON, p19.SPLIT),
                           method="L-BFGS-B", bounds=bnds)
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return key, None, float("nan")
    return key, [float(v) for v in best.x], float(best.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true",
                    help="reads the smoke file of P19: checks that the script runs")
    args = ap.parse_args()

    src = "p19_recovery_ml_smoke.json" if args.smoke else "p19_recovery_ml.json"
    with open(os.path.join(RESULTS_DIR, src)) as f:
        j19 = json.load(f)
    assert j19["smoke_run"] == bool(args.smoke) and j19["lambda"] == 0.0
    n_agents, reps = j19["n_agents"], j19["n_replications"]
    with open(os.path.join(RESULTS_DIR,
                           "p16_split_and_affect_sensitivity.json")) as f:
        p16j = json.load(f)
    n_trials = [p16j["per_subject"]["M1_lam0"][i]["n_trials"] for i in range(n_agents)]
    names = {m: list(p10.PARAM_NAMES[m]) for m in MODELS}
    print(f"agents {n_agents}, replications {reps}, workers {args.workers}, "
          f"coupling fitted at {LAM_ON}")

    # ---- regenerate P19's data and prove they are the same
    data, worst = {}, 0.0
    for gi, g in enumerate(MODELS):
        truth = np.array(j19["truth"][g])
        for rep in range(reps):
            ch, rw, _ = p19.simulate(truth, g, n_trials, p19.SEED0 + 100 * gi + rep)
            data[(g, rep)] = (ch, rw)
            st = j19["per_fit"][f"{g}|{rep}|{g}"]
            for b in range(n_agents):
                v = p16.nll(st["params"][b], ch[b], rw[b], g, 0.0, p19.SPLIT)
                bic = p10.N_PARAMS[g] * np.log(ch[b].size) + 2.0 * v
                worst = max(worst, abs(bic - st["bic"][b]))
    print(f"self-check, regenerated data against P19: {worst:.2e}")
    assert worst < 1e-6, "the regenerated data differ from P19's"

    tasks = []
    for gi, g in enumerate(MODELS):
        for rep in range(reps):
            ch, rw = data[(g, rep)]
            for b in range(n_agents):
                seed = 20000 + 1_000_000 * gi + 10_000 * rep + b
                tasks.append(((g, rep, b), ch[b], rw[b], g, seed))
    print(f"fits to run: {len(tasks)}")
    t0 = time.time()
    out = {}
    with Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(fit_on, tasks, chunksize=8)):
            out[res[0]] = res[1:]
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(tasks)}  {el/60:.1f} min elapsed, "
                      f"about {el/(k+1)*(len(tasks)-k-1)/60:.0f} min left")
    n_failed = sum(1 for v in out.values() if v[0] is None)
    print(f"done in {(time.time()-t0)/60:.1f} min; failed fits: {n_failed}")

    models = {}
    print("\nMedians: true / fitted with the coupling off / fitted with it on")
    for g in MODELS:
        truth = np.vstack([np.array(j19["truth"][g])] * reps)
        off = np.vstack([np.array(j19["per_fit"][f"{g}|{rep}|{g}"]["params"])
                         for rep in range(reps)])
        on = np.vstack([np.array([out[(g, rep, b)][0] for b in range(n_agents)])
                        for rep in range(reps)])
        nll_off, nll_on, off_better = [], [], []
        for rep in range(reps):
            ch, rw = data[(g, rep)]
            st = j19["per_fit"][f"{g}|{rep}|{g}"]
            a = np.array([p16.nll(st["params"][b], ch[b], rw[b], g, 0.0, p19.SPLIT)
                          for b in range(n_agents)])
            c = np.array([out[(g, rep, b)][1] for b in range(n_agents)])
            nll_off.append(float(a.sum())); nll_on.append(float(c.sum()))
            off_better.append(int(np.sum(a < c)))
        e = {"nll_sum_off_per_replication": nll_off,
             "nll_sum_on_per_replication": nll_on,
             "n_agents_off_better_per_replication": off_better,
             "likelihood_detects_wrong_term": bool(all(a < c for a, c in
                                                       zip(nll_off, nll_on))),
             "parameters": {}}
        for jx, nm in enumerate(names[g]):
            t_med, off_med, on_med = (float(np.median(v[:, jx])) for v in (truth, off, on))
            shift = on_med - off_med
            if nm.startswith("alpha"):
                biased = abs(shift) > READ["alpha_shift"]
            else:
                biased = abs(shift) > READ["relative_shift_other"] * max(abs(off_med), 1e-9)
            e["parameters"][nm] = {
                "true_median": t_med, "median_coupling_off": off_med,
                "median_coupling_on": on_med, "shift_on_minus_off": float(shift),
                "biased_by_fixed_term": bool(biased),
                "n_lower_with_coupling_on": int(np.sum(on[:, jx] < off[:, jx] - 1e-9)),
                "n": int(on.shape[0]),
                "pearson_r_true_vs_on": p19.corr(truth[:, jx], on[:, jx]),
                "pearson_r_true_vs_off": p19.corr(truth[:, jx], off[:, jx])}
            if nm == "persistence":
                e["parameters"][nm]["share_positive_on"] = float(np.mean(on[:, jx] > 0))
                e["parameters"][nm]["share_positive_off"] = float(np.mean(off[:, jx] > 0))
            print(f"  {g} {nm:12s} {t_med:7.3f} / {off_med:7.3f} / {on_med:7.3f}"
                  f"   {'BIASED' if biased else 'unchanged'}")
        print(f"     summed NLL off {nll_off}  on {nll_on}")
        models[g] = e

    m5 = models["M5"]["parameters"]["alpha_pos"]
    reading = {
        "biased_parameters": [f"{g}.{nm}" for g in MODELS for nm in names[g]
                              if models[g]["parameters"][nm]["biased_by_fixed_term"]],
        "unchanged_parameters": [f"{g}.{nm}" for g in MODELS for nm in names[g]
                                 if not models[g]["parameters"][nm]["biased_by_fixed_term"]],
        "M5_alpha_pos_shift": m5["shift_on_minus_off"],
        "M5_alpha_pos_same_direction_as_participants": bool(m5["shift_on_minus_off"] < 0),
        "likelihood_detects_wrong_term_for": [g for g in MODELS
                                              if models[g]["likelihood_detects_wrong_term"]],
    }
    print()
    for k, v in reading.items():
        print(f"{k}: {v}")

    result = {"procedure": "structures fitted with the affect coupling at its default "
                           "to synthetic data generated with the coupling off (P19 data)",
              "registered": False, "smoke_run": bool(args.smoke),
              "lambda_generating": 0.0, "lambda_fitted": LAM_ON,
              "n_agents": n_agents, "n_replications": reps, "n_starts": p19.N_STARTS,
              "reading_thresholds": READ,
              "self_check_max_abs_diff": float(worst),
              "n_fits": len(tasks), "n_failed_fits": int(n_failed),
              "models": models, "reading": reading,
              "per_fit_on": {f"{g}|{rep}": [out[(g, rep, b)][0] for b in range(n_agents)]
                             for g in MODELS for rep in range(reps)}}
    name = "p20_fixed_term_bias_smoke.json" if args.smoke else "p20_fixed_term_bias.json"
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
