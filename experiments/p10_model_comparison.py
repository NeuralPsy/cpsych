"""P10: model comparison on the corrected sample, with the published model added.

Supersedes P8. Two changes, both responses to specific objections:

1. SAMPLE. P8 selected participants with PreciseYrs >= 18 from the public
   demographics file (322 rows), giving 117 and then 114 with a usable choice
   file. The source study (Eckstein et al. 2022, eLife 11:e75474, and the
   original task report, Eckstein et al. 2022, Dev Cogn Neurosci 55:101106)
   states: "Two participants from the undergraduate sample were excluded
   because they were older than 30". Those two are in the archive and were in
   P8's 114 (ID 425, 31.0 years; ID 433, 50.0 years). This script applies the
   source's own window, 18 <= age <= 30, which gives the published adult
   sample for this task: 57 undergraduates (18-28) plus 55 community adults
   (25-30) = 112.

2. MODEL SPACE. P8 compared four models and claimed to span the published
   structure. It did not. The published winning model for this task has FOUR
   free parameters -- alpha+, alpha-, beta, perseverance -- and in addition
   performs counterfactual updating of the UNCHOSEN option, at rates tied to
   the factual rates (alpha+ = counterfactual alpha+, alpha- = counterfactual
   alpha-). P8 had no counterfactual updating in any model, so the published
   structure was never fitted. M5 here is that model. M6 isolates
   counterfactual updating on its own.

Models:
  M1  alpha, beta                               (base, as calibrated in P2.4)
  M2  alpha, beta, persistence
  M3  alpha_pos, alpha_neg, beta
  M4  alpha_pos, alpha_neg, beta, persistence
  M5  alpha_pos, alpha_neg, beta, persistence + counterfactual  [PUBLISHED]
  M6  alpha, beta + counterfactual

M1-M4 are the registered set. M5 and M6 are an UNREGISTERED extension, added
after the fact; the registered criterion below is evaluated on M1-M4 only and
the extension is reported separately and labelled as such.

Valence split: M3 and M4 split on the sign of the prediction error, as in P8.
M5 and M6 split on the sign of the OUTCOME, as the published model does
("applied to only positive and only negative outcomes"). With V initialised at
0.5 and binary reward the two rules almost always agree; M5 follows the source.

Counterfactual rule (from the source): after outcome r on the chosen action,
the unchosen action is updated toward the inverse outcome (1 - r), at the rate
for the valence of that inverse outcome. Reward sensitivity rho_eff multiplies
the counterfactual outcome as it does the factual one. Mood is driven by the
factual prediction error only.

Dynamics otherwise replicate cpsych.modules.rl_mood.RLMoodModule exactly:
  V init 0.5, m init 0, gamma = exp(-1/tau_m)
  choice_prob = softmax(beta*V0, beta*V1)[1]
  rho_eff = 1 + lam*m
  delta   = rho_eff*r - V_chosen
  V[chosen] += alpha*delta
  m = gamma*(m + delta)
tau_m and lam are held at the calibrated values (INTERNAL exposure in spec).

Persistence here is added to the choice logit, so it is on the scale of
beta * P in the published parameterisation, not of P itself. The two are not
numerically comparable and no comparison is made.

PRE-REGISTERED CRITERION (fixed before P8 was run; reproduced verbatim, and
evaluated here on M1-M4 only):
  Reading 1 ("parameter moved, model adequate"): the free-fit median alpha
    falls in [0.60, 0.95] AND the winning model is M1.
  Reading 2 ("model misspecified"): the free-fit median alpha falls outside
    [0.60, 0.95] OR a model other than M1 wins by dBIC > 10.
  Any other combination is recorded as INDETERMINATE and reported as such.

Rule 1: every number lands in results/p10_model_comparison.json. Nothing is
transcribed by hand.

Usage:  python experiments/p10_model_comparison.py [--smoke N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eckstein2022")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Calibrated values held fixed (P2.4). Spec exposure: INTERNAL.
TAU_M, LAM = 2.163, 0.49
GAMMA = float(np.exp(-1.0 / TAU_M))

# Age window of the source study's adult sample.
AGE_MIN, AGE_MAX = 18.0, 30.0

# Reference values the fit is compared against.
# Published values for this task (eLife 11:e75474, Figure 3A text):
#   alpha+ mean 0.77 (sd 0.11), alpha- mean 0.62 (sd 0.14), 1/beta mean 0.33
#   (sd 0.15). Note the published quantity is 1/beta, not beta.
ALPHA_OURS = 0.15        # calibrated on the probabilistic protocol
BETA_OURS = 2.50
ALPHA_POS_PUBLISHED = 0.77
ALPHA_NEG_PUBLISHED = 0.62
INV_BETA_PUBLISHED = 0.33

# Spec bound on alpha in cpsych.modules.rl_mood (Param("alpha", .., 0.01, 0.50)).
ALPHA_SPEC_MAX = 0.50
# The free fit is deliberately allowed past the spec bound, so that the
# question "where does the data put alpha" is not answered by the bound.
ALPHA_FIT_BOUNDS = (0.01, 0.99)
BETA_FIT_BOUNDS = (0.10, 20.0)
PERSIST_BOUNDS = (-2.0, 2.0)

PREREG = {"alpha_lo": 0.60, "alpha_hi": 0.95, "dbic_decisive": 10.0,
          "evaluated_on": ["M1", "M2", "M3", "M4"]}

REGISTERED_MODELS = ["M1", "M2", "M3", "M4"]
EXTENSION_MODELS = ["M5", "M6"]
ALL_MODELS = REGISTERED_MODELS + EXTENSION_MODELS


# --------------------------------------------------------------------------
# data

def load_adults(limit=None):
    demo = pd.read_csv(os.path.join(DATA_DIR,
                                    "EcksteinEtAl_eLife2022_Demographics.csv"))
    age = demo["PreciseYrs"]
    keep = (age >= AGE_MIN) & (age <= AGE_MAX)
    ids = demo[keep]["ID"].values
    excluded_over_max = demo[age > AGE_MAX][["ID", "PreciseYrs"]]
    subjects = []
    for sid in ids:
        path = os.path.join(DATA_DIR, f"PS_{sid}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if len(df) < 50:
            continue
        ch = pd.to_numeric(df["selected_box"], errors="coerce").values
        rw = pd.to_numeric(df["reward"], errors="coerce").values
        ok = ~(np.isnan(ch) | np.isnan(rw))
        ch, rw = ch[ok].astype(int), rw[ok].astype(float)
        if ch.size < 50:
            continue
        u = np.unique(ch)
        if not np.all(np.isin(u, [0, 1])):      # recode if not already 0/1
            ch = (ch == u.max()).astype(int)
        subjects.append({"id": int(sid), "choice": ch, "reward": rw})
        if limit and len(subjects) >= limit:
            break
    prov = {
        "demographics_rows": int(len(demo)),
        "age_window": [AGE_MIN, AGE_MAX],
        "n_in_age_window": int(keep.sum()),
        "n_with_usable_file": len(subjects),
        "min_trials_required": 50,
        "excluded_above_age_max": [
            {"id": int(r.ID), "age": float(r.PreciseYrs)}
            for r in excluded_over_max.itertuples()],
        "trial_counts": {
            "min": int(min(s["choice"].size for s in subjects)),
            "max": int(max(s["choice"].size for s in subjects)),
            "median": float(np.median([s["choice"].size for s in subjects])),
        } if subjects else None,
    }
    return subjects, prov


# --------------------------------------------------------------------------
# likelihood

def _unpack(params, model):
    """-> (a_pos, a_neg, beta, persist, counterfactual, split_on_outcome)"""
    if model == "M1":
        return params[0], params[0], params[1], 0.0, False, False
    if model == "M2":
        return params[0], params[0], params[1], params[2], False, False
    if model == "M3":
        return params[0], params[1], params[2], 0.0, False, False
    if model == "M4":
        return params[0], params[1], params[2], params[3], False, False
    if model == "M5":
        return params[0], params[1], params[2], params[3], True, True
    if model == "M6":
        return params[0], params[0], params[1], 0.0, True, True
    raise ValueError(model)


def neg_loglik(params, choice, reward, model):
    """Negative log-likelihood of an observed choice sequence."""
    a_pos, a_neg, beta, persist, counterfactual, split_on_outcome = _unpack(
        params, model)

    V = np.array([0.5, 0.5])
    m = 0.0
    prev = -1
    ll = 0.0
    for t in range(choice.size):
        logit = beta * (V[1] - V[0])
        if persist and prev >= 0:
            logit += persist * (2.0 * prev - 1.0)
        logit = np.clip(logit, -50.0, 50.0)
        p1 = 1.0 / (1.0 + np.exp(-logit))
        p = p1 if choice[t] == 1 else 1.0 - p1
        ll += np.log(max(p, 1e-12))

        r = reward[t]
        c = choice[t]
        rho_eff = 1.0 + LAM * m
        delta = rho_eff * r - V[c]
        if split_on_outcome:
            rate = a_pos if r > 0.5 else a_neg
        else:
            rate = a_pos if delta >= 0 else a_neg
        V[c] += rate * delta

        if counterfactual:
            u = 1 - c
            r_cf = 1.0 - r
            delta_cf = rho_eff * r_cf - V[u]
            rate_cf = a_pos if r_cf > 0.5 else a_neg
            V[u] += rate_cf * delta_cf

        m = GAMMA * (m + delta)   # mood driven by the factual error only
        prev = c
    return -ll


N_PARAMS = {"M1": 2, "M2": 3, "M3": 3, "M4": 4, "M5": 4, "M6": 2}
PARAM_NAMES = {
    "M1": ["alpha", "beta"],
    "M2": ["alpha", "beta", "persistence"],
    "M3": ["alpha_pos", "alpha_neg", "beta"],
    "M4": ["alpha_pos", "alpha_neg", "beta", "persistence"],
    "M5": ["alpha_pos", "alpha_neg", "beta", "persistence"],
    "M6": ["alpha", "beta"],
}


def starts(model, rng, n=8):
    out = []
    for _ in range(n):
        a1 = rng.uniform(0.05, 0.85)
        a2 = rng.uniform(0.05, 0.85)
        b = rng.uniform(0.5, 8.0)
        p = rng.uniform(-0.5, 0.8)
        out.append({"M1": [a1, b], "M2": [a1, b, p],
                    "M3": [a1, a2, b], "M4": [a1, a2, b, p],
                    "M5": [a1, a2, b, p], "M6": [a1, b]}[model])
    return out


def bounds_for(model):
    A, B, P = ALPHA_FIT_BOUNDS, BETA_FIT_BOUNDS, PERSIST_BOUNDS
    return {"M1": [A, B], "M2": [A, B, P],
            "M3": [A, A, B], "M4": [A, A, B, P],
            "M5": [A, A, B, P], "M6": [A, B]}[model]


def fit_subject(subj, model, rng):
    best = None
    for x0 in starts(model, rng):
        try:
            res = minimize(neg_loglik, x0,
                           args=(subj["choice"], subj["reward"], model),
                           method="L-BFGS-B", bounds=bounds_for(model))
        except Exception:
            continue
        if res.success or np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        return None
    n = subj["choice"].size
    k = N_PARAMS[model]
    return {"params": [float(v) for v in best.x],
            "nll": float(best.fun),
            "n_trials": int(n),
            "bic": float(k * np.log(n) + 2.0 * best.fun)}


# --------------------------------------------------------------------------

def winner_counts(per_model, models):
    """Per-subject best model among `models`."""
    freq = {k: 0 for k in models}
    ids = set()
    for k in models:
        ids |= {f["id"] for f in per_model[k]["per_subject"]}
    for sid in sorted(ids):
        bic_by = {}
        for k in models:
            hit = [f for f in per_model[k]["per_subject"] if f["id"] == sid]
            if hit:
                bic_by[k] = hit[0]["bic"]
        if bic_by:
            freq[min(bic_by, key=bic_by.get)] += 1
    return freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="fit only N subjects, for a run-check (not a result)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    subjects, prov = load_adults(limit=args.smoke or None)
    print(f"Demographics rows: {prov['demographics_rows']}")
    print(f"In age window {AGE_MIN}-{AGE_MAX}: {prov['n_in_age_window']}")
    print(f"Excluded above {AGE_MAX}: {prov['excluded_above_age_max']}")
    print(f"Subjects fitted: {len(subjects)}"
          + ("  [SMOKE RUN - not a result]" if args.smoke else ""))
    if prov["trial_counts"]:
        print(f"Trials per subject: {prov['trial_counts']}")
    print()

    per_model = {}
    for model in ALL_MODELS:
        fits = []
        for i, s in enumerate(subjects):
            f = fit_subject(s, model, rng)
            if f:
                f["id"] = s["id"]
                fits.append(f)
            if (i + 1) % 20 == 0:
                print(f"  {model}: {i+1}/{len(subjects)}")
        if not fits:
            continue
        arr = np.array([f["params"] for f in fits])
        per_model[model] = {
            "n_subjects": len(fits),
            "n_params": N_PARAMS[model],
            "param_names": PARAM_NAMES[model],
            "registered": model in REGISTERED_MODELS,
            "bic_sum": float(sum(f["bic"] for f in fits)),
            "nll_sum": float(sum(f["nll"] for f in fits)),
            "param_median": [float(v) for v in np.median(arr, axis=0)],
            "param_iqr": [[float(np.percentile(arr[:, j], 25)),
                           float(np.percentile(arr[:, j], 75))]
                          for j in range(arr.shape[1])],
            "per_subject": fits,
        }
        print(f"  {model}: BIC={per_model[model]['bic_sum']:.1f} "
              f"median {PARAM_NAMES[model]}="
              f"{[round(v,3) for v in per_model[model]['param_median']]}")

    # --- winner within the registered set, and overall
    reg = [k for k in REGISTERED_MODELS if k in per_model]
    winner_reg = min(reg, key=lambda k: per_model[k]["bic_sum"])
    best_reg = per_model[winner_reg]["bic_sum"]
    for k in reg:
        per_model[k]["dbic_within_registered"] = float(
            per_model[k]["bic_sum"] - best_reg)

    winner_all = min(per_model, key=lambda k: per_model[k]["bic_sum"])
    best_all = per_model[winner_all]["bic_sum"]
    for k in per_model:
        per_model[k]["dbic"] = float(per_model[k]["bic_sum"] - best_all)

    freq_reg = winner_counts(per_model, reg)
    freq_all = winner_counts(per_model, list(per_model))

    # --- free-fit alpha from the base model
    m1 = per_model.get("M1")
    alpha_med = m1["param_median"][0] if m1 else float("nan")
    alpha_iqr = m1["param_iqr"][0] if m1 else [float("nan")] * 2
    alphas = (np.array([f["params"][0] for f in m1["per_subject"]])
              if m1 else np.array([]))
    beta_med = m1["param_median"][1] if m1 else float("nan")
    beta_iqr = m1["param_iqr"][1] if m1 else [float("nan")] * 2

    # --- registered verdict, on the registered models only
    in_window = PREREG["alpha_lo"] <= alpha_med <= PREREG["alpha_hi"]
    m1_wins = winner_reg == "M1"
    other_decisive = (not m1_wins) and (
        per_model["M1"]["dbic_within_registered"] > PREREG["dbic_decisive"])
    if in_window and m1_wins:
        verdict = "READING_1_parameter_moved"
    elif (not in_window) or other_decisive:
        verdict = "READING_2_model_misspecified"
    else:
        verdict = "INDETERMINATE"

    result = {
        "supersedes": "p8_model_comparison.json",
        "procedure": "per-subject maximum-likelihood fit on observed choice "
                     "sequences; 8 random starts, L-BFGS-B; BIC = k*ln(n) + 2*nll",
        "sample_provenance": prov,
        "smoke_run": bool(args.smoke),
        "seed": args.seed,
        "fixed": {"tau_m": TAU_M, "lam": LAM, "gamma": GAMMA},
        "reference": {"alpha_ours": ALPHA_OURS, "beta_ours": BETA_OURS,
                      "alpha_pos_published": ALPHA_POS_PUBLISHED,
                      "alpha_neg_published": ALPHA_NEG_PUBLISHED,
                      "inv_beta_published": INV_BETA_PUBLISHED,
                      "alpha_spec_max": ALPHA_SPEC_MAX},
        "preregistered_criterion": PREREG,
        "registered_models": REGISTERED_MODELS,
        "extension_models": EXTENSION_MODELS,
        "extension_note": "M5 is the published winning model for this task "
                          "(alpha+, alpha-, beta, perseverance, with "
                          "counterfactual updating of the unchosen option at "
                          "tied rates). M6 isolates counterfactual updating. "
                          "Neither was in the registered set; both are "
                          "reported as an unregistered extension.",
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "per_subject"}
                   for k, v in per_model.items()},
        "winner_within_registered": winner_reg,
        "winner_overall": winner_all,
        "subject_level_best_model_counts_registered": freq_reg,
        "subject_level_best_model_counts_all": freq_all,
        "free_fit_M1": {
            "alpha_median": alpha_med,
            "alpha_iqr": alpha_iqr,
            "beta_median": beta_med,
            "beta_iqr": beta_iqr,
            "frac_above_spec_max": (float(np.mean(alphas > ALPHA_SPEC_MAX))
                                    if alphas.size else None),
            "n_above_spec_max": (int(np.sum(alphas > ALPHA_SPEC_MAX))
                                 if alphas.size else None),
            "n_at_upper_bound": (int(np.sum(alphas >= ALPHA_FIT_BOUNDS[1]))
                                 if alphas.size else None),
            "frac_in_prereg_window": (float(np.mean(
                (alphas >= PREREG["alpha_lo"]) & (alphas <= PREREG["alpha_hi"])))
                if alphas.size else None),
        },
        "verdict_registered": verdict,
        "per_subject": {k: v["per_subject"] for k, v in per_model.items()},
    }

    print()
    print(f"Winner within registered set (M1-M4): {winner_reg}")
    print(f"Winner overall (M1-M6):               {winner_all}")
    print(f"Per-subject winners, registered: {freq_reg}")
    print(f"Per-subject winners, all:        {freq_all}")
    print(f"Free-fit M1: alpha median={alpha_med:.3f} IQR={alpha_iqr}, "
          f"beta median={beta_med:.3f} IQR={beta_iqr}")
    print(f"  ours alpha={ALPHA_OURS} beta={BETA_OURS}; "
          f"published alpha+={ALPHA_POS_PUBLISHED} alpha-={ALPHA_NEG_PUBLISHED} "
          f"1/beta={INV_BETA_PUBLISHED}; spec max alpha={ALPHA_SPEC_MAX}")
    print(f"  above spec max: {result['free_fit_M1']['n_above_spec_max']}"
          f"/{len(subjects)}; at optimiser bound: "
          f"{result['free_fit_M1']['n_at_upper_bound']}")
    print(f"REGISTERED VERDICT (on M1-M4): {verdict}")

    if args.smoke:
        print("\nSmoke run - results NOT written.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "p10_model_comparison.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
