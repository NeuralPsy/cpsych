"""P13: do P10's conclusions survive wider optimiser bounds?

Why this exists. In P10 many fitted values sit on an optimiser bound, and not
only the learning rate the manuscript reports. Under the published structure
(M5) persistence is at +2 for 27 of 112 participants and beta at 20 for 17;
55 of 112 have at least one parameter on a bound. The manuscript itself says
values piling up at a bound are a misspecification signal. So the bounds must
either be shown not to matter, or be reported.

What this does. Refits all six structures of P10 with the same likelihood,
the same sample and the same random starts, changing only two bounds:
    beta          (0.10, 20)   ->  (0.10, 200)
    persistence   (-2, 2)      ->  (-20, 20)
The learning-rate bound stays at (0.01, 0.99): a learning rate has a natural
ceiling at 1, and 0.99 was set on purpose (see P10).

Each participant is also started from their own P10 solution, so the new fit
can never be worse than the old one. Any difference is therefore due to the
bounds and not to optimiser luck.

P10's JSON is NOT overwritten. Results go to a separate file, with every P10
quantity recomputed and set beside the original.

READING, fixed before this script was first run. P10's conclusions are
"robust to the bounds" only if ALL of these hold:
  (a) the overall BIC winner is unchanged, and so is the winner among the
      registered four;
  (b) every median learning rate moves by less than 0.05;
  (c) the share of participants with negative persistence in M2, and with
      positive persistence in M5, each moves by less than 5 percentage points;
  (d) the ordering "beta moves less than some learning rate" holds in every
      structure, as the manuscript claims.
If any fails, the manuscript's numbers must be replaced by these, not
footnoted. Whatever happens, the counts at each bound are reported for both
sets of bounds.

A warning about what wider bounds cannot fix. For a participant who almost
always repeats a rewarded choice, the likelihood is nearly flat in beta above
some value: beta = 20 and beta = 200 predict the same choices. Such a
participant will run to the new bound too. That is not misspecification; it
is a parameter the data bound from below only. The output therefore reports,
for every value on a bound, how much likelihood the wider bound bought
("nll_gain"). A gain near zero means "flat", a large gain means the old bound
was biting.

This is an UNREGISTERED sensitivity analysis.

Rule 1: every number lands in results/p13_bounds_sensitivity.json.

Usage:  python experiments/p13_bounds_sensitivity.py [--smoke N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import p10_model_comparison as p10          # noqa: E402

RESULTS_DIR = os.path.join(HERE, "..", "results")

OLD = {"alpha": p10.ALPHA_FIT_BOUNDS, "beta": p10.BETA_FIT_BOUNDS,
       "persistence": p10.PERSIST_BOUNDS}
NEW = {"alpha": p10.ALPHA_FIT_BOUNDS, "beta": (0.10, 200.0),
       "persistence": (-20.0, 20.0)}

TOL = 1e-3                      # "on the bound" means within this of it
READING = {"alpha_median_shift_max": 0.05, "sign_share_shift_max_pct": 5.0}
ALPHA_CAL, BETA_CAL = 0.15, 2.50


def kind(name):
    return "alpha" if name.startswith("alpha") else name


def bounds_new(model):
    return [NEW[kind(n)] for n in p10.PARAM_NAMES[model]]


def fit_subject(subj, model, rng, warm):
    """P10's fit, wider bounds, plus a warm start from the P10 solution."""
    bnds = bounds_new(model)
    x0s = p10.starts(model, rng) + [list(warm)]
    best = None
    for x0 in x0s:
        x0 = [min(max(v, lo), hi) for v, (lo, hi) in zip(x0, bnds)]
        try:
            res = minimize(p10.neg_loglik, x0,
                           args=(subj["choice"], subj["reward"], model),
                           method="L-BFGS-B", bounds=bnds)
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    # never return something worse than the P10 solution itself
    warm_nll = float(p10.neg_loglik(list(warm), subj["choice"],
                                    subj["reward"], model))
    if best is None or warm_nll < best.fun:
        x, fun = list(warm), warm_nll
    else:
        x, fun = [float(v) for v in best.x], float(best.fun)
    n = subj["choice"].size
    return {"params": x, "nll": fun, "n_trials": int(n),
            "bic": float(p10.N_PARAMS[model] * np.log(n) + 2.0 * fun)}


def at_bound_counts(arr, names, bounds):
    out = {}
    for j, nm in enumerate(names):
        lo, hi = bounds[kind(nm)]
        out[nm] = {"n_at_lower": int(np.sum(arr[:, j] <= lo + TOL)),
                   "n_at_upper": int(np.sum(arr[:, j] >= hi - TOL)),
                   "bounds": [lo, hi]}
    any_hit = np.zeros(arr.shape[0], dtype=bool)
    for j, nm in enumerate(names):
        lo, hi = bounds[kind(nm)]
        any_hit |= (arr[:, j] <= lo + TOL) | (arr[:, j] >= hi - TOL)
    out["_any_parameter"] = int(any_hit.sum())
    return out


def summarise(fits, model, bounds):
    names = p10.PARAM_NAMES[model]
    arr = np.array([f["params"] for f in fits])
    s = {"param_names": names,
         "param_median": [float(v) for v in np.median(arr, axis=0)],
         "param_iqr": [[float(np.percentile(arr[:, j], 25)),
                        float(np.percentile(arr[:, j], 75))]
                       for j in range(arr.shape[1])],
         "bic_sum": float(sum(f["bic"] for f in fits)),
         "nll_sum": float(sum(f["nll"] for f in fits)),
         "at_bound": at_bound_counts(arr, names, bounds)}
    for j, nm in enumerate(names):
        if kind(nm) == "alpha":
            s.setdefault("alpha_above_spec_max", {})[nm] = int(
                np.sum(arr[:, j] > p10.ALPHA_SPEC_MAX))
            s.setdefault("alpha_ratio_to_calibrated", {})[nm] = float(
                np.median(arr[:, j]) / ALPHA_CAL)
        if nm == "beta":
            s["beta_ratio_to_calibrated"] = float(
                np.median(arr[:, j]) / BETA_CAL)
        if nm == "persistence":
            s["persistence_sign"] = {
                "n_negative": int(np.sum(arr[:, j] < 0)),
                "n_positive": int(np.sum(arr[:, j] > 0)),
                "pct_negative": float(100 * np.mean(arr[:, j] < 0)),
                "pct_positive": float(100 * np.mean(arr[:, j] > 0))}
    if model == "M5":
        ip, im = names.index("alpha_pos"), names.index("alpha_neg")
        s["either_alpha_above_spec_max"] = int(np.sum(
            (arr[:, ip] > p10.ALPHA_SPEC_MAX) | (arr[:, im] > p10.ALPHA_SPEC_MAX)))
    return s


def winners(per_subject, models, ids):
    freq = {m: 0 for m in models}
    for i in ids:
        bic = {m: per_subject[m][i]["bic"] for m in models}
        freq[min(bic, key=bic.get)] += 1
    return freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="fit only N subjects, for a run-check (not a result)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    subjects, prov = p10.load_adults(limit=args.smoke or None)
    ids = [s["id"] for s in subjects]
    print(f"Subjects: {len(subjects)}"
          + ("  [SMOKE RUN - not a result]" if args.smoke else ""))

    with open(os.path.join(RESULTS_DIR, "p10_model_comparison.json")) as f:
        old_json = json.load(f)
    if not args.smoke:
        assert old_json["sample_provenance"]["n_with_usable_file"] == len(ids), (
            "P10 JSON and this run disagree on the sample size")

    rng = np.random.default_rng(args.seed)
    models = p10.ALL_MODELS
    new_ps, old_ps, summary = {}, {}, {}
    for model in models:
        old_by_id = {f["id"]: f for f in old_json["per_subject"][model]}
        assert set(ids) <= set(old_by_id), f"{model}: ids missing from P10"
        old_ps[model] = {i: old_by_id[i] for i in ids}
        new_ps[model] = {}
        for k, s in enumerate(subjects):
            f = fit_subject(s, model, rng, old_by_id[s["id"]]["params"])
            f["id"] = s["id"]
            new_ps[model][s["id"]] = f
            if (k + 1) % 20 == 0:
                print(f"  {model}: {k+1}/{len(subjects)}")

        old_fits = [old_ps[model][i] for i in ids]
        new_fits = [new_ps[model][i] for i in ids]
        so = summarise(old_fits, model, OLD)
        sn = summarise(new_fits, model, NEW)

        # how much likelihood the wider bound bought, for those who were on
        # a beta or persistence bound in P10
        names = p10.PARAM_NAMES[model]
        gain = np.array([o["nll"] - n["nll"]
                         for o, n in zip(old_fits, new_fits)])
        oa = np.array([f["params"] for f in old_fits])
        on_old = np.zeros(len(ids), dtype=bool)
        for j, nm in enumerate(names):
            if kind(nm) in ("beta", "persistence"):
                lo, hi = OLD[kind(nm)]
                on_old |= (oa[:, j] <= lo + TOL) | (oa[:, j] >= hi - TOL)
        nll_gain = {
            "n_on_old_beta_or_persistence_bound": int(on_old.sum()),
            "gain_median_those": (float(np.median(gain[on_old]))
                                  if on_old.any() else None),
            "gain_max_those": (float(gain[on_old].max())
                               if on_old.any() else None),
            "n_gain_above_1": int(np.sum(gain > 1.0)),
            "n_gain_above_3": int(np.sum(gain > 3.0)),
            "gain_sum_all": float(gain.sum()),
            "min_gain": float(gain.min()),   # must be >= 0 by construction
        }
        summary[model] = {"registered": model in p10.REGISTERED_MODELS,
                          "p10_bounds": so, "wide_bounds": sn,
                          "nll_gain": nll_gain,
                          "median_shift": [float(a - b) for a, b in zip(
                              sn["param_median"], so["param_median"])]}
        print(f"  {model}: BIC {so['bic_sum']:.1f} -> {sn['bic_sum']:.1f}; "
              f"medians {[round(v, 3) for v in so['param_median']]} -> "
              f"{[round(v, 3) for v in sn['param_median']]}; any-at-bound "
              f"{so['at_bound']['_any_parameter']} -> "
              f"{sn['at_bound']['_any_parameter']}")

    def ranking(key):
        return sorted(models, key=lambda m: summary[m][key]["bic_sum"])

    reg = p10.REGISTERED_MODELS
    comp = {}
    for key, ps in (("p10_bounds", old_ps), ("wide_bounds", new_ps)):
        order = ranking(key)
        best = summary[order[0]][key]["bic_sum"]
        reg_order = [m for m in order if m in reg]
        comp[key] = {
            "bic_ranking": order,
            "dbic": {m: float(summary[m][key]["bic_sum"] - best) for m in order},
            "winner_overall": order[0],
            "winner_registered": reg_order[0],
            "dbic_M1_within_registered": float(
                summary["M1"][key]["bic_sum"]
                - summary[reg_order[0]][key]["bic_sum"]),
            "subject_winners_all": winners(ps, models, ids),
            "subject_winners_registered": winners(ps, reg, ids),
            "M5_better_than_M2": int(sum(
                ps["M5"][i]["bic"] < ps["M2"][i]["bic"] for i in ids)),
            "M5_better_than_M2_by_10": int(sum(
                ps["M2"][i]["bic"] - ps["M5"][i]["bic"] > 10 for i in ids)),
            "M2_better_than_M5_by_10": int(sum(
                ps["M5"][i]["bic"] - ps["M2"][i]["bic"] > 10 for i in ids)),
        }

    # --- the pre-specified reading
    a_same_winner = (comp["p10_bounds"]["winner_overall"]
                     == comp["wide_bounds"]["winner_overall"]
                     and comp["p10_bounds"]["winner_registered"]
                     == comp["wide_bounds"]["winner_registered"])
    alpha_shifts = {}
    for m in models:
        for j, nm in enumerate(p10.PARAM_NAMES[m]):
            if kind(nm) == "alpha":
                alpha_shifts[f"{m}.{nm}"] = abs(summary[m]["median_shift"][j])
    b_alpha = max(alpha_shifts.values()) < READING["alpha_median_shift_max"]
    sign_shift = {
        "M2_pct_negative": abs(
            summary["M2"]["wide_bounds"]["persistence_sign"]["pct_negative"]
            - summary["M2"]["p10_bounds"]["persistence_sign"]["pct_negative"]),
        "M5_pct_positive": abs(
            summary["M5"]["wide_bounds"]["persistence_sign"]["pct_positive"]
            - summary["M5"]["p10_bounds"]["persistence_sign"]["pct_positive"]),
    }
    c_sign = max(sign_shift.values()) < READING["sign_share_shift_max_pct"]
    ordering = {}
    for m in models:
        s = summary[m]["wide_bounds"]
        ordering[m] = bool(max(s["alpha_ratio_to_calibrated"].values())
                           > s["beta_ratio_to_calibrated"])
    d_order = all(ordering.values())
    robust = bool(a_same_winner and b_alpha and c_sign and d_order)

    result = {
        "procedure": "P10 refitted with wider bounds on beta and persistence; "
                     "same likelihood, sample and random starts, plus a warm "
                     "start from each participant's P10 solution",
        "registered": False,
        "smoke_run": bool(args.smoke),
        "seed": args.seed,
        "n_subjects": len(ids),
        "sample_provenance": prov,
        "bounds": {"p10": {k: list(v) for k, v in OLD.items()},
                   "wide": {k: list(v) for k, v in NEW.items()}},
        "on_bound_tolerance": TOL,
        "models": summary,
        "comparison": comp,
        "reading": {
            "thresholds": READING,
            "a_winners_unchanged": bool(a_same_winner),
            "b_alpha_medians_stable": bool(b_alpha),
            "b_alpha_median_abs_shift": alpha_shifts,
            "c_persistence_sign_stable": bool(c_sign),
            "c_sign_share_abs_shift_pct": sign_shift,
            "d_beta_moves_less_than_some_alpha": ordering,
            "d_holds_everywhere": bool(d_order),
            "robust_to_bounds": robust,
        },
        "per_subject_wide": {m: [new_ps[m][i] for i in ids] for m in models},
    }

    print()
    for key in ("p10_bounds", "wide_bounds"):
        c = comp[key]
        print(f"{key:11s}: winner {c['winner_overall']} "
              f"(registered: {c['winner_registered']}); "
              f"ranking {c['bic_ranking']}")
    print(f"(a) winners unchanged:            {a_same_winner}")
    print(f"(b) alpha medians stable:         {b_alpha}  "
          f"max shift {max(alpha_shifts.values()):.3f}")
    print(f"(c) persistence sign stable:      {c_sign}  {sign_shift}")
    print(f"(d) beta moves less than an alpha: {d_order}")
    print(f"ROBUST TO BOUNDS: {robust}")

    if args.smoke:
        print("\nSmoke run - results NOT written.")
        return
    path = os.path.join(RESULTS_DIR, "p13_bounds_sensitivity.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
