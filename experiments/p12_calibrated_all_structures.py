"""P12: do the calibrated values fail under every model structure?

Why this exists. The manuscript says the calibrated values (alpha = 0.15,
beta = 2.50) failed "on every structure". P11 only simulated them in the base
module (condition A). This script tests the claim for all six structures of
P10, in two ways: by likelihood on the observed choice sequences, and by
simulated behaviour.

What "calibrated values under a structure" means. The calibration fixed two
numbers, alpha and beta. It says nothing about persistence, and it has one
learning rate, not two. So under each structure:
  - every learning rate is clamped to 0.15 (alpha+ = alpha- = 0.15),
  - beta is clamped to 2.50,
  - persistence, where the structure has it, is the ONLY free parameter and is
    fitted per participant by maximum likelihood, within P10's bounds.
This gives each structure its best chance to rescue the calibrated values
with its own extra mechanism.

Consequence, stated up front so nobody finds it later: with both learning
rates equal, the valence split does nothing. So
  M3 is identical to M1, and M4 is identical to M2.
Six structures give FOUR distinct conditions: M1, M2, M5, M6. The output
records this explicitly. The manuscript should say "four distinct conditions
spanning the six structures", not "six".

Outputs, per structure:
  nll_calibrated   summed negative log-likelihood at the clamped values
  nll_free         the same from P10's free fit (read from its JSON)
  bic_calibrated   with k = number of free parameters here (0 or 1)
  dbic             bic_calibrated - bic_free   (> 0: the free fit is better
                   even after paying for its extra parameters)
  n_free_better    participants whose own BIC prefers the free fit
  behaviour        simulated win-stay, lose-shift, accuracy, and how many of
                   the three fall inside the observed interquartile range

CRITERION (fixed before this script was first run; same as P11):
  a structure "reproduces behaviour at the calibrated values" only if all
  three observables fall inside the observed interquartile ranges.
  "The calibrated values fail on every structure" is supported only if no
  structure meets that, AND dbic > 10 for every structure.
  Anything else is reported as it falls.

CONTROL: the M1 simulation uses P11's seed and must reproduce P11's
condition A exactly. If it does not, nothing else here is trustworthy.

This is an UNREGISTERED addition, made after the manuscript was drafted, to
check a sentence in it.

Rule 1: every number lands in results/p12_calibrated_all_structures.json.

Usage:  python experiments/p12_calibrated_all_structures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import p10_model_comparison as p10          # noqa: E402  likelihood, loader
import p11_free_fit_behaviour as p11        # noqa: E402  simulation, data

RESULTS_DIR = os.path.join(HERE, "..", "results")

ALPHA_CAL, BETA_CAL = 0.15, 2.50
DBIC_DECISIVE = 10.0

# structure -> (has persistence, counterfactual, identical-to)
STRUCTURES = {
    "M1": {"persist": False, "cf": False, "same_as": None},
    "M2": {"persist": True,  "cf": False, "same_as": None},
    "M3": {"persist": False, "cf": False, "same_as": "M1"},
    "M4": {"persist": True,  "cf": False, "same_as": "M2"},
    "M5": {"persist": True,  "cf": True,  "same_as": None},
    "M6": {"persist": False, "cf": True,  "same_as": None},
}
SIM_SEED = {"M1": 42, "M2": 52, "M5": 55, "M6": 56}   # M1 = P11 condition A


def clamped_params(model, persistence=0.0):
    """Parameter vector for p10.neg_loglik with alpha, beta at calibration."""
    a, b, p = ALPHA_CAL, BETA_CAL, persistence
    return {"M1": [a, b], "M2": [a, b, p], "M3": [a, a, b],
            "M4": [a, a, b, p], "M5": [a, a, b, p], "M6": [a, b]}[model]


def fit_clamped(subj, model):
    """NLL at the calibrated values; persistence fitted if the model has it."""
    ch, rw = subj["choice"], subj["reward"]
    if not STRUCTURES[model]["persist"]:
        return 0.0, float(p10.neg_loglik(clamped_params(model), ch, rw, model))
    lo, hi = p10.PERSIST_BOUNDS
    res = minimize_scalar(
        lambda p: p10.neg_loglik(clamped_params(model, p), ch, rw, model),
        bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    # also check p = 0 and both bounds, so the answer is never worse than those
    cands = [(float(res.x), float(res.fun))]
    for p in (0.0, lo, hi):
        cands.append((p, float(p10.neg_loglik(clamped_params(model, p),
                                              ch, rw, model))))
    return min(cands, key=lambda c: c[1])


def main():
    subjects, prov = p10.load_adults()
    B = len(subjects)
    print(f"Subjects: {B}")

    with open(os.path.join(RESULTS_DIR, "p10_model_comparison.json")) as f:
        free = json.load(f)
    assert free["sample_provenance"]["n_with_usable_file"] == B, (
        "P10 JSON and this run disagree on the sample size")
    with open(os.path.join(RESULTS_DIR, "p11_free_fit_behaviour.json")) as f:
        p11_json = json.load(f)

    # observed behaviour, computed by P11's own code
    e_ws, e_ls, e_acc = p11.empirical(p11.load_adults())
    emp = {"ws": float(np.nanmean(e_ws)), "ls": float(np.nanmean(e_ls)),
           "acc": float(np.nanmean(e_acc)),
           "ws_iqr": [float(np.nanpercentile(e_ws, 25)),
                      float(np.nanpercentile(e_ws, 75))],
           "ls_iqr": [float(np.nanpercentile(e_ls, 25)),
                      float(np.nanpercentile(e_ls, 75))],
           "acc_iqr": [float(np.nanpercentile(e_acc, 25)),
                       float(np.nanpercentile(e_acc, 75))]}

    ids = [s["id"] for s in subjects]
    out = {}
    for model, spec in STRUCTURES.items():
        free_ps = {f["id"]: f for f in free["per_subject"][model]}
        assert set(free_ps) == set(ids), f"{model}: ids differ from P10"

        persist, nll = [], []
        for s in subjects:
            p, v = fit_clamped(s, model)
            persist.append(p); nll.append(v)
        persist, nll = np.array(persist), np.array(nll)
        n_tr = np.array([s["choice"].size for s in subjects])
        k_here = 1 if spec["persist"] else 0
        bic_here = k_here * np.log(n_tr) + 2.0 * nll
        bic_free = np.array([free_ps[i]["bic"] for i in ids])
        nll_free = np.array([free_ps[i]["nll"] for i in ids])

        entry = {
            "same_as": spec["same_as"],
            "clamped": {"alpha": ALPHA_CAL, "beta": BETA_CAL},
            "free_parameters_here": ["persistence"] if spec["persist"] else [],
            "counterfactual": spec["cf"],
            "nll_calibrated": float(nll.sum()),
            "nll_free": float(nll_free.sum()),
            "bic_calibrated": float(bic_here.sum()),
            "bic_free": float(bic_free.sum()),
            "dbic": float(bic_here.sum() - bic_free.sum()),
            "n_free_better": int(np.sum(bic_free < bic_here)),
            "n_free_better_by_10": int(np.sum(bic_here - bic_free
                                              > DBIC_DECISIVE)),
            "mean_choice_prob_calibrated": float(np.mean(np.exp(-nll / n_tr))),
            "mean_choice_prob_free": float(np.mean(np.exp(-nll_free / n_tr))),
        }
        if spec["persist"]:
            lo, hi = p10.PERSIST_BOUNDS
            entry["persistence"] = {
                "median": float(np.median(persist)),
                "iqr": [float(np.percentile(persist, 25)),
                        float(np.percentile(persist, 75))],
                "n_negative": int(np.sum(persist < 0)),
                "n_positive": int(np.sum(persist > 0)),
                "n_at_lower_bound": int(np.sum(persist <= lo + 1e-4)),
                "n_at_upper_bound": int(np.sum(persist >= hi - 1e-4)),
                "bounds": [lo, hi],
            }

        # behaviour: simulate once per distinct condition
        if spec["same_as"] is None:
            ws, ls, acc = p11.simulate(
                ALPHA_CAL, BETA_CAL, B,
                persistence=persist if spec["persist"] else None,
                counterfactual=spec["cf"], seed=SIM_SEED[model])
            beh = {"ws": ws, "ls": ls, "acc": acc,
                   "ws_in_iqr": bool(emp["ws_iqr"][0] <= ws <= emp["ws_iqr"][1]),
                   "ls_in_iqr": bool(emp["ls_iqr"][0] <= ls <= emp["ls_iqr"][1]),
                   "acc_in_iqr": bool(emp["acc_iqr"][0] <= acc
                                      <= emp["acc_iqr"][1])}
            beh["n_in_iqr"] = int(beh["ws_in_iqr"] + beh["ls_in_iqr"]
                                  + beh["acc_in_iqr"])
            beh["seed"] = SIM_SEED[model]
        else:
            beh = dict(out[spec["same_as"]]["behaviour"])
            beh["copied_from"] = spec["same_as"]
        entry["behaviour"] = beh
        out[model] = entry
        print(f"  {model}: dBIC={entry['dbic']:9.1f}  free better for "
              f"{entry['n_free_better']}/{B}  ws={beh['ws']:.3f} "
              f"ls={beh['ls']:.3f} acc={beh['acc']:.3f}  "
              f"in IQR {beh['n_in_iqr']}/3"
              + (f"  [= {spec['same_as']}]" if spec["same_as"] else ""))

    # identity checks: M3 must equal M1, M4 must equal M2, in likelihood
    identity = {
        "M3_vs_M1_nll_abs_diff": abs(out["M3"]["nll_calibrated"]
                                     - out["M1"]["nll_calibrated"]),
        "M4_vs_M2_nll_abs_diff": abs(out["M4"]["nll_calibrated"]
                                     - out["M2"]["nll_calibrated"]),
    }
    identity["holds"] = bool(identity["M3_vs_M1_nll_abs_diff"] < 1e-6
                             and identity["M4_vs_M2_nll_abs_diff"] < 1e-3)

    # control against P11 condition A
    ref = p11_json["conditions"]["A_calibrated"]
    got = out["M1"]["behaviour"]
    control = {"p11_A": ref,
               "p12_M1": {k: got[k] for k in ("ws", "ls", "acc")},
               "abs_diff": {k: abs(ref[k] - got[k]) for k in ("ws", "ls", "acc")}}
    control["reproduces_p11"] = all(v < 1e-9 for v in control["abs_diff"].values())

    distinct = [m for m, s in STRUCTURES.items() if s["same_as"] is None]
    none_reproduces = all(out[m]["behaviour"]["n_in_iqr"] < 3 for m in distinct)
    all_decisive = all(out[m]["dbic"] > DBIC_DECISIVE for m in distinct)
    if none_reproduces and all_decisive:
        verdict = "calibrated values fail on every structure"
    elif none_reproduces:
        verdict = ("calibrated values fail behaviourally on every structure, "
                   "but not decisively by likelihood on all")
    else:
        ok = [m for m in distinct if out[m]["behaviour"]["n_in_iqr"] == 3]
        verdict = ("calibrated values REPRODUCE behaviour under: "
                   + ", ".join(ok))

    result = {
        "procedure": "alpha and beta clamped at the calibrated values under "
                     "each P10 structure; persistence, where present, fitted "
                     "per participant; likelihood on observed sequences and "
                     "simulated behaviour compared with the free fits of P10 "
                     "and the observed data",
        "registered": False,
        "n_subjects": B,
        "sample_provenance": prov,
        "calibrated": {"alpha": ALPHA_CAL, "beta": BETA_CAL},
        "criterion": {"behaviour": "all three observables inside observed IQR",
                      "dbic_decisive": DBIC_DECISIVE},
        "distinct_conditions": distinct,
        "note_on_identity": "with alpha+ = alpha- the valence split is inert: "
                            "M3 = M1 and M4 = M2. Six structures give four "
                            "distinct conditions.",
        "identity_check": identity,
        "data": emp,
        "structures": out,
        "control_vs_p11": control,
        "verdict": verdict,
    }

    print()
    print(f"Identity M3=M1, M4=M2: {'OK' if identity['holds'] else 'FAILED'}")
    print(f"Control vs P11 condition A: "
          f"{'OK' if control['reproduces_p11'] else 'MISMATCH'}")
    print(f"VERDICT: {verdict}")

    path = os.path.join(RESULTS_DIR, "p12_calibrated_all_structures.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
