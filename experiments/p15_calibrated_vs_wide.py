"""P15: calibrated values under each structure, compared with the P13 fits.

Replaces P12. P12 asked the right question but compared against the wrong
fits: it read the free fits from P10 (inverse temperature bounded at 20,
persistence at +-2), while every other number in the paper now comes from P13
(bounds 200 and +-20). It also fitted the clamped persistence within +-2.
P12 stays in the archive; its margins are not like-for-like and must not be
quoted.

What is the same as P12: alpha (both rates) clamped at 0.15, beta at 2.50,
persistence the only free parameter where the structure has one; M3 = M1 and
M4 = M2 at these values, so four distinct conditions.

What is different:
  - the comparator is P13's per-participant fits;
  - persistence is fitted within P13's bounds;
  - per-participant arrays (id, clamped persistence, clamped nll, BIC both
    ways) are stored, so every aggregate can be audited from the JSON;
  - behaviour is NOT judged here. One simulation per condition is kept only
    as the control against P11. The behavioural verdict comes from P17, which
    runs many seeds.

Outputs, per structure:
  nll_calibrated, nll_free(P13), bic_calibrated, bic_free(P13), dbic,
  n_free_better, n_free_better_by_10, persistence summary, per_subject.

CRITERION (fixed before this script was first run):
  "the free fit is decisively preferred" for a structure if dbic > 10.
  Reported as it falls.

UNREGISTERED addition, made after two independent reviews of P12-P14.

Rule 1: every number lands in results/p15_calibrated_vs_wide.json.

Usage:  python experiments/p15_calibrated_vs_wide.py
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

import p10_model_comparison as p10          # noqa: E402
import p11_free_fit_behaviour as p11        # noqa: E402

RESULTS_DIR = os.path.join(HERE, "..", "results")
ALPHA_CAL, BETA_CAL = 0.15, 2.50
DBIC_DECISIVE = 10.0
TOL = 1e-3

STRUCTURES = {
    "M1": {"persist": False, "cf": False, "same_as": None},
    "M2": {"persist": True,  "cf": False, "same_as": None},
    "M3": {"persist": False, "cf": False, "same_as": "M1"},
    "M4": {"persist": True,  "cf": False, "same_as": "M2"},
    "M5": {"persist": True,  "cf": True,  "same_as": None},
    "M6": {"persist": False, "cf": True,  "same_as": None},
}


def clamped_params(model, persistence=0.0):
    a, b, p = ALPHA_CAL, BETA_CAL, persistence
    return {"M1": [a, b], "M2": [a, b, p], "M3": [a, a, b],
            "M4": [a, a, b, p], "M5": [a, a, b, p], "M6": [a, b]}[model]


def fit_clamped(subj, model, pbounds):
    ch, rw = subj["choice"], subj["reward"]
    if not STRUCTURES[model]["persist"]:
        return 0.0, float(p10.neg_loglik(clamped_params(model), ch, rw, model))
    lo, hi = pbounds
    # The nll is convex in persistence when alpha and beta are fixed (the
    # value path does not depend on it), so a bounded scalar search finds the
    # minimum. Endpoints and zero are checked as well.
    res = minimize_scalar(
        lambda p: p10.neg_loglik(clamped_params(model, p), ch, rw, model),
        bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    cands = [(float(res.x), float(res.fun))]
    for p in (0.0, lo, hi):
        cands.append((p, float(p10.neg_loglik(clamped_params(model, p),
                                              ch, rw, model))))
    return min(cands, key=lambda c: c[1])


def main():
    subjects, prov = p10.load_adults()
    B = len(subjects)
    ids = [s["id"] for s in subjects]
    print(f"Subjects: {B}")

    with open(os.path.join(RESULTS_DIR, "p13_bounds_sensitivity.json")) as f:
        p13 = json.load(f)
    assert not p13["smoke_run"], "P13 JSON is a smoke run"
    assert p13["n_subjects"] == B, "P13 and this run disagree on the sample"
    pbounds = tuple(p13["bounds"]["wide"]["persistence"])
    with open(os.path.join(RESULTS_DIR, "p11_free_fit_behaviour.json")) as f:
        p11_json = json.load(f)

    out = {}
    for model, spec in STRUCTURES.items():
        free = {f["id"]: f for f in p13["per_subject_wide"][model]}
        assert [f["id"] for f in p13["per_subject_wide"][model]] == ids, (
            f"{model}: participant order differs from P13")

        pers, nll = [], []
        for s in subjects:
            p, v = fit_clamped(s, model, pbounds)
            pers.append(p); nll.append(v)
        pers, nll = np.array(pers), np.array(nll)
        n_tr = np.array([s["choice"].size for s in subjects])
        k = 1 if spec["persist"] else 0
        bic_cal = k * np.log(n_tr) + 2.0 * nll
        bic_free = np.array([free[i]["bic"] for i in ids])
        nll_free = np.array([free[i]["nll"] for i in ids])

        e = {
            "same_as": spec["same_as"],
            "clamped": {"alpha": ALPHA_CAL, "beta": BETA_CAL},
            "free_parameters_here": ["persistence"] if spec["persist"] else [],
            "counterfactual": spec["cf"],
            "comparator": "p13_bounds_sensitivity.json / per_subject_wide",
            "nll_calibrated": float(nll.sum()),
            "nll_free": float(nll_free.sum()),
            "bic_calibrated": float(bic_cal.sum()),
            "bic_free": float(bic_free.sum()),
            "dbic": float(bic_cal.sum() - bic_free.sum()),
            "decisive": bool(bic_cal.sum() - bic_free.sum() > DBIC_DECISIVE),
            "n_free_better": int(np.sum(bic_free < bic_cal)),
            "n_free_better_by_10": int(np.sum(bic_cal - bic_free > DBIC_DECISIVE)),
            "n_calibrated_better_by_10": int(np.sum(bic_free - bic_cal
                                                    > DBIC_DECISIVE)),
            "mean_choice_prob_calibrated": float(np.mean(np.exp(-nll / n_tr))),
            "mean_choice_prob_free": float(np.mean(np.exp(-nll_free / n_tr))),
            "per_subject": [
                {"id": int(i), "persistence": float(p), "nll": float(v),
                 "bic": float(bc), "bic_free": float(bf), "n_trials": int(n)}
                for i, p, v, bc, bf, n in zip(ids, pers, nll, bic_cal,
                                              bic_free, n_tr)],
        }
        if spec["persist"]:
            lo, hi = pbounds
            e["persistence"] = {
                "median": float(np.median(pers)),
                "iqr": [float(np.percentile(pers, 25)),
                        float(np.percentile(pers, 75))],
                "n_negative": int(np.sum(pers < 0)),
                "n_positive": int(np.sum(pers > 0)),
                "n_at_lower_bound": int(np.sum(pers <= lo + TOL)),
                "n_at_upper_bound": int(np.sum(pers >= hi - TOL)),
                "n_beyond_2": int(np.sum(np.abs(pers) > 2.0)),
                "bounds": [lo, hi],
            }
        out[model] = e
        print(f"  {model}: dBIC={e['dbic']:9.1f}  free better "
              f"{e['n_free_better']}/{B}  by>10 {e['n_free_better_by_10']}"
              + (f"  [= {spec['same_as']}]" if spec["same_as"] else ""))

    identity = {
        "M3_vs_M1_nll_abs_diff": abs(out["M3"]["nll_calibrated"]
                                     - out["M1"]["nll_calibrated"]),
        "M4_vs_M2_nll_abs_diff": abs(out["M4"]["nll_calibrated"]
                                     - out["M2"]["nll_calibrated"]),
    }
    identity["holds"] = bool(identity["M3_vs_M1_nll_abs_diff"] < 1e-6
                             and identity["M4_vs_M2_nll_abs_diff"] < 1e-3)

    # control: the code path is the one P11 used (same seed, same arguments)
    ws, ls, acc = p11.simulate(ALPHA_CAL, BETA_CAL, B, seed=42)
    ref = p11_json["conditions"]["A_calibrated"]
    control = {"p11_A": ref, "here": {"ws": ws, "ls": ls, "acc": acc},
               "abs_diff": {"ws": abs(ref["ws"] - ws), "ls": abs(ref["ls"] - ls),
                            "acc": abs(ref["acc"] - acc)},
               "note": "shows only that the simulation code path is unchanged"}
    control["reproduces_p11"] = all(v < 1e-9 for v in control["abs_diff"].values())

    distinct = [m for m, s in STRUCTURES.items() if s["same_as"] is None]
    result = {
        "procedure": "alpha and beta clamped at the calibrated values under "
                     "each structure; persistence, where present, fitted per "
                     "participant within the P13 bounds; likelihood compared "
                     "with the P13 free fits",
        "replaces": "p12_calibrated_all_structures.json (comparator was P10)",
        "registered": False,
        "n_subjects": B,
        "sample_provenance": prov,
        "calibrated": {"alpha": ALPHA_CAL, "beta": BETA_CAL},
        "persistence_bounds": list(pbounds),
        "dbic_decisive": DBIC_DECISIVE,
        "distinct_conditions": distinct,
        "identity_check": identity,
        "structures": out,
        "summary": {
            "dbic_min": float(min(out[m]["dbic"] for m in out)),
            "dbic_max": float(max(out[m]["dbic"] for m in out)),
            "n_free_better_min": int(min(out[m]["n_free_better"] for m in out)),
            "n_free_better_max": int(max(out[m]["n_free_better"] for m in out)),
            "decisive_everywhere": bool(all(out[m]["decisive"] for m in out)),
        },
        "control_vs_p11": control,
        "behaviour": "not judged here; see p17_seed_sensitivity.json",
    }
    print()
    print(f"Identity M3=M1, M4=M2: {'OK' if identity['holds'] else 'FAILED'}")
    print(f"Control vs P11 condition A: "
          f"{'OK' if control['reproduces_p11'] else 'MISMATCH'}")
    s = result["summary"]
    print(f"dBIC range {s['dbic_min']:.0f} to {s['dbic_max']:.0f}; free better "
          f"for {s['n_free_better_min']} to {s['n_free_better_max']} of {B}; "
          f"decisive everywhere: {s['decisive_everywhere']}")

    path = os.path.join(RESULTS_DIR, "p15_calibrated_vs_wide.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
