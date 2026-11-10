"""P18: the transfer test with the affect loop switched off (lambda = 0).

Why this exists. Three facts, found after P16:
  1. The calibration that produced alpha = 0.15, beta = 2.50 was run with
     lambda = 0 (see rl_mood_calibration.json: "grid search alpha x beta,
     lambda=0"). Every later stage of the transfer test ran with
     lambda = 0.49. So the calibrated values were not "applied unchanged":
     the affect coupling was switched on between calibration and test.
  2. The published model has no affect loop. With lambda = 0 the "published
     structure" is the published model; with lambda = 0.49 it is not.
  3. P16 refitted all six structures with lambda = 0. Every one of them fits
     better than with lambda = 0.49, and the published structure's positive
     learning rate moves from 0.31 to 0.72. By P16's pre-specified reading
     the results are not robust to the affect loop.
The two affect parameters are also the two the platform cannot recover at
all (r = 0.009 and 0.005), so holding them at fixed non-zero values inside a
fit has no support.

This script therefore repeats the two remaining stages at lambda = 0, so that
calibration, free fit and behavioural check all use one model:
  PART A (as P15)  calibrated values held under each structure, persistence
                   fitted where present, compared with P16's lambda = 0 fits.
  PART B (as P17)  every behavioural condition over many seeds, simulated
                   with lambda = 0, two readings (interquartile range; two
                   standard errors of the observed mean).

The free fits themselves are read from P16 ("M*_lam0"); nothing is refitted.

CRITERIA, fixed before this script was first run (same as P15 and P17):
  Part A: the free fit is decisively preferred for a structure if dbic > 10.
  Part B: IQR reading  - "reproduces" if all three observables are inside the
                         observed interquartile range in >= 95 per cent of
                         seeds, "fails" if in <= 5 per cent, else
                         "indeterminate".
          mean reading - "reproduces" if all three simulated means are within
                         two standard errors of the observed means.

CHECK ON THE SIMULATOR: the simulator reads the affect coupling from a module
constant. This script sets it to zero and verifies, before simulating, that
(a) with the constant restored to 0.49 the simulator still reproduces P11's
condition A exactly, and (b) with it at zero the affect state has no effect
on choices: two runs that differ only in tau_m give identical behaviour.

UNREGISTERED addition, made after two independent reviews.

Rule 1: every number lands in results/p18_transfer_lambda0.json.

Usage:  python experiments/p18_transfer_lambda0.py [--seeds 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import p10_model_comparison as p10              # noqa: E402
import p11_free_fit_behaviour as p11            # noqa: E402
import p13_bounds_sensitivity as p13            # noqa: E402
import p16_split_and_affect_sensitivity as p16  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "..", "results")
ALPHA_CAL, BETA_CAL = 0.15, 2.50
LAM0 = 0.0
DBIC_DECISIVE = 10.0
TOL = 1e-3
SEED0 = 1000
OBS = ("ws", "ls", "acc")
PASS_SHARE, FAIL_SHARE, N_SE = 0.95, 0.05, 2.0

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


def nll0(params, subj, model):
    return float(p16.nll(params, subj["choice"], subj["reward"], model, LAM0,
                         "outcome"))


def fit_clamped(subj, model, pbounds):
    if not STRUCTURES[model]["persist"]:
        return 0.0, nll0(clamped_params(model), subj, model)
    lo, hi = pbounds
    res = minimize_scalar(lambda p: nll0(clamped_params(model, p), subj, model),
                          bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    cands = [(float(res.x), float(res.fun))]
    for p in (0.0, lo, hi):
        cands.append((p, nll0(clamped_params(model, p), subj, model)))
    return min(cands, key=lambda c: c[1])


def simulator_checks(B, p11_json):
    """(a) unchanged code path at 0.49; (b) affect inert at lambda = 0."""
    lam_main, tau_main = p11.LAM, p11.TAU_M
    a = p11.simulate(ALPHA_CAL, BETA_CAL, B, seed=42)
    ref = p11_json["conditions"]["A_calibrated"]
    same_path = max(abs(ref["ws"] - a[0]), abs(ref["ls"] - a[1]),
                    abs(ref["acc"] - a[2]))
    p11.LAM = LAM0
    x = p11.simulate(0.6, 4.0, B, seed=7)
    p11.TAU_M = 5.0
    y = p11.simulate(0.6, 4.0, B, seed=7)
    p11.TAU_M = tau_main
    inert = max(abs(u - v) for u, v in zip(x, y))
    p11.LAM = lam_main
    return {"reproduces_p11_A_at_lambda_main_abs_diff": float(same_path),
            "affect_inert_at_lambda0_abs_diff": float(inert),
            "ok": bool(same_path < 1e-9 and inert < 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    seeds = list(range(SEED0, SEED0 + args.seeds))

    subjects, prov = p10.load_adults()
    B = len(subjects)
    ids = [s["id"] for s in subjects]
    print(f"Subjects: {B};  seeds per condition: {len(seeds)}")

    with open(os.path.join(RESULTS_DIR,
                           "p16_split_and_affect_sensitivity.json")) as f:
        p16j = json.load(f)
    with open(os.path.join(RESULTS_DIR, "p11_free_fit_behaviour.json")) as f:
        p11_json = json.load(f)
    assert not p16j["smoke_run"] and p16j["n_subjects"] == B
    free = {m: p16j["per_subject"][m + "_lam0"] for m in p10.ALL_MODELS}
    for m in free:
        assert [f["id"] for f in free[m]] == ids, f"{m}: order differs from P16"
    pbounds = tuple(p13.NEW["persistence"])

    chk = simulator_checks(B, p11_json)
    print(f"Simulator checks: {'OK' if chk['ok'] else 'FAILED'}  {chk}")
    assert chk["ok"], "simulator checks failed"

    # ------------------------------------------------------------- PART A
    print("\nPART A: calibrated values against the lambda = 0 free fits")
    part_a = {}
    for model, spec in STRUCTURES.items():
        pers, nll = [], []
        for s in subjects:
            p, v = fit_clamped(s, model, pbounds)
            pers.append(p); nll.append(v)
        pers, nll = np.array(pers), np.array(nll)
        n_tr = np.array([s["choice"].size for s in subjects])
        k = 1 if spec["persist"] else 0
        bic_cal = k * np.log(n_tr) + 2.0 * nll
        bic_free = np.array([f["bic"] for f in free[model]])
        nll_free = np.array([f["nll"] for f in free[model]])
        e = {"same_as": spec["same_as"], "counterfactual": spec["cf"],
             "free_parameters_here": ["persistence"] if spec["persist"] else [],
             "nll_calibrated": float(nll.sum()),
             "nll_free": float(nll_free.sum()),
             "bic_calibrated": float(bic_cal.sum()),
             "bic_free": float(bic_free.sum()),
             "dbic": float(bic_cal.sum() - bic_free.sum()),
             "decisive": bool(bic_cal.sum() - bic_free.sum() > DBIC_DECISIVE),
             "n_free_better": int(np.sum(bic_free < bic_cal)),
             "n_free_better_by_10": int(np.sum(bic_cal - bic_free
                                               > DBIC_DECISIVE)),
             "n_calibrated_better_by_10": int(np.sum(bic_free - bic_cal
                                                     > DBIC_DECISIVE)),
             "per_subject": [{"id": int(i), "persistence": float(p),
                              "nll": float(v), "bic": float(bc),
                              "bic_free": float(bf)}
                             for i, p, v, bc, bf in zip(ids, pers, nll, bic_cal,
                                                        bic_free)]}
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
                "n_beyond_2": int(np.sum(np.abs(pers) > 2.0))}
        part_a[model] = e
        print(f"  {model}: dBIC={e['dbic']:9.1f}  free better "
              f"{e['n_free_better']}/{B}"
              + (f"  [= {spec['same_as']}]" if spec["same_as"] else ""))
    identity = {
        "M3_vs_M1": abs(part_a["M3"]["nll_calibrated"]
                        - part_a["M1"]["nll_calibrated"]),
        "M4_vs_M2": abs(part_a["M4"]["nll_calibrated"]
                        - part_a["M2"]["nll_calibrated"])}
    identity["holds"] = bool(identity["M3_vs_M1"] < 1e-6
                             and identity["M4_vs_M2"] < 1e-3)

    # ------------------------------------------------------------- PART B
    print("\nPART B: behaviour over seeds, simulated with lambda = 0")

    def P(model):
        return np.array([f["params"] for f in free[model]])

    def pers_cal(model):
        return np.array([f["persistence"] for f in part_a[model]["per_subject"]])

    m1, m2, m3, m4, m5, m6 = (P(m) for m in p10.ALL_MODELS)
    a_med, b_med = float(np.median(m1[:, 0])), float(np.median(m1[:, 1]))
    conditions = {
        "K1_calibrated_base": dict(alpha_pos=ALPHA_CAL, beta=BETA_CAL),
        "K2_calibrated_plus_persistence": dict(
            alpha_pos=ALPHA_CAL, beta=BETA_CAL, persistence=pers_cal("M2")),
        "K5_calibrated_published_model": dict(
            alpha_pos=ALPHA_CAL, beta=BETA_CAL, persistence=pers_cal("M5"),
            counterfactual=True),
        "K6_calibrated_counterfactual_only": dict(
            alpha_pos=ALPHA_CAL, beta=BETA_CAL, counterfactual=True),
        "B_freefit_median_base": dict(alpha_pos=a_med, beta=b_med),
        "C_freefit_per_subject_base": dict(alpha_pos=m1[:, 0], beta=m1[:, 1]),
        "D_freefit_per_subject_persistence": dict(
            alpha_pos=m2[:, 0], beta=m2[:, 1], persistence=m2[:, 2]),
        "G_freefit_per_subject_valence_split": dict(
            alpha_pos=m3[:, 0], beta=m3[:, 2], alpha_neg=m3[:, 1]),
        "H_freefit_per_subject_valence_split_persistence": dict(
            alpha_pos=m4[:, 0], beta=m4[:, 2], alpha_neg=m4[:, 1],
            persistence=m4[:, 3]),
        "E_freefit_per_subject_published_model": dict(
            alpha_pos=m5[:, 0], beta=m5[:, 2], alpha_neg=m5[:, 1],
            persistence=m5[:, 3], counterfactual=True),
        "F_freefit_per_subject_counterfactual_only": dict(
            alpha_pos=m6[:, 0], beta=m6[:, 1], counterfactual=True),
    }

    data = p11.load_adults()
    assert len(data) == B
    e = dict(zip(OBS, p11.empirical(data)))
    emp = {}
    for k in OBS:
        v = e[k][~np.isnan(e[k])]
        emp[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                  "se": float(v.std(ddof=1) / np.sqrt(v.size)), "n": int(v.size),
                  "iqr": [float(np.percentile(v, 25)),
                          float(np.percentile(v, 75))]}

    lam_main = p11.LAM
    p11.LAM = LAM0
    part_b = {}
    try:
        for name, kw in conditions.items():
            sims = np.array([p11.simulate(B=B, seed=s, **kw) for s in seeds])
            inside = np.column_stack([
                (sims[:, j] >= emp[k]["iqr"][0]) & (sims[:, j] <= emp[k]["iqr"][1])
                for j, k in enumerate(OBS)])
            share_all = float(np.mean(inside.all(axis=1)))
            c = {"n_seeds": len(seeds),
                 "share_seeds_all_three_in_iqr": share_all}
            differs = []
            for j, k in enumerate(OBS):
                mu, sd = float(sims[:, j].mean()), float(sims[:, j].std(ddof=1))
                se = float(np.sqrt(emp[k]["se"] ** 2 + sd ** 2))
                z = (mu - emp[k]["mean"]) / se
                c[k] = {"mean": mu, "sd_across_seeds": sd,
                        "share_seeds_in_iqr": float(inside[:, j].mean()),
                        "diff_from_observed_mean": mu - emp[k]["mean"],
                        "combined_se": se, "z": float(z),
                        "within_2se": bool(abs(z) <= N_SE)}
                if abs(z) > N_SE:
                    differs.append(k)
            c["verdict_iqr"] = ("reproduces" if share_all >= PASS_SHARE else
                                "fails" if share_all <= FAIL_SHARE else
                                "indeterminate")
            c["verdict_mean"] = ("reproduces" if not differs
                                 else "differs: " + ", ".join(differs))
            part_b[name] = c
            print(f"  {name:48s} ws {c['ws']['mean']:.3f} ls {c['ls']['mean']:.3f} "
                  f"acc {c['acc']['mean']:.3f}  IQR: {c['verdict_iqr']} "
                  f"({100*share_all:.0f}%)  mean: {c['verdict_mean']}")
    finally:
        p11.LAM = lam_main

    cal = [k for k in part_b if k.startswith("K")]
    fr = [k for k in part_b if not k.startswith("K")]
    summary = {
        "dbic_min": float(min(part_a[m]["dbic"] for m in part_a)),
        "dbic_max": float(max(part_a[m]["dbic"] for m in part_a)),
        "n_free_better_min": int(min(part_a[m]["n_free_better"] for m in part_a)),
        "n_free_better_max": int(max(part_a[m]["n_free_better"] for m in part_a)),
        "decisive_everywhere": bool(all(part_a[m]["decisive"] for m in part_a)),
        "calibrated_reproducing_by_iqr": [k for k in cal if part_b[k]["verdict_iqr"] == "reproduces"],
        "calibrated_reproducing_by_mean": [k for k in cal if part_b[k]["verdict_mean"] == "reproduces"],
        "free_reproducing_by_iqr": [k for k in fr if part_b[k]["verdict_iqr"] == "reproduces"],
        "free_indeterminate_by_iqr": [k for k in fr if part_b[k]["verdict_iqr"] == "indeterminate"],
        "free_failing_by_iqr": [k for k in fr if part_b[k]["verdict_iqr"] == "fails"],
        "free_reproducing_by_mean": [k for k in fr if part_b[k]["verdict_mean"] == "reproduces"],
    }
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")

    result = {
        "procedure": "transfer test at lambda = 0: calibrated values held "
                     "under each structure against P16's lambda = 0 fits, and "
                     "behaviour over many seeds simulated with lambda = 0",
        "registered": False,
        "lambda": LAM0,
        "n_subjects": B,
        "sample_provenance": prov,
        "calibrated": {"alpha": ALPHA_CAL, "beta": BETA_CAL},
        "persistence_bounds": list(pbounds),
        "seeds": [seeds[0], seeds[-1]],
        "n_seeds": len(seeds),
        "thresholds": {"dbic_decisive": DBIC_DECISIVE, "pass_share": PASS_SHARE,
                       "fail_share": FAIL_SHARE, "n_se": N_SE},
        "simulator_checks": chk,
        "identity_check": identity,
        "free_fit_medians_M1": {"alpha": a_med, "beta": b_med},
        "observed": emp,
        "part_a_calibrated": part_a,
        "part_b_behaviour": part_b,
        "summary": summary,
    }
    path = os.path.join(RESULTS_DIR, "p18_transfer_lambda0.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
