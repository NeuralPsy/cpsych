"""P14: stage 3 again, at the wide-bounds fits of P13.

Why this exists. P13 refitted the six structures with wider bounds on beta
and persistence, and by its own pre-specified reading the P10 numbers are to
be replaced by the P13 ones. P11 (stage 3: does the model reproduce behaviour
at its own fitted values?) was run on the P10 fits. If the tables move to
P13, stage 3 has to move with them. This script is P11, unchanged, reading
P13's per-participant fits instead of P10's.

Same code path as P11: its simulate(), its empirical(), its sample loader,
its seeds. Only the fitted parameters differ. P11's JSON is NOT overwritten.

Conditions (Eckstein protocol, 120 trials):
  A  calibrated parameters                    alpha=0.15, beta=2.50
  B  free-fit medians (M1, P13)
  C  per-subject free fits (M1, P13)
  D  per-subject free fits (M2, P13)          plus persistence
  E  per-subject free fits (M5, P13)          published structure
  F  per-subject free fits (M6, P13)          counterfactual only  [new]

F is added because P12 simulated M6 at the calibrated values; this gives the
matching row at the fitted values.

CRITERION (P11's, unchanged, fixed before this script was first run):
  a condition reproduces behaviour only if win-stay, lose-shift and accuracy
  all fall inside the observed interquartile ranges.

CONTROL: condition A uses P11's seed and must reproduce P11's condition A
exactly. If it does not, nothing else here is trustworthy.

A note on very large fitted values. Some P13 fits have beta in the hundreds.
Simulating at beta = 200 is simulating a near-deterministic chooser, which is
what the likelihood says about that participant; nothing is clipped here
beyond what P11's simulate() already does.

UNREGISTERED: follows from P13, which is itself an unregistered sensitivity
analysis.

Rule 1: every number lands in results/p14_free_fit_behaviour_wide.json.

Usage:  python experiments/p14_free_fit_behaviour_wide.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import p11_free_fit_behaviour as p11        # noqa: E402

RESULTS_DIR = os.path.join(HERE, "..", "results")
ALPHA_CAL, BETA_CAL = p11.ALPHA_CAL, p11.BETA_CAL


def main():
    data = p11.load_adults()
    B = len(data)
    print(f"Adults (age {p11.AGE_MIN}-{p11.AGE_MAX}, usable file): {B}")

    with open(os.path.join(RESULTS_DIR, "p13_bounds_sensitivity.json")) as f:
        p13 = json.load(f)
    with open(os.path.join(RESULTS_DIR, "p11_free_fit_behaviour.json")) as f:
        p11_json = json.load(f)
    assert not p13["smoke_run"], "P13 JSON is a smoke run"
    assert p13["n_subjects"] == B, "P13 and this run disagree on the sample size"

    def fits(model):
        ps = p13["per_subject_wide"][model]
        assert len(ps) == B
        # P11 passes per-subject arrays in P10's order; P13 kept that order.
        assert [f["id"] for f in ps] == [f["id"] for f in
                                         p13["per_subject_wide"]["M1"]]
        return np.array([f["params"] for f in ps])

    m1, m2, m5, m6 = fits("M1"), fits("M2"), fits("M5"), fits("M6")
    a_med, b_med = float(np.median(m1[:, 0])), float(np.median(m1[:, 1]))

    e_ws, e_ls, e_acc = p11.empirical(data)
    emp = {"ws": float(np.nanmean(e_ws)), "ls": float(np.nanmean(e_ls)),
           "acc": float(np.nanmean(e_acc)),
           "ws_iqr": [float(np.nanpercentile(e_ws, 25)),
                      float(np.nanpercentile(e_ws, 75))],
           "ls_iqr": [float(np.nanpercentile(e_ls, 25)),
                      float(np.nanpercentile(e_ls, 75))],
           "acc_iqr": [float(np.nanpercentile(e_acc, 25)),
                       float(np.nanpercentile(e_acc, 75))]}

    conds = {}
    conds["A_calibrated"] = p11.simulate(ALPHA_CAL, BETA_CAL, B, seed=42)
    conds["B_freefit_median"] = p11.simulate(a_med, b_med, B, seed=43)
    conds["C_freefit_per_subject"] = p11.simulate(m1[:, 0], m1[:, 1], B,
                                                  seed=44)
    conds["D_freefit_per_subject_persistence"] = p11.simulate(
        m2[:, 0], m2[:, 1], B, persistence=m2[:, 2], seed=45)
    conds["E_freefit_per_subject_published_structure"] = p11.simulate(
        m5[:, 0], m5[:, 2], B, alpha_neg=m5[:, 1], persistence=m5[:, 3],
        counterfactual=True, seed=46)
    conds["F_freefit_per_subject_counterfactual_only"] = p11.simulate(
        m6[:, 0], m6[:, 1], B, counterfactual=True, seed=47)

    verdicts, miss = {}, {}
    for k, (ws, ls, acc) in conds.items():
        v = {"ws_in_iqr": bool(emp["ws_iqr"][0] <= ws <= emp["ws_iqr"][1]),
             "ls_in_iqr": bool(emp["ls_iqr"][0] <= ls <= emp["ls_iqr"][1]),
             "acc_in_iqr": bool(emp["acc_iqr"][0] <= acc <= emp["acc_iqr"][1])}
        v["n_in_iqr"] = int(v["ws_in_iqr"] + v["ls_in_iqr"] + v["acc_in_iqr"])
        verdicts[k] = v
        # how far outside the range, for any observable that misses
        d = {}
        for name, val in (("ws", ws), ("ls", ls), ("acc", acc)):
            lo, hi = emp[f"{name}_iqr"]
            d[name] = float(max(lo - val, val - hi, 0.0))
        miss[k] = d

    # control against P11 condition A
    ref = p11_json["conditions"]["A_calibrated"]
    got = conds["A_calibrated"]
    control = {"p11_A": ref,
               "p14_A": {"ws": got[0], "ls": got[1], "acc": got[2]},
               "abs_diff": {"ws": abs(ref["ws"] - got[0]),
                            "ls": abs(ref["ls"] - got[1]),
                            "acc": abs(ref["acc"] - got[2])}}
    control["reproduces_p11"] = all(v < 1e-9
                                    for v in control["abs_diff"].values())

    # P11 values for the conditions it had, side by side
    vs_p11 = {}
    for k in conds:
        if k in p11_json["conditions"]:
            old = p11_json["conditions"][k]
            new = conds[k]
            vs_p11[k] = {"p11": old,
                         "diff": {"ws": new[0] - old["ws"],
                                  "ls": new[1] - old["ls"],
                                  "acc": new[2] - old["acc"]},
                         "p11_n_in_iqr": p11_json["inside_observed_iqr"][k]
                                                 ["n_in_iqr"]}

    print()
    print(f"Control vs P11 condition A: "
          f"{'OK' if control['reproduces_p11'] else 'MISMATCH'}")
    print()
    print(f"{'condition':46s} {'win-stay':>9s} {'lose-shift':>11s} "
          f"{'accuracy':>9s}  in IQR")
    print(f"{'DATA':46s} {emp['ws']:9.3f} {emp['ls']:11.3f} {emp['acc']:9.3f}")
    for k, (ws, ls, acc) in conds.items():
        print(f"{k:46s} {ws:9.3f} {ls:11.3f} {acc:9.3f}  "
              f"{verdicts[k]['n_in_iqr']}/3")

    free = [k for k in conds if k != "A_calibrated"]
    all_ok = [k for k in free if verdicts[k]["n_in_iqr"] == 3]
    conclusion = ("free-fit parameters reproduce behaviour under: "
                  + ", ".join(all_ok) if all_ok
                  else "no free-fit condition reproduces all three observables")
    print()
    print(f"CONCLUSION: {conclusion}")

    result = {
        "procedure": "P11 repeated at the wide-bounds fits of P13; same "
                     "simulation code, sample and seeds",
        "registered": False,
        "fits_from": "p13_bounds_sensitivity.json (per_subject_wide)",
        "n_subjects": B,
        "age_window": [p11.AGE_MIN, p11.AGE_MAX],
        "n_trials": p11.N_TRIALS,
        "fixed": {"tau_m": p11.TAU_M, "lam": p11.LAM},
        "free_fit_medians_M1": {"alpha": a_med, "beta": b_med},
        "data": emp,
        "conditions": {k: {"ws": v[0], "ls": v[1], "acc": v[2]}
                       for k, v in conds.items()},
        "inside_observed_iqr": verdicts,
        "distance_outside_iqr": miss,
        "compared_with_p11": vs_p11,
        "control_vs_p11": control,
        "conclusion": conclusion,
    }
    path = os.path.join(RESULTS_DIR, "p14_free_fit_behaviour_wide.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
