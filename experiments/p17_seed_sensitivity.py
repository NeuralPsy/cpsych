"""P17: the behavioural check over many simulation seeds.

Why this exists. P11, P12 and P14 judged whether a model "reproduces
behaviour" from ONE simulation per condition. An independent review measured
the seed-to-seed spread of the simulated group mean at about 0.003 for
win-stay, and found two verdicts sitting inside it: one condition "failed" by
0.0019 and another "passed" by 0.0028. A verdict that flips with the seed is
not a result. This script repeats every condition over many seeds.

Conditions (Eckstein protocol, 120 trials, 112 simulated agents):
  calibrated values (alpha 0.15, beta 2.50), persistence from P15 where the
  structure has it:
    K1  base module                      (= valence split, M3)
    K2  base plus persistence            (= valence split plus persistence, M4)
    K5  published structure
    K6  counterfactual revision only
  fitted values from P13:
    B   free-fit medians, base module
    C   per-participant fits, base module
    D   per-participant fits, base plus persistence
    E   per-participant fits, published structure
    F   per-participant fits, counterfactual revision only

For each condition and each observable (win-stay, lose-shift, accuracy):
  mean and SD across seeds, the share of seeds falling inside the observed
  interquartile range, and the distance of the simulated mean from the
  OBSERVED MEAN in units of its standard error.

Two readings are reported, because the first is permissive:
  IQR reading (the criterion used so far): the observable lies inside the
    interquartile range of individual participants. The lose-shift range is
    wide (about 0.43 to 0.71), so this is a low bar.
  Mean reading (stricter): the simulated mean lies within two standard errors
    of the observed mean, the standard error combining between-participant
    and between-seed variation.

VERDICTS, fixed before this script was first run. For each condition:
  IQR reading:  "reproduces" if all three observables are inside the range in
                at least 95 per cent of seeds; "fails" if in at most 5 per
                cent; otherwise "indeterminate".
  Mean reading: "reproduces" if all three simulated means are within two
                standard errors of the observed means; otherwise "differs",
                naming the observables that differ.
  Both are reported for every condition. Neither is revised afterwards.

UNREGISTERED addition, made after two independent reviews.

Rule 1: every number lands in results/p17_seed_sensitivity.json.

Usage:  python experiments/p17_seed_sensitivity.py [--seeds 200]
"""
from __future__ import annotations

import argparse
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
SEED0 = 1000
OBS = ("ws", "ls", "acc")
PASS_SHARE, FAIL_SHARE, N_SE = 0.95, 0.05, 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    seeds = list(range(SEED0, SEED0 + args.seeds))

    data = p11.load_adults()
    B = len(data)
    print(f"Adults: {B};  seeds per condition: {len(seeds)}")

    with open(os.path.join(RESULTS_DIR, "p13_bounds_sensitivity.json")) as f:
        p13 = json.load(f)
    with open(os.path.join(RESULTS_DIR, "p15_calibrated_vs_wide.json")) as f:
        p15 = json.load(f)
    assert not p13["smoke_run"]
    assert p13["n_subjects"] == B and p15["n_subjects"] == B

    ids = [f["id"] for f in p13["per_subject_wide"]["M1"]]

    def fits(model):
        ps = p13["per_subject_wide"][model]
        assert [f["id"] for f in ps] == ids
        return np.array([f["params"] for f in ps])

    def clamped_persistence(model):
        ps = p15["structures"][model]["per_subject"]
        assert [f["id"] for f in ps] == ids, f"{model}: order differs from P13"
        return np.array([f["persistence"] for f in ps])

    m1, m2, m5, m6 = fits("M1"), fits("M2"), fits("M5"), fits("M6")
    a_med, b_med = float(np.median(m1[:, 0])), float(np.median(m1[:, 1]))

    conditions = {
        "K1_calibrated_base": dict(alpha_pos=ALPHA_CAL, beta=BETA_CAL),
        "K2_calibrated_plus_persistence": dict(
            alpha_pos=ALPHA_CAL, beta=BETA_CAL,
            persistence=clamped_persistence("M2")),
        "K5_calibrated_published_structure": dict(
            alpha_pos=ALPHA_CAL, beta=BETA_CAL,
            persistence=clamped_persistence("M5"), counterfactual=True),
        "K6_calibrated_counterfactual_only": dict(
            alpha_pos=ALPHA_CAL, beta=BETA_CAL, counterfactual=True),
        "B_freefit_median_base": dict(alpha_pos=a_med, beta=b_med),
        "C_freefit_per_subject_base": dict(alpha_pos=m1[:, 0], beta=m1[:, 1]),
        "D_freefit_per_subject_persistence": dict(
            alpha_pos=m2[:, 0], beta=m2[:, 1], persistence=m2[:, 2]),
        "E_freefit_per_subject_published_structure": dict(
            alpha_pos=m5[:, 0], beta=m5[:, 2], alpha_neg=m5[:, 1],
            persistence=m5[:, 3], counterfactual=True),
        "F_freefit_per_subject_counterfactual_only": dict(
            alpha_pos=m6[:, 0], beta=m6[:, 1], counterfactual=True),
    }

    # observed behaviour: mean, standard error, interquartile range
    e = dict(zip(OBS, p11.empirical(data)))
    emp = {}
    for k in OBS:
        v = e[k][~np.isnan(e[k])]
        emp[k] = {"mean": float(v.mean()),
                  "sd": float(v.std(ddof=1)),
                  "se": float(v.std(ddof=1) / np.sqrt(v.size)),
                  "n": int(v.size),
                  "iqr": [float(np.percentile(v, 25)),
                          float(np.percentile(v, 75))]}

    out = {}
    for name, kw in conditions.items():
        sims = np.array([p11.simulate(B=B, seed=s, **kw) for s in seeds])
        inside = np.column_stack([
            (sims[:, j] >= emp[k]["iqr"][0]) & (sims[:, j] <= emp[k]["iqr"][1])
            for j, k in enumerate(OBS)])
        share_all = float(np.mean(inside.all(axis=1)))
        c = {"n_seeds": len(seeds),
             "share_seeds_all_three_in_iqr": share_all,
             "mean_n_in_iqr": float(inside.sum(axis=1).mean())}
        differs = []
        for j, k in enumerate(OBS):
            mu, sd = float(sims[:, j].mean()), float(sims[:, j].std(ddof=1))
            se = float(np.sqrt(emp[k]["se"] ** 2 + sd ** 2))
            z = (mu - emp[k]["mean"]) / se
            c[k] = {"mean": mu, "sd_across_seeds": sd,
                    "min": float(sims[:, j].min()),
                    "max": float(sims[:, j].max()),
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
        out[name] = c
        print(f"  {name:44s} ws {c['ws']['mean']:.3f}+-{c['ws']['sd_across_seeds']:.3f} "
              f"ls {c['ls']['mean']:.3f}+-{c['ls']['sd_across_seeds']:.3f} "
              f"acc {c['acc']['mean']:.3f}+-{c['acc']['sd_across_seeds']:.3f}  "
              f"IQR: {c['verdict_iqr']} ({100*share_all:.0f}%)  "
              f"mean: {c['verdict_mean']}")

    cal = [k for k in out if k.startswith("K")]
    free = [k for k in out if not k.startswith("K")]
    summary = {
        "calibrated_reproducing_by_iqr": [k for k in cal
                                          if out[k]["verdict_iqr"] == "reproduces"],
        "calibrated_reproducing_by_mean": [k for k in cal
                                           if out[k]["verdict_mean"] == "reproduces"],
        "free_reproducing_by_iqr": [k for k in free
                                    if out[k]["verdict_iqr"] == "reproduces"],
        "free_indeterminate_by_iqr": [k for k in free
                                      if out[k]["verdict_iqr"] == "indeterminate"],
        "free_failing_by_iqr": [k for k in free
                                if out[k]["verdict_iqr"] == "fails"],
        "free_reproducing_by_mean": [k for k in free
                                     if out[k]["verdict_mean"] == "reproduces"],
    }

    print()
    print(f"observed: " + "  ".join(
        f"{k} {emp[k]['mean']:.3f} (se {emp[k]['se']:.3f}, IQR "
        f"{emp[k]['iqr'][0]:.3f}-{emp[k]['iqr'][1]:.3f})" for k in OBS))
    for k, v in summary.items():
        print(f"{k}: {v}")

    result = {
        "procedure": "every behavioural condition simulated over many seeds; "
                     "two readings (inside the observed interquartile range; "
                     "within two standard errors of the observed mean)",
        "registered": False,
        "n_subjects": B,
        "n_trials": p11.N_TRIALS,
        "seeds": [seeds[0], seeds[-1]],
        "n_seeds": len(seeds),
        "thresholds": {"pass_share": PASS_SHARE, "fail_share": FAIL_SHARE,
                       "n_se": N_SE},
        "free_fit_medians_M1": {"alpha": a_med, "beta": b_med},
        "observed": emp,
        "conditions": out,
        "summary": summary,
    }
    path = os.path.join(RESULTS_DIR, "p17_seed_sensitivity.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
