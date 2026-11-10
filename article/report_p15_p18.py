#!/usr/bin/env python3
"""Print the numbers of P15-P18 straight from the result files.

P16 (check 2) and P18 are the primary transfer analysis, with the affect
coupling off (lambda = 0). P13, P15 and P17 are the same analysis with the
coupling at its default (lambda = 0.49) and serve as the sensitivity analysis.

Usage:  python article/report_p15_p18.py results
"""
import json
import os
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results"


def load(name):
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return json.load(f)


p15 = load("p15_calibrated_vs_wide.json")
p16 = load("p16_split_and_affect_sensitivity.json")
p17 = load("p17_seed_sensitivity.json")
p18 = load("p18_transfer_lambda0.json")
c1, c2 = p16["check1_split_rule"], p16["check2_lambda_zero"]
NAMES = {"M1": "base", "M2": "base + persistence", "M3": "separate rates",
         "M4": "separate rates + persistence", "M5": "published structure",
         "M6": "counterfactual only"}

print("=" * 78)
print("PRIMARY ANALYSIS: affect coupling off (lambda = 0)")
print("=" * 78)
print(f"participants {p18['n_subjects']}, seeds {p18['n_seeds']}")

print("\nFree fits (P16, check 2): median [IQR], share above 0.50, at upper bound")
for m, v in c2["models"].items():
    print(f"  {m} {NAMES[m]}   BIC {v['bic_sum']:.1f}   dBIC {c2['dbic'][m]:.1f}"
          f"   best for {c2['subject_winners_all'][m]}")
    P = np.array([s["params"] for s in p16["per_subject"][m + "_lam0"]])
    for i, n in enumerate(v["param_names"]):
        x = P[:, i]
        line = (f"      {n:12s} {np.median(x):8.3f} "
                f"[{np.percentile(x, 25):.3f}, {np.percentile(x, 75):.3f}]")
        if n.startswith("alpha"):
            line += (f"   x{np.median(x) / 0.15:.1f}   >0.50: {(x > 0.5).sum()}"
                     f"   at 0.99: {(x >= 0.99 - 1e-3).sum()}"
                     f"   at 0.01: {(x <= 0.01 + 1e-3).sum()}")
        elif n == "beta":
            line += f"   x{np.median(x) / 2.5:.1f}   >20: {(x > 20).sum()}"
        elif n == "persistence":
            line += (f"   negative: {(x < 0).sum()}   positive: {(x > 0).sum()}"
                     f"   |p|>2: {(np.abs(x) > 2).sum()}")
        print(line)
    if "_any_parameter" in v.get("at_bound", {}):
        print(f"      any parameter at a bound: {v['at_bound']['_any_parameter']}")
print(f"  registered four, individual winners: {c2['subject_winners_registered']}")
print(f"  published structure better than base + persistence for "
      f"{c2['M5_better_than_M2']} participants")

print("\nCalibrated values against the free fit (P18, part A)")
for m, e in p18["part_a_calibrated"].items():
    print(f"  {m} {NAMES[m]:30s} dBIC {e['dbic']:8.1f}   free better "
          f"{e['n_free_better']}   by >10: {e['n_free_better_by_10']}")

print("\nBehaviour over seeds (P18, part B)")
o = p18["observed"]
print("  observed: " + "  ".join(
    f"{k} {o[k]['mean']:.3f} (SE {o[k]['se']:.4f}, IQR "
    f"{o[k]['iqr'][0]:.3f}-{o[k]['iqr'][1]:.3f})" for k in ("ws", "ls", "acc")))
for name, c in p18["part_b_behaviour"].items():
    print(f"  {name:50s} " + " ".join(
        f"{k} {c[k]['mean']:.3f} (z {c[k]['z']:+.1f})" for k in ("ws", "ls", "acc"))
        + f"   IQR: {c['verdict_iqr']} "
          f"({100 * c['share_seeds_all_three_in_iqr']:.0f}%)   mean: {c['verdict_mean']}")

print()
print("=" * 78)
print("SENSITIVITY: affect coupling at its default (lambda = 0.49)")
print("=" * 78)
print("Change in summed BIC from lambda = 0.49 to lambda = 0 (negative = better at 0):")
for m, v in c2["bic_vs_P13"].items():
    print(f"  {m} {NAMES[m]:30s} {v:8.1f}")
print(f"ranking at 0.49: {c2['P13_ranking']}")
print(f"ranking at 0:    {c2['bic_ranking']}")
print(f"reading: {c2['reading']}")

print("\nCalibrated values against the free fit at lambda = 0.49 (P15)")
print(f"  {p15['summary']}")

print("\nBehaviour over seeds at lambda = 0.49 (P17)")
for name, c in p17["conditions"].items():
    print(f"  {name:50s} " + " ".join(
        f"{k} {c[k]['mean']:.3f}" for k in ("ws", "ls", "acc") if k in c)
        + f"   IQR: {c.get('verdict_iqr')}   mean: {c.get('verdict_mean')}")

print("\nSplit rule (P16, check 1): prediction-error sign against outcome sign")
for m, d in c1["disagreement_at_P13_fits"].items():
    print(f"  {m}: rules disagree on {100 * d['share_trials_disagree']:.1f}% of trials")
print(f"  ranking with the outcome split: {c1['ranking_with_outcome_split']}")
print(f"  bound claim survives: {c1['bound_claim_survives']}")

print("\nMissing rows (P16, check 3)")
c3 = p16["check3_missing_rows"]
print(f"  files {c3['n_files']}, with missing rows {c3['n_files_with_missing_rows']}, "
      f"tables must be redone: {c3['tables_must_be_redone']}")
