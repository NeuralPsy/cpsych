#!/usr/bin/env python3
"""Print every number of the transfer analysis that comes from P12, P13 and P14.

P12  calibrated values held under each of the six structures
P13  the six structures refitted with wider optimiser bounds
P14  the behavioural check (stage 3) at the P13 fits

P13 and P14 supersede P10 and P11 as the source of the reported fits and of
the behavioural check; report_p10_p11.py still prints the earlier ones, which
stay in the archive. Nothing here is recomputed from data and nothing is
transcribed by hand: every figure is read from the three JSON files written
by the experiment scripts, plus P10's JSON for the before/after comparison.

Usage:
    python article/report_p12_p14.py [results_dir]
"""
import json
import os
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results"
CAL_ALPHA, CAL_BETA = 0.15, 2.50
SPEC_MAX = 0.50
LABEL = {"M1": "base module", "M2": "base + persistence",
         "M3": "valence split", "M4": "valence split + persistence",
         "M5": "published structure", "M6": "counterfactual only"}


def load(name):
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return json.load(f)


def line(ch="-", w=86):
    print(ch * w)


def head(title):
    print()
    line("=")
    print(title)
    line("=")


p10 = load("p10_model_comparison.json")
p12 = load("p12_calibrated_all_structures.json")
p13 = load("p13_bounds_sensitivity.json")
p14 = load("p14_free_fit_behaviour_wide.json")

per = p13["per_subject_wide"]
n = p13["n_subjects"]


def names(m):
    return p13["models"][m]["wide_bounds"]["param_names"]


def param(m, name):
    i = names(m).index(name)
    return np.array([f["params"][i] for f in per[m]])


# ------------------------------------------------------------------ P12
head("P12  CALIBRATED VALUES UNDER EACH STRUCTURE  "
     f"(alpha = {CAL_ALPHA}, beta = {CAL_BETA}; persistence fitted where present)")
d = p12["data"]
print(f"  observed            ws {d['ws']:.3f}  ls {d['ls']:.3f}  acc {d['acc']:.3f}")
print(f"  observed IQR        ws {d['ws_iqr'][0]:.3f}-{d['ws_iqr'][1]:.3f}  "
      f"ls {d['ls_iqr'][0]:.3f}-{d['ls_iqr'][1]:.3f}  "
      f"acc {d['acc_iqr'][0]:.3f}-{d['acc_iqr'][1]:.3f}")
line()
print(f"  {'structure':30s} {'ws':>6s} {'ls':>6s} {'acc':>6s} {'in IQR':>7s} "
      f"{'dBIC':>8s} {'free better':>12s} {'by >10':>7s}")
for m, e in p12["structures"].items():
    b = e["behaviour"]
    same = f"  (= {e['same_as']})" if e["same_as"] else ""
    print(f"  {m + ' ' + LABEL[m]:30s} {b['ws']:6.3f} {b['ls']:6.3f} "
          f"{b['acc']:6.3f} {b['n_in_iqr']:>5d}/3 {e['dbic']:8.0f} "
          f"{e['n_free_better']:>8d}/{n} {e['n_free_better_by_10']:>7d}{same}")
line()
print(f"  distinct conditions              {p12['distinct_conditions']}")
ic = p12["identity_check"]
print(f"  M3 = M1 and M4 = M2 in likelihood  {ic['holds']}")
print(f"  control vs P11 condition A       "
      f"{p12['control_vs_p11']['reproduces_p11']}")
for m, e in p12["structures"].items():
    if "persistence" in e and not e["same_as"]:
        p = e["persistence"]
        print(f"  {m} persistence at clamped alpha, beta: median {p['median']:+.3f}, "
              f"positive for {p['n_positive']}/{n}")
print(f"  VERDICT: {p12['verdict']}")

# ------------------------------------------------------------------ P13
head("P13  FREE FITS WITH WIDER BOUNDS")
bw, bo = p13["bounds"]["wide"], p13["bounds"]["p10"]
print(f"  bounds, P10 -> P13   beta {bo['beta']} -> {bw['beta']}   "
      f"persistence {bo['persistence']} -> {bw['persistence']}   "
      f"alpha {bw['alpha']} (unchanged)")
line()
print(f"  {'structure':30s} {'rate':>10s} {'median [IQR]':>24s} {'x cal':>6s} "
      f"{'> 0.50':>7s} {'at 0.99':>8s}")
for m in ("M1", "M2", "M3", "M4", "M6", "M5"):
    for nm in names(m):
        if not nm.startswith("alpha"):
            continue
        x = param(m, nm)
        q = np.percentile(x, [25, 50, 75])
        print(f"  {m + ' ' + LABEL[m]:30s} {nm:>10s} "
              f"{q[1]:8.3f} [{q[0]:.3f}, {q[2]:.3f}] {q[1] / CAL_ALPHA:6.1f} "
              f"{int((x > SPEC_MAX).sum()):>7d} {int((x >= 0.99).sum()):>8d}")
line()
ap, an = param("M5", "alpha_pos"), param("M5", "alpha_neg")
print(f"  M5: at least one rate above {SPEC_MAX}      "
      f"{int(((ap > SPEC_MAX) | (an > SPEC_MAX)).sum())}/{n}")
for m in ("M3", "M4", "M5"):
    print(f"  {m}: alpha_pos at 0.99                 "
          f"{100 * float((param(m, 'alpha_pos') >= 0.99).mean()):.0f} %")
a1 = param("M1", "alpha")
print(f"  M1: alpha displacement              {np.median(a1) - CAL_ALPHA:.3f}")
print(f"  M5: alpha_pos displacement          {np.median(ap) - CAL_ALPHA:.3f}")
line()
print(f"  {'structure':30s} {'beta median [IQR]':>28s} {'x cal':>6s} {'> 20':>6s}")
for m in ("M1", "M2", "M3", "M4", "M6", "M5"):
    x = param(m, "beta")
    q = np.percentile(x, [25, 50, 75])
    print(f"  {m + ' ' + LABEL[m]:30s} {q[1]:10.3f} [{q[0]:.3f}, {q[2]:.3f}] "
          f"{q[1] / CAL_BETA:6.1f} {int((x > 20).sum()):>6d}")
line()
for m in ("M2", "M4", "M5"):
    x = param(m, "persistence")
    q = np.percentile(x, [25, 50, 75])
    print(f"  {m} persistence: median {q[1]:+.3f} [{q[0]:+.3f}, {q[2]:+.3f}]  "
          f"negative {100 * float((x < 0).mean()):.0f} %  "
          f"positive {100 * float((x > 0).mean()):.0f} %  "
          f"beyond 2: {int((np.abs(x) > 2).sum())}")

head("P13  MODEL COMPARISON")
c = p13["comparison"]["wide_bounds"]
print(f"  {'structure':30s} {'BIC':>8s} {'dBIC':>6s} {'best for':>9s} "
      f"{'BIC gain vs P10':>16s}")
for m in c["bic_ranking"]:
    w = p13["models"][m]["wide_bounds"]["bic_sum"]
    gain = p10["models"][m]["bic_sum"] - w
    print(f"  {m + ' ' + LABEL[m]:30s} {w:8.0f} {c['dbic'][m]:6.0f} "
          f"{c['subject_winners_all'][m]:>9d} {gain:16.1f}")
line()
print(f"  winner overall / among registered   {c['winner_overall']} / "
      f"{c['winner_registered']}")
print(f"  M1 behind the registered winner     "
      f"{c['dbic_M1_within_registered']:.0f}")
print(f"  individual winners, registered set  {c['subject_winners_registered']}")
print(f"  M5 better than M2                   {c['M5_better_than_M2']}/{n}"
      f"   by >10: {c['M5_better_than_M2_by_10']}"
      f"   M2 better by >10: {c['M2_better_than_M5_by_10']}")

head("P13  PRE-SPECIFIED READING")
r = p13["reading"]
print(f"  (a) winners unchanged                    {r['a_winners_unchanged']}")
print(f"  (b) alpha medians within "
      f"{r['thresholds']['alpha_median_shift_max']}            "
      f"{r['b_alpha_medians_stable']}   largest shift "
      f"{max(r['b_alpha_median_abs_shift'].values()):.3f}")
print(f"  (c) persistence sign shares within "
      f"{r['thresholds']['sign_share_shift_max_pct']:.0f} pts  "
      f"{r['c_persistence_sign_stable']}")
print(f"  (d) beta moves less than some alpha      {r['d_holds_everywhere']}")
print(f"  robust to bounds                         {r['robust_to_bounds']}")

# ------------------------------------------------------------------ P14
head("P14  BEHAVIOUR AT THE P13 FITS")
d = p14["data"]
print(f"  {'condition':48s} {'ws':>6s} {'ls':>6s} {'acc':>6s} {'in IQR':>7s}")
print(f"  {'observed':48s} {d['ws']:6.3f} {d['ls']:6.3f} {d['acc']:6.3f}")
for k, v in p14["conditions"].items():
    print(f"  {k:48s} {v['ws']:6.3f} {v['ls']:6.3f} {v['acc']:6.3f} "
          f"{p14['inside_observed_iqr'][k]['n_in_iqr']:>5d}/3")
line()
for k, v in p14["distance_outside_iqr"].items():
    out = {a: round(b, 3) for a, b in v.items() if b > 0}
    if out and k != "A_calibrated":
        print(f"  {k}: outside the observed range by {out}")
print(f"  control vs P11 condition A          "
      f"{p14['control_vs_p11']['reproduces_p11']}")
print(f"  CONCLUSION: {p14['conclusion']}")
