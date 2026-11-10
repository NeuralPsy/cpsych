#!/usr/bin/env python3
"""Print every number of the transfer analysis from the stored result files.

Reads results/p10_model_comparison.json and results/p11_free_fit_behaviour.json
and prints the sample provenance, the free-fit parameters under each model
structure, the model comparison, and the behavioural check. Nothing is
recomputed and nothing is transcribed by hand: every figure comes straight out
of the two JSON files written by the experiment scripts.

Usage:
    python article/report_p10_p11.py [results_dir]
"""
import json
import os
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results"

CAL_ALPHA, CAL_BETA = 0.15, 2.50


def load(name):
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return json.load(f)


p10 = load("p10_model_comparison.json")
p11 = load("p11_free_fit_behaviour.json")
res = load("rl_mood_resolution.json")

per = p10["per_subject"]
models = p10["models"]
n = len(per["M1"])


def param(model, name):
    i = models[model]["param_names"].index(name)
    return np.array([f["params"][i] for f in per[model]])


def line(ch="-", w=78):
    print(ch * w)


# ---------------------------------------------------------------- sample
line("=")
print("SAMPLE")
line("=")
sp = p10["sample_provenance"]
print(f"  demographic records in archive   {sp['demographics_rows']}")
print(f"  age window                       "
      f"{sp['age_window'][0]:.0f}-{sp['age_window'][1]:.0f}")
print(f"  inside the window                {sp['n_in_age_window']}")
print(f"  with a usable choice file        {sp['n_with_usable_file']}")
print(f"  excluded as older than {sp['age_window'][1]:.0f}        "
      f"{[(e['id'], e['age']) for e in sp['excluded_above_age_max']]}")
tc = sp["trial_counts"]
print(f"  trials per participant           "
      f"{tc['min']}-{tc['max']}, median {tc['median']:.0f}")

# ------------------------------------------------- free fit, base module
print()
line("=")
print("FREE FIT, BASE MODULE (M1)")
line("=")
a1, b1 = param("M1", "alpha"), param("M1", "beta")
print(f"  alpha  median {np.median(a1):.3f}  "
      f"IQR [{np.percentile(a1, 25):.3f}, {np.percentile(a1, 75):.3f}]  "
      f"{np.median(a1) / CAL_ALPHA:.1f}x calibrated ({CAL_ALPHA})")
print(f"  beta   median {np.median(b1):.3f}  "
      f"IQR [{np.percentile(b1, 25):.3f}, {np.percentile(b1, 75):.3f}]  "
      f"{np.median(b1) / CAL_BETA:.1f}x calibrated ({CAL_BETA})")
spec = p10["reference"]["alpha_spec_max"]
print(f"  above the declared maximum {spec}  "
      f"{(a1 > spec).sum()} of {n}")
print(f"  at the optimiser's bound 0.99    {(a1 >= 0.99).sum()} of {n}")
mdd = abs(res["alpha"]["mdd"])
print(f"  displacement of alpha            {np.median(a1) - CAL_ALPHA:.3f}  "
      f"= {(np.median(a1) - CAL_ALPHA) / mdd:.1f}x the minimum detectable "
      f"difference ({mdd})")

# --------------------------------------- learning rate by model structure
print()
line("=")
print("LEARNING RATE REQUIRED, BY MODEL STRUCTURE")
line("=")
print(f"  {'model':34s} {'rate':10s} {'median':>8s} {'IQR':>18s} "
      f"{'x cal':>6s} {'>0.50':>8s} {'at .99':>7s}")
line()
LABEL = {
    "M1": "base module",
    "M2": "base plus persistence",
    "M3": "valence split",
    "M4": "valence split plus persistence",
    "M6": "counterfactual revision only",
    "M5": "published structure",
}
for m in ("M1", "M2", "M3", "M4", "M6", "M5"):
    if m not in per:
        continue
    for nm in models[m]["param_names"]:
        if not nm.startswith("alpha"):
            continue
        v = param(m, nm)
        print(f"  {LABEL[m]:34s} {nm:10s} {np.median(v):8.3f} "
              f"{'[' + format(np.percentile(v, 25), '.3f') + ', '
                 + format(np.percentile(v, 75), '.3f') + ']':>18s} "
              f"{np.median(v) / CAL_ALPHA:6.1f} "
              f"{str((v > spec).sum()) + '/' + str(n):>8s} "
              f"{(v >= 0.99).sum():7d}")

# ------------------------------------------------------ model comparison
print()
line("=")
print("MODEL COMPARISON")
line("=")
fr_all = p10["subject_level_best_model_counts_all"]
fr_reg = p10["subject_level_best_model_counts_registered"]
print(f"  {'model':34s} {'k':>2s} {'reg':>4s} {'BIC':>8s} {'dBIC':>6s} "
      f"{'best for':>9s}")
line()
for m in sorted(models, key=lambda k: models[k]["bic_sum"]):
    print(f"  {LABEL.get(m, m):34s} {models[m]['n_params']:2d} "
          f"{'yes' if models[m]['registered'] else 'no':>4s} "
          f"{models[m]['bic_sum']:8.0f} {models[m]['dbic']:6.0f} "
          f"{fr_all.get(m, 0):9d}")
print()
print(f"  winner among the registered four   "
      f"{p10['winner_within_registered']}  "
      f"(individual winners {dict(fr_reg)})")
print(f"  winner overall                     {p10['winner_overall']}")
print(f"  registered verdict                 {p10['verdict_registered']}")

for m in ("M2", "M4", "M5"):
    if m in per and "persistence" in models[m]["param_names"]:
        p = param(m, "persistence")
        print(f"  persistence, {LABEL[m]:32s} median {np.median(p):+.3f}, "
              f"negative for {100 * (p < 0).mean():.0f}% of participants")

# ---------------------------------------------------------- behaviour
print()
line("=")
print("BEHAVIOUR AT THE FITTED PARAMETERS")
line("=")
c = p11.get("control_vs_p6")
if c:
    print(f"  control against the earlier P6 run: "
          f"{'reproduced' if c['reproduces_p6'] else 'MISMATCH'} "
          f"(differences "
          f"{', '.join(f'{k} {v:.4f}' for k, v in c['abs_diff'].items())})")
d = p11["data"]
print(f"  n = {p11['n_subjects']}, {p11['n_trials']} simulated trials")
print()
print(f"  {'condition':46s} {'WS':>7s} {'LS':>7s} {'ACC':>7s}  in IQR")
print(f"  {'observed data':46s} {d['ws']:7.3f} {d['ls']:7.3f} {d['acc']:7.3f}")
for k, v in p11["conditions"].items():
    nq = p11["inside_observed_iqr"][k]["n_in_iqr"]
    print(f"  {k:46s} {v['ws']:7.3f} {v['ls']:7.3f} {v['acc']:7.3f}  {nq}/3")
print()
print(f"  observed IQR   win-stay [{d['ws_iqr'][0]:.3f}, {d['ws_iqr'][1]:.3f}]"
      f"   lose-shift [{d['ls_iqr'][0]:.3f}, {d['ls_iqr'][1]:.3f}]"
      f"   accuracy [{d['acc_iqr'][0]:.3f}, {d['acc_iqr'][1]:.3f}]")
print()
for note in p11.get("known_protocol_deviations", []):
    print(f"  protocol deviation: {note}")
