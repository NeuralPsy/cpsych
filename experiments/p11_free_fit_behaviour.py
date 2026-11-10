"""P11: does the model reproduce behaviour at its own free-fit parameters?

Supersedes P9. Same two corrections as P10:
  1. Sample restricted to the source study's own adult window, 18 <= age <= 30
     (the source excluded two participants older than 30; both were in P9).
  2. A condition added at the fitted parameters of M5, the published winning
     model for this task (alpha+, alpha-, beta, perseverance, with
     counterfactual updating of the unchosen option at tied rates).

BIC says which model fits the choice sequence best. It does not say whether
any of them reproduces behaviour. This script asks that directly: simulate at
the free-fit parameters and compare win-stay, lose-shift and accuracy with the
data.

Conditions (all on the Eckstein protocol, 120 trials):
  CONTROL  calibrated parameters at B=114     [must reproduce P6 exactly]
  A  calibrated parameters                    alpha=0.15, beta=2.50
  B  free-fit medians (M1)                    alpha, beta from P10 M1 medians
  C  per-subject free fits (M1)               each subject's own alpha, beta
  D  per-subject free fits (M2)               plus that subject's persistence
  E  per-subject free fits (M5)               published structure

CONTROL re-runs P9's condition A at P9's own B (114) and seed, so that a
disagreement with results/p6_rl_comparison.json can be distinguished from the
effect of the corrected sample. It is reported first and explicitly. If it
fails, nothing else here is trustworthy.

Two known deviations of the simulated protocol from the published one, both
disclosed rather than corrected (correcting them would invalidate the P6
control):
  - the source rewarded the first correct choice after every switch with
    certainty; this environment does not.
  - the archive's choice sequences run 124-131 trials while the published task
    length is 120; the fits in P10 use the archive sequences as they are, and
    the simulations here use 120 trials.

Rule 1: every number lands in results/p11_free_fit_behaviour.json.

Usage:  python experiments/p11_free_fit_behaviour.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpsych.modules.rl_mood import RLMoodModule
from cpsych.environments.reversal_bandit import ReversalBandit

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eckstein2022")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

TAU_M, LAM = 2.163, 0.49
N_TRIALS = 120
ALPHA_CAL, BETA_CAL = 0.15, 2.50
AGE_MIN, AGE_MAX = 18.0, 30.0
P9_B = 114          # P9's sample size, for the control only


def load_adults(age_min=AGE_MIN, age_max=AGE_MAX):
    demo = pd.read_csv(os.path.join(DATA_DIR,
                                    "EcksteinEtAl_eLife2022_Demographics.csv"))
    age = demo["PreciseYrs"]
    ids = demo[(age >= age_min) & (age <= age_max)]["ID"].values
    out = {}
    for sid in ids:
        p = os.path.join(DATA_DIR, f"PS_{sid}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            if len(df) >= 50:
                out[int(sid)] = df
    return out


def obs_from_sequences(choices, rewards, correct, n):
    """win-stay, lose-shift, accuracy for one sequence (same rule as P6)."""
    nw = ns = nwst = nlsh = 0
    for t in range(1, n):
        if rewards[t - 1] == 1:
            nw += 1
            if choices[t] == choices[t - 1]:
                nwst += 1
        else:
            ns += 1
            if choices[t] != choices[t - 1]:
                nlsh += 1
    return (nwst / nw if nw else np.nan,
            nlsh / ns if ns else np.nan,
            float(np.mean(correct)))


def empirical(data):
    ws, ls, acc = [], [], []
    for df in data.values():
        n = len(df)
        a, b, c = obs_from_sequences(df["selected_box"].values,
                                     df["reward"].values,
                                     df["ACC"].values.astype(float), n)
        ws.append(a); ls.append(b); acc.append(c)
    return np.array(ws), np.array(ls), np.array(acc)


def _as_vec(v, B, name):
    v = np.atleast_1d(np.asarray(v, dtype=float))
    if v.size == 1:
        v = np.repeat(v, B)
    assert v.size == B, f"{name} length {v.size} != B {B}"
    return v


def simulate(alpha_pos, beta, B, alpha_neg=None, persistence=None,
             counterfactual=False, seed=42):
    """Simulate B agents on the Eckstein protocol.

    alpha_pos/alpha_neg/beta/persistence: scalar or per-subject array.
    B is always explicit, so a scalar run cannot silently simulate one agent.
    alpha_neg None  -> single learning rate (alpha_pos used throughout).
    counterfactual  -> also update the unchosen arm toward the inverse
                       outcome, at the rate for that inverse outcome's
                       valence, as in the published model.

    Dynamics come from RLMoodModule.trial_update. For a valence split, the
    module's alpha slot is set per trial to the rate the outcome calls for, so
    the module stays the single source of the update rule.
    """
    a_pos = _as_vec(alpha_pos, B, "alpha_pos")
    a_neg = a_pos.copy() if alpha_neg is None else _as_vec(alpha_neg, B, "alpha_neg")
    bet = _as_vec(beta, B, "beta")
    persist = (np.zeros(B) if persistence is None
               else _as_vec(persistence, B, "persistence"))
    use_persist = bool(np.any(persist != 0.0))

    rng = np.random.default_rng(seed)
    mod = RLMoodModule()
    env = ReversalBandit(p_reward=0.75, p_incorrect=0.0,
                         reversal_mode="criterion", criterion_range=(7, 15))
    env.reset(B, rng)

    x = mod.default_state(B)
    theta = mod.default_params(B)
    theta[:, 1] = bet
    theta[:, 2] = TAU_M
    theta[:, 3] = LAM
    mask = np.ones(B, dtype=bool)

    ch = np.zeros((N_TRIALS, B))
    rw = np.zeros((N_TRIALS, B))
    cr = np.zeros((N_TRIALS, B))
    prev_choice = np.full(B, 0.5)

    for t in range(N_TRIALS):
        env.next_trial(rng, mask)
        if use_persist:
            V0, V1 = x[:, 0], x[:, 1]
            logit = bet * (V1 - V0) + persist * (2.0 * prev_choice - 1.0)
            p1 = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))
        else:
            p1 = mod.outputs(x, theta)["choice_prob"]
        choice = (rng.random(B) < p1).astype(float)
        cr[t] = choice == env._correct_arm
        r = env.reward(choice, rng)["r"]
        ch[t] = choice
        rw[t] = r

        rho_eff = 1.0 + LAM * x[:, 2]          # before the update
        theta[:, 0] = np.where(r > 0.5, a_pos, a_neg)
        x = mod.trial_update(x, theta, mask, {"r": r, "choice": choice})

        if counterfactual:
            r_cf = 1.0 - r
            rate_cf = np.where(r_cf > 0.5, a_pos, a_neg)
            unchosen = (1.0 - choice).astype(int)
            V_un = x[np.arange(B), unchosen]
            x[np.arange(B), unchosen] = V_un + rate_cf * (rho_eff * r_cf - V_un)

        prev_choice = choice

    ws, ls, acc = [], [], []
    for b in range(B):
        a, c, d = obs_from_sequences(ch[:, b], rw[:, b], cr[:, b], N_TRIALS)
        ws.append(a); ls.append(c); acc.append(d)
    return (float(np.nanmean(ws)), float(np.nanmean(ls)), float(np.nanmean(acc)))


def main():
    data = load_adults()
    B = len(data)
    print(f"Adults (age {AGE_MIN}-{AGE_MAX}, usable file): {B}")

    e_ws, e_ls, e_acc = empirical(data)
    emp = {"ws": float(np.nanmean(e_ws)),
           "ls": float(np.nanmean(e_ls)),
           "acc": float(np.nanmean(e_acc)),
           "ws_iqr": [float(np.nanpercentile(e_ws, 25)), float(np.nanpercentile(e_ws, 75))],
           "ls_iqr": [float(np.nanpercentile(e_ls, 25)), float(np.nanpercentile(e_ls, 75))],
           "acc_iqr": [float(np.nanpercentile(e_acc, 25)), float(np.nanpercentile(e_acc, 75))]}

    p10_path = os.path.join(RESULTS_DIR, "p10_model_comparison.json")
    with open(p10_path) as f:
        p10 = json.load(f)
    assert p10["sample_provenance"]["n_with_usable_file"] == B, (
        "P10 and P11 disagree on the sample size")
    m1 = np.array([s["params"] for s in p10["per_subject"]["M1"]])
    m2 = np.array([s["params"] for s in p10["per_subject"]["M2"]])
    m5 = np.array([s["params"] for s in p10["per_subject"]["M5"]])
    a_med, b_med = float(np.median(m1[:, 0])), float(np.median(m1[:, 1]))

    conds = {}
    conds["A_calibrated"] = simulate(ALPHA_CAL, BETA_CAL, B, seed=42)
    conds["B_freefit_median"] = simulate(a_med, b_med, B, seed=43)
    conds["C_freefit_per_subject"] = simulate(m1[:, 0], m1[:, 1], B, seed=44)
    conds["D_freefit_per_subject_persistence"] = simulate(
        m2[:, 0], m2[:, 1], B, persistence=m2[:, 2], seed=45)
    conds["E_freefit_per_subject_published_structure"] = simulate(
        m5[:, 0], m5[:, 2], B, alpha_neg=m5[:, 1], persistence=m5[:, 3],
        counterfactual=True, seed=46)

    # control: does the calibrated condition still reproduce P6, at P9's B?
    p6_path = os.path.join(RESULTS_DIR, "p6_rl_comparison.json")
    control = None
    if os.path.exists(p6_path):
        with open(p6_path) as f:
            p6 = json.load(f)
        ref = p6["model"]
        got = simulate(ALPHA_CAL, BETA_CAL, P9_B, seed=42)
        control = {
            "note": f"calibrated parameters at B={P9_B} and seed 42, matching "
                    f"P6/P9 exactly, so that agreement with P6 is tested "
                    f"independently of the corrected sample",
            "p6_model": {"ws": ref["ws"], "ls": ref["ls"], "acc": ref["acc"]},
            "p11_control": {"ws": got[0], "ls": got[1], "acc": got[2]},
            "abs_diff": {"ws": abs(ref["ws"] - got[0]),
                         "ls": abs(ref["ls"] - got[1]),
                         "acc": abs(ref["acc"] - got[2])},
        }
        control["reproduces_p6"] = all(v < 0.02 for v in control["abs_diff"].values())

    print()
    if control:
        ok = "OK" if control["reproduces_p6"] else "MISMATCH"
        print(f"Control (calibrated at B={P9_B} vs P6): {ok}  "
              f"diffs={ {k: round(v,4) for k,v in control['abs_diff'].items()} }")
    print()
    print(f"{'condition':46s} {'win-stay':>9s} {'lose-shift':>11s} {'accuracy':>9s}")
    print(f"{'DATA':46s} {emp['ws']:9.3f} {emp['ls']:11.3f} {emp['acc']:9.3f}")
    for k, (ws, ls, acc) in conds.items():
        print(f"{k:46s} {ws:9.3f} {ls:11.3f} {acc:9.3f}")

    verdicts = {}
    for k, (ws, ls, acc) in conds.items():
        verdicts[k] = {
            "ws_in_iqr": bool(emp["ws_iqr"][0] <= ws <= emp["ws_iqr"][1]),
            "ls_in_iqr": bool(emp["ls_iqr"][0] <= ls <= emp["ls_iqr"][1]),
            "acc_in_iqr": bool(emp["acc_iqr"][0] <= acc <= emp["acc_iqr"][1]),
        }
        verdicts[k]["n_in_iqr"] = sum(verdicts[k][x] for x in
                                      ("ws_in_iqr", "ls_in_iqr", "acc_in_iqr"))

    print()
    for k, v in verdicts.items():
        print(f"  {k:46s} inside observed IQR on {v['n_in_iqr']}/3 observables")

    best = max(verdicts, key=lambda k: verdicts[k]["n_in_iqr"])
    conclusion = ("free-fit parameters still fail to reproduce behaviour"
                  if verdicts[best]["n_in_iqr"] < 3
                  else "free-fit parameters reproduce behaviour")
    print()
    print(f"CONCLUSION: {conclusion}")

    result = {
        "supersedes": "p9_free_fit_behaviour.json",
        "procedure": "simulate the RL module on the Eckstein protocol at "
                     "calibrated and free-fit parameters; compare win-stay, "
                     "lose-shift and accuracy with the observed data",
        "n_subjects": B,
        "age_window": [AGE_MIN, AGE_MAX],
        "n_trials": N_TRIALS,
        "env": {"mode": "criterion", "p_reward": 0.75, "p_incorrect": 0.0,
                "criterion_rewards": [7, 15]},
        "known_protocol_deviations": [
            "the source rewarded the first correct choice after a switch with "
            "certainty; this environment does not",
            "archive choice sequences run 124-131 trials, the published task "
            "length is 120; simulations use 120",
        ],
        "fixed": {"tau_m": TAU_M, "lam": LAM},
        "free_fit_medians_M1": {"alpha": a_med, "beta": b_med},
        "data": emp,
        "conditions": {k: {"ws": v[0], "ls": v[1], "acc": v[2]}
                       for k, v in conds.items()},
        "inside_observed_iqr": verdicts,
        "control_vs_p6": control,
        "conclusion": conclusion,
    }
    out = os.path.join(RESULTS_DIR, "p11_free_fit_behaviour.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
