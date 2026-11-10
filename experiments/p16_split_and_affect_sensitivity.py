"""P16: three checks raised by independent review of P10-P14.

CHECK 1 - the valence split in M3 and M4.
  P10 splits the learning rate in M3/M4 on the SIGN OF THE PREDICTION ERROR
  and in M5/M6 on the OUTCOME (rewarded or not). The manuscript said the two
  rules "almost always agree". That holds only while reward is effectively
  binary. Here it is not: the affect loop scales reward by 1 + lambda*m, so a
  rewarded trial can carry a negative prediction error. At learning rates
  near 1, where M3/M4 land, a reviewer measured disagreement on roughly one
  trial in six on synthetic sequences. If so, the "alpha+" of M3/M4 is not a
  rate for rewarded trials and must not be compared with the alpha+ of M5.
  This script:
    (a) counts, on the participants' own sequences at the P13 fitted values,
        how often the two rules disagree;
    (b) refits M3 and M4 with the OUTCOME split (called M3o, M4o), same
        bounds and starts as P13, so that all valence-split structures use
        one definition.

CHECK 2 - the affect loop inside every fitted structure.
  All six structures carry the platform's fixed affect loop (lambda = 0.49,
  tau_m = 2.163). The published model has none, so "the published model
  exactly as specified" was not exact. This script refits all six structures
  with lambda = 0 (plain reinforcement learning). With lambda = 0 the two
  split rules coincide, so this family is also free of the problem in check 1.

CHECK 3 - missing trials in the data files.
  The fitting loader (P10) drops rows with a missing choice or reward. The
  function that computes observed behaviour (P11) does not. If any file has
  missing rows, observed win-stay, lose-shift and accuracy were computed on
  different trials from the fits. This script counts missing rows per file
  and recomputes the observed values both ways.

Nothing here overwrites an earlier result.

READING, fixed before this script was first run.
  Check 1: the manuscript's statement "without counterfactual revision the
    positive learning rate reached the bound for 70 per cent of participants"
    survives only if, under the outcome split, alpha+ is at the 0.99 bound
    for at least 50 per cent of participants in BOTH M3o and M4o. Otherwise
    it is withdrawn and the M3/M4 rows are reported under the outcome split
    with whatever they show.
  Check 2: results are "robust to the affect loop" only if, against P13,
    (a) the BIC winner is unchanged,
    (b) the M1 alpha median and both M5 alpha medians each move by < 0.05,
    (c) the share negative in M2 persistence and the share positive in M5
        persistence each move by < 5 percentage points.
    If any fails, the lambda = 0 values are reported beside the main ones
    and the dependence is stated.
  Check 3: if any analysed file has a missing row, the observed values are
    recomputed on clean rows and the difference reported; a change of more
    than 0.001 in any observed mean or quartile means the behavioural tables
    must be redone on clean rows.

UNREGISTERED addition, made after two independent reviews.

Rule 1: every number lands in results/p16_split_and_affect_sensitivity.json.

Usage:  python experiments/p16_split_and_affect_sensitivity.py [--smoke N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import p10_model_comparison as p10          # noqa: E402
import p11_free_fit_behaviour as p11        # noqa: E402
import p13_bounds_sensitivity as p13        # noqa: E402

RESULTS_DIR = os.path.join(HERE, "..", "results")
ALPHA_CAL, BETA_CAL = 0.15, 2.50
READING = {"alpha_shift_max": 0.05, "sign_share_shift_max_pct": 5.0,
           "bound_share_min": 0.50, "observed_shift_max": 0.001}

# the split rule P10 uses for each structure
P10_SPLIT = {"M1": "pe", "M2": "pe", "M3": "pe", "M4": "pe",
             "M5": "outcome", "M6": "outcome"}


def nll(params, choice, reward, model, lam, split, trace=None):
    """P10's likelihood with the affect coupling and the split rule exposed.

    lam = p10.LAM and split = P10_SPLIT[model] reproduce p10.neg_loglik
    exactly; main() asserts this before anything is fitted.
    trace, if a dict, collects per-trial diagnostics for check 1.
    """
    a_pos, a_neg, beta, persist, counterfactual, _ = p10._unpack(params, model)
    V = np.array([0.5, 0.5])
    m = 0.0
    prev = -1
    ll = 0.0
    n_dis = n_rew = n_rew_neg = 0
    for t in range(choice.size):
        logit = beta * (V[1] - V[0])
        if persist and prev >= 0:
            logit += persist * (2.0 * prev - 1.0)
        logit = np.clip(logit, -50.0, 50.0)
        p1 = 1.0 / (1.0 + np.exp(-logit))
        p = p1 if choice[t] == 1 else 1.0 - p1
        ll += np.log(max(p, 1e-12))

        r = reward[t]
        c = choice[t]
        rho_eff = 1.0 + lam * m
        delta = rho_eff * r - V[c]
        by_outcome = r > 0.5
        by_pe = delta >= 0
        if trace is not None:
            n_dis += int(by_outcome != by_pe)
            n_rew += int(by_outcome)
            n_rew_neg += int(by_outcome and not by_pe)
        rate = a_pos if (by_outcome if split == "outcome" else by_pe) else a_neg
        V[c] += rate * delta

        if counterfactual:
            u = 1 - c
            r_cf = 1.0 - r
            delta_cf = rho_eff * r_cf - V[u]
            rate_cf = a_pos if r_cf > 0.5 else a_neg
            V[u] += rate_cf * delta_cf

        m = p10.GAMMA * (m + delta)
        prev = c
    if trace is not None:
        trace.update(n_trials=int(choice.size), n_disagree=n_dis,
                     n_rewarded=n_rew, n_rewarded_negative_pe=n_rew_neg)
    return -ll


def self_test(subjects):
    """The generalised likelihood must equal P10's at P10's settings."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for s in subjects[:5]:
        for model in p10.ALL_MODELS:
            for x0 in p10.starts(model, rng, n=3):
                a = p10.neg_loglik(x0, s["choice"], s["reward"], model)
                b = nll(x0, s["choice"], s["reward"], model, p10.LAM,
                        P10_SPLIT[model])
                worst = max(worst, abs(a - b))
    return worst


def fit(subj, model, rng, warm, lam, split):
    bnds = p13.bounds_new(model)
    best = None
    for x0 in p10.starts(model, rng) + [list(warm)]:
        x0 = [min(max(v, lo), hi) for v, (lo, hi) in zip(x0, bnds)]
        try:
            res = minimize(nll, x0, args=(subj["choice"], subj["reward"],
                                          model, lam, split),
                           method="L-BFGS-B", bounds=bnds)
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:                       # every start failed: keep the warm start
        x = [min(max(v, lo), hi) for v, (lo, hi) in zip(warm, bnds)]
        fun = float(nll(x, subj["choice"], subj["reward"], model, lam, split))
    else:
        x, fun = [float(v) for v in best.x], float(best.fun)
    n = subj["choice"].size
    return {"id": subj["id"], "params": x, "nll": fun, "n_trials": int(n),
            "bic": float(p10.N_PARAMS[model] * np.log(n) + 2.0 * fun)}


def fit_family(subjects, models, lam, split_of, warm_from, seed, label):
    rng = np.random.default_rng(seed)
    per, summ = {}, {}
    for model in models:
        warm = {f["id"]: f["params"] for f in warm_from[model]}
        fits = []
        for k, s in enumerate(subjects):
            fits.append(fit(s, model, rng, warm[s["id"]], lam, split_of[model]))
            if (k + 1) % 20 == 0:
                print(f"  [{label}] {model}: {k+1}/{len(subjects)}")
        per[model] = fits
        summ[model] = p13.summarise(fits, model, p13.NEW)
        summ[model]["split"] = split_of[model]
        summ[model]["lambda"] = lam
        print(f"  [{label}] {model}: BIC {summ[model]['bic_sum']:.1f}  medians "
              f"{[round(v, 3) for v in summ[model]['param_median']]}")
    return per, summ


def share_at_upper(fits, model, name):
    i = p10.PARAM_NAMES[model].index(name)
    x = np.array([f["params"][i] for f in fits])
    hi = p13.NEW[p13.kind(name)][1]
    return float(np.mean(x >= hi - p13.TOL))


def check3_missing(ids):
    """Missing rows per analysed file, and observed behaviour both ways."""
    files, raw_obs, clean_obs = [], [], []
    for sid in ids:
        df = pd.read_csv(os.path.join(p10.DATA_DIR, f"PS_{sid}.csv"))
        ch = pd.to_numeric(df["selected_box"], errors="coerce")
        rw = pd.to_numeric(df["reward"], errors="coerce")
        ac = pd.to_numeric(df["ACC"], errors="coerce")
        bad = ch.isna() | rw.isna()
        codes = sorted(set(ch.dropna().astype(int).tolist()))
        files.append({"id": int(sid), "n_rows": int(len(df)),
                      "n_missing_choice": int(ch.isna().sum()),
                      "n_missing_reward": int(rw.isna().sum()),
                      "n_missing_acc": int(ac.isna().sum()),
                      "n_rows_dropped_by_fit_loader": int(bad.sum()),
                      "choice_codes": codes})
        raw_obs.append(p11.obs_from_sequences(
            df["selected_box"].values, df["reward"].values,
            df["ACC"].values.astype(float), len(df)))
        ok = ~(bad | ac.isna())
        clean_obs.append(p11.obs_from_sequences(
            ch[ok].values, rw[ok].values, ac[ok].values.astype(float),
            int(ok.sum())))

    def stats(obs):
        a = np.array(obs, dtype=float)
        out = {}
        for j, nm in enumerate(("ws", "ls", "acc")):
            out[nm] = float(np.nanmean(a[:, j]))
            out[nm + "_iqr"] = [float(np.nanpercentile(a[:, j], 25)),
                                float(np.nanpercentile(a[:, j], 75))]
            out[nm + "_n_nan"] = int(np.isnan(a[:, j]).sum())
        return out

    raw, clean = stats(raw_obs), stats(clean_obs)
    diffs = []
    for k in ("ws", "ls", "acc"):
        diffs.append(abs(raw[k] - clean[k]))
        diffs += [abs(a - b) for a, b in zip(raw[k + "_iqr"], clean[k + "_iqr"])]
    n_any = sum(1 for f in files if f["n_rows_dropped_by_fit_loader"]
                or f["n_missing_acc"])
    odd = [f["id"] for f in files if f["choice_codes"] not in ([0, 1], [0], [1])]
    return {
        "n_files": len(files),
        "n_files_with_missing_rows": n_any,
        "total_rows_dropped_by_fit_loader": int(
            sum(f["n_rows_dropped_by_fit_loader"] for f in files)),
        "files_with_choice_codes_other_than_0_1": odd,
        "observed_raw_rows": raw,
        "observed_clean_rows": clean,
        "max_abs_difference": float(max(diffs)) if diffs else 0.0,
        "tables_must_be_redone": bool(
            diffs and max(diffs) > READING["observed_shift_max"]),
        "files_with_missing": [f for f in files
                               if f["n_rows_dropped_by_fit_loader"]
                               or f["n_missing_acc"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    subjects, prov = p10.load_adults(limit=args.smoke or None)
    ids = [s["id"] for s in subjects]
    print(f"Subjects: {len(subjects)}"
          + ("  [SMOKE RUN - not a result]" if args.smoke else ""))

    with open(os.path.join(RESULTS_DIR, "p13_bounds_sensitivity.json")) as f:
        p13j = json.load(f)
    assert not p13j["smoke_run"]
    wide = {m: [f for f in p13j["per_subject_wide"][m] if f["id"] in set(ids)]
            for m in p10.ALL_MODELS}
    for m in wide:
        assert [f["id"] for f in wide[m]] == ids, f"{m}: order differs from P13"
    if not args.smoke:
        assert p13j["n_subjects"] == len(ids)

    worst = self_test(subjects)
    print(f"Self-test against p10.neg_loglik: max abs difference {worst:.2e}")
    assert worst < 1e-9, "generalised likelihood does not reproduce P10's"

    # ------------------------------------------------ check 1a: disagreement
    print("\nCheck 1a: how often the two split rules disagree (real sequences)")
    dis = {}
    for model in ("M3", "M4"):
        tot = {"n_trials": 0, "n_disagree": 0, "n_rewarded": 0,
               "n_rewarded_negative_pe": 0}
        per_subj = []
        for s, f in zip(subjects, wide[model]):
            tr = {}
            nll(f["params"], s["choice"], s["reward"], model, p10.LAM, "pe", tr)
            for k in tot:
                tot[k] += tr[k]
            per_subj.append(tr["n_disagree"] / tr["n_trials"])
        dis[model] = {
            **tot,
            "share_trials_disagree": tot["n_disagree"] / tot["n_trials"],
            "share_rewarded_with_negative_pe":
                tot["n_rewarded_negative_pe"] / max(tot["n_rewarded"], 1),
            "per_subject_share_median": float(np.median(per_subj)),
            "per_subject_share_iqr": [float(np.percentile(per_subj, 25)),
                                      float(np.percentile(per_subj, 75))],
            "n_subjects_above_5pct": int(np.sum(np.array(per_subj) > 0.05)),
        }
        print(f"  {model}: rules disagree on "
              f"{100*dis[model]['share_trials_disagree']:.1f}% of trials; "
              f"{100*dis[model]['share_rewarded_with_negative_pe']:.1f}% of "
              f"rewarded trials have a negative prediction error")

    # ------------------------------------------------ check 1b: outcome split
    print("\nCheck 1b: M3 and M4 refitted with the outcome split")
    split_o = {"M3": "outcome", "M4": "outcome"}
    per_o, sum_o = fit_family(subjects, ["M3", "M4"], p10.LAM, split_o, wide,
                              args.seed, "outcome split")
    c1 = {}
    for model in ("M3", "M4"):
        old = p13.summarise(wide[model], model, p13.NEW)
        c1[model + "o"] = {
            "pe_split_P13": old,
            "outcome_split": sum_o[model],
            "share_alpha_pos_at_bound_pe": share_at_upper(wide[model], model,
                                                          "alpha_pos"),
            "share_alpha_pos_at_bound_outcome": share_at_upper(per_o[model],
                                                               model, "alpha_pos"),
            "bic_change": sum_o[model]["bic_sum"] - old["bic_sum"],
        }
    stands = all(c1[k]["share_alpha_pos_at_bound_outcome"]
                 >= READING["bound_share_min"] for k in c1)
    # ranking of the six with M3o, M4o in place of M3, M4
    bic6 = {m: p13j["models"][m]["wide_bounds"]["bic_sum"] for m in p10.ALL_MODELS}
    bic6o = dict(bic6, M3=sum_o["M3"]["bic_sum"], M4=sum_o["M4"]["bic_sum"])
    per6o = {m: {f["id"]: f for f in (per_o[m] if m in per_o else wide[m])}
             for m in p10.ALL_MODELS}
    rank_o = sorted(bic6o, key=bic6o.get)
    check1 = {
        "disagreement_at_P13_fits": dis,
        "refit": c1,
        "bound_claim_survives": bool(stands),
        "ranking_with_outcome_split": rank_o,
        "dbic_with_outcome_split": {m: bic6o[m] - bic6o[rank_o[0]] for m in rank_o},
        "subject_winners_with_outcome_split": p13.winners(per6o, p10.ALL_MODELS,
                                                          ids),
        "subject_winners_registered_with_outcome_split": p13.winners(
            per6o, p10.REGISTERED_MODELS, ids),
    }

    # ------------------------------------------------ check 2: lambda = 0
    print("\nCheck 2: all six structures with the affect loop switched off")
    split_0 = {m: "outcome" for m in p10.ALL_MODELS}
    per_0, sum_0 = fit_family(subjects, p10.ALL_MODELS, 0.0, split_0, wide,
                              args.seed, "lambda=0")
    rank_0 = sorted(p10.ALL_MODELS, key=lambda m: sum_0[m]["bic_sum"])
    rank_13 = sorted(p10.ALL_MODELS, key=lambda m: bic6[m])
    per0 = {m: {f["id"]: f for f in per_0[m]} for m in p10.ALL_MODELS}

    def med(summary, model, name):
        return summary[model]["param_median"][p10.PARAM_NAMES[model].index(name)]

    s13 = {m: p13.summarise(wide[m], m, p13.NEW) for m in p10.ALL_MODELS}
    shifts = {
        "M1.alpha": abs(med(sum_0, "M1", "alpha") - med(s13, "M1", "alpha")),
        "M5.alpha_pos": abs(med(sum_0, "M5", "alpha_pos")
                            - med(s13, "M5", "alpha_pos")),
        "M5.alpha_neg": abs(med(sum_0, "M5", "alpha_neg")
                            - med(s13, "M5", "alpha_neg")),
    }
    sign = {
        "M2_pct_negative": abs(sum_0["M2"]["persistence_sign"]["pct_negative"]
                               - s13["M2"]["persistence_sign"]["pct_negative"]),
        "M5_pct_positive": abs(sum_0["M5"]["persistence_sign"]["pct_positive"]
                               - s13["M5"]["persistence_sign"]["pct_positive"]),
    }
    a_ok = rank_0[0] == rank_13[0]
    b_ok = max(shifts.values()) < READING["alpha_shift_max"]
    c_ok = max(sign.values()) < READING["sign_share_shift_max_pct"]
    check2 = {
        "models": sum_0,
        "bic_ranking": rank_0,
        "dbic": {m: sum_0[m]["bic_sum"] - sum_0[rank_0[0]]["bic_sum"]
                 for m in rank_0},
        "bic_vs_P13": {m: sum_0[m]["bic_sum"] - bic6[m] for m in p10.ALL_MODELS},
        "subject_winners_all": p13.winners(per0, p10.ALL_MODELS, ids),
        "subject_winners_registered": p13.winners(per0, p10.REGISTERED_MODELS,
                                                  ids),
        "M5_better_than_M2": int(sum(per0["M5"][i]["bic"] < per0["M2"][i]["bic"]
                                     for i in ids)),
        "P13_ranking": rank_13,
        "reading": {"a_winner_unchanged": bool(a_ok),
                    "b_alpha_medians_stable": bool(b_ok),
                    "b_abs_shift": shifts,
                    "c_sign_stable": bool(c_ok),
                    "c_abs_shift_pct": sign,
                    "robust_to_affect_loop": bool(a_ok and b_ok and c_ok)},
    }

    # ------------------------------------------------ check 3: missing rows
    print("\nCheck 3: missing rows in the analysed files")
    check3 = check3_missing(ids)
    print(f"  files with missing rows: {check3['n_files_with_missing_rows']} of "
          f"{check3['n_files']}; largest change in an observed value: "
          f"{check3['max_abs_difference']:.4f}")

    result = {
        "procedure": "three sensitivity checks on the transfer analysis: split "
                     "rule in M3/M4, affect loop in all structures, missing "
                     "rows in the data files",
        "registered": False,
        "smoke_run": bool(args.smoke),
        "seed": args.seed,
        "n_subjects": len(ids),
        "sample_provenance": prov,
        "bounds": {k: list(v) for k, v in p13.NEW.items()},
        "fixed": {"tau_m": p10.TAU_M, "lambda_main": p10.LAM},
        "reading_thresholds": READING,
        "self_test_max_abs_diff": worst,
        "check1_split_rule": check1,
        "check2_lambda_zero": check2,
        "check3_missing_rows": check3,
        "per_subject": {"M3o": per_o["M3"], "M4o": per_o["M4"],
                        **{m + "_lam0": per_0[m] for m in p10.ALL_MODELS}},
    }

    print()
    print(f"CHECK 1: '70 per cent at the bound' survives the outcome split: "
          f"{stands}")
    for k in c1:
        print(f"   {k}: alpha+ at bound {100*c1[k]['share_alpha_pos_at_bound_pe']:.0f}% "
              f"(PE split) -> {100*c1[k]['share_alpha_pos_at_bound_outcome']:.0f}% "
              f"(outcome split)")
    r = check2["reading"]
    print(f"CHECK 2: robust to the affect loop: {r['robust_to_affect_loop']}  "
          f"(winner {a_ok}, alpha {b_ok} {({k: round(v, 3) for k, v in shifts.items()})}, "
          f"sign {c_ok})")
    print(f"CHECK 3: behavioural tables must be redone: "
          f"{check3['tables_must_be_redone']}")

    if args.smoke:
        print("\nSmoke run - results NOT written.")
        return
    path = os.path.join(RESULTS_DIR, "p16_split_and_affect_sensitivity.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
