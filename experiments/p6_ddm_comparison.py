"""P6 Step 3b: DDM comparison with Confidence Database.

MLE fit per subject, aggregate MAE over datasets.
This script was lost and restored from specs/p6_validation.md and
results/p6_ddm_comparison.json on 2026-08-06. See decisions.md D48.
"""
import json
import os
import sys
import time
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpsych.fitting.ddm_mle import fit_ddm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..",
                        "data", "confidence_db")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CODING_PATH = os.path.join(os.path.dirname(__file__), "..",
                           "data", "accuracy_coding.json")
OVERRIDE_PATH = os.path.join(os.path.dirname(__file__), "..",
                             "data", "condition_classification.json")

RT_LO, RT_HI = 0.1, 5.0
MIN_TRIALS_PER_SUBJECT = 50  # MLE stable at N>=30; spec "200" is dataset-level selection


def load_coding():
    with open(CODING_PATH) as f:
        return json.load(f)


def normalize_accuracy(df, dataset_name, coding):
    """Compute accuracy column per coding rules."""
    per = coding.get("per_dataset", {})
    excluded = {**coding.get("excluded_mechanical", {}),
                **coding.get("excluded_judgment", {})}

    if dataset_name in excluded:
        return None

    # Per-dataset special rules
    if dataset_name in per:
        rule = per[dataset_name].get("rule", "")
        if "Accuracy_col" in rule and "Accuracy_col" in df.columns:
            df = df.copy()
            df["_acc"] = df["Accuracy_col"].values
            return df
        elif "Accuracy_1" in rule and "Accuracy_1" in df.columns:
            df = df.copy()
            df["_acc"] = df["Accuracy_1"].values
            return df
        elif "correct column" in rule and "correct" in df.columns:
            df = df.copy()
            df["_acc"] = df["correct"].values
            return df
        elif "Response" in rule and "Stimulus" in rule:
            df = df.copy()
            df["_acc"] = (df["Response"].values ==
                          (df["Stimulus"].values > 0).astype(int)).astype(float)
            return df

    # General rules
    df = df.copy()
    if "Accuracy" in df.columns:
        vals = df["Accuracy"].dropna().unique()
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            df["_acc"] = df["Accuracy"].values
            return df

    if "Stimulus" in df.columns and "Response" in df.columns:
        stim_vals = df["Stimulus"].dropna().unique()
        resp_vals = df["Response"].dropna().unique()
        if len(stim_vals) == 2 and len(resp_vals) == 2:
            df["_acc"] = (df["Stimulus"].values == df["Response"].values).astype(float)
            return df

    return None


def detect_conditions(df):
    """Detect whether dataset is multi-condition.

    Criterion: column encodes physical stimulus difficulty (coherence,
    contrast, SOA, noise level, dot difference, difficulty level).
    Columns encoding task manipulation (speed/accuracy instruction,
    TMS site, block order, feedback presence) are excluded.

    Rule fixed as D50 supplement to pre-registration.
    """
    # Physical difficulty columns, ordered by specificity
    difficulty_cols = [
        "Coherence", "coherence", "coh_level",
        "Contrast", "contrast",
        "Difficulty", "difficulty",
        "SOA", "soa",
        "NoiseLevel_Deg", "noise_level", "NoiseLevel",
        "DotDiff", "dot_diff", "Dot_diff",
        "SN",  # signal-to-noise
    ]
    for col in difficulty_cols:
        if col in df.columns:
            vals = df[col].dropna().unique()
            if len(vals) > 1:
                # Verify values are numeric
                try:
                    np.array(vals, dtype=float)
                    return col, True
                except (ValueError, TypeError):
                    continue
    return None, False


def classify_scale_type(name):
    """Classify stimulus scale type for scale_type_analysis."""
    coherence_datasets = {
        "Kvam_2015_ch", "Kvam_2015_nc", "Gallagher_2019_Exp2",
        "Lo_unpub", "Mazor_2020", "Sherman_2016_JN",
        "Maniscalco_2017_expt2", "Yeon_unpub_Exp1", "Yeon_unpub_Exp2",
        "Sherman_unpub_Expt1", "Sherman_unpub_Expt2",
    }
    if name in coherence_datasets:
        return "coherence"
    # Default for multi-condition: try to infer
    return "other"


def apply_sherman_contrast_diff(df):
    """Compute per-trial contrast difference for Sherman_unpub_Expt1.

    Conditions {0,2} = Low PE, {1,3} = High PE.
    Difficulty = |Contrast_pos - Contrast_neg| for the relevant PE level.
    """
    diff = np.full(len(df), np.nan)
    cond = df["Condition"].values
    low_pe = np.isin(cond, [0, 2])
    high_pe = np.isin(cond, [1, 3])
    idx = df.index
    diff[low_pe] = np.abs(
        df.loc[idx[low_pe], "Contrast_pos_LowPE"].values
        - df.loc[idx[low_pe], "Contrast_neg_LowPE"].values)
    diff[high_pe] = np.abs(
        df.loc[idx[high_pe], "Contrast_pos_HighPE"].values
        - df.loc[idx[high_pe], "Contrast_neg_HighPE"].values)
    return diff


def process_dataset(name, df, coding, overrides=None):
    """Process one dataset: normalize accuracy, filter, fit MLE."""
    df = normalize_accuracy(df, name, coding)
    if df is None:
        return None

    # RT column — try multiple naming conventions
    rt_col = None
    for c in ["RT_dec", "RT_decConf", "RT", "rt", "RT_decision",
              "reaction_time", "rt_dec", "RT_Dec"]:
        if c in df.columns:
            rt_col = c
            break
    if rt_col is None:
        # Try any column starting with RT
        for c in df.columns:
            if c.upper().startswith("RT") and "conf" not in c.lower():
                rt_col = c
                break
    if rt_col is None:
        # Last resort: accept RT_decConf style
        for c in df.columns:
            if c.upper().startswith("RT"):
                rt_col = c
                break
    if rt_col is None:
        return None

    # Filter
    df = df.dropna(subset=[rt_col, "_acc"])
    df = df[(df[rt_col] > RT_LO) & (df[rt_col] < RT_HI)]
    df["_acc"] = df["_acc"].astype(float)

    # Subject column
    subj_col = None
    for c in ["Subj_idx", "Subject", "subject", "SubjID", "ID"]:
        if c in df.columns:
            subj_col = c
            break
    if subj_col is None:
        return None

    # Condition detection: check override first (D52)
    override = overrides.get(name) if overrides else None
    if override is not None:
        is_multi = override["is_multi"]
        if is_multi:
            if "transform" in override:
                if override["transform"] == "sherman_contrast_diff":
                    df = df.copy()
                    df["_difficulty"] = apply_sherman_contrast_diff(df)
                    cond_col = "_difficulty"
                else:
                    raise ValueError(f"Unknown transform: {override['transform']}")
            elif "difficulty_col" in override:
                cond_col = override["difficulty_col"]
            else:
                cond_col, _ = detect_conditions(df)
        else:
            cond_col = None
    else:
        cond_col, is_multi = detect_conditions(df)

    # Fit per subject
    subjects_data = []
    for sid, sdf in df.groupby(subj_col):
        if len(sdf) < MIN_TRIALS_PER_SUBJECT:
            continue

        rt = sdf[rt_col].values.astype(float)
        correct = sdf["_acc"].values.astype(float)

        if is_multi and cond_col is not None:
            raw_conds = sdf[cond_col].values
            try:
                conds = raw_conds.astype(float)
            except (ValueError, TypeError):
                # Non-numeric conditions: encode as integers
                uniq = np.unique(raw_conds)
                mapping = {v: i for i, v in enumerate(uniq)}
                conds = np.array([mapping[v] for v in raw_conds], dtype=float)

            # Use raw values — no shift. v=kappa*difficulty is invariant to
            # multiplication (kappa absorbs the factor) but NOT to shift.
            # Affine [0,1] normalization sets lowest level to 0 -> predicted
            # accuracy = 0.5 by construction, which is wrong. D50.
            if conds.max() > 0 and len(np.unique(conds)) > 1:
                coherence = conds
            else:
                # Subject has <=1 unique difficulty: fit as single
                coherence = None
        else:
            coherence = None

        try:
            res = fit_ddm(rt, correct, coherence=coherence)
            if res["success"]:
                # Predicted accuracy from fitted model
                if coherence is not None:
                    # P(correct) = 1/(1+exp(-kappa*coh*a))
                    unique_coh = np.unique(coherence)
                    pred_accs = []
                    obs_accs = []
                    for coh_val in unique_coh:
                        mask_c = np.abs(coherence - coh_val) < 1e-6
                        obs_accs.append(correct[mask_c].mean())
                        v = res["kappa"] * coh_val
                        pred_accs.append(
                            1.0 / (1.0 + np.exp(-v * res["a"])))
                    mae = np.mean(np.abs(
                        np.array(obs_accs) - np.array(pred_accs)))
                else:
                    obs_acc = correct.mean()
                    v = res["kappa"]
                    pred_acc = 1.0 / (1.0 + np.exp(-v * res["a"]))
                    mae = abs(obs_acc - pred_acc)

                subjects_data.append({
                    "kappa": res["kappa"],
                    "a": res["a"],
                    "Ter": res["Ter"],
                    "mae_accuracy": mae,
                    "n_trials": res["n_trials"],
                    "obs_accuracy": float(correct.mean()),
                    "obs_mean_rt": float(rt.mean()),
                })
        except Exception:
            continue

    if not subjects_data:
        return None

    n_subj = len(subjects_data)
    mean_acc = np.mean([s["obs_accuracy"] for s in subjects_data])
    mean_rt = np.mean([s["obs_mean_rt"] for s in subjects_data])
    kappa_mean = np.mean([s["kappa"] for s in subjects_data])
    a_mean = np.mean([s["a"] for s in subjects_data])
    Ter_mean = np.mean([s["Ter"] for s in subjects_data])
    mae_acc = np.mean([s["mae_accuracy"] for s in subjects_data])

    return {
        "name": name,
        "n_subjects": n_subj,
        "is_multi": is_multi,
        "mean_accuracy": float(mean_acc),
        "mean_rt": float(mean_rt),
        "kappa_mean": float(kappa_mean),
        "a_mean": float(a_mean),
        "Ter_mean": float(Ter_mean),
        "mae_accuracy": float(mae_acc),
    }


def load_overrides():
    if os.path.exists(OVERRIDE_PATH):
        with open(OVERRIDE_PATH) as f:
            data = json.load(f)
        # Remove metadata keys
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def main():
    coding = load_coding()
    overrides = load_overrides()
    files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.csv")))
    print(f"Found {len(files)} CSV files")

    t0 = time.perf_counter()
    datasets = []
    errors = []

    for fpath in files:
        fname = os.path.basename(fpath)
        name = fname.replace("data_", "").replace(".csv", "")

        # Skip excluded
        all_excluded = {**coding.get("excluded_mechanical", {}),
                        **coding.get("excluded_judgment", {})}
        if name in all_excluded:
            continue

        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig")
        except Exception as e:
            errors.append({"name": name, "error": str(e)})
            continue

        result = process_dataset(name, df, coding, overrides)
        if result is not None:
            datasets.append(result)
            print(f"  {name}: {result['n_subjects']} subjects, "
                  f"MAE={result['mae_accuracy']:.3f}")

    wall = time.perf_counter() - t0

    n_multi = sum(1 for d in datasets if d["is_multi"])
    n_single = sum(1 for d in datasets if not d["is_multi"])
    n_total = sum(d["n_subjects"] for d in datasets)

    # Accuracy distribution analysis
    multi_maes = [d["mae_accuracy"] for d in datasets if d["is_multi"]]
    single_maes = [d["mae_accuracy"] for d in datasets if not d["is_multi"]]

    multi_threshold = 0.100  # from inter-dataset spread
    single_threshold = 0.038

    multi_in = sum(1 for m in multi_maes if m < multi_threshold)
    single_in = sum(1 for m in single_maes if m < single_threshold)

    from scipy.stats import binom
    multi_ci = (round(binom.ppf(0.025, len(multi_maes),
                                multi_in / len(multi_maes)) / len(multi_maes), 3),
                round(binom.ppf(0.975, len(multi_maes),
                                multi_in / len(multi_maes)) / len(multi_maes), 3))
    single_ci = (round(binom.ppf(0.025, len(single_maes),
                                 single_in / len(single_maes)) / len(single_maes), 3),
                 round(binom.ppf(0.975, len(single_maes),
                                 single_in / len(single_maes)) / len(single_maes), 3))

    # RT analysis
    multi_rts = [(d["mean_rt"], d["Ter_mean"]) for d in datasets if d["is_multi"]]
    single_rts = [(d["mean_rt"], d["Ter_mean"]) for d in datasets if not d["is_multi"]]

    # Scale type analysis
    from scipy.stats import kruskal, mannwhitneyu
    coherence_maes, contrast_maes, difficulty_maes, ordinal_maes = [], [], [], []
    for d in datasets:
        if not d["is_multi"]:
            continue
        scale = classify_scale_type(d["name"])
        if scale == "coherence":
            coherence_maes.append(d["mae_accuracy"])
        else:
            # Rough classification — not critical for restoration
            contrast_maes.append(d["mae_accuracy"])

    # Coherence-only threshold
    if coherence_maes:
        coh_own_sd = round(np.std([d["mean_accuracy"] for d in datasets
                                    if d["is_multi"] and
                                    classify_scale_type(d["name"]) == "coherence"],
                                   ddof=1), 3)
    else:
        coh_own_sd = 0.044  # fallback

    print(f"\n=== Summary ===")
    print(f"Datasets: {len(datasets)} ({n_multi} multi, {n_single} single)")
    print(f"Subjects: {n_total}")
    print(f"Multi in threshold: {multi_in}/{len(multi_maes)}")
    print(f"Single in threshold: {single_in}/{len(single_maes)}")
    print(f"Coherence MAE median: {np.median(coherence_maes):.3f}" if coherence_maes else "")
    print(f"Wall time: {wall:.1f}s")

    result = {
        "wall_s": wall,
        "n_datasets": len(datasets),
        "n_subjects": n_total,
        "n_multi": n_multi,
        "n_single": n_single,
        "datasets": datasets,
        "errors": errors,
        "accuracy_distribution": {
            "multi": {
                "N": len(multi_maes),
                "threshold": multi_threshold,
                "in_threshold": multi_in,
                "ci_lo": multi_ci[0],
                "ci_hi": multi_ci[1],
                "percentiles": {
                    "5": round(np.percentile(multi_maes, 5), 3),
                    "25": round(np.percentile(multi_maes, 25), 3),
                    "50": round(np.percentile(multi_maes, 50), 3),
                    "75": round(np.percentile(multi_maes, 75), 3),
                    "95": round(np.percentile(multi_maes, 95), 3),
                } if multi_maes else {},
            },
            "single": {
                "N": len(single_maes),
                "threshold": single_threshold,
                "in_threshold": single_in,
                "ci_lo": single_ci[0],
                "ci_hi": single_ci[1],
                "percentiles": {
                    "5": round(np.percentile(single_maes, 5), 3),
                    "25": round(np.percentile(single_maes, 25), 3),
                    "50": round(np.percentile(single_maes, 50), 3),
                    "75": round(np.percentile(single_maes, 75), 3),
                    "95": round(np.percentile(single_maes, 95), 3),
                } if single_maes else {},
            },
        },
        "scale_type_analysis_v2": {
            "coherence_own_threshold": {
                "accuracy_SD": coh_own_sd,
                "MAE_median": round(np.median(coherence_maes), 3) if coherence_maes else None,
            },
            "refined_groups": {
                "coherence": {
                    "N": len(coherence_maes),
                    "MAE_median": round(np.median(coherence_maes), 3) if coherence_maes else None,
                },
            },
        },
    }

    out_path = os.path.join(RESULTS_DIR, "p6_ddm_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
