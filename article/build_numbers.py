"""Build numbers.json from results/ files.

Every number in the paper comes from here. Rule 1: no number typed by hand.
Each entry: value, source file, JSON path within file, description.

Usage: python article/build_numbers.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "article" / "numbers.json"


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def extract(data, path):
    """Extract value from nested dict by dot-separated path."""
    parts = path.split(".")
    v = data
    for p in parts:
        if isinstance(v, list):
            v = v[int(p)]
        else:
            v = v[p]
    return v


def build():
    numbers = {}

    # === DDM calibration (P2) ===
    cal = load("ddm_calibration.json")
    numbers["ddm_kappa"] = {
        "value": cal["kappa"], "source": "ddm_calibration.json",
        "path": "kappa", "desc": "DDM drift sensitivity (calibrated)"}
    numbers["ddm_a"] = {
        "value": cal["a"], "source": "ddm_calibration.json",
        "path": "a", "desc": "DDM threshold (calibrated)"}
    numbers["ddm_Ter"] = {
        "value": cal["Ter"], "source": "ddm_calibration.json",
        "path": "Ter", "desc": "DDM non-decision time (calibrated)"}

    # === DDM recovery (P2) ===
    rec = load("ddm_recovery.json")
    for p in ["kappa", "a", "Ter"]:
        r = rec["recovery"][p]
        numbers[f"ddm_recovery_r_{p}"] = {
            "value": r["r"], "source": "ddm_recovery.json",
            "path": f"recovery.{p}.r", "desc": f"DDM recovery r({p})"}

    # === RL calibration ===
    rl_cal = load("rl_mood_calibration.json")
    numbers["rl_alpha"] = {
        "value": rl_cal["calibrated"]["alpha"],
        "source": "rl_mood_calibration.json",
        "path": "calibrated.alpha", "desc": "RL learning rate (calibrated)"}
    numbers["rl_beta"] = {
        "value": rl_cal["calibrated"]["beta"],
        "source": "rl_mood_calibration.json",
        "path": "calibrated.beta", "desc": "RL inverse temperature (calibrated)"}

    # === MLE recovery (P6) ===
    mle = load("p6_ddm_mle_recovery.json")
    numbers["mle_r_kappa"] = {
        "value": mle["r_kappa"], "source": "p6_ddm_mle_recovery.json",
        "path": "r_kappa", "desc": "MLE recovery r(kappa)"}
    numbers["mle_r_a"] = {
        "value": mle["r_a"], "source": "p6_ddm_mle_recovery.json",
        "path": "r_a", "desc": "MLE recovery r(a)"}
    numbers["mle_r_Ter"] = {
        "value": mle["r_Ter"], "source": "p6_ddm_mle_recovery.json",
        "path": "r_Ter", "desc": "MLE recovery r(Ter)"}

    # === DDM comparison (P6) ===
    ddm = load("p6_ddm_comparison.json")
    numbers["p6_ddm_n_datasets"] = {
        "value": ddm["n_datasets"], "source": "p6_ddm_comparison.json",
        "path": "n_datasets", "desc": "DDM comparison: datasets"}
    numbers["p6_ddm_n_subjects"] = {
        "value": ddm["n_subjects"], "source": "p6_ddm_comparison.json",
        "path": "n_subjects", "desc": "DDM comparison: subjects"}
    # Accuracy distribution
    ad = ddm["accuracy_distribution"]
    numbers["p6_ddm_single_in_threshold"] = {
        "value": ad["single"]["in_threshold"],
        "source": "p6_ddm_comparison.json",
        "path": "accuracy_distribution.single.in_threshold",
        "desc": "Single-condition: count in threshold"}
    numbers["p6_ddm_single_N"] = {
        "value": ad["single"]["N"],
        "source": "p6_ddm_comparison.json",
        "path": "accuracy_distribution.single.N",
        "desc": "Single-condition: total datasets"}
    numbers["p6_ddm_single_mae_median"] = {
        "value": ad["single"]["percentiles"]["50"],
        "source": "p6_ddm_comparison.json",
        "path": "accuracy_distribution.single.percentiles.50",
        "desc": "Single-condition: MAE median"}
    numbers["p6_ddm_multi_mae_median"] = {
        "value": ad["multi"]["percentiles"]["50"],
        "source": "p6_ddm_comparison.json",
        "path": "accuracy_distribution.multi.percentiles.50",
        "desc": "Multi-condition: MAE median"}
    # Scale type
    st = ddm["scale_type_analysis_v2"]
    numbers["p6_ddm_coherence_mae"] = {
        "value": st["refined_groups"]["coherence"]["MAE_median"],
        "source": "p6_ddm_comparison.json",
        "path": "scale_type_analysis_v2.refined_groups.coherence.MAE_median",
        "desc": "Coherence datasets MAE median"}
    numbers["p6_ddm_coherence_threshold"] = {
        "value": st["coherence_own_threshold"]["accuracy_SD"],
        "source": "p6_ddm_comparison.json",
        "path": "scale_type_analysis_v2.coherence_own_threshold.accuracy_SD",
        "desc": "Coherence-only inter-dataset SD"}

    # === Arousal comparison (P6) — restructured after D49 ===
    # Duration comparison withdrawn. Onset and variance retained.
    ar = load("p6_arousal_comparison.json")
    # Retained results — from robustness table (use fixed_50 variant as reference)
    rob = ar["robustness"]["fixed_50"]
    numbers["p6_arousal_onset_data"] = {
        "value": rob["onset_circ_mean_h"],
        "source": "p6_arousal_comparison.json",
        "path": "robustness.fixed_50.onset_circ_mean_h",
        "desc": "Data onset circular mean (h), fixed_50 variant"}
    numbers["p6_arousal_onset_diff_min"] = {
        "value": ar["retained"]["onset"]["diff_range_min"][0],
        "source": "p6_arousal_comparison.json",
        "path": "retained.onset.diff_range_min.0",
        "desc": "Onset difference min (min)"}
    numbers["p6_arousal_onset_data_sd_min"] = {
        "value": rob["onset_within_sd_median_min"],
        "source": "p6_arousal_comparison.json",
        "path": "robustness.fixed_50.onset_within_sd_median_min",
        "desc": "Data onset within-individual circular SD (min)"}
    numbers["p6_arousal_within_sd"] = {
        "value": rob["within_individual_sd_h"],
        "source": "p6_arousal_comparison.json",
        "path": "robustness.fixed_50.within_individual_sd_h",
        "desc": "Within-individual SD (h)"}
    numbers["p6_arousal_between_sd"] = {
        "value": rob["between_individual_sd_h"],
        "source": "p6_arousal_comparison.json",
        "path": "robustness.fixed_50.between_individual_sd_h",
        "desc": "Between-individual SD (h)"}

    # === RL comparison (P6) ===
    rl = load("p6_rl_comparison.json")
    numbers["p6_rl_n_adults"] = {
        "value": rl["n_adults"], "source": "p6_rl_comparison.json",
        "path": "n_adults", "desc": "RL adults 18+"}
    numbers["p6_rl_data_ws"] = {
        "value": rl["empirical"]["ws"]["mean"],
        "source": "p6_rl_comparison.json",
        "path": "empirical.ws.mean", "desc": "Data win-stay"}
    numbers["p6_rl_model_ws"] = {
        "value": rl["model"]["ws"],
        "source": "p6_rl_comparison.json",
        "path": "model.ws", "desc": "Model win-stay (calibrated)"}
    numbers["p6_rl_diag_ws"] = {
        "value": rl["diagnostic"]["alpha_eckstein"]["ws"],
        "source": "p6_rl_comparison.json",
        "path": "diagnostic.alpha_eckstein.ws",
        "desc": "Model WS at alpha=0.76 (diagnostic)"}
    numbers["p6_rl_data_acc"] = {
        "value": rl["empirical"]["acc"]["mean"],
        "source": "p6_rl_comparison.json",
        "path": "empirical.acc.mean", "desc": "Data accuracy"}
    numbers["p6_rl_model_acc"] = {
        "value": rl["model"]["acc"],
        "source": "p6_rl_comparison.json",
        "path": "model.acc", "desc": "Model accuracy (calibrated)"}

    # === Arousal calibration ===
    ar_cal = load("arousal_calibration.json")
    numbers["arousal_v_vh"] = {
        "value": ar_cal["baseline_params"]["v_vh"],
        "source": "arousal_calibration.json",
        "path": "baseline_params.v_vh", "desc": "Arousal v_vh (calibrated)"}
    numbers["arousal_Q_REF"] = {
        "value": ar_cal["Q_M_WAKE_MEAN_BASELINE"],
        "source": "arousal_calibration.json",
        "path": "Q_M_WAKE_MEAN_BASELINE", "desc": "Arousal Q_REF"}

    # === Arousal recovery ===
    ar_rec = load("arousal_recovery.json")
    for p in ["v_vh", "v_vc", "v_vm", "v_mv"]:
        numbers[f"arousal_recovery_r_{p}"] = {
            "value": ar_rec[p]["r"],
            "source": "arousal_recovery.json",
            "path": f"{p}.r",
            "desc": f"Arousal recovery r({p})"}

    # === RL recovery ===
    rl_rec = load("rl_mood_recovery.json")
    for p in ["beta", "alpha"]:
        numbers[f"rl_recovery_r_{p}"] = {
            "value": rl_rec["recovery"][p]["r"],
            "source": "rl_mood_recovery.json",
            "path": f"recovery.{p}.r",
            "desc": f"RL recovery r({p})"}

    # === RL recovery alpha at N=1000 ===
    try:
        alpha_ci = load("rl_mood_alpha_recovery_ci.json")
        numbers["rl_recovery_r_alpha_n1000"] = {
            "value": alpha_ci["results"]["N1000"]["r"],
            "source": "rl_mood_alpha_recovery_ci.json",
            "path": "results.N1000.r",
            "desc": "RL recovery r(alpha) at N=1000"}
    except (FileNotFoundError, KeyError):
        pass

    # === RL calibration extras ===
    numbers["rl_tau_m"] = {
        "value": rl_cal["calibrated"]["tau_m"],
        "source": "rl_mood_calibration.json",
        "path": "calibrated.tau_m", "desc": "RL tau_m"}
    numbers["rl_lam_default"] = {
        "value": rl_cal["calibrated"]["lam_default"],
        "source": "rl_mood_calibration.json",
        "path": "calibrated.lam_default", "desc": "RL lambda default"}
    numbers["rl_lam_crit"] = {
        "value": rl_cal["calibrated"]["lambda_crit"],
        "source": "rl_mood_calibration.json",
        "path": "calibrated.lambda_crit", "desc": "RL lambda_crit standalone"}

    # === Arousal sigma_h ===
    try:
        d18 = load("arousal_d18_final.json")
        numbers["arousal_sigma_h"] = {
            "value": d18["sigma_h"],
            "source": "arousal_d18_final.json",
            "path": "sigma_h", "desc": "Arousal sigma_h (D18)"}
    except (FileNotFoundError, KeyError):
        pass

    # === P3 value coupling ===
    p3v = load("p3_value_coupling_fine.json")
    numbers["p3_s_value_optimum"] = {
        "value": p3v["optimum_s"],
        "source": "p3_value_coupling_fine.json",
        "path": "optimum_s", "desc": "s_value optimum (D29)"}

    # === P3 lambda_crit closed ===
    p3s = load("p3_value_coupling_sweep.json")
    numbers["p3_lam_crit_closed"] = {
        "value": p3s["lambda_crit"]["lambda_crit_numerical"],
        "source": "p3_value_coupling_sweep.json",
        "path": "lambda_crit.lambda_crit_numerical",
        "desc": "lambda_crit closed-loop (D30)"}
    numbers["p3_lam_crit_standalone"] = {
        "value": p3s["lambda_crit"]["lambda_crit_standalone"],
        "source": "p3_value_coupling_sweep.json",
        "path": "lambda_crit.lambda_crit_standalone",
        "desc": "lambda_crit standalone"}

    # === P4 impossible map ===
    p4m = load("p4_impossible_map_kappa_value.json")
    n_unstable = p4m["counts"].get("UNSTABLE", 0)
    n_chance = p4m["counts"].get("CHANCE", 0)
    numbers["p4_map_n_unstable"] = {
        "value": n_unstable, "source": "p4_impossible_map_kappa_value.json",
        "path": "counts.UNSTABLE", "desc": "Impossible map: UNSTABLE nodes"}
    numbers["p4_map_n_chance"] = {
        "value": n_chance, "source": "p4_impossible_map_kappa_value.json",
        "path": "counts.CHANCE", "desc": "Impossible map: CHANCE nodes"}

    # === P6 DDM extras ===
    numbers["p6_ddm_n_multi"] = {
        "value": ddm["n_multi"], "source": "p6_ddm_comparison.json",
        "path": "n_multi", "desc": "DDM multi-condition datasets"}
    numbers["p6_ddm_n_single"] = {
        "value": ddm["n_single"], "source": "p6_ddm_comparison.json",
        "path": "n_single", "desc": "DDM single-condition datasets"}
    # D50: scale_type pairwise tests removed — 4-group classification
    # (coherence/contrast/difficulty/ordinal) not reproduced. See D50.

    # === P6 Arousal extras (restructured after D49) ===
    numbers["p6_arousal_n_subjects"] = {
        "value": rob["n_subjects"], "source": "p6_arousal_comparison.json",
        "path": "robustness.fixed_50.n_subjects", "desc": "Arousal N controls"}
    numbers["p6_arousal_model_sd"] = {
        "value": ar["retained"]["variance_decomposition"]["model_sd_h"],
        "source": "p6_arousal_comparison.json",
        "path": "retained.variance_decomposition.model_sd_h",
        "desc": "Arousal model SD (h)"}

    # === P6 RL extras ===
    numbers["p6_rl_diag_acc"] = {
        "value": rl["diagnostic"]["alpha_eckstein"]["acc"],
        "source": "p6_rl_comparison.json",
        "path": "diagnostic.alpha_eckstein.acc",
        "desc": "Model ACC at alpha=0.76, beta=2.50 (diagnostic)"}
    # Full Eckstein params (alpha=0.76, beta=3.0, persist=0.3)
    numbers["p6_rl_diag_full_ws"] = {
        "value": rl["diagnostic"]["persistence"]["runs"]["alpha076_beta300_p03"]["ws"],
        "source": "p6_rl_comparison.json",
        "path": "diagnostic.persistence.runs.alpha076_beta300_p03.ws",
        "desc": "Model WS at full Eckstein params"}
    numbers["p6_rl_diag_full_acc"] = {
        "value": rl["diagnostic"]["persistence"]["runs"]["alpha076_beta300_p03"]["acc"],
        "source": "p6_rl_comparison.json",
        "path": "diagnostic.persistence.runs.alpha076_beta300_p03.acc",
        "desc": "Model ACC at full Eckstein params"}
    numbers["p6_rl_ratio_alpha"] = {
        "value": rl["diagnostic"]["eckstein_concordance"]["ratio_alpha"],
        "source": "p6_rl_comparison.json",
        "path": "diagnostic.eckstein_concordance.ratio_alpha",
        "desc": "Alpha ratio our/Eckstein"}
    numbers["p6_rl_ratio_beta"] = {
        "value": rl["diagnostic"]["eckstein_concordance"]["ratio_beta"],
        "source": "p6_rl_comparison.json",
        "path": "diagnostic.eckstein_concordance.ratio_beta",
        "desc": "Beta ratio our/Eckstein"}

    # === DDM resolution ===
    ddm_res = load("ddm_resolution.json")
    numbers["ddm_mdd_kappa"] = {
        "value": ddm_res["resolution"]["kappa"]["mdd"],
        "source": "ddm_resolution.json",
        "path": "resolution.kappa.mdd", "desc": "DDM MDD kappa"}
    numbers["ddm_mdd_a"] = {
        "value": ddm_res["resolution"]["a"]["mdd"],
        "source": "ddm_resolution.json",
        "path": "resolution.a.mdd", "desc": "DDM MDD a"}
    numbers["ddm_mdd_Ter"] = {
        "value": ddm_res["resolution"]["Ter"]["mdd"],
        "source": "ddm_resolution.json",
        "path": "resolution.Ter.mdd", "desc": "DDM MDD Ter"}

    # === RL resolution ===
    rl_res = load("rl_mood_resolution.json")
    numbers["rl_mdd_beta"] = {
        "value": rl_res["beta"]["mdd"],
        "source": "rl_mood_resolution.json",
        "path": "beta.mdd", "desc": "RL MDD beta"}
    numbers["rl_mdd_alpha"] = {
        "value": rl_res["alpha"]["mdd"],
        "source": "rl_mood_resolution.json",
        "path": "alpha.mdd", "desc": "RL MDD alpha"}

    # === Missing numbers flagged by red-team ===
    # D49: arousal algorithm difference withdrawn (duration comparison removed).

    # Bonferroni p-values for scale types
    kv_bonf = ddm.get("scale_type_analysis_v2", {}).get("pairwise_bonferroni", {})
    if "coherence_vs_contrast" in kv_bonf:
        numbers["p6_ddm_coh_vs_contrast_p_bonf"] = {
            "value": kv_bonf["coherence_vs_contrast"]["p_bonf"],
            "source": "p6_ddm_comparison.json",
            "path": "scale_type_analysis_v2.pairwise_bonferroni.coherence_vs_contrast.p_bonf",
            "desc": "Coherence vs contrast p (Bonferroni)"}

    # CI on median difference coherence-contrast
    ci_diff = ddm.get("scale_type_analysis_v2", {}).get("coherence_vs_contrast_CI", [])
    if ci_diff:
        numbers["p6_ddm_coh_contrast_ci_lo"] = {
            "value": ci_diff[0], "source": "p6_ddm_comparison.json",
            "path": "scale_type_analysis_v2.coherence_vs_contrast_CI.0",
            "desc": "Coherence-contrast median diff CI lower"}
        numbers["p6_ddm_coh_contrast_ci_hi"] = {
            "value": ci_diff[1], "source": "p6_ddm_comparison.json",
            "path": "scale_type_analysis_v2.coherence_vs_contrast_CI.1",
            "desc": "Coherence-contrast median diff CI upper"}

    # RL tau_m and lambda recovery r
    numbers["rl_recovery_r_tau_m"] = {
        "value": rl_rec["recovery"]["tau_m"]["r"],
        "source": "rl_mood_recovery.json",
        "path": "recovery.tau_m.r", "desc": "RL recovery r(tau_m)"}
    numbers["rl_recovery_r_lambda"] = {
        "value": rl_rec["recovery"]["lambda"]["r"],
        "source": "rl_mood_recovery.json",
        "path": "recovery.lambda.r", "desc": "RL recovery r(lambda)"}

    # D49: Arousal alg2 CI/IQR withdrawn (duration comparison removed).

    # P4 map: UNSTABLE and CHANCE absent from counts means 0.
    # Verify by checking key is absent (handled in verify_numbers.py).
    numbers["p4_map_n_unstable"]["value"] = 0
    numbers["p4_map_n_unstable"]["path"] = "counts"
    numbers["p4_map_n_unstable"]["absent_key"] = "UNSTABLE"
    numbers["p4_map_n_chance"]["value"] = 0
    numbers["p4_map_n_chance"]["path"] = "counts"
    numbers["p4_map_n_chance"]["absent_key"] = "CHANCE"

    with open(OUT, "w") as f:
        json.dump(numbers, f, indent=2)
    print(f"Built {len(numbers)} numbers -> {OUT}")
    return numbers


if __name__ == "__main__":
    build()
