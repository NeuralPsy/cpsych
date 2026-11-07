"""P6 Step 4: Arousal comparison with OBF-Psychiatric actigraphy.

Controls only. Sleep onset timing (uncalibrated prediction), onset
variability (structural gap), variance decomposition (data property).

Duration comparison WITHDRAWN: sleep scoring algorithm not reproducible
(D48, A15). Three scoring variants tested for robustness of retained
results. See decisions.md D48, D49.

This script was lost and restored on 2026-08-06. See decisions.md D48.

Reference for Cole-Kripke attempt: Cole RJ et al. (1992) Sleep 15:461-469.
Cole-Kripke inapplicable: coefficients calibrated on AMI device, OBF uses
Actiwatch AW4 with incompatible activity count scale (0-4245 vs 0-300).
"""
import json
import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..",
                        "data", "obf_psychiatric",
                        "OBF-Psychiatric Dataset", "control")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Model predictions (all params fixed/exploratory, not calibrated to these data)
MODEL_ONSET_H = 23.0
MODEL_ONSET_SD_MIN = 21.5
MODEL_SD_H = 0.50


# ── Three scoring variants for robustness check ──────────────────────

def _smooth_median(activity, window=11):
    n = len(activity)
    out = np.zeros(n)
    half = window // 2
    for i in range(n):
        out[i] = np.median(activity[max(0, i - half):min(n, i + half + 1)])
    return out


def _score_and_merge(smoothed, threshold, min_ep=30, merge_gap=15):
    """Score sleep with gap merging and min episode filtering."""
    n = len(smoothed)
    raw = smoothed <= threshold

    # Merge short awakenings
    sleep = raw.copy()
    i = 0
    while i < n:
        if not sleep[i]:
            j = i
            while j < n and not sleep[j]:
                j += 1
            if (j - i) < merge_gap and i > 0 and j < n:
                sleep[i:j] = True
            i = j
        else:
            i += 1

    # Remove short episodes
    final = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if sleep[i]:
            j = i
            while j < n and sleep[j]:
                j += 1
            if (j - i) >= min_ep:
                final[i:j] = True
            i = j
        else:
            i += 1
    return final


def threshold_p15_nonzero(activity, timestamps):
    """Variant 1: 15th percentile of nonzero activity (per subject)."""
    nz = activity[activity > 0]
    return np.percentile(nz, 15) if len(nz) > 0 else 1.0


def threshold_fixed_50(activity, timestamps):
    """Variant 2: fixed threshold = 50."""
    return 50.0


def threshold_p15_night_nonzero(activity, timestamps):
    """Variant 3: 15th percentile of nonzero nighttime activity."""
    hours = np.array([t.hour for t in timestamps])
    night = (hours >= 22) | (hours < 6)
    nact = activity[night & (activity > 0)]
    return np.percentile(nact, 15) if len(nact) > 0 else 1.0


VARIANTS = [
    ("p15_nonzero", threshold_p15_nonzero),
    ("fixed_50", threshold_fixed_50),
    ("p15_night_nonzero", threshold_p15_night_nonzero),
]


# ── Episode extraction ───────────────────────────────────────────────

def extract_nightly_main(sleep, timestamps):
    """Extract per-night main sleep episode (longest in 19:00-12:00 window).

    No onset filtering — all episodes reported as found.
    """
    n = len(sleep)
    hours = np.array([t.hour + t.minute / 60 for t in timestamps])
    dates = np.array([t.date() for t in timestamps])
    unique_dates = sorted(set(dates))

    results = []
    for d_idx in range(len(unique_dates) - 1):
        d1 = unique_dates[d_idx]
        d2 = unique_dates[d_idx + 1]

        mask = np.zeros(n, dtype=bool)
        for k in range(n):
            if dates[k] == d1 and hours[k] >= 19:
                mask[k] = True
            elif dates[k] == d2 and hours[k] < 12:
                mask[k] = True

        night_sleep = sleep & mask
        best_dur = 0
        best_onset_idx = None
        i = 0
        while i < n:
            if night_sleep[i]:
                j = i
                while j < n and night_sleep[j]:
                    j += 1
                if (j - i) > best_dur:
                    best_dur = j - i
                    best_onset_idx = i
                i = j
            else:
                i += 1

        if best_dur >= 60 and best_onset_idx is not None:
            results.append({
                "duration_h": best_dur / 60,
                "onset_h": hours[best_onset_idx],
            })

    return results


# ── Circular statistics ──────────────────────────────────────────────

def circular_mean(hours, period=24.0):
    angles = 2 * np.pi * np.array(hours) / period
    return (np.arctan2(np.mean(np.sin(angles)),
                       np.mean(np.cos(angles))) % (2 * np.pi)) * period / (2 * np.pi)


def circular_sd(hours, period=24.0):
    angles = 2 * np.pi * np.array(hours) / period
    R = np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
    R = min(R, 1.0 - 1e-10)
    return np.sqrt(-2 * np.log(R)) * period / (2 * np.pi)


# ── Main ─────────────────────────────────────────────────────────────

def analyze_variant(files, name, threshold_fn):
    """Run one scoring variant, return per-subject episodes."""
    per_subject = {}
    for fpath in files:
        sid = os.path.basename(fpath).replace(".csv", "")
        df = pd.read_csv(fpath, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(df["timestamp"].values)
        activity = df["activity"].values.astype(float)

        smoothed = _smooth_median(activity)
        threshold = threshold_fn(activity, ts)
        sleep = _score_and_merge(smoothed, threshold)
        episodes = extract_nightly_main(sleep, ts)

        valid = [e for e in episodes if 3 < e["duration_h"] < 16]
        if len(valid) >= 3:
            per_subject[sid] = {
                "durs": [e["duration_h"] for e in valid],
                "onsets": [e["onset_h"] for e in valid],
            }

    return per_subject


def compute_stats(per_subject):
    """Compute onset, variance decomposition from per-subject data."""
    # Per-subject duration means (for variance decomposition)
    subj_means = [np.mean(v["durs"]) for v in per_subject.values()]

    # Within-individual SD
    within_sds = [np.std(v["durs"], ddof=1) for v in per_subject.values()
                  if len(v["durs"]) >= 2]
    within_sd = round(np.mean(within_sds), 2) if within_sds else 0.0
    between_sd = round(np.std(subj_means, ddof=1), 2)

    # Onset: all onsets pooled
    all_onsets = []
    for v in per_subject.values():
        all_onsets.extend(v["onsets"])
    onset_cmean = round(circular_mean(all_onsets), 2) if all_onsets else None
    onset_csd_min = round(circular_sd(all_onsets) * 60) if all_onsets else None

    # Onset within-individual variability (median across subjects)
    onset_within = []
    for v in per_subject.values():
        if len(v["onsets"]) >= 3:
            onset_within.append(circular_sd(v["onsets"]) * 60)
    onset_within_median = round(np.median(onset_within)) if onset_within else None

    return {
        "n_subjects": len(per_subject),
        "onset_circ_mean_h": onset_cmean,
        "onset_circ_sd_min": onset_csd_min,
        "onset_within_sd_median_min": onset_within_median,
        "within_individual_sd_h": within_sd,
        "between_individual_sd_h": between_sd,
        "within_over_between": round(within_sd / between_sd, 2) if between_sd > 0 else None,
        "mean_duration_h": round(np.mean(subj_means), 2),
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "control_*.csv")))
    print(f"{len(files)} control files")

    robustness = {}
    for vname, vfn in VARIANTS:
        print(f"\nScoring variant: {vname}")
        ps = analyze_variant(files, vname, vfn)
        stats = compute_stats(ps)
        robustness[vname] = stats
        print(f"  N={stats['n_subjects']}, "
              f"onset={stats['onset_circ_mean_h']}h, "
              f"onset_within_SD={stats['onset_within_sd_median_min']} min, "
              f"within_SD={stats['within_individual_sd_h']}h, "
              f"between_SD={stats['between_individual_sd_h']}h, "
              f"dur_mean={stats['mean_duration_h']}h")

    # Robustness ranges
    onsets = [r["onset_circ_mean_h"] for r in robustness.values()
              if r["onset_circ_mean_h"] is not None]
    onset_vars = [r["onset_within_sd_median_min"] for r in robustness.values()
                  if r["onset_within_sd_median_min"] is not None]
    within_sds = [r["within_individual_sd_h"] for r in robustness.values()]
    between_sds = [r["between_individual_sd_h"] for r in robustness.values()]

    print(f"\n=== Robustness across 3 variants ===")
    print(f"  Onset circ mean: [{min(onsets):.1f}, {max(onsets):.1f}]h "
          f"(model: {MODEL_ONSET_H}h)")
    print(f"  Onset within-SD: [{min(onset_vars):.0f}, {max(onset_vars):.0f}] min "
          f"(model: {MODEL_ONSET_SD_MIN} min)")
    print(f"  Duration within-SD: [{min(within_sds):.2f}, {max(within_sds):.2f}]h")
    print(f"  Duration between-SD: [{min(between_sds):.2f}, {max(between_sds):.2f}]h")

    # Onset difference
    onset_diffs = []
    for o in onsets:
        d = (MODEL_ONSET_H - o) * 60
        if abs(d) > 720:
            d -= 1440 * np.sign(d)
        onset_diffs.append(int(d))

    result = {
        "withdrawn": {
            "mean_duration": "Withdrawn: sleep scoring algorithm not reproducible (D48, A15). "
                             "Cole-Kripke inapplicable (device scale mismatch). "
                             "No published algorithm for Actiwatch AW4 without recalibration.",
        },
        "retained": {
            "onset": {
                "note": "Sleep onset not calibrated to these data — true prediction at fixed params.",
                "model_onset_h": MODEL_ONSET_H,
                "data_values_h": {v: robustness[v]["onset_circ_mean_h"]
                                  for v in robustness},
                "diff_range_min": [min(onset_diffs), max(onset_diffs)],
            },
            "onset_variability": {
                "note": "Structural gap: model lacks external variability sources.",
                "data_range_min": [min(onset_vars), max(onset_vars)],
                "model_min": MODEL_ONSET_SD_MIN,
                "ratio_range": [round(min(onset_vars) / MODEL_ONSET_SD_MIN, 1),
                                round(max(onset_vars) / MODEL_ONSET_SD_MIN, 1)],
            },
            "variance_decomposition": {
                "note": "Data property, not model comparison. Within > between at all variants.",
                "within_sd_range_h": [min(within_sds), max(within_sds)],
                "between_sd_range_h": [min(between_sds), max(between_sds)],
                "model_sd_h": MODEL_SD_H,
            },
        },
        "robustness": robustness,
        "pre_registration_gap": (
            "Specification required fixing the scoring algorithm but did not require "
            "specifying the device, activity count scale, or coefficient provenance. "
            "An algorithm is fully specified only together with the device it was "
            "calibrated on."
        ),
    }

    out_path = os.path.join(RESULTS_DIR, "p6_arousal_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
