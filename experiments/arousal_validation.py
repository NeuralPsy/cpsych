"""Parameter recovery and deferred observables for arousal module.

Spec: specs/arousal.md v2.3.
Run: PYTHONPATH=src python experiments/arousal_validation.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpsych.modules.arousal import (
    ArousalModule, _I, _sigmoid, Q_THRESH, T_TRIM,
    load_arousal_calibration,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
Q_REF = load_arousal_calibration(RESULTS_DIR / "arousal_calibration.json")


# ================================================================
# Batch simulation with online transition detection
# ================================================================

def batch_simulate(theta, dt=1.0, T_days=7, seed=42, noise=True,
                   clamp_vm_min=None, clamp_start_h=None,
                   clamp_end_h=None):
    """Simulate B replicates. Returns transition log and gain swings.

    clamp_vm_min: if set, clamp V_m >= this value during [clamp_start_h, clamp_end_h].
    """
    mod = ArousalModule(q_m_wake_mean_baseline=Q_REF)
    B = theta.shape[0]
    rng = np.random.default_rng(seed)

    x = mod.default_state(B)
    if not noise:
        theta = theta.copy()
        theta[:, _I["sigma_vv"]] = 0.0
        theta[:, _I["sigma_vm"]] = 0.0
        theta[:, _I["sigma_h"]] = 0.0
    u = np.zeros((B, 0))

    T = int(T_days * 86400 / dt)
    max_trans = 100
    debounce_s = 60  # 1 min: prevent double-counting same transition

    trans_times = np.full((B, max_trans), np.nan)
    trans_types = np.zeros((B, max_trans), dtype=np.int8)
    trans_H = np.full((B, max_trans), np.nan)
    trans_count = np.zeros(B, dtype=int)
    last_trans_step = np.full(B, -debounce_s / dt - 1)  # debounce tracker

    Qm0 = _sigmoid(x[:, 1], theta[:, _I["Q_max"]],
                    theta[:, _I["theta_sig"]], theta[:, _I["sigma_q"]])
    was_wake = Qm0 > Q_THRESH

    wake_qm_max = np.full(B, -np.inf)
    wake_qm_min = np.full(B, np.inf)
    gain_swings = np.full((B, 10), np.nan)
    swing_idx = np.zeros(B, dtype=int)

    for step in range(T):
        dx = mod.drift(x, theta, u)
        x = x + dx * dt
        if noise:
            sigma = mod.diffusion(x, theta)
            dW = rng.standard_normal(x.shape)
            x = x + sigma * np.sqrt(dt) * dW

        # Deprivation clamp
        if clamp_vm_min is not None:
            t_h = (step + 1) * dt / 3600.0
            if clamp_start_h <= t_h <= clamp_end_h:
                x[:, 1] = np.maximum(x[:, 1], clamp_vm_min)

        Qm = _sigmoid(x[:, 1], theta[:, _I["Q_max"]],
                       theta[:, _I["theta_sig"]], theta[:, _I["sigma_q"]])
        is_wake = Qm > Q_THRESH

        wm = is_wake
        wake_qm_max[wm] = np.maximum(wake_qm_max[wm], Qm[wm])
        wake_qm_min[wm] = np.minimum(wake_qm_min[wm], Qm[wm])

        changed = is_wake != was_wake
        if np.any(changed):
            t_h = (step + 1) * dt / 3600.0
            for b in np.where(changed)[0]:
                # Debounce: skip if too close to previous transition
                if (step - last_trans_step[b]) * dt < debounce_s:
                    continue

                last_trans_step[b] = step
                was_wake[b] = is_wake[b]

                c = int(trans_count[b])
                if c < max_trans:
                    trans_times[b, c] = t_h
                    trans_types[b, c] = 1 if is_wake[b] else -1
                    trans_H[b, c] = float(x[b, 2])
                    trans_count[b] = c + 1

                if not is_wake[b] and wake_qm_max[b] > 0:
                    si = int(swing_idx[b])
                    if si < 10:
                        gain_swings[b, si] = (
                            (wake_qm_max[b] - wake_qm_min[b])
                            / wake_qm_max[b])
                        swing_idx[b] = si + 1

                if is_wake[b]:
                    wake_qm_max[b] = -np.inf
                    wake_qm_min[b] = np.inf

    return {
        "trans_times": trans_times,
        "trans_types": trans_types,
        "trans_H": trans_H,
        "trans_count": trans_count,
        "gain_swings": gain_swings,
    }


def extract_observables(sim, skip_first_n=1):
    """Extract (B, 35) observable matrix from simulation results.

    Skips first skip_first_n transitions of each type (transient).
    Observables: sleep_onset_1..7, sleep_offset_1..7,
                 sleep_dur_1..7, wake_dur_1..7, gain_swing_1..7.
    """
    B = sim["trans_count"].shape[0]
    obs = np.full((B, 35), np.nan)

    for b in range(B):
        n = int(sim["trans_count"][b])
        onsets = []   # W->S times
        offsets = []  # S->W times
        for i in range(n):
            if sim["trans_types"][b, i] == -1:
                onsets.append(sim["trans_times"][b, i])
            else:
                offsets.append(sim["trans_times"][b, i])

        # Skip first cycle
        onsets = onsets[skip_first_n:]
        offsets = offsets[skip_first_n:]

        for j in range(min(7, len(onsets))):
            obs[b, j] = onsets[j]
        for j in range(min(7, len(offsets))):
            obs[b, 7 + j] = offsets[j]

        # Sleep durations: onset to next offset
        for j in range(min(7, len(onsets), len(offsets))):
            obs[b, 14 + j] = offsets[j] - onsets[j]
        # Wake durations: offset to next onset
        for j in range(min(7, len(offsets), len(onsets) - 1)):
            if j < len(offsets):
                obs[b, 21 + j] = onsets[j + 1] - offsets[j]

        # Gain swings (skip first)
        swings = sim["gain_swings"][b]
        valid = swings[~np.isnan(swings)]
        valid = valid[skip_first_n:] if len(valid) > skip_first_n else valid
        for j in range(min(7, len(valid))):
            obs[b, 28 + j] = valid[j]

    return obs


# ================================================================
# Parameter Recovery
# ================================================================

def run_parameter_recovery():
    """4-parameter recovery: v_vh, v_vc, v_vm, v_mv.

    Grid: 5^4 = 625 nodes × 20 reps = 12500 replicates.
    One-pass design: deterministic predictions for all 625 nodes,
    then noisy data for all 12500 replicates. Recovery = nearest
    node in observable space, no re-simulation.
    """
    print("=" * 60)
    print("PARAMETER RECOVERY (4 params)")
    print("=" * 60)

    PARAM_NAMES = ["v_vh", "v_vc", "v_vm", "v_mv"]
    param_grids = {
        "v_vh": np.linspace(0.08, 0.50, 5),
        "v_vc": np.linspace(-10.0, -1.0, 5),
        "v_vm": np.linspace(-2.70, -0.50, 5),
        "v_mv": np.linspace(-5.0, -0.90, 5),
    }
    n_reps = 20

    grid = np.array([(vh, vc, vm, mv)
                     for vh in param_grids["v_vh"]
                     for vc in param_grids["v_vc"]
                     for vm in param_grids["v_vm"]
                     for mv in param_grids["v_mv"]])
    n_pts = len(grid)  # 625
    B_noisy = n_pts * n_reps  # 12500

    print(f"Grid: 5^4 = {n_pts} points, {n_reps} reps = {B_noisy} total")

    checkpoint_path = RESULTS_DIR / "arousal_recovery_checkpoint.json"

    # ---- Delete stale checkpoints to ensure clean computation ----
    # Checkpoint files are not versioned and silently persist across code
    # changes, producing non-reproducible results if loaded.
    det_obs_path = RESULTS_DIR / "arousal_recovery_det_obs.npy"
    det_cycling_path = RESULTS_DIR / "arousal_recovery_det_cycling.npy"
    noisy_obs_path = RESULTS_DIR / "arousal_recovery_noisy_obs.npy"
    noisy_true_path = RESULTS_DIR / "arousal_recovery_noisy_true.npy"
    for p in [det_obs_path, det_cycling_path, noisy_obs_path,
              noisy_true_path, checkpoint_path]:
        if p.exists():
            p.unlink()

    # ---- Phase 1: Deterministic predictions ----
    print(f"Running deterministic predictions (B={n_pts})...")
    t0 = time.time()
    mod = ArousalModule(q_m_wake_mean_baseline=Q_REF)
    theta_det = mod.default_params(n_pts)
    for i in range(n_pts):
        theta_det[i, _I["v_vh"]] = grid[i, 0]
        theta_det[i, _I["v_vc"]] = grid[i, 1]
        theta_det[i, _I["v_vm"]] = grid[i, 2]
        theta_det[i, _I["v_mv"]] = grid[i, 3]

    sim_det = batch_simulate(theta_det, noise=False, seed=0)
    obs_det = extract_observables(sim_det)
    cycling = sim_det["trans_count"] >= 10
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"  Cycling grid points: {cycling.sum()}/{n_pts}")

    # ---- Phase 2: Noisy simulations ----
    print(f"Running noisy simulations (B={B_noisy})...")
    t0 = time.time()
    mod = ArousalModule(q_m_wake_mean_baseline=Q_REF)
    theta_noisy = mod.default_params(B_noisy)
    theta_true = np.zeros((B_noisy, 4))
    for i in range(n_pts):
        for r in range(n_reps):
            b = i * n_reps + r
            theta_noisy[b, _I["v_vh"]] = grid[i, 0]
            theta_noisy[b, _I["v_vc"]] = grid[i, 1]
            theta_noisy[b, _I["v_vm"]] = grid[i, 2]
            theta_noisy[b, _I["v_mv"]] = grid[i, 3]
            theta_true[b] = grid[i]

    sim_noisy = batch_simulate(theta_noisy, noise=True, seed=42)
    obs_noisy = extract_observables(sim_noisy)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- Phase 3: Recovery by nearest-node lookup ----
    print("Analyzing recovery (nearest-node in observable space)...")

    # Standardize observables using noisy statistics
    obs_std = np.nanstd(obs_noisy, axis=0)
    obs_std[obs_std < 1e-10] = 1.0

    obs_det_std = obs_det / obs_std[np.newaxis, :]
    obs_noisy_std = obs_noisy / obs_std[np.newaxis, :]

    # NaN mask for det (n_pts, 35)
    det_valid = ~np.isnan(obs_det_std)
    obs_det_z = np.where(det_valid, obs_det_std, 0.0)

    theta_hat = np.zeros((B_noisy, 4))
    for b in range(B_noisy):
        obs_b = obs_noisy_std[b]
        b_valid = ~np.isnan(obs_b)
        # Distance only over mutually valid features; normalize by count
        both = b_valid[np.newaxis, :] & det_valid       # (n_pts, 35)
        diff = np.where(both, obs_b[np.newaxis, :] - obs_det_z, 0.0)
        n_valid = np.sum(both, axis=1)
        dists = np.where(n_valid > 0,
                         np.sum(diff ** 2, axis=1) / n_valid, 1e12)
        theta_hat[b] = grid[np.argmin(dists)]

    # ---- Phase 4: Compute statistics ----
    results = {"param_grids": {k: v.tolist() for k, v in param_grids.items()}}

    # Per-parameter r and RMSE
    print(f"\n{'Param':>8} {'r':>8} {'RMSE':>10} {'Verdict':>10}")
    print("-" * 40)
    for p, pname in enumerate(PARAM_NAMES):
        r = float(np.corrcoef(theta_true[:, p], theta_hat[:, p])[0, 1])
        rmse = float(np.sqrt(np.mean((theta_true[:, p] - theta_hat[:, p]) ** 2)))
        verdict = "EXPOSED" if r >= 0.7 else "internal"
        results[pname] = {"r": round(r, 4), "rmse": round(rmse, 4),
                          "verdict": verdict}
        print(f"{pname:>8} {r:>8.4f} {rmse:>10.4f} {verdict:>10}")

    # Cross-correlation matrix: r(theta_true_i, theta_hat_j)
    n_p = len(PARAM_NAMES)
    cross = np.zeros((n_p, n_p))
    print(f"\n{'':>12}", end="")
    for pn in PARAM_NAMES:
        print(f"{pn+'_hat':>12}", end="")
    print()
    for i in range(n_p):
        print(f"{PARAM_NAMES[i]+'_true':>12}", end="")
        for j in range(n_p):
            r = float(np.corrcoef(theta_true[:, i], theta_hat[:, j])[0, 1])
            cross[i, j] = r
            print(f"{r:>12.4f}", end="")
        print()
    results["cross_correlation"] = cross.tolist()

    # Apply cross-correlation disqualification
    for i in range(n_p):
        for j in range(n_p):
            if i != j and abs(cross[i, j]) >= 0.6:
                pn = PARAM_NAMES[j]
                print(f"  WARNING: |r({PARAM_NAMES[i]}_true, {pn}_hat)| "
                      f"= {abs(cross[i,j]):.4f} >= 0.6")
                if results[pn]["verdict"] == "EXPOSED":
                    results[pn]["verdict"] = "internal (cross-corr)"

    # Confusion matrices
    print("\n--- Confusion matrices (true rows, estimated cols) ---")
    for p, pname in enumerate(PARAM_NAMES):
        vals = param_grids[pname]
        n_v = len(vals)
        cm = np.zeros((n_v, n_v), dtype=int)
        for b in range(B_noisy):
            true_idx = int(np.argmin(np.abs(vals - theta_true[b, p])))
            hat_idx = int(np.argmin(np.abs(vals - theta_hat[b, p])))
            cm[true_idx, hat_idx] += 1
        print(f"\n{pname} (n_reps={n_reps} per cell, "
              f"n_other={n_pts // n_v} other combos):")
        print(f"{'true\\est':>10}", end="")
        for v in vals:
            print(f"{v:>8.2f}", end="")
        print()
        for i in range(n_v):
            print(f"{vals[i]:>10.2f}", end="")
            for j in range(n_v):
                print(f"{cm[i,j]:>8d}", end="")
            print()
        results[pname]["confusion"] = cm.tolist()

    # Per-node theta_hat means (for shape-of-recovery analysis)
    print("\n--- Per-node recovery (theta_hat means across 20 reps) ---")
    for p, pname in enumerate(PARAM_NAMES):
        vals = param_grids[pname]
        # Marginal: for each true value of this param, mean theta_hat
        node_true = []
        node_hat_mean = []
        node_hat_std = []
        for v in vals:
            mask = np.abs(theta_true[:, p] - v) < 1e-6
            if mask.sum() > 0:
                node_true.append(float(v))
                node_hat_mean.append(float(np.mean(theta_hat[mask, p])))
                node_hat_std.append(float(np.std(theta_hat[mask, p])))
        results[pname]["node_true"] = node_true
        results[pname]["node_hat_mean"] = node_hat_mean
        results[pname]["node_hat_std"] = node_hat_std
        print(f"\n{pname}:")
        print(f"  {'true':>8} {'hat_mean':>10} {'hat_std':>10}")
        for t, m, s in zip(node_true, node_hat_mean, node_hat_std):
            print(f"  {t:>8.3f} {m:>10.3f} {s:>10.3f}")

    # Final verdicts
    print("\n--- FINAL VERDICT ---")
    any_exposed = False
    for pn in PARAM_NAMES:
        v = results[pn]["verdict"]
        r = results[pn]["r"]
        print(f"  {pn}: {v} (r={r:.4f})")
        if v == "EXPOSED":
            any_exposed = True

    if not any_exposed:
        print("\n  *** NO PARAMETER PASSES THRESHOLD — ESCALATION REQUIRED ***")
    results["any_exposed"] = any_exposed

    # Save full results
    out_path = RESULTS_DIR / "arousal_recovery.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    with open(out_path, "w") as f:
        json.dump(json.loads(json.dumps(results, default=convert)),
                  f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ================================================================
# Deferred Observables
# ================================================================

def _extract_paired_cycles(sim, b=0, skip_before_h=24.0):
    """Extract properly paired sleep-wake cycles from transition log.

    Returns list of dicts: {onset_h, offset_h, sleep_dur_h, H_at_onset}.
    Pairing: each W->S (onset) paired with the NEXT S->W (offset).
    """
    n = int(sim["trans_count"][b])
    transitions = []
    for i in range(n):
        t = sim["trans_times"][b, i]
        if t < skip_before_h:
            continue
        transitions.append({
            "time": t,
            "type": int(sim["trans_types"][b, i]),  # -1=W->S, +1=S->W
            "H": float(sim["trans_H"][b, i]),
        })

    cycles = []
    i = 0
    while i < len(transitions):
        if transitions[i]["type"] == -1:  # W->S = sleep onset
            onset = transitions[i]
            # Find next S->W
            for j in range(i + 1, len(transitions)):
                if transitions[j]["type"] == 1:  # S->W = wake onset
                    offset = transitions[j]
                    cycles.append({
                        "onset_h": onset["time"],
                        "offset_h": offset["time"],
                        "sleep_dur_h": offset["time"] - onset["time"],
                        "H_at_onset": onset["H"],
                    })
                    i = j + 1
                    break
            else:
                break
        else:
            i += 1
    return cycles


def deferred_deprivation_and_rebound():
    """Deferred #1 and #2: sleep deprivation response and H rebound."""
    print("\n" + "=" * 60)
    print("DEFERRED #1 + #2: DEPRIVATION & REBOUND")
    print("=" * 60)

    mod = ArousalModule(q_m_wake_mean_baseline=Q_REF)
    theta = mod.default_params(1)
    theta[0, _I["sigma_vv"]] = 0.0
    theta[0, _I["sigma_vm"]] = 0.0
    theta[0, _I["sigma_h"]] = 0.0

    # Baseline 7-day run
    sim_base = batch_simulate(theta, noise=False, T_days=7, seed=0)
    cycles_base = _extract_paired_cycles(sim_base, b=0, skip_before_h=24.0)

    if len(cycles_base) < 3:
        print("ERROR: fewer than 3 baseline cycles")
        return {"passed": False, "reason": "not enough baseline cycles"}

    base_sleep_dur = np.mean([c["sleep_dur_h"] for c in cycles_base])
    base_onset_tod = np.mean([c["onset_h"] % 24 for c in cycles_base])
    base_H = np.mean([c["H_at_onset"] for c in cycles_base])

    print(f"Baseline ({len(cycles_base)} cycles):")
    print(f"  Sleep onset ToD: ~{base_onset_tod:.1f}h")
    print(f"  Sleep duration: ~{base_sleep_dur:.2f}h")
    print(f"  H at onset: ~{base_H:.2f}")
    onset_strs = [f"{c['onset_h']:.1f}" for c in cycles_base]
    print(f"  Onsets: {onset_strs}")

    # Deprivation: clamp V_m >= 1.0 (normal wake equilibrium).
    # V_m = 1.0 gives Q_m ≈ 4.7 (normal wake level), so H continues
    # to accumulate at the normal wake rate. V_m >= 0 would give
    # Q_m = 3.45, where H would actually decrease (wrong model).
    #
    # Start just before 2nd bedtime, run 18h = ~34h total continuous
    # wakefulness. Release in mid-afternoon (circadian alertness).
    second_onset = cycles_base[1]["onset_h"]  # ~71h
    clamp_start = second_onset - 0.5  # just before normal bedtime
    clamp_duration = 18.0
    clamp_end = clamp_start + clamp_duration
    third_onset = cycles_base[2]["onset_h"]  # ~95h (expected without dep.)

    print(f"\nDeprivation: clamp V_m >= 1.0 from {clamp_start:.1f}h to "
          f"{clamp_end:.1f}h ({clamp_duration:.0f}h forced wake)")
    print(f"Expected next normal onset: ~{third_onset:.1f}h")

    sim_dep = batch_simulate(theta, noise=False, T_days=7, seed=0,
                             clamp_vm_min=1.0,
                             clamp_start_h=clamp_start,
                             clamp_end_h=clamp_end)

    # Find recovery sleep: first W->S after clamp_end
    cycles_dep = _extract_paired_cycles(sim_dep, b=0, skip_before_h=clamp_end)

    if not cycles_dep:
        print("  ERROR: no recovery sleep detected!")
        return {"passed": False, "reason": "no recovery sleep"}

    rec = cycles_dep[0]
    onset_advance = third_onset - rec["onset_h"]
    dur_excess = rec["sleep_dur_h"] - base_sleep_dur
    h_excess_pct = (rec["H_at_onset"] - base_H) / base_H * 100

    print(f"\nRecovery sleep:")
    print(f"  Onset: {rec['onset_h']:.2f}h (expected ~{third_onset:.2f}h)")
    print(f"  Onset advance: {onset_advance:.2f}h (target >= 1h)")
    print(f"  Duration: {rec['sleep_dur_h']:.2f}h (baseline ~{base_sleep_dur:.2f}h)")
    print(f"  Duration excess: {dur_excess:.2f}h (target >= 1h)")
    print(f"  H at onset: {rec['H_at_onset']:.2f} (baseline ~{base_H:.2f})")
    print(f"  H excess: {h_excess_pct:.1f}% (target >= 20%)")

    dep_pass = onset_advance >= 1.0 and dur_excess >= 1.0
    reb_pass = h_excess_pct >= 20.0
    print(f"\n  Deprivation (#1): {'PASS' if dep_pass else 'FAIL'}")
    print(f"  Rebound (#2): {'PASS' if reb_pass else 'FAIL'}")

    return {
        "deprivation": {
            "onset_advance_h": round(onset_advance, 2),
            "duration_excess_h": round(dur_excess, 2),
            "passed": dep_pass,
        },
        "rebound": {
            "H_recovery": round(float(rec["H_at_onset"]), 2),
            "H_baseline": round(float(base_H), 2),
            "H_excess_pct": round(h_excess_pct, 1),
            "passed": reb_pass,
        },
    }


def deferred_transition_sharpness():
    """Deferred #3: transition time 10% -> 90% of gain < 30 min."""
    print("\n" + "=" * 60)
    print("DEFERRED #3: TRANSITION SHARPNESS")
    print("=" * 60)

    mod = ArousalModule(q_m_wake_mean_baseline=Q_REF)
    x = mod.default_state(1)
    theta = mod.default_params(1)
    theta[0, _I["sigma_vv"]] = 0.0
    theta[0, _I["sigma_vm"]] = 0.0
    theta[0, _I["sigma_h"]] = 0.0
    u = np.zeros((1, 0))

    # Run 3 days, record gain near S->W transitions
    T = 3 * 86400
    was_wake = False
    transition_times = []
    # Record gain at 1s resolution around transitions
    gain_log = []  # (time_h, gain)

    for step in range(T):
        dx = mod.drift(x, theta, u)
        x = x + dx * 1.0
        Qm = _sigmoid(x[0, 1], theta[0, _I["Q_max"]],
                       theta[0, _I["theta_sig"]], theta[0, _I["sigma_q"]])
        g = Qm / Q_REF
        t_h = (step + 1) / 3600.0
        is_wake = Qm > Q_THRESH

        if is_wake and not was_wake and t_h > 24:
            transition_times.append(t_h)

        # Record near transitions (±2h)
        for tt in transition_times:
            if abs(t_h - tt) < 2.0:
                gain_log.append((t_h, g))
                break

        was_wake = is_wake

    # Analyze each transition
    rise_times = []
    for tt in transition_times:
        nearby = [(t, g) for t, g in gain_log if abs(t - tt) < 1.5]
        if len(nearby) < 10:
            continue
        gains = np.array([g for _, g in nearby])
        times = np.array([t for t, _ in nearby])
        g_max = gains.max()
        g_10 = 0.10 * g_max
        g_90 = 0.90 * g_max

        t_10 = times[np.argmax(gains >= g_10)] if np.any(gains >= g_10) else np.nan
        t_90 = times[np.argmax(gains >= g_90)] if np.any(gains >= g_90) else np.nan
        if not np.isnan(t_10) and not np.isnan(t_90):
            rise_min = (t_90 - t_10) * 60  # hours -> minutes
            rise_times.append(rise_min)

    if rise_times:
        mean_rise = np.mean(rise_times)
        max_rise = np.max(rise_times)
    else:
        mean_rise = max_rise = np.nan

    passed = max_rise < 30.0 if not np.isnan(max_rise) else False
    print(f"S->W transitions found: {len(transition_times)}")
    print(f"Rise times (10%->90%): {[f'{r:.1f}' for r in rise_times]} min")
    print(f"Mean: {mean_rise:.1f} min, Max: {max_rise:.1f} min (target < 30)")
    print(f"Transition sharpness (#3): {'PASS' if passed else 'FAIL'}")

    return {
        "rise_times_min": [round(r, 1) for r in rise_times],
        "mean_min": round(mean_rise, 1),
        "max_min": round(max_rise, 1),
        "passed": passed,
    }


def deferred_spurious_transitions():
    """Deferred #4: noise doesn't cause spurious transitions."""
    print("\n" + "=" * 60)
    print("DEFERRED #4: SPURIOUS TRANSITIONS FROM NOISE")
    print("=" * 60)

    B = 100
    mod = ArousalModule(q_m_wake_mean_baseline=Q_REF)

    # Deterministic reference
    theta_det = mod.default_params(1)
    theta_det[0, _I["sigma_vv"]] = 0.0
    theta_det[0, _I["sigma_vm"]] = 0.0
    theta_det[0, _I["sigma_h"]] = 0.0
    sim_det = batch_simulate(theta_det, noise=False, T_days=7, seed=0)
    n_det = int(sim_det["trans_count"][0])
    det_times = sim_det["trans_times"][0, :n_det]

    print(f"Deterministic: {n_det} transitions")

    # Noisy runs
    theta_noisy = mod.default_params(B)
    print(f"Running B={B} noisy simulations...")
    t0 = time.time()
    sim_noisy = batch_simulate(theta_noisy, noise=True, T_days=7, seed=123)
    print(f"  Done in {time.time()-t0:.1f}s")

    noisy_counts = sim_noisy["trans_count"].astype(int)
    median_count = float(np.median(noisy_counts))
    count_ratio = abs(median_count - n_det) / n_det

    print(f"Noisy median transitions: {median_count} "
          f"(det={n_det}, diff={count_ratio*100:.1f}%)")

    # Check timing: what fraction of noisy transitions are >2h from any
    # deterministic transition?
    n_outside = 0
    n_total = 0
    for b in range(B):
        nc = int(noisy_counts[b])
        for i in range(nc):
            t_noisy = sim_noisy["trans_times"][b, i]
            if t_noisy < 24:  # skip transient
                continue
            n_total += 1
            min_dist = np.min(np.abs(det_times - t_noisy))
            if min_dist > 2.0:
                n_outside += 1

    frac_outside = n_outside / n_total if n_total > 0 else 0.0

    count_pass = count_ratio <= 0.10
    timing_pass = frac_outside <= 0.05
    passed = count_pass and timing_pass

    print(f"Transitions >2h from baseline: {n_outside}/{n_total} "
          f"= {frac_outside*100:.1f}% (target <= 5%)")
    print(f"Count criterion (<=10% diff): {'PASS' if count_pass else 'FAIL'}")
    print(f"Timing criterion (<=5% outside): {'PASS' if timing_pass else 'FAIL'}")
    print(f"Spurious transitions (#4): {'PASS' if passed else 'FAIL'}")

    return {
        "det_count": n_det,
        "noisy_median": median_count,
        "count_diff_pct": round(count_ratio * 100, 1),
        "outside_2h_pct": round(frac_outside * 100, 1),
        "count_pass": count_pass,
        "timing_pass": timing_pass,
        "passed": passed,
    }


# ================================================================
# Main
# ================================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "recovery"

    if mode == "recovery":
        run_parameter_recovery()
    elif mode == "deferred":
        all_results = {}
        all_results["deferred_1_2"] = deferred_deprivation_and_rebound()
        all_results["deferred_3"] = deferred_transition_sharpness()
        all_results["deferred_4"] = deferred_spurious_transitions()

        out_path = RESULTS_DIR / "arousal_deferred.json"
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        with open(out_path, "w") as f:
            json.dump(json.loads(json.dumps(all_results, default=convert)),
                      f, indent=2)
        print(f"\nDeferred results saved to {out_path}")
    else:
        print(f"Unknown mode: {mode}. Use 'recovery' or 'deferred'.")


if __name__ == "__main__":
    main()
