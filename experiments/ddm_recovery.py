"""DDM parameter recovery.

Grid: kappa × a × Ter = 5³ = 125 points × 20 replicates = 2500 total.
Each replicate: ~3500 trials (matching monkey N count).
Observables: per-coherence accuracy (6) + per-coherence mean RT (6) = 12.

Procedure (specs/ddm.md v1.2):
  For each replicate: simulate DDM with known (kappa, a, Ter),
  compute 12 observables, then estimate theta_hat via grid search.
  Report r(theta, theta_hat) per parameter.

Outputs: results/ddm_recovery.json
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# --- Grid ---
N_PER_PARAM = 5
N_REPLICATES = 20
N_TRIALS = 3500  # ~590 per coherence (matching monkey N)
COHERENCES = np.array([0.0, 0.032, 0.064, 0.128, 0.256, 0.512])
DT = 0.001
SEED = 42

kappa_grid = np.linspace(5.0, 30.0, N_PER_PARAM)
a_grid = np.linspace(0.5, 3.0, N_PER_PARAM)
Ter_grid = np.linspace(0.1, 0.5, N_PER_PARAM)

print(f"Grid: kappa={kappa_grid}")
print(f"       a={a_grid}")
print(f"       Ter={Ter_grid}")
print(f"Points: {N_PER_PARAM**3} × {N_REPLICATES} = {N_PER_PARAM**3 * N_REPLICATES} replicates")
print(f"Trials per replicate: {N_TRIALS}")

# --- Build parameter array ---
params_list = []
for k in kappa_grid:
    for a in a_grid:
        for t in Ter_grid:
            params_list.append((k, a, t))
params_arr = np.array(params_list)  # (125, 3)
n_points = len(params_arr)

# Expand with replicates
# Total B = n_points * N_REPLICATES
B_total = n_points * N_REPLICATES
print(f"Total B = {B_total}")

# Repeat each parameter set N_REPLICATES times
theta_true = np.repeat(params_arr, N_REPLICATES, axis=0)  # (B_total, 3)

# --- Simulate ---
rng = np.random.default_rng(SEED)

# Per-replicate accumulators
# For each replicate, track per-coherence counts
n_correct = np.zeros((B_total, 6))  # per coherence
n_trials_per_coh = np.zeros((B_total, 6))
sum_rt = np.zeros((B_total, 6))

# Current trial state
evidence = np.zeros(B_total)
trial_clock = np.zeros(B_total)
# Current stimulus
coh_idx = rng.integers(0, 6, size=B_total)
coherence = COHERENCES[coh_idx]
direction = rng.choice(np.array([-1.0, 1.0]), size=B_total)

# Trial count per replicate
trial_count = np.zeros(B_total, dtype=int)
# Track which replicates are still running
all_done_mask = np.zeros(B_total, dtype=bool)

# T_max per trial (computational safeguard)
T_MAX = 5.0 * (3.0 / 2) ** 2  # 11.25 sec

print("\nSimulating...")
t0 = time.time()

kappa = theta_true[:, 0]
a = theta_true[:, 1]
Ter = theta_true[:, 2]
half_a = a / 2

step = 0
max_steps = int(N_TRIALS * T_MAX / DT)  # absolute upper bound

while not np.all(all_done_mask):
    # Drift + diffusion (only for active replicates)
    active = ~all_done_mask
    v = kappa[active] * coherence[active] * direction[active]
    dW = rng.standard_normal(np.sum(active))
    evidence[active] += v * DT + dW * np.sqrt(DT)
    trial_clock[active] += DT

    # Check trial completion
    boundary_hit = np.abs(evidence) >= half_a
    timeout = trial_clock >= T_MAX
    trial_ended = (boundary_hit | timeout) & ~all_done_mask

    if np.any(trial_ended):
        idx = np.where(trial_ended)[0]

        for i in idx:
            ci = coh_idx[i]
            # Choice
            if np.abs(evidence[i]) >= half_a[i]:
                choice = 1.0 if evidence[i] >= half_a[i] else -1.0
            else:
                # timeout — mark as truncated, exclude from stats
                trial_count[i] += 1
                if trial_count[i] >= N_TRIALS:
                    all_done_mask[i] = True
                # Reset and new trial
                evidence[i] = 0.0
                trial_clock[i] = 0.0
                coh_idx[i] = rng.integers(0, 6)
                coherence[i] = COHERENCES[coh_idx[i]]
                direction[i] = rng.choice([-1.0, 1.0])
                continue

            correct = float(choice == direction[i])
            rt = trial_clock[i] + Ter[i]

            n_trials_per_coh[i, ci] += 1
            n_correct[i, ci] += correct
            sum_rt[i, ci] += rt

            trial_count[i] += 1
            if trial_count[i] >= N_TRIALS:
                all_done_mask[i] = True

            # Reset
            evidence[i] = 0.0
            trial_clock[i] = 0.0
            coh_idx[i] = rng.integers(0, 6)
            coherence[i] = COHERENCES[coh_idx[i]]
            direction[i] = rng.choice([-1.0, 1.0])

    step += 1
    if step % 100000 == 0:
        pct = np.mean(all_done_mask) * 100
        elapsed = time.time() - t0
        print(f"  step {step:,}: {pct:.0f}% done, "
              f"min_trials={np.min(trial_count)}, "
              f"elapsed={elapsed:.0f}s")

    if step > max_steps:
        print("WARNING: max steps reached")
        break

elapsed = time.time() - t0
print(f"Simulation done: {step:,} steps, {elapsed:.1f}s")
print(f"Min trial count: {np.min(trial_count)}")

# --- Compute observables ---
# Per-coherence accuracy and mean RT
acc_obs = np.zeros((B_total, 6))
rt_obs = np.zeros((B_total, 6))
for ci in range(6):
    mask = n_trials_per_coh[:, ci] > 0
    acc_obs[mask, ci] = n_correct[mask, ci] / n_trials_per_coh[mask, ci]
    rt_obs[mask, ci] = sum_rt[mask, ci] / n_trials_per_coh[mask, ci]
    acc_obs[~mask, ci] = np.nan
    rt_obs[~mask, ci] = np.nan

# Combined observable vector: [acc_0..5, rt_0..5]
obs = np.hstack([acc_obs, rt_obs])  # (B_total, 12)

# --- Grid search recovery ---
print("\nRunning grid search recovery...")

# For each parameter point, compute predicted observables
# using analytical DDM formulas
pred_obs = np.zeros((n_points, 12))
for pi, (k, aa, t) in enumerate(params_arr):
    p = k * aa
    for ci, coh in enumerate(COHERENCES):
        # Accuracy
        pred_acc = 1.0 / (1.0 + np.exp(-p * coh)) if coh > 0 else 0.5
        # Mean DT
        if coh > 0:
            pc = p * coh
            h = np.tanh(pc / 2) / (2 * pc)
        else:
            h = 0.25
        pred_rt = aa ** 2 * h + t
        pred_obs[pi, ci] = pred_acc
        pred_obs[pi, 6 + ci] = pred_rt

# For each replicate, find best-matching parameter point
theta_hat = np.zeros((B_total, 3))

for ri in range(B_total):
    obs_ri = obs[ri]
    valid = ~np.isnan(obs_ri)
    if np.sum(valid) < 6:
        theta_hat[ri] = np.nan
        continue

    # Standardized L2 distance
    diffs = pred_obs[:, valid] - obs_ri[valid]
    # Normalize by std across grid points
    std = np.std(pred_obs[:, valid], axis=0)
    std[std < 1e-10] = 1.0
    dist = np.sum((diffs / std) ** 2, axis=1)
    best = np.argmin(dist)
    theta_hat[ri] = params_arr[best]

# --- Compute correlations ---
print("\nRecovery results:")
param_names = ["kappa", "a", "Ter"]
results = {}

for pi, pname in enumerate(param_names):
    true = theta_true[:, pi]
    est = theta_hat[:, pi]
    valid = ~np.isnan(est)
    r = np.corrcoef(true[valid], est[valid])[0, 1]
    rmse = np.sqrt(np.mean((true[valid] - est[valid]) ** 2))
    print(f"  {pname}: r = {r:.4f}, RMSE = {rmse:.4f}")
    results[pname] = {"r": round(float(r), 4), "rmse": round(float(rmse), 4)}

# Cross-correlations
print("\nCross-correlations:")
cross = {}
for pi, pn1 in enumerate(param_names):
    for pj, pn2 in enumerate(param_names):
        if pi >= pj:
            continue
        true_i = theta_true[:, pi]
        est_j = theta_hat[:, pj]
        valid = ~np.isnan(est_j)
        r = np.corrcoef(true_i[valid], est_j[valid])[0, 1]
        print(f"  r({pn1}_true, {pn2}_hat) = {r:.4f}")
        cross[f"{pn1}_true_vs_{pn2}_hat"] = round(float(r), 4)

# --- Save ---
out = {
    "grid": {
        "kappa": kappa_grid.tolist(),
        "a": a_grid.tolist(),
        "Ter": Ter_grid.tolist(),
        "n_per_param": N_PER_PARAM,
        "n_replicates": N_REPLICATES,
        "n_trials": N_TRIALS,
        "n_points": n_points,
        "B_total": B_total,
    },
    "recovery": results,
    "cross_correlations": cross,
    "classification": {},
    "simulation": {
        "steps": step,
        "elapsed_sec": round(elapsed, 1),
        "min_trial_count": int(np.min(trial_count)),
    },
}

# Classify
for pname in param_names:
    r = results[pname]["r"]
    # Check cross-correlation
    max_cross = 0
    for key, val in cross.items():
        if pname in key:
            max_cross = max(max_cross, abs(val))
    if r >= 0.7 and max_cross < 0.6:
        cls = "fitted"
    elif r >= 0.7:
        cls = "fitted (cross-corr warning)"
    else:
        cls = "exploratory"
    out["classification"][pname] = cls
    print(f"  {pname}: {cls}")

outpath = ROOT / "results" / "ddm_recovery.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}")
