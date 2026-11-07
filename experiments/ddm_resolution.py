"""DDM parameter resolution analysis (vectorized).

For each parameter: sweep on fine grid near calibrated value, batch-simulate
many replicates, estimate minimum detectable difference (MDD).

Outputs: results/ddm_resolution.json
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
cal = json.load(open(ROOT / "results" / "ddm_calibration.json"))

COHERENCES = np.array([0.0, 0.032, 0.064, 0.128, 0.256, 0.512])
DT = 0.001
N_TRIALS = 3500
N_REPS = 50
N_FINE = 21
T_MAX = 11.25
SEED = 314

kappa_cal = cal["kappa"]
a_cal = cal["a"]
Ter_cal = cal["Ter"]

print(f"Calibrated: kappa={kappa_cal}, a={a_cal}, Ter={Ter_cal}")


def analytical_obs(kappa, a, Ter):
    """Analytical predicted observables: 6 accuracy + 6 mean RT."""
    p = kappa * a
    obs = np.zeros(12)
    for ci, coh in enumerate(COHERENCES):
        obs[ci] = 1.0 / (1.0 + np.exp(-p * coh)) if coh > 0 else 0.5
        if coh > 0:
            pc = p * coh
            h = np.tanh(pc / 2) / (2 * pc)
        else:
            h = 0.25
        obs[6 + ci] = a ** 2 * h + Ter
    return obs


def batch_simulate(kappa_arr, a_arr, Ter_arr, n_trials, rng):
    """Simulate DDM for B replicates in parallel. Return (B, 12) observables."""
    B = len(kappa_arr)
    half_a = a_arr / 2

    n_correct = np.zeros((B, 6))
    n_per_coh = np.zeros((B, 6))
    sum_rt = np.zeros((B, 6))

    evidence = np.zeros(B)
    trial_clock = np.zeros(B)
    coh_idx = rng.integers(0, 6, size=B)
    coherence = COHERENCES[coh_idx]
    direction = rng.choice(np.array([-1.0, 1.0]), size=B)

    trial_count = np.zeros(B, dtype=int)
    all_done = np.zeros(B, dtype=bool)

    step = 0
    while not np.all(all_done):
        active = ~all_done
        v = kappa_arr[active] * coherence[active] * direction[active]
        dW = rng.standard_normal(np.sum(active))
        evidence[active] += v * DT + dW * np.sqrt(DT)
        trial_clock[active] += DT

        boundary_hit = np.abs(evidence) >= half_a
        timeout = trial_clock >= T_MAX
        trial_ended = (boundary_hit | timeout) & ~all_done

        if np.any(trial_ended):
            idx = np.where(trial_ended)[0]
            for i in idx:
                ci = coh_idx[i]
                if np.abs(evidence[i]) >= half_a[i]:
                    choice = 1.0 if evidence[i] >= half_a[i] else -1.0
                    correct = float(choice == direction[i])
                    rt = trial_clock[i] + Ter_arr[i]
                    n_per_coh[i, ci] += 1
                    n_correct[i, ci] += correct
                    sum_rt[i, ci] += rt
                # else timeout — skip

                trial_count[i] += 1
                if trial_count[i] >= n_trials:
                    all_done[i] = True

                evidence[i] = 0.0
                trial_clock[i] = 0.0
                coh_idx[i] = rng.integers(0, 6)
                coherence[i] = COHERENCES[coh_idx[i]]
                direction[i] = rng.choice([-1.0, 1.0])

        step += 1

    mask = n_per_coh > 0
    acc = np.where(mask, n_correct / np.maximum(n_per_coh, 1), np.nan)
    rt = np.where(mask, sum_rt / np.maximum(n_per_coh, 1), np.nan)
    return np.hstack([acc, rt])


results = {}
t0 = time.time()

for pname, pcal, prange_frac in [
    ("kappa", kappa_cal, 0.30),
    ("a", a_cal, 0.30),
    ("Ter", Ter_cal, 0.30),
]:
    lo = pcal * (1 - prange_frac)
    hi = pcal * (1 + prange_frac)
    grid = np.linspace(lo, hi, N_FINE)
    step_size = grid[1] - grid[0]

    print(f"\n=== {pname}: [{lo:.4f}, {hi:.4f}], step={step_size:.4f} ===")

    # Build batch: N_FINE × N_REPS replicates
    B = N_FINE * N_REPS
    kappa_batch = np.full(B, kappa_cal)
    a_batch = np.full(B, a_cal)
    Ter_batch = np.full(B, Ter_cal)

    for gi, pval in enumerate(grid):
        sl = slice(gi * N_REPS, (gi + 1) * N_REPS)
        if pname == "kappa":
            kappa_batch[sl] = pval
        elif pname == "a":
            a_batch[sl] = pval
        else:
            Ter_batch[sl] = pval

    rng = np.random.default_rng(SEED)
    print(f"  Simulating {B} replicates x {N_TRIALS} trials...")
    obs = batch_simulate(kappa_batch, a_batch, Ter_batch, N_TRIALS, rng)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # Build prediction grid
    pred_all = np.zeros((N_FINE, 12))
    for gi, pval in enumerate(grid):
        if pname == "kappa":
            pred_all[gi] = analytical_obs(pval, a_cal, Ter_cal)
        elif pname == "a":
            pred_all[gi] = analytical_obs(kappa_cal, pval, Ter_cal)
        else:
            pred_all[gi] = analytical_obs(kappa_cal, a_cal, pval)

    # Recovery
    theta_hat_idx = np.zeros((N_FINE, N_REPS), dtype=int)
    for gi in range(N_FINE):
        for ri in range(N_REPS):
            obs_ri = obs[gi * N_REPS + ri]
            valid = ~np.isnan(obs_ri)
            diffs = pred_all[:, valid] - obs_ri[valid]
            std = np.std(pred_all[:, valid], axis=0)
            std[std < 1e-10] = 1.0
            dist = np.sum((diffs / std) ** 2, axis=1)
            theta_hat_idx[gi, ri] = np.argmin(dist)

    # MDD analysis
    baseline_idx = N_FINE // 2
    print(f"  Discrimination from baseline (index {baseline_idx}, val={grid[baseline_idx]:.4f}):")

    disc_data = []
    for gi in range(N_FINE):
        if gi == baseline_idx:
            continue
        correct_rate = np.mean(theta_hat_idx[gi] != baseline_idx)
        exact_rate = np.mean(theta_hat_idx[gi] == gi)
        delta = grid[gi] - grid[baseline_idx]
        delta_pct = delta / pcal * 100
        disc_data.append({
            "grid_val": round(float(grid[gi]), 4),
            "delta": round(float(delta), 4),
            "delta_pct": round(float(delta_pct), 1),
            "discrimination": round(float(correct_rate), 3),
            "exact_recovery": round(float(exact_rate), 3),
        })
        if abs(gi - baseline_idx) <= 5:
            print(f"    delta={delta:+.4f} ({delta_pct:+.1f}%): "
                  f"disc={correct_rate:.0%}, exact={exact_rate:.0%}")

    # Find MDD (both directions)
    mdd = None
    for dd in disc_data:
        if dd["discrimination"] >= 0.80 and abs(dd["delta"]) < (mdd or float("inf")):
            mdd = abs(dd["delta"])
            mdd_pct = abs(dd["delta_pct"])

    if mdd is None:
        mdd = abs(grid[-1] - grid[baseline_idx])
        mdd_pct = mdd / pcal * 100
        print(f"  MDD not reached at 80%")

    print(f"  MDD({pname}) = {mdd:.4f} ({mdd_pct:.1f}% of calibrated)")

    results[pname] = {
        "calibrated": round(float(pcal), 4),
        "mdd": round(float(mdd), 4),
        "mdd_pct": round(float(mdd_pct), 1),
        "grid_step": round(float(step_size), 4),
        "discrimination_profile": disc_data,
    }

elapsed = time.time() - t0
print(f"\nTotal: {elapsed:.0f}s")

out = {
    "resolution": results,
    "config": {
        "n_fine": N_FINE, "n_reps": N_REPS, "n_trials": N_TRIALS,
        "range_fraction": 0.30,
    },
    "interpretation": "MDD = minimum detectable difference at 80% discrimination "
                      "with N_TRIALS=3500 trials. Recommended knob step.",
}

outpath = ROOT / "results" / "ddm_resolution.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved to {outpath}")
