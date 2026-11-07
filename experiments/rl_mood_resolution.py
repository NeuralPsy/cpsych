"""RL parameter resolution analysis (MDD via Mann-Whitney U).

RESTORED. Original script lost; restored from spec (specs/sandbox.md)
and output format (results/rl_mood_resolution.json).

For each fitted RL parameter (beta, alpha): sweep on grid around
calibrated value, generate N_REPS replicates at calibrated and at
shifted value, compare win_stay_rate distributions with Mann-Whitney U
test, estimate minimum detectable difference (MDD) at POWER threshold.

Observable: win_stay_rate -- primary behavioral summary for RL.
Environment: ReversalBandit (geom mode, p_reward=0.75, volatility=0.3).

Outputs: results/rl_mood_resolution.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cpsych.modules.rl_mood import RLMoodModule
from cpsych.environments.reversal_bandit import ReversalBandit

N_TRIALS = 500
N_REPS = 50
POWER = 0.8
SEED = 42
P_REWARD = 0.75
VOLATILITY = 0.3
TAU_M = 2.163   # from Rutledge gamma=0.63

# Calibrated values (from rl_mood_calibration.json)
ALPHA_CAL = 0.15
BETA_CAL = 2.5


def run_agent_batch(alphas, betas, n_trials, seed):
    """Run batch of RW agents, return win_stay_rate per replicate.

    Parameters
    ----------
    alphas, betas : (B,) arrays
    n_trials : int
    seed : int

    Returns
    -------
    win_stay : (B,) array of win-stay rates
    """
    B = len(alphas)
    rng = np.random.default_rng(seed)
    mod = RLMoodModule()
    env = ReversalBandit(p_reward=P_REWARD, volatility=VOLATILITY)
    env.reset(B, rng)

    x = mod.default_state(B)
    theta = mod.default_params(B)
    theta[:, 0] = alphas
    theta[:, 1] = betas
    theta[:, 2] = TAU_M
    theta[:, 3] = 0.0     # lambda=0 (no feedback during resolution)
    theta[:, 4] = 0.0     # no noise
    mask = np.ones(B, dtype=bool)

    choices = np.zeros((n_trials, B))
    rewards = np.zeros((n_trials, B))

    for t in range(n_trials):
        env.next_trial(rng)
        out = mod.outputs(x, theta)
        p1 = out["choice_prob"]
        choice = (rng.random(B) < p1).astype(float)
        choices[t] = choice
        rew = env.reward(choice, rng)
        r = rew["r"]
        rewards[t] = r
        env_result = {"r": r, "choice": choice}
        x = mod.trial_update(x, theta, mask, env_result)

    # Win-stay rate per replicate
    ws_num = np.zeros(B)
    ws_den = np.zeros(B)
    for t in range(1, n_trials):
        prev_win = rewards[t - 1] == 1
        repeated = choices[t] == choices[t - 1]
        ws_num += (prev_win & repeated).astype(float)
        ws_den += prev_win.astype(float)
    win_stay = np.where(ws_den > 0, ws_num / ws_den, 0.5)
    return win_stay


def estimate_power(ws_baseline, ws_shifted, n_boot, rng):
    """Estimate power via bootstrap resampling of Mann-Whitney U test.

    Resample N_REPS from each group with replacement, run Mann-Whitney U,
    repeat n_boot times. Power = fraction significant at alpha=0.05.
    """
    n_sig = 0
    n_bl = len(ws_baseline)
    n_sh = len(ws_shifted)
    for _ in range(n_boot):
        idx_bl = rng.integers(0, n_bl, size=N_REPS)
        idx_sh = rng.integers(0, n_sh, size=N_REPS)
        sample_bl = ws_baseline[idx_bl]
        sample_sh = ws_shifted[idx_sh]
        _, p = mannwhitneyu(sample_bl, sample_sh, alternative='two-sided')
        if p < 0.05:
            n_sig += 1
    return n_sig / n_boot


def mdd_for_param(param_name, calibrated, grid):
    """Run MDD analysis for one parameter.

    Generates baseline at calibrated values, then for each grid point
    generates shifted samples. Compares via bootstrap Mann-Whitney U.
    """
    print(f"\n=== {param_name}: calibrated={calibrated}, "
          f"grid={grid[0]:.2f}..{grid[-1]:.2f} ===")

    rng_boot = np.random.default_rng(SEED + 7000)
    rng_seeds = np.random.default_rng(SEED + 3000)
    n_boot = N_REPS  # 50 bootstrap reps for power estimation

    # Generate baseline pool (larger than N_REPS for stable bootstrap)
    B_pool = N_REPS * 10
    alphas_bl = np.full(B_pool, ALPHA_CAL)
    betas_bl = np.full(B_pool, BETA_CAL)
    print(f"  Generating baseline pool: B={B_pool}...")
    ws_baseline = run_agent_batch(alphas_bl, betas_bl, N_TRIALS, SEED)

    discrimination = []
    for gi, pval in enumerate(grid):
        if abs(pval - calibrated) < 1e-10:
            continue

        delta = pval - calibrated

        # Build shifted arrays
        alphas_sh = np.full(B_pool, ALPHA_CAL)
        betas_sh = np.full(B_pool, BETA_CAL)
        if param_name == "beta":
            betas_sh[:] = pval
        else:
            alphas_sh[:] = pval

        seed_sh = int(rng_seeds.integers(0, 2**31))
        ws_shifted = run_agent_batch(alphas_sh, betas_sh, N_TRIALS,
                                     seed_sh)

        # Bootstrap power
        power_est = estimate_power(ws_baseline, ws_shifted, n_boot,
                                   rng_boot)
        mean_diff = float(np.mean(ws_shifted) - np.mean(ws_baseline))

        disc_entry = {
            "delta": round(float(delta), 2),
            "value": round(float(pval), 2),
            "power": round(power_est, 3),
            "mean_diff": round(mean_diff, 4),
        }
        discrimination.append(disc_entry)
        print(f"  delta={delta:+.2f} -> val={pval:.2f}: "
              f"power={power_est:.3f}, mean_diff={mean_diff:+.4f}")

    # Find MDD: smallest |delta| with power >= POWER
    mdd_delta = None
    mdd_value = None
    mdd_power = None
    for dd in sorted(discrimination, key=lambda d: abs(d["delta"])):
        if dd["power"] >= POWER:
            mdd_delta = dd["delta"]
            mdd_value = dd["value"]
            mdd_power = dd["power"]
            break

    if mdd_delta is None:
        largest = max(discrimination, key=lambda d: abs(d["delta"]))
        mdd_delta = largest["delta"]
        mdd_value = largest["value"]
        mdd_power = largest["power"]
        print(f"  WARNING: MDD not reached at {POWER}")

    print(f"  MDD({param_name}) = {mdd_delta:+.2f} "
          f"(value={mdd_value:.2f}, power={mdd_power:.3f})")

    return {
        "calibrated": calibrated,
        "mdd": mdd_delta,
        "mdd_value": mdd_value,
        "power": mdd_power,
        "grid": [round(float(g), 2) for g in grid],
        "discrimination": discrimination,
        "N_TRIALS": N_TRIALS,
        "N_REPS": N_REPS,
        "POWER": POWER,
    }


def main():
    t0 = time.time()

    # Beta: grid 1.5 to 3.5, step 0.2
    beta_grid = np.arange(1.5, 3.5 + 0.1, 0.2)
    beta_grid = np.round(beta_grid, 1)
    results_beta = mdd_for_param("beta", BETA_CAL, beta_grid)

    # Alpha: grid 0.05 to 0.25, step 0.02
    alpha_grid = np.arange(0.05, 0.25 + 0.01, 0.02)
    alpha_grid = np.round(alpha_grid, 2)
    results_alpha = mdd_for_param("alpha", ALPHA_CAL, alpha_grid)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s")

    output = {
        "beta": results_beta,
        "alpha": results_alpha,
    }

    outpath = ROOT / "results" / "rl_mood_resolution.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
