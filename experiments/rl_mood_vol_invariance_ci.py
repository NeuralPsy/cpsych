"""Bootstrap CI on volatility invariance |diff|.

|recovery_speed(vol=0.1) - recovery_speed(vol=1.0)| with CI.
PASS requires CI upper bound < 2.

Output: results/rl_mood_vol_invariance_ci.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsych.modules.rl_mood import RLMoodModule, lambda_crit
from cpsych.environments.reversal_bandit import ReversalBandit


def run_and_get_recovery(alpha, beta, tau_m, lam, n_trials, B, p_reward,
                         volatility, seed):
    """Run agent, return per-replicate recovery speed."""
    rng = np.random.default_rng(seed)
    mod = RLMoodModule()
    env = ReversalBandit(p_reward=p_reward, volatility=volatility)
    env.reset(B, rng)

    x = mod.default_state(B)
    theta = mod.default_params(B)
    theta[:, 0] = alpha
    theta[:, 1] = beta
    theta[:, 2] = tau_m
    theta[:, 3] = lam
    theta[:, 4] = 0.0
    mask = np.ones(B, dtype=bool)

    choices = np.zeros((n_trials, B))
    correct_arms = np.zeros((n_trials, B))

    for t in range(n_trials):
        env.next_trial(rng)
        out = mod.outputs(x, theta)
        choice = (rng.random(B) < out["choice_prob"]).astype(float)
        choices[t] = choice
        rew = env.reward(choice, rng)
        correct_arms[t] = env._correct_arm
        x = mod.trial_update(x, theta, mask, {"r": rew["r"], "choice": choice})

    # Recovery speed per replicate
    reversals = np.zeros((n_trials, B), dtype=bool)
    for t in range(1, n_trials):
        reversals[t] = correct_arms[t] != correct_arms[t-1]

    recovery = np.full(B, np.nan)
    for b in range(B):
        rev_times = np.where(reversals[:, b])[0]
        if len(rev_times) == 0:
            continue
        rec_trials = []
        for rt in rev_times:
            c = (choices[rt:, b] == correct_arms[rt:, b]).astype(float)
            found = False
            for start in range(min(len(c) - 9, 40)):
                if c[start:start+10].mean() >= 0.70:
                    rec_trials.append(start)
                    found = True
                    break
            if not found:
                rec_trials.append(40)
        recovery[b] = np.mean(rec_trials) if rec_trials else np.nan

    return recovery


def main():
    alpha = 0.15
    beta = 2.50
    tau_m = 2.163
    lc = lambda_crit(tau_m, r_mean=0.60)
    lam = 0.5 * lc
    n_trials = 500
    B = 500  # more replicates for tighter CI
    p_reward = 0.75
    seed = 42

    print(f"Running volatility invariance with B={B}...")

    rec_stable = run_and_get_recovery(
        alpha, beta, tau_m, lam, n_trials, B, p_reward, 0.1, seed)
    rec_volatile = run_and_get_recovery(
        alpha, beta, tau_m, lam, n_trials, B, p_reward, 1.0, seed)

    # Filter NaN (replicates with no reversals at vol=0.1)
    valid_s = ~np.isnan(rec_stable)
    valid_v = ~np.isnan(rec_volatile)
    print(f"  Valid replicates: stable={valid_s.sum()}, volatile={valid_v.sum()}")

    # Point estimate
    mean_s = float(np.nanmean(rec_stable))
    mean_v = float(np.nanmean(rec_volatile))
    diff = abs(mean_s - mean_v)
    print(f"  Mean recovery: stable={mean_s:.2f}, volatile={mean_v:.2f}, |diff|={diff:.2f}")

    # Bootstrap CI on |diff|
    n_boot = 5000
    rng = np.random.default_rng(123)
    diffs = np.zeros(n_boot)
    rs = rec_stable[valid_s]
    rv = rec_volatile[valid_v]
    for i in range(n_boot):
        idx_s = rng.integers(0, len(rs), size=len(rs))
        idx_v = rng.integers(0, len(rv), size=len(rv))
        diffs[i] = abs(np.mean(rs[idx_s]) - np.mean(rv[idx_v]))

    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    pass_flag = ci_hi < 2.0

    print(f"  |diff| = {diff:.2f}, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  PASS (CI hi < 2.0): {'YES' if pass_flag else 'NO'}")

    output = {
        "procedure": f"B={B}, N={n_trials}, 5000 bootstrap resamples",
        "mean_recovery_stable_vol01": round(mean_s, 3),
        "mean_recovery_volatile_vol10": round(mean_v, 3),
        "abs_diff": round(diff, 3),
        "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
        "threshold": 2.0,
        "pass": bool(pass_flag),
        "n_valid_stable": int(valid_s.sum()),
        "n_valid_volatile": int(valid_v.sum()),
    }

    out_path = (Path(__file__).resolve().parents[1] /
                "results" / "rl_mood_vol_invariance_ci.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
