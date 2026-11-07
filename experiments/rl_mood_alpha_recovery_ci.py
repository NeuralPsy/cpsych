"""Alpha recovery: bootstrap CI on r, and N-dependence.

1. Bootstrap CI on r(alpha, alpha_hat) from existing 5^4 x 20 recovery.
2. Re-run recovery at N=1000 and N=2000 to find threshold N where r >= 0.70.

Output: results/rl_mood_alpha_recovery_ci.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsych.modules.rl_mood import RLMoodModule, lambda_crit
from cpsych.environments.reversal_bandit import ReversalBandit


def run_agent_batch(alphas, betas, tau_ms, lams, n_trials,
                    p_reward, volatility, seed):
    B = len(alphas)
    rng = np.random.default_rng(seed)
    mod = RLMoodModule()
    env = ReversalBandit(p_reward=p_reward, volatility=volatility)
    env.reset(B, rng)

    x = mod.default_state(B)
    theta = mod.default_params(B)
    theta[:, 0] = alphas
    theta[:, 1] = betas
    theta[:, 2] = tau_ms
    theta[:, 3] = lams
    theta[:, 4] = 0.0
    mask = np.ones(B, dtype=bool)

    choices = np.zeros((n_trials, B))
    rewards = np.zeros((n_trials, B))
    correct_arms = np.zeros((n_trials, B))

    for t in range(n_trials):
        env.next_trial(rng)
        out = mod.outputs(x, theta)
        p1 = out["choice_prob"]
        choice = (rng.random(B) < p1).astype(float)
        choices[t] = choice
        rew = env.reward(choice, rng)
        r = rew["r"]
        rewards[t] = r
        correct_arms[t] = env._correct_arm
        x = mod.trial_update(x, theta, mask, {"r": r, "choice": choice})

    return choices, rewards, correct_arms


def compute_obs(choices, rewards, correct_arms):
    n_trials, B = choices.shape
    ws = np.zeros(B)
    ws_den = np.zeros(B)
    ls = np.zeros(B)
    ls_den = np.zeros(B)

    for t in range(1, n_trials):
        prev_win = rewards[t-1] == 1
        prev_loss = rewards[t-1] == 0
        repeated = choices[t] == choices[t-1]
        ws += (prev_win & repeated).astype(float)
        ws_den += prev_win.astype(float)
        ls += (prev_loss & ~repeated).astype(float)
        ls_den += prev_loss.astype(float)

    win_stay = np.where(ws_den > 0, ws / ws_den, 0.5)
    lose_switch = np.where(ls_den > 0, ls / ls_den, 0.5)

    reversals = np.zeros((n_trials, B), dtype=bool)
    for t in range(1, n_trials):
        reversals[t] = correct_arms[t] != correct_arms[t-1]

    recovery = np.full(B, 20.0)
    for b in range(B):
        rev_times = np.where(reversals[:, b])[0]
        if len(rev_times) == 0:
            continue
        rec_trials = []
        for rt in rev_times:
            correct = (choices[rt:, b] == correct_arms[rt:, b]).astype(float)
            found = False
            for start in range(min(len(correct) - 9, 40)):
                if correct[start:start+10].mean() >= 0.70:
                    rec_trials.append(start)
                    found = True
                    break
            if not found:
                rec_trials.append(40)
        recovery[b] = np.mean(rec_trials) if rec_trials else 20.0

    ac = np.zeros(B)
    for b in range(B):
        c = choices[:, b]
        if np.std(c) > 1e-10:
            ac[b] = np.corrcoef(c[:-1], c[1:])[0, 1]

    return np.column_stack([win_stay, lose_switch, recovery, ac])


def run_recovery(n_trials, seed=42):
    """Full recovery pipeline, return r(alpha)."""
    n_per = 5
    n_reps = 20
    p_reward = 0.75
    volatility = 0.3

    alpha_grid = np.linspace(0.05, 0.35, n_per)
    beta_grid = np.linspace(1.0, 6.0, n_per)
    tau_m_grid = np.linspace(1.0, 10.0, n_per)
    lam_grid = np.linspace(0.0, 0.4, n_per)

    true_params = []
    for a in alpha_grid:
        for b in beta_grid:
            for tm in tau_m_grid:
                lc = lambda_crit(tm, r_mean=0.60)
                for l in lam_grid:
                    stable = l < lc
                    true_params.append((a, b, tm, l, stable))

    n_points = len(true_params)
    B_total = n_points * n_reps

    alphas = np.zeros(B_total)
    betas = np.zeros(B_total)
    tau_ms = np.zeros(B_total)
    lams = np.zeros(B_total)
    stable_flags = np.zeros(B_total, dtype=bool)

    for i, (a, b, tm, l, s) in enumerate(true_params):
        idx = slice(i * n_reps, (i + 1) * n_reps)
        alphas[idx] = a
        betas[idx] = b
        tau_ms[idx] = tm
        lams[idx] = l
        stable_flags[idx] = s

    choices, rewards, correct_arms = run_agent_batch(
        alphas, betas, tau_ms, lams, n_trials, p_reward, volatility, seed)
    obs = compute_obs(choices, rewards, correct_arms)

    ref_obs = np.zeros((n_points, 4))
    for i in range(n_points):
        idx = slice(i * n_reps, (i + 1) * n_reps)
        ref_obs[i] = obs[idx].mean(axis=0)

    obs_std = obs.std(axis=0)
    obs_std[obs_std < 1e-10] = 1.0
    ref_norm = ref_obs / obs_std

    estimated_alpha = np.zeros(B_total)
    for i in range(B_total):
        obs_i = obs[i] / obs_std
        dists = np.sum((ref_norm - obs_i)**2, axis=1)
        best = np.argmin(dists)
        estimated_alpha[i] = true_params[best][0]

    mask = stable_flags
    true_a = alphas[mask]
    est_a = estimated_alpha[mask]
    r = float(np.corrcoef(true_a, est_a)[0, 1])
    return r, true_a, est_a


def bootstrap_ci(true_a, est_a, n_boot=2000, alpha=0.05, seed=123):
    rng = np.random.default_rng(seed)
    n = len(true_a)
    rs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[i] = np.corrcoef(true_a[idx], est_a[idx])[0, 1]
    lo = np.percentile(rs, 100 * alpha / 2)
    hi = np.percentile(rs, 100 * (1 - alpha / 2))
    return float(lo), float(hi), rs


def main():
    results = {}

    # N=500 (original)
    print("Recovery at N=500 (original)...")
    r_500, true_a_500, est_a_500 = run_recovery(500)
    lo_500, hi_500, _ = bootstrap_ci(true_a_500, est_a_500)
    results["N500"] = {
        "r": round(r_500, 4), "ci_lo": round(lo_500, 4),
        "ci_hi": round(hi_500, 4),
    }
    print(f"  r={r_500:.4f} [{lo_500:.4f}, {hi_500:.4f}]")

    # N=1000
    print("Recovery at N=1000...")
    r_1000, true_a_1000, est_a_1000 = run_recovery(1000)
    lo_1000, hi_1000, _ = bootstrap_ci(true_a_1000, est_a_1000)
    results["N1000"] = {
        "r": round(r_1000, 4), "ci_lo": round(lo_1000, 4),
        "ci_hi": round(hi_1000, 4),
    }
    print(f"  r={r_1000:.4f} [{lo_1000:.4f}, {hi_1000:.4f}]")

    # N=2000
    print("Recovery at N=2000...")
    r_2000, true_a_2000, est_a_2000 = run_recovery(2000)
    lo_2000, hi_2000, _ = bootstrap_ci(true_a_2000, est_a_2000)
    results["N2000"] = {
        "r": round(r_2000, 4), "ci_lo": round(lo_2000, 4),
        "ci_hi": round(hi_2000, 4),
    }
    print(f"  r={r_2000:.4f} [{lo_2000:.4f}, {hi_2000:.4f}]")

    # Find threshold
    fitted_at = None
    for key in ["N500", "N1000", "N2000"]:
        if results[key]["ci_lo"] >= 0.70:
            fitted_at = key
            break

    output = {
        "procedure": "5^4 x 20 recovery at 3 trial counts, bootstrap CI (2000 resamples)",
        "results": results,
        "fitted_threshold": fitted_at,
        "verdict": f"alpha fitted at {fitted_at}" if fitted_at else
                   "alpha exploratory at all tested N",
    }

    out_path = (Path(__file__).resolve().parents[1] /
                "results" / "rl_mood_alpha_recovery_ci.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
