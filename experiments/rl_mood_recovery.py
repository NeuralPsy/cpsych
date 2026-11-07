"""Parameter recovery for RL+Mood module.

Grid: alpha x beta x tau_m x lambda = 5^4 = 625 points x 20 replicates.
500 trials per replicate. Observables: win-stay, lose-switch,
recovery speed, choice autocorrelation.

Output: results/rl_mood_recovery.json
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
    """Run batch of agents. alphas etc are (B,) arrays."""
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
        env_result = {"r": r, "choice": choice}
        x = mod.trial_update(x, theta, mask, env_result)

    return choices, rewards, correct_arms


def compute_obs_batch(choices, rewards, correct_arms):
    """Compute observables for each replicate."""
    n_trials, B = choices.shape
    ws = np.zeros(B)
    ws_den = np.zeros(B)
    ls = np.zeros(B)
    ls_den = np.zeros(B)

    for t in range(1, n_trials):
        prev_win = rewards[t-1] == 1
        prev_loss = rewards[t-1] == 0
        repeated = choices[t] == choices[t-1]
        switched = ~repeated
        ws += (prev_win & repeated).astype(float)
        ws_den += prev_win.astype(float)
        ls += (prev_loss & switched).astype(float)
        ls_den += prev_loss.astype(float)

    win_stay = np.where(ws_den > 0, ws / ws_den, 0.5)
    lose_switch = np.where(ls_den > 0, ls / ls_den, 0.5)

    # Recovery speed per replicate
    reversals = np.zeros((n_trials, B), dtype=bool)
    for t in range(1, n_trials):
        reversals[t] = correct_arms[t] != correct_arms[t-1]

    recovery = np.full(B, 20.0)  # default if no reversals
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

    # Choice autocorrelation
    ac = np.zeros(B)
    for b in range(B):
        c = choices[:, b]
        if np.std(c) > 1e-10:
            ac[b] = np.corrcoef(c[:-1], c[1:])[0, 1]

    return np.column_stack([win_stay, lose_switch, recovery, ac])


def main():
    # Grid
    n_per_param = 5
    n_reps = 20
    n_trials = 500
    p_reward = 0.75
    volatility = 0.3
    seed = 42

    alpha_grid = np.linspace(0.05, 0.35, n_per_param)
    beta_grid = np.linspace(1.0, 6.0, n_per_param)
    tau_m_grid = np.linspace(1.0, 10.0, n_per_param)
    lam_grid = np.linspace(0.0, 0.4, n_per_param)

    # Precompute lambda_crit for each tau_m to flag unstable points
    lc_grid = lambda_crit(tau_m_grid, r_mean=0.60)

    print(f"Grid: {n_per_param}^4 = {n_per_param**4} points x {n_reps} reps "
          f"= {n_per_param**4 * n_reps} total agents")

    # Build parameter arrays
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

    print(f"B_total = {B_total}")
    print(f"Stable points: {np.sum(stable_flags) / n_reps:.0f}/{n_points}")

    # Run all agents
    choices, rewards, correct_arms = run_agent_batch(
        alphas, betas, tau_ms, lams, n_trials, p_reward, volatility, seed)

    # Compute observables
    obs = compute_obs_batch(choices, rewards, correct_arms)
    # obs shape: (B_total, 4) = [win_stay, lose_switch, recovery, choice_ac]

    # Grid search recovery: for each replicate, find best-matching grid point
    # Precompute reference observables for each grid point (mean over reps)
    ref_obs = np.zeros((n_points, 4))
    for i in range(n_points):
        idx = slice(i * n_reps, (i + 1) * n_reps)
        ref_obs[i] = obs[idx].mean(axis=0)

    # Standardize
    obs_std = obs.std(axis=0)
    obs_std[obs_std < 1e-10] = 1.0
    ref_norm = ref_obs / obs_std

    # For each replicate, find closest grid point
    estimated = np.zeros((B_total, 4))  # estimated (alpha, beta, tau_m, lam)
    for i in range(B_total):
        obs_i = obs[i] / obs_std
        dists = np.sum((ref_norm - obs_i)**2, axis=1)
        best = np.argmin(dists)
        a, b, tm, l, _ = true_params[best]
        estimated[i] = [a, b, tm, l]

    true_arr = np.column_stack([alphas, betas, tau_ms, lams])
    param_names = ["alpha", "beta", "tau_m", "lambda"]

    # Only use stable points for recovery assessment
    stable_mask = stable_flags

    results = {}
    print("\nRecovery results (stable points only):")
    print(f"{'param':>8} {'r':>8} {'cross_max':>10} {'class':>12}")
    for j, name in enumerate(param_names):
        t = true_arr[stable_mask, j]
        e = estimated[stable_mask, j]
        if np.std(t) < 1e-10:
            r = 0.0
        else:
            r = float(np.corrcoef(t, e)[0, 1])

        # Cross-correlations
        cross = {}
        for k, name2 in enumerate(param_names):
            if k == j:
                continue
            t2 = true_arr[stable_mask, k]
            if np.std(t2) < 1e-10 or np.std(e) < 1e-10:
                cross[name2] = 0.0
            else:
                cross[name2] = float(np.corrcoef(t2, e)[0, 1])

        max_cross = max(abs(v) for v in cross.values())
        fitted = r >= 0.7 and max_cross < 0.6
        cls = "fitted" if fitted else "exploratory"

        results[name] = {
            "r": round(r, 4),
            "cross_correlations": {k: round(v, 4) for k, v in cross.items()},
            "max_cross": round(max_cross, 4),
            "class": cls,
        }
        print(f"{name:>8} {r:8.4f} {max_cross:10.4f} {cls:>12}")

    output = {
        "procedure": f"grid {n_per_param}^4 x {n_reps} reps, {n_trials} trials, "
                     f"p_reward={p_reward}, vol={volatility}, seed={seed}",
        "grid": {
            "alpha": [round(float(a), 4) for a in alpha_grid],
            "beta": [round(float(b), 4) for b in beta_grid],
            "tau_m": [round(float(t), 4) for t in tau_m_grid],
            "lambda": [round(float(l), 4) for l in lam_grid],
        },
        "n_points": n_points,
        "n_stable": int(np.sum(stable_flags) / n_reps),
        "n_reps": n_reps,
        "n_trials": n_trials,
        "observables": ["win_stay", "lose_switch", "recovery_speed", "choice_autocorr"],
        "recovery": results,
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "rl_mood_recovery.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
