"""Calibrate alpha, beta for RL+Mood module.

Procedure (specs/rl_mood.md v1.3):
  Step 1-4: grid search alpha x beta with lambda=0.
    Target: win-stay ~ 0.70, recovery_speed ~ 8-12 trials.
  Step 5: tau_m = -1/ln(0.63) = 2.163 (fixed from Rutledge gamma).
  Lambda: uncalibrated exploratory. Default = 0.5 * lambda_crit.

Output: results/rl_mood_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsych.modules.rl_mood import RLMoodModule, lambda_crit
from cpsych.environments.reversal_bandit import ReversalBandit


def run_agent(alpha, beta, tau_m, lam, n_trials, B, p_reward, volatility,
              seed):
    """Run RW+mood agent, return behavioral observables."""
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
    theta[:, 4] = 0.0  # no noise for calibration
    mask = np.ones(B, dtype=bool)

    choices = np.zeros((n_trials, B))
    rewards = np.zeros((n_trials, B))
    correct_arms = np.zeros((n_trials, B))

    for t in range(n_trials):
        env.next_trial(rng)

        # Softmax choice
        out = mod.outputs(x, theta)
        p1 = out["choice_prob"]
        choice = (rng.random(B) < p1).astype(float)
        choices[t] = choice

        # Reward
        rew = env.reward(choice, rng)
        r = rew["r"]
        rewards[t] = r
        correct_arms[t] = env._correct_arm

        # Trial update
        env_result = {"r": r, "choice": choice}
        x = mod.trial_update(x, theta, mask, env_result)

    return choices, rewards, correct_arms


def compute_observables(choices, rewards, correct_arms):
    """Compute win-stay, lose-switch, recovery speed, choice autocorr."""
    n_trials, B = choices.shape

    # Win-stay: P(repeat choice | previous reward = 1)
    ws_num = np.zeros(B)
    ws_den = np.zeros(B)
    for t in range(1, n_trials):
        prev_win = rewards[t-1] == 1
        repeated = choices[t] == choices[t-1]
        ws_num += (prev_win & repeated).astype(float)
        ws_den += prev_win.astype(float)
    win_stay = np.where(ws_den > 0, ws_num / ws_den, np.nan)

    # Lose-switch: P(switch choice | previous reward = 0)
    ls_num = np.zeros(B)
    ls_den = np.zeros(B)
    for t in range(1, n_trials):
        prev_loss = rewards[t-1] == 0
        switched = choices[t] != choices[t-1]
        ls_num += (prev_loss & switched).astype(float)
        ls_den += prev_loss.astype(float)
    lose_switch = np.where(ls_den > 0, ls_num / ls_den, np.nan)

    # Recovery speed: trials until 70% accuracy in sliding window of 10
    # after reversal (change in correct_arm)
    reversals = np.zeros((n_trials, B), dtype=bool)
    for t in range(1, n_trials):
        reversals[t] = correct_arms[t] != correct_arms[t-1]

    recovery_trials = []
    for b in range(B):
        rev_times = np.where(reversals[:, b])[0]
        if len(rev_times) == 0:
            continue
        for rt in rev_times:
            # Find first window of 10 trials after reversal with >=7 correct
            correct = (choices[rt:, b] == correct_arms[rt:, b]).astype(float)
            for start in range(len(correct) - 9):
                if correct[start:start+10].mean() >= 0.70:
                    recovery_trials.append(start)
                    break

    recovery_speed = np.mean(recovery_trials) if recovery_trials else np.nan

    # Choice autocorrelation
    ac = np.zeros(B)
    for b in range(B):
        c = choices[:, b]
        if np.std(c) > 0:
            ac[b] = np.corrcoef(c[:-1], c[1:])[0, 1]
    choice_autocorr = np.nanmean(ac)

    return {
        "win_stay": float(np.nanmean(win_stay)),
        "lose_switch": float(np.nanmean(lose_switch)),
        "recovery_speed": float(recovery_speed),
        "choice_autocorr": float(choice_autocorr),
    }


def main():
    # Grid search alpha x beta
    alphas = np.linspace(0.05, 0.40, 15)
    betas = np.linspace(1.0, 8.0, 15)
    tau_m = 2.163  # from gamma=0.63
    lam = 0.0  # no feedback during calibration
    n_trials = 500
    B = 50  # replicates per grid point
    p_reward = 0.75
    volatility = 0.3
    seed = 42

    target_ws = 0.70
    target_rec = 10.0  # middle of [8, 12]

    print(f"Grid search: {len(alphas)}x{len(betas)} = {len(alphas)*len(betas)} points")
    print(f"Targets: win-stay={target_ws}, recovery={target_rec}")

    best_dist = float('inf')
    best_alpha = None
    best_beta = None
    best_obs = None
    all_results = []

    for ai, alpha in enumerate(alphas):
        for bi, beta in enumerate(betas):
            choices, rewards, correct_arms = run_agent(
                alpha, beta, tau_m, lam, n_trials, B,
                p_reward, volatility, seed)
            obs = compute_observables(choices, rewards, correct_arms)

            # Standardized distance
            ws_err = (obs["win_stay"] - target_ws) / 0.05
            rec_err = (obs["recovery_speed"] - target_rec) / 2.0
            dist = ws_err**2 + rec_err**2

            all_results.append({
                "alpha": round(float(alpha), 4),
                "beta": round(float(beta), 4),
                **{k: round(v, 4) for k, v in obs.items()},
                "distance": round(float(dist), 4),
            })

            if dist < best_dist:
                best_dist = dist
                best_alpha = float(alpha)
                best_beta = float(beta)
                best_obs = obs

        if (ai + 1) % 5 == 0:
            print(f"  Row {ai+1}/{len(alphas)} done. "
                  f"Best so far: a={best_alpha:.3f}, b={best_beta:.3f}, "
                  f"ws={best_obs['win_stay']:.3f}, rec={best_obs['recovery_speed']:.1f}")

    # Compute lambda_crit and default lambda
    lc = lambda_crit(tau_m, r_mean=0.60)
    lam_default = 0.5 * lc

    # Jacobian check: numerical derivatives at best point
    eps = 0.01
    obs_center = best_obs
    choices_a, rewards_a, correct_a = run_agent(
        best_alpha + eps, best_beta, tau_m, 0.0, n_trials, B,
        p_reward, volatility, seed)
    obs_a = compute_observables(choices_a, rewards_a, correct_a)
    choices_b, rewards_b, correct_b = run_agent(
        best_alpha, best_beta + eps, tau_m, 0.0, n_trials, B,
        p_reward, volatility, seed)
    obs_b = compute_observables(choices_b, rewards_b, correct_b)

    J = np.array([
        [(obs_a["win_stay"] - obs_center["win_stay"]) / eps,
         (obs_b["win_stay"] - obs_center["win_stay"]) / eps],
        [(obs_a["recovery_speed"] - obs_center["recovery_speed"]) / eps,
         (obs_b["recovery_speed"] - obs_center["recovery_speed"]) / eps],
    ])
    det_J = float(np.linalg.det(J))

    print(f"\nCalibration result:")
    print(f"  alpha = {best_alpha:.4f}")
    print(f"  beta  = {best_beta:.4f}")
    print(f"  tau_m = {tau_m:.4f} (from gamma=0.63)")
    print(f"  lam   = {lam_default:.4f} (0.5 * lambda_crit = 0.5 * {lc:.4f})")
    print(f"  win_stay     = {best_obs['win_stay']:.4f} (target {target_ws})")
    print(f"  recovery     = {best_obs['recovery_speed']:.1f} (target {target_rec})")
    print(f"  lose_switch  = {best_obs['lose_switch']:.4f}")
    print(f"  choice_ac    = {best_obs['choice_autocorr']:.4f}")
    print(f"  Jacobian det = {det_J:.6f} ({'nonsingular' if abs(det_J) > 0.001 else 'SINGULAR'})")

    output = {
        "procedure": "grid search alpha x beta, lambda=0, 500 trials, B=50, "
                     "p_reward=0.75, vol=0.3, seed=42",
        "calibrated": {
            "alpha": round(best_alpha, 4),
            "beta": round(best_beta, 4),
            "tau_m": round(tau_m, 4),
            "lam_default": round(lam_default, 4),
            "lambda_crit": round(lc, 4),
        },
        "observables": {k: round(v, 4) for k, v in best_obs.items()},
        "targets": {"win_stay": target_ws, "recovery_speed": target_rec},
        "jacobian": {
            "matrix": J.tolist(),
            "det": round(det_J, 6),
            "nonsingular": bool(abs(det_J) > 0.001),
        },
        "grid": {
            "alphas": [round(float(a), 4) for a in alphas],
            "betas": [round(float(b), 4) for b in betas],
            "n_points": len(alphas) * len(betas),
        },
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "rl_mood_calibration.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
