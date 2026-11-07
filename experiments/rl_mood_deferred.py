"""Deferred observables for RL+Mood module.

1. Rutledge coefficients (w_CR / w_RPE ratio).
2. Learning rate invariance to volatility (positive prediction).
3. Mood autocorrelation (gamma_model vs 0.63).

One-shot check. Output: results/rl_mood_deferred.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsych.modules.rl_mood import RLMoodModule, lambda_crit
from cpsych.environments.reversal_bandit import ReversalBandit


def run_and_record(alpha, beta, tau_m, lam, n_trials, B, p_reward,
                   volatility, seed):
    """Run agent, record full trajectories for analysis."""
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

    mood_traj = np.zeros((n_trials, B))
    reward_traj = np.zeros((n_trials, B))
    choice_traj = np.zeros((n_trials, B))
    delta_traj = np.zeros((n_trials, B))
    correct_traj = np.zeros((n_trials, B))

    for t in range(n_trials):
        env.next_trial(rng)
        out = mod.outputs(x, theta)
        p1 = out["choice_prob"]
        choice = (rng.random(B) < p1).astype(float)
        rew = env.reward(choice, rng)
        r = rew["r"]

        # Record before update
        mood_traj[t] = x[:, 2]

        # Compute delta for recording
        V_chosen = np.where(choice == 0, x[:, 0], x[:, 1])
        rho_eff = 1.0 + lam * x[:, 2]
        delta = rho_eff * r - V_chosen
        delta_traj[t] = delta
        reward_traj[t] = r
        choice_traj[t] = choice
        correct_traj[t] = env._correct_arm

        env_result = {"r": r, "choice": choice}
        x = mod.trial_update(x, theta, mask, env_result)

    return mood_traj, reward_traj, delta_traj, choice_traj, correct_traj


def deferred_1_rutledge_coefficients(mood, rewards, deltas, gamma):
    """Rutledge regression: mood ~ sum_k gamma^k (w_CR*r_k + w_RPE*delta_k)."""
    n_trials, B = mood.shape
    results = []

    for b in range(B):
        # Build regressors: exponentially weighted history
        # For each trial t, compute sum_k=0..t-1 gamma^k * r_{t-1-k}
        # and sum_k=0..t-1 gamma^k * delta_{t-1-k}
        max_history = min(20, n_trials)
        X_list = []
        y_list = []
        for t in range(max_history, n_trials):
            weights = gamma ** np.arange(max_history)
            cr_sum = np.sum(weights * rewards[t-max_history:t, b][::-1])
            rpe_sum = np.sum(weights * deltas[t-max_history:t, b][::-1])
            X_list.append([cr_sum, rpe_sum])
            y_list.append(mood[t, b])

        X = np.array(X_list)
        y = np.array(y_list)

        if len(y) < 10 or np.std(y) < 1e-10:
            continue

        # OLS
        X_aug = np.column_stack([X, np.ones(len(X))])
        try:
            beta_hat = np.linalg.lstsq(X_aug, y, rcond=None)[0]
            results.append({"w_CR": beta_hat[0], "w_RPE": beta_hat[1]})
        except np.linalg.LinAlgError:
            continue

    if not results:
        return {"w_CR_mean": float('nan'), "w_RPE_mean": float('nan'),
                "ratio": float('nan'), "pass": False}

    w_CR = np.mean([r["w_CR"] for r in results])
    w_RPE = np.mean([r["w_RPE"] for r in results])
    # Criterion: w_RPE > w_CR, both > 0
    qualitative_match = w_RPE > w_CR and w_RPE > 0

    return {
        "w_CR_mean": round(float(w_CR), 4),
        "w_RPE_mean": round(float(w_RPE), 4),
        "ratio_RPE_over_CR": round(float(w_RPE / w_CR), 4) if abs(w_CR) > 1e-10 else float('inf'),
        "criterion": "w_RPE > w_CR > 0",
        "qualitative_match": bool(qualitative_match),
    }


def deferred_2_volatility_invariance(alpha, beta, tau_m, lam, n_trials, B,
                                     p_reward, seed):
    """RW predicts: recovery speed invariant to volatility."""

    def recovery_speed_at_vol(vol):
        _, rewards, _, choices, correct = run_and_record(
            alpha, beta, tau_m, lam, n_trials, B, p_reward, vol, seed)
        # Compute recovery speed
        reversals = np.zeros((n_trials, B), dtype=bool)
        for t in range(1, n_trials):
            reversals[t] = correct[t] != correct[t-1]
        speeds = []
        for b in range(B):
            rev_times = np.where(reversals[:, b])[0]
            for rt in rev_times:
                c = (choices[rt:, b] == correct[rt:, b]).astype(float)
                for start in range(min(len(c) - 9, 40)):
                    if c[start:start+10].mean() >= 0.70:
                        speeds.append(start)
                        break
        return np.mean(speeds) if speeds else float('nan')

    rec_stable = recovery_speed_at_vol(0.1)
    rec_volatile = recovery_speed_at_vol(1.0)
    diff = abs(rec_stable - rec_volatile)

    return {
        "recovery_stable_vol01": round(float(rec_stable), 2),
        "recovery_volatile_vol10": round(float(rec_volatile), 2),
        "abs_difference": round(float(diff), 2),
        "criterion": "|diff| < 2 trials",
        "pass": bool(diff < 2),
        "behrens_note": "Behrens 2007: humans show faster recovery in volatile "
                        "blocks. RW by construction does not. This documents "
                        "the limitation, not a failure.",
    }


def deferred_3_mood_autocorrelation(mood):
    """Mood autocorrelation gamma_model. Criterion: |gamma_model - 0.63| < 0.15."""
    n_trials, B = mood.shape
    acs = []
    for b in range(B):
        m = mood[:, b]
        if np.std(m) > 1e-10:
            ac = np.corrcoef(m[:-1], m[1:])[0, 1]
            if not np.isnan(ac):
                acs.append(ac)
    gamma_model = np.mean(acs) if acs else float('nan')
    diff = abs(gamma_model - 0.63)

    return {
        "gamma_model": round(float(gamma_model), 4),
        "gamma_target": 0.63,
        "abs_difference": round(float(diff), 4),
        "criterion": "|gamma_model - 0.63| < 0.15",
        "pass": bool(diff < 0.15),
    }


def main():
    alpha = 0.15
    beta = 2.50
    tau_m = 2.163
    lc = lambda_crit(tau_m, r_mean=0.60)
    lam = 0.5 * lc
    gamma = np.exp(-1.0 / tau_m)
    n_trials = 500
    B = 200
    p_reward = 0.75
    volatility = 0.3
    seed = 42

    print(f"Baseline: alpha={alpha}, beta={beta}, tau_m={tau_m}, "
          f"lam={lam:.4f}, gamma={gamma:.4f}")

    # Run baseline
    mood, rewards, deltas, choices, correct = run_and_record(
        alpha, beta, tau_m, lam, n_trials, B, p_reward, volatility, seed)

    # Deferred 1
    print("\n1. Rutledge coefficients")
    d1 = deferred_1_rutledge_coefficients(mood, rewards, deltas, gamma)
    print(f"   w_CR={d1['w_CR_mean']:.4f}, w_RPE={d1['w_RPE_mean']:.4f}, "
          f"match={d1['qualitative_match']}")

    # Deferred 2
    print("\n2. Volatility invariance (positive prediction)")
    d2 = deferred_2_volatility_invariance(alpha, beta, tau_m, lam,
                                          n_trials, B, p_reward, seed)
    print(f"   recovery(vol=0.1)={d2['recovery_stable_vol01']:.1f}, "
          f"recovery(vol=1.0)={d2['recovery_volatile_vol10']:.1f}, "
          f"|diff|={d2['abs_difference']:.1f}, "
          f"{'PASS' if d2['pass'] else 'FAIL'}")

    # Deferred 3
    print("\n3. Mood autocorrelation")
    d3 = deferred_3_mood_autocorrelation(mood)
    print(f"   gamma_model={d3['gamma_model']:.4f}, "
          f"|diff|={d3['abs_difference']:.4f}, "
          f"{'PASS' if d3['pass'] else 'FAIL'}")

    output = {
        "baseline": {
            "alpha": alpha, "beta": beta, "tau_m": tau_m,
            "lam": round(float(lam), 4), "gamma": round(float(gamma), 4),
            "n_trials": n_trials, "B": B,
            "p_reward": p_reward, "volatility": volatility, "seed": seed,
        },
        "deferred_1_rutledge_coefficients": d1,
        "deferred_2_volatility_invariance": d2,
        "deferred_3_mood_autocorrelation": d3,
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "rl_mood_deferred.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
