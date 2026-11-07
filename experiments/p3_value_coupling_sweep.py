"""P3.4: Sweep s_value for value_diff → drift_bias coupling.

Two observables:
1. Accuracy gain vs open-loop (no coupling)
2. Post-reversal recovery speed (trials to 70% correct)

Expected: accuracy grows with s, recovery speed non-monotonic.
At high s, strong value bias slows recovery after reversal.

Also: numerical lam_crit in closed-loop composite.

Results: results/p3_value_coupling_sweep.json
"""
import json
import sys
import os

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpsych.modules.ddm import DDMModule, T_MAX_TRIAL
from cpsych.modules.rl_mood import RLMoodModule


def simulate_trials(n_trials, s_value, coherence, p_reward, seed,
                    volatility=0.3, lam=0.0, dt=0.001):
    """Simulate DDM+RL for n_trials with per-trial tracking.

    Uses modules directly in a trial loop for per-trial access.
    """
    rng = np.random.default_rng(seed)
    ddm = DDMModule()
    rl = RLMoodModule()

    # Single replicate for per-trial tracking
    x_ddm = ddm.default_state(1)
    x_rl = rl.default_state(1)
    theta_ddm = ddm.default_params(1)
    theta_rl = rl.default_params(1)
    theta_rl[0, 3] = lam  # lambda

    # Environment state
    correct_arm = float(rng.integers(0, 2))
    trials_in_block = 0

    results_per_trial = []
    reversal_points = []

    for trial in range(n_trials):
        # Check reversal
        trials_in_block += 1
        if volatility > 0:
            p_rev = volatility / 20.0
            if rng.random() < p_rev:
                old_arm = correct_arm
                correct_arm = 1.0 - correct_arm
                reversal_points.append(trial)
                trials_in_block = 0

        # Direction and coherence
        direction = 2.0 * correct_arm - 1.0
        coh = coherence

        # Apply coupling: drift_bias = s * value_diff
        V0, V1 = x_rl[0, 0], x_rl[0, 1]
        value_diff = V1 - V0
        theta_ddm_eff = theta_ddm.copy()
        theta_ddm_eff[0, 3] = s_value * value_diff  # drift_bias

        # Reset DDM
        x_ddm[0, :] = 0.0

        # Simulate DDM until trial_done
        u = np.array([[coh, direction]])
        is_timeout = False
        for step in range(int(T_MAX_TRIAL / dt) + 100):
            dx = ddm.drift(x_ddm, theta_ddm_eff, u)
            sigma = ddm.diffusion(x_ddm, theta_ddm_eff)
            dW = rng.standard_normal(x_ddm.shape)
            x_ddm = x_ddm + dx * dt + sigma * np.sqrt(dt) * dW

            if ddm.trial_done(x_ddm, theta_ddm_eff)[0]:
                out = ddm.outputs(x_ddm, theta_ddm_eff)
                is_timeout = bool(out["is_timeout"][0] > 0.5)
                break

        # Get choice
        out = ddm.outputs(x_ddm, theta_ddm_eff)
        ddm_choice = out["choice"][0]  # ±1
        arm = 1.0 if ddm_choice >= 0 else 0.0

        # Reward
        chose_correct = (arm == correct_arm)
        p = p_reward if chose_correct else (1.0 - p_reward)
        r = float(rng.random() < p)

        # RL trial_update
        env_result = {
            "r": np.array([r]),
            "choice": np.array([arm]),
            "is_timeout": np.array([float(is_timeout)]),
        }
        mask = np.ones(1, dtype=bool)
        x_rl = rl.trial_update(x_rl, theta_rl, mask, env_result)

        results_per_trial.append({
            "correct": bool(chose_correct),
            "reward": r,
            "V0": float(x_rl[0, 0]),
            "V1": float(x_rl[0, 1]),
            "m": float(x_rl[0, 2]),
            "value_diff": float(x_rl[0, 1] - x_rl[0, 0]),
            "is_timeout": is_timeout,
        })

    return results_per_trial, reversal_points


def compute_accuracy(trial_data, window=20):
    """Sliding window accuracy."""
    correct = [t["correct"] for t in trial_data]
    acc = []
    for i in range(len(correct)):
        start = max(0, i - window + 1)
        acc.append(np.mean(correct[start:i+1]))
    return acc


def compute_recovery_speed(trial_data, reversal_points, threshold=0.70,
                            window=4, max_trials=50):
    """Trials to reach threshold accuracy after each reversal."""
    correct = [t["correct"] for t in trial_data]
    speeds = []
    for rev in reversal_points:
        if rev + max_trials >= len(correct):
            continue
        for k in range(1, max_trials + 1):
            start = rev + max(1, k - window + 1)
            end = rev + k + 1
            if end > len(correct):
                break
            win = correct[start:end]
            if len(win) >= window and np.mean(win[-window:]) >= threshold:
                speeds.append(k)
                break
        else:
            speeds.append(max_trials)  # didn't recover
    return speeds


def run_sweep():
    """Main sweep over s_value."""
    s_values = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    n_trials = 500
    n_seeds = 20
    coherence = 0.128
    p_reward = 0.75
    volatility = 0.3

    results = {"s_values": s_values, "n_trials": n_trials,
               "n_seeds": n_seeds, "coherence": coherence,
               "p_reward": p_reward, "volatility": volatility,
               "per_s": []}

    for s in s_values:
        print(f"  s={s:.1f}...", end=" ", flush=True)
        accuracies = []
        recovery_speeds = []
        max_m_vals = []

        for seed in range(n_seeds):
            trial_data, rev_points = simulate_trials(
                n_trials, s, coherence, p_reward, seed=seed + 100,
                volatility=volatility)

            # Overall accuracy (last 100 trials)
            last_correct = [t["correct"] for t in trial_data[-100:]]
            accuracies.append(np.mean(last_correct))

            # Recovery speed
            speeds = compute_recovery_speed(trial_data, rev_points)
            if speeds:
                recovery_speeds.append(np.mean(speeds))

            # Max |m|
            m_vals = [abs(t["m"]) for t in trial_data]
            max_m_vals.append(max(m_vals))

        entry = {
            "s": s,
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "recovery_mean": float(np.mean(recovery_speeds)) if recovery_speeds else None,
            "recovery_std": float(np.std(recovery_speeds)) if recovery_speeds else None,
            "recovery_n": len(recovery_speeds),
            "max_m_mean": float(np.mean(max_m_vals)),
            "max_m_max": float(np.max(max_m_vals)),
        }
        results["per_s"].append(entry)
        print(f"acc={entry['accuracy_mean']:.3f}, "
              f"rec={entry['recovery_mean']:.1f}" if entry['recovery_mean'] else "rec=N/A")

    # Open-loop baseline (s=0)
    baseline = results["per_s"][0]
    results["baseline_accuracy"] = baseline["accuracy_mean"]

    # Accuracy gain
    for entry in results["per_s"]:
        entry["accuracy_gain"] = entry["accuracy_mean"] - baseline["accuracy_mean"]

    return results


def run_lambda_crit_sweep():
    """Numerical lam_crit in closed-loop composite.

    Sweep lam, check for mood divergence.
    """
    lambda_values = np.arange(0.0, 2.5, 0.1).tolist()
    n_trials = 500
    n_seeds = 30
    s_value = 5.0
    coherence = 0.128
    p_reward = 0.75
    divergence_threshold = 10.0  # max|m| threshold

    results = {"lambda_values": lambda_values, "s_value": s_value,
               "n_trials": n_trials, "n_seeds": n_seeds,
               "divergence_threshold": divergence_threshold,
               "per_lambda": []}

    for lam in lambda_values:
        print(f"  lam={lam:.2f}...", end=" ", flush=True)
        n_diverged = 0
        max_m_vals = []

        for seed in range(n_seeds):
            trial_data, _ = simulate_trials(
                n_trials, s_value, coherence, p_reward,
                seed=seed + 200, volatility=0.0, lam=lam)

            max_m = max(abs(t["m"]) for t in trial_data)
            max_m_vals.append(max_m)
            if max_m > divergence_threshold:
                n_diverged += 1

        frac_diverged = n_diverged / n_seeds
        entry = {
            "lambda": lam,
            "frac_diverged": frac_diverged,
            "max_m_mean": float(np.mean(max_m_vals)),
            "max_m_p95": float(np.percentile(max_m_vals, 95)),
        }
        results["per_lambda"].append(entry)
        print(f"div={frac_diverged:.0%}, max_m_p95={entry['max_m_p95']:.2f}")

    # Find critical lambda: first lambda where frac_diverged > 0.5
    for entry in results["per_lambda"]:
        if entry["frac_diverged"] > 0.5:
            results["lambda_crit_numerical"] = entry["lambda"]
            break
    else:
        results["lambda_crit_numerical"] = None

    # Standalone value
    from cpsych.modules.rl_mood import lambda_crit
    results["lambda_crit_standalone"] = float(lambda_crit(2.163, 0.60))

    return results


if __name__ == "__main__":
    print("=== P3.4 value coupling sweep ===")
    print("\n1. Accuracy + recovery speed vs s_value:")
    sweep = run_sweep()

    print("\n2. lam_crit in closed-loop composite:")
    lam_results = run_lambda_crit_sweep()

    combined = {
        "sweep": sweep,
        "lambda_crit": lam_results,
    }

    out_path = os.path.join(os.path.dirname(__file__), '..',
                             'results', 'p3_value_coupling_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n=== Sweep summary ===")
    print(f"{'s':>5} {'acc_gain':>9} {'recovery':>9} {'max|m|':>8}")
    for e in sweep["per_s"]:
        rec = f"{e['recovery_mean']:.1f}" if e['recovery_mean'] else "N/A"
        print(f"{e['s']:5.1f} {e['accuracy_gain']:+9.4f} {rec:>9} {e['max_m_mean']:8.2f}")

    print(f"\n=== lam_crit ===")
    print(f"Standalone: {lam_results['lambda_crit_standalone']:.4f}")
    print(f"Closed-loop (numerical): {lam_results['lambda_crit_numerical']}")
