"""P3.4: Fine-grid sweep of s_value around optimum.

Coarse sweep showed peak accuracy at s~2. This refines to s in [0.5, 6]
step 0.5, with more seeds for precision.

Results: results/p3_value_coupling_fine.json
"""
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpsych.modules.ddm import DDMModule, T_MAX_TRIAL
from cpsych.modules.rl_mood import RLMoodModule


def simulate_trials(n_trials, s_value, coherence, p_reward, seed,
                    volatility=0.3, dt=0.001):
    """Simulate DDM+RL for n_trials with per-trial tracking."""
    rng = np.random.default_rng(seed)
    ddm = DDMModule()
    rl = RLMoodModule()

    x_ddm = ddm.default_state(1)
    x_rl = rl.default_state(1)
    theta_ddm = ddm.default_params(1)
    theta_rl = rl.default_params(1)

    correct_arm = float(rng.integers(0, 2))
    trials_in_block = 0

    correct_seq = []
    reversal_points = []

    for trial in range(n_trials):
        trials_in_block += 1
        if volatility > 0:
            p_rev = volatility / 20.0
            if rng.random() < p_rev:
                correct_arm = 1.0 - correct_arm
                reversal_points.append(trial)
                trials_in_block = 0

        direction = 2.0 * correct_arm - 1.0
        V0, V1 = x_rl[0, 0], x_rl[0, 1]
        theta_ddm_eff = theta_ddm.copy()
        theta_ddm_eff[0, 3] = s_value * (V1 - V0)

        x_ddm[0, :] = 0.0
        u = np.array([[coherence, direction]])
        for step in range(int(T_MAX_TRIAL / dt) + 100):
            dx = ddm.drift(x_ddm, theta_ddm_eff, u)
            sigma = ddm.diffusion(x_ddm, theta_ddm_eff)
            dW = rng.standard_normal(x_ddm.shape)
            x_ddm = x_ddm + dx * dt + sigma * np.sqrt(dt) * dW
            if ddm.trial_done(x_ddm, theta_ddm_eff)[0]:
                break

        out = ddm.outputs(x_ddm, theta_ddm_eff)
        arm = 1.0 if out["choice"][0] >= 0 else 0.0
        is_timeout = float(out["is_timeout"][0] > 0.5)

        chose_correct = (arm == correct_arm)
        p = p_reward if chose_correct else (1.0 - p_reward)
        r = float(rng.random() < p)

        env_result = {
            "r": np.array([r]),
            "choice": np.array([arm]),
            "is_timeout": np.array([is_timeout]),
        }
        x_rl = rl.trial_update(x_rl, theta_rl, np.ones(1, dtype=bool),
                                env_result)

        correct_seq.append(bool(chose_correct))

    return correct_seq, reversal_points


def compute_recovery(correct_seq, reversal_points, threshold=0.70,
                      window=4, max_trials=50):
    """Trials to reach threshold accuracy after each reversal."""
    speeds = []
    for rev in reversal_points:
        if rev + max_trials >= len(correct_seq):
            continue
        for k in range(1, max_trials + 1):
            start = rev + max(1, k - window + 1)
            end = rev + k + 1
            if end > len(correct_seq):
                break
            win = correct_seq[start:end]
            if len(win) >= window and np.mean(win[-window:]) >= threshold:
                speeds.append(k)
                break
        else:
            speeds.append(max_trials)
    return speeds


def main():
    s_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
    n_trials = 500
    n_seeds = 40
    coherence = 0.128
    p_reward = 0.75
    volatility = 0.3

    results = {
        "s_values": s_values, "n_trials": n_trials,
        "n_seeds": n_seeds, "coherence": coherence,
        "p_reward": p_reward, "volatility": volatility,
        "per_s": [],
    }

    for s in s_values:
        print(f"  s={s:.1f}...", end=" ", flush=True)
        accs = []
        recs = []

        for seed in range(n_seeds):
            correct_seq, rev_pts = simulate_trials(
                n_trials, s, coherence, p_reward, seed=seed + 100,
                volatility=volatility)

            accs.append(np.mean(correct_seq[-200:]))
            speeds = compute_recovery(correct_seq, rev_pts)
            if speeds:
                recs.append(np.mean(speeds))

        acc_mean = float(np.mean(accs))
        acc_se = float(np.std(accs) / np.sqrt(len(accs)))
        rec_mean = float(np.mean(recs)) if recs else None
        rec_se = float(np.std(recs) / np.sqrt(len(recs))) if len(recs) > 1 else None

        entry = {
            "s": s,
            "accuracy_mean": acc_mean,
            "accuracy_se": acc_se,
            "recovery_mean": rec_mean,
            "recovery_se": rec_se,
        }
        results["per_s"].append(entry)
        rec_str = f"{rec_mean:.2f}" if rec_mean else "N/A"
        print(f"acc={acc_mean:.4f}+-{acc_se:.4f}, rec={rec_str}")

    # Find optimum
    baseline = results["per_s"][0]["accuracy_mean"]
    for e in results["per_s"]:
        e["accuracy_gain"] = e["accuracy_mean"] - baseline

    gains = [(e["s"], e["accuracy_gain"]) for e in results["per_s"]]
    best_s, best_gain = max(gains, key=lambda x: x[1])
    results["optimum_s"] = best_s
    results["optimum_gain"] = best_gain
    results["baseline_accuracy"] = baseline

    # Degradation onset: first s where gain < 0
    for e in results["per_s"]:
        if e["s"] > 0 and e["accuracy_gain"] < 0:
            results["degradation_onset_s"] = e["s"]
            break

    out_path = os.path.join(os.path.dirname(__file__), '..',
                             'results', 'p3_value_coupling_fine.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print(f"\n=== Fine sweep summary ===")
    print(f"Baseline accuracy: {baseline:.4f}")
    print(f"Optimum: s={best_s:.1f}, gain={best_gain:+.4f}")
    print(f"Degradation onset: s={results.get('degradation_onset_s', 'N/A')}")
    print(f"\n{'s':>5} {'gain':>8} {'acc':>8} {'recovery':>9}")
    for e in results["per_s"]:
        rec = f"{e['recovery_mean']:.2f}" if e['recovery_mean'] else "N/A"
        print(f"{e['s']:5.1f} {e['accuracy_gain']:+8.4f} "
              f"{e['accuracy_mean']:8.4f} {rec:>9}")


if __name__ == "__main__":
    print("=== P3.4 fine s_value sweep ===\n")
    main()
