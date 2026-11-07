"""Numerical verification of lambda_crit(tau_m).

Corrected derivation:
The update sequence per trial is:
  1. Decay:  m' = gamma * m_n   where gamma = exp(-dt/tau_m)
  2. Delta:  delta = (1 + lam*m') * r - V_chosen
  3. Update: m_{n+1} = m' + delta

Substituting:
  m_{n+1} = gamma*m_n + (1 + lam*gamma*m_n)*r - V
           = gamma*m_n*(1 + lam*r) + r - V

Eigenvalue = gamma * (1 + lam * E[r])
Critical:  gamma * (1 + lam_crit * r_mean) = 1
  =>  lam_crit = (1/gamma - 1) / r_mean = (1 - gamma) / (gamma * r_mean)

The naive formula lam_crit = (1-gamma)/r_mean was wrong by a factor 1/gamma
because it placed the feedback AFTER the decay, not inside the decayed term.

r_mean depends on agent accuracy and p_reward:
  r_mean = accuracy*p_reward + (1-accuracy)*(1-p_reward)
For conservative bound (perfect agent): r_mean = p_reward.

Output: results/rl_mood_lambda_crit.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpsych.environments.reversal_bandit import ReversalBandit


def run_rl_agent(alpha, beta, tau_m, lam, n_trials, B, p_reward, volatility,
                 seed):
    """Run RW+mood agent. Return max|m| per replicate and mean reward rate."""
    rng = np.random.default_rng(seed)
    env = ReversalBandit(p_reward=p_reward, volatility=volatility)
    env.reset(B, rng)

    V0 = np.full(B, 0.5)
    V1 = np.full(B, 0.5)
    m = np.zeros(B)
    max_abs_m = np.zeros(B)
    total_r = np.zeros(B)

    dt = 1.0

    for t in range(n_trials):
        m = m * np.exp(-dt / tau_m)
        env.next_trial(rng)

        logits = np.column_stack([beta * V0, beta * V1])
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        choice = (rng.random(B) < probs[:, 1]).astype(float)

        rew = env.reward(choice, rng)
        r = rew["r"]
        total_r += r

        rho_eff = 1.0 + lam * m
        V_chosen = np.where(choice == 0, V0, V1)
        delta = rho_eff * r - V_chosen

        V0 = np.where(choice == 0, V0 + alpha * delta, V0)
        V1 = np.where(choice == 1, V1 + alpha * delta, V1)

        m = m + delta
        max_abs_m = np.maximum(max_abs_m, np.abs(m))

    r_mean = float(np.mean(total_r / n_trials))
    return max_abs_m, r_mean


def find_numerical_crit(tau_m, lam_max, n_steps, alpha, beta,
                        n_trials, B, p_reward, volatility, seed,
                        threshold_frac=0.5, threshold_m=50.0):
    """Find smallest lam where >threshold_frac have max|m|>threshold_m."""
    lam_values = np.linspace(0, lam_max, n_steps)
    for lam in lam_values:
        max_m, _ = run_rl_agent(alpha, beta, tau_m, lam, n_trials, B,
                                p_reward, volatility, seed)
        frac = np.mean(max_m > threshold_m)
        if frac > threshold_frac:
            return float(lam)
    return None


def lam_crit_corrected(tau_m, dt, r_mean):
    """Corrected formula: eigenvalue = gamma*(1 + lam*r_mean) = 1."""
    gamma = np.exp(-dt / tau_m)
    return (1.0 - gamma) / (gamma * r_mean)


def lam_crit_naive(tau_m, dt, r_mean):
    """Naive (wrong) formula from v1.1."""
    return (1.0 - np.exp(-dt / tau_m)) / r_mean


def main():
    alpha = 0.15
    beta = 3.0
    p_reward = 0.75
    volatility = 0.3
    n_trials = 500
    B = 200
    seed = 42
    dt = 1.0

    # First, measure r_mean at lambda=0 (no feedback) to get baseline
    _, r_mean_baseline = run_rl_agent(alpha, beta, 2.16, 0.0,
                                       n_trials, B, p_reward, volatility, seed)
    print(f"Baseline r_mean (lam=0): {r_mean_baseline:.4f}")

    tau_m_values = [1.0, 2.16, 5.0, 10.0, 20.0]

    results = []
    print(f"\n{'tau_m':>6} {'naive':>8} {'corrected':>10} {'numerical':>10} "
          f"{'corr/num':>9} {'r_mean':>7}")
    print("=" * 60)

    for tau_m in tau_m_values:
        gamma = np.exp(-dt / tau_m)
        naive = lam_crit_naive(tau_m, dt, r_mean_baseline)
        corrected = lam_crit_corrected(tau_m, dt, r_mean_baseline)

        # Wide sweep to find actual threshold
        lam_max = max(2.0 * corrected, 5.0)
        lam_crit_num = find_numerical_crit(
            tau_m, lam_max, 200, alpha, beta,
            n_trials, B, p_reward, volatility, seed,
            threshold_m=50.0)

        # Measure r_mean at lambda just below threshold
        if lam_crit_num is not None:
            _, r_mean_at_crit = run_rl_agent(
                alpha, beta, tau_m, 0.8 * lam_crit_num,
                n_trials, B, p_reward, volatility, seed)
            ratio = corrected / lam_crit_num
        else:
            r_mean_at_crit = None
            ratio = None

        entry = {
            "tau_m": tau_m,
            "gamma": round(float(gamma), 4),
            "naive": round(naive, 4),
            "corrected": round(corrected, 4),
            "numerical": round(lam_crit_num, 4) if lam_crit_num else None,
            "ratio_corrected_over_numerical": round(ratio, 3) if ratio else None,
            "r_mean_at_crit": round(r_mean_at_crit, 4) if r_mean_at_crit else None,
        }
        results.append(entry)

        num_str = f"{lam_crit_num:.4f}" if lam_crit_num else "N/A"
        ratio_str = f"{ratio:.3f}" if ratio else "N/A"
        rm_str = f"{r_mean_at_crit:.4f}" if r_mean_at_crit else "N/A"
        print(f"{tau_m:6.2f} {naive:8.4f} {corrected:10.4f} {num_str:>10} "
              f"{ratio_str:>9} {rm_str:>7}")

    # Stability test at baseline tau_m=2.16
    print("\nStability test at tau_m=2.16")
    print("=" * 60)
    tau_m_base = 2.16
    corrected_base = lam_crit_corrected(tau_m_base, dt, r_mean_baseline)

    lam_crit_precise = find_numerical_crit(
        tau_m_base, 3.0 * corrected_base, 300, alpha, beta,
        n_trials, B, p_reward, volatility, seed,
        threshold_m=50.0)

    if lam_crit_precise is not None:
        max_m_below, _ = run_rl_agent(
            alpha, beta, tau_m_base, 0.8 * lam_crit_precise,
            n_trials, B, p_reward, volatility, seed)
        stable_frac = float(np.mean(max_m_below < 50))

        max_m_above, _ = run_rl_agent(
            alpha, beta, tau_m_base, 1.2 * lam_crit_precise,
            n_trials, B, p_reward, volatility, seed)
        diverge_frac = float(np.mean(max_m_above > 50))

        print(f"  lam_crit_numerical = {lam_crit_precise:.4f}")
        print(f"  corrected formula  = {corrected_base:.4f}")
        print(f"  Below (lam={0.8*lam_crit_precise:.4f}): "
              f"{stable_frac:.1%} stable  "
              f"{'PASS' if stable_frac > 0.90 else 'FAIL'}")
        print(f"  Above (lam={1.2*lam_crit_precise:.4f}): "
              f"{diverge_frac:.1%} diverged  "
              f"{'PASS' if diverge_frac > 0.50 else 'FAIL'}")

        stability_test = {
            "tau_m": tau_m_base,
            "lam_crit_formula": round(corrected_base, 4),
            "lam_crit_numerical": round(lam_crit_precise, 4),
            "below_crit": {
                "lambda": round(0.8 * lam_crit_precise, 4),
                "frac_stable": round(stable_frac, 4),
                "pass_flag": bool(stable_frac > 0.90),
            },
            "above_crit": {
                "lambda": round(1.2 * lam_crit_precise, 4),
                "frac_diverged": round(diverge_frac, 4),
                "pass_flag": bool(diverge_frac > 0.50),
            },
        }
    else:
        stability_test = {"error": "lam_crit not found"}

    output = {
        "procedure": "sweep lam at fixed tau_m, 500 trials, B=200, "
                     "p_reward=0.75, vol=0.3, seed=42",
        "corrected_formula": "lam_crit = (1 - gamma) / (gamma * r_mean), "
                             "gamma = exp(-dt/tau_m)",
        "naive_formula_bug": "v1.1 had lam_crit = (1-gamma)/r_mean, missing "
                             "factor 1/gamma. Decay before update, not after.",
        "r_mean_baseline": round(r_mean_baseline, 4),
        "alpha": alpha,
        "beta": beta,
        "dt": dt,
        "sweep_results": results,
        "stability_test": stability_test,
    }

    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    out_path = (Path(__file__).resolve().parents[1] /
                "results" / "rl_mood_lambda_crit.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
