"""P4 Step 1: Triple composite stability analysis.

Analysis A: Composite Jacobian structure verification.
  - 9x9 Jacobian at wake equilibrium with all couplings.
  - Verify eigenvalues = arousal eigenvalues ∪ {0,0,0,0,0}.
  - Sweep strengths (s_kappa, s_alpha, s_value) to confirm.

Analysis B: Numerical stability sweep.
  - Run full composite at various strength combinations.
  - Verify all states finite, bounded.

Results: results/p4_stability.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpsych.core.composite_setup import (
    make_triple_modules, make_triple_couplings,
    make_triple_integrator, wake_equilibrium,
    triple_linearization_point,
)
from cpsych.core.stability import (
    composite_jacobian, numerical_jacobian, max_real_eigenvalue,
)


def analysis_a_jacobian():
    """Composite Jacobian structure verification."""
    print("=== Analysis A: Composite Jacobian ===\n")

    # Get linearization point
    print("Computing wake equilibrium (2 days)...")
    modules = make_triple_modules()
    wake_eq = wake_equilibrium(modules["arousal"], T_settle_days=2)
    print(f"  gain = {wake_eq['gain']:.4f}, Qm = {wake_eq['Qm']:.4f}")

    modules, x0, theta0, u0 = triple_linearization_point(wake_eq)

    results = {
        "wake_equilibrium": {
            "gain": wake_eq["gain"],
            "Qm": wake_eq["Qm"],
            "x_arousal": wake_eq["x0"][0].tolist(),
        },
        "jacobian_structure": {},
        "strength_sweep": [],
    }

    # Arousal-only Jacobian for reference eigenvalues
    J_arousal = numerical_jacobian(
        modules["arousal"], wake_eq["x0"], wake_eq["theta0"],
        np.zeros((1, 0)))
    arousal_eigs = np.sort(np.real(np.linalg.eigvals(J_arousal)))
    print(f"  Arousal eigenvalues: {arousal_eigs}")

    # Full 9x9 Jacobian at default strengths
    couplings = make_triple_couplings(s_kappa=2.0, s_alpha=2.0, s_value=3.0)
    J_full = composite_jacobian(modules, x0, theta0, u0, couplings)
    full_eigs = np.sort(np.real(np.linalg.eigvals(J_full)))
    print(f"  Full 9x9 eigenvalues: {full_eigs}")
    print(f"  Max Re(eig): {max_real_eigenvalue(J_full):.6f}")

    results["jacobian_structure"] = {
        "shape": list(J_full.shape),
        "arousal_eigenvalues": arousal_eigs.tolist(),
        "full_eigenvalues": full_eigs.tolist(),
        "max_real_eigenvalue": float(max_real_eigenvalue(J_full)),
        "arousal_max_real": float(np.max(arousal_eigs)),
    }

    # Strength sweep: 5^3 = 125 points
    print("\nStrength sweep (5x5x5 = 125 points)...")
    s_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    sweep_results = []
    all_match = True

    for sk in s_values:
        for sa in s_values:
            for sv in s_values:
                c = make_triple_couplings(sk, sa, sv)
                J = composite_jacobian(modules, x0, theta0, u0, c)
                eigs = np.sort(np.real(np.linalg.eigvals(J)))
                max_re = float(np.max(eigs))

                # Check: max Re(eig) should match arousal max
                # (DDM and RL contribute zeros)
                match = abs(max_re - float(np.max(arousal_eigs))) < 0.01
                if not match:
                    all_match = False

                sweep_results.append({
                    "s_kappa": sk, "s_alpha": sa, "s_value": sv,
                    "max_real_eigenvalue": round(max_re, 6),
                    "eigenvalues": [round(e, 6) for e in eigs.tolist()],
                    "matches_arousal": match,
                })

    results["strength_sweep"] = sweep_results
    results["all_match_arousal"] = all_match
    n_match = sum(1 for r in sweep_results if r["matches_arousal"])
    print(f"  {n_match}/125 match arousal eigenvalues")
    if not all_match:
        mismatches = [r for r in sweep_results if not r["matches_arousal"]]
        print(f"  WARNING: {len(mismatches)} mismatches:")
        for m in mismatches[:5]:
            print(f"    s=({m['s_kappa']},{m['s_alpha']},{m['s_value']}): "
                  f"max_re={m['max_real_eigenvalue']}")

    return results


def analysis_b_numerical():
    """Numerical stability sweep."""
    print("\n=== Analysis B: Numerical Stability ===\n")

    results = {"configurations": [], "all_stable": True}

    s_kappa_vals = [0.0, 2.0, 4.0]
    s_alpha_vals = [0.0, 2.0, 4.0]
    s_value_vals = [0.0, 3.0, 7.0, 10.0, 15.0]

    total = len(s_kappa_vals) * len(s_alpha_vals) * len(s_value_vals)
    print(f"Running {total} configurations (B=10, T=86400)...")

    i = 0
    for sk in s_kappa_vals:
        for sa in s_alpha_vals:
            for sv in s_value_vals:
                i += 1
                print(f"  [{i}/{total}] s=({sk},{sa},{sv})...",
                      end="", flush=True)

                integ = make_triple_integrator(
                    B=10, seed=42, obs_dt=86400.0,
                    s_kappa=sk, s_alpha=sa, s_value=sv)
                state = integ.run(T=86400.0)

                # Check all states finite
                all_finite = True
                max_states = {}
                for name in ["arousal", "ddm", "rl_mood"]:
                    x = state.x[name]
                    finite = np.all(np.isfinite(x))
                    if not finite:
                        all_finite = False
                    max_states[name] = float(np.max(np.abs(x)))

                # Check RL values bounded
                V0 = state.x["rl_mood"][:, 0]
                V1 = state.x["rl_mood"][:, 1]
                m = state.x["rl_mood"][:, 2]
                rl_bounded = (
                    np.all(np.abs(V0) < 2.0) and
                    np.all(np.abs(V1) < 2.0) and
                    np.all(np.abs(m) < 10.0)
                )

                config_result = {
                    "s_kappa": sk, "s_alpha": sa, "s_value": sv,
                    "all_finite": bool(all_finite),
                    "rl_bounded": bool(rl_bounded),
                    "max_abs_state": max_states,
                    "trial_count_median": int(np.median(state.trial_count)),
                }
                results["configurations"].append(config_result)

                status = "OK" if all_finite and rl_bounded else "FAIL"
                if not (all_finite and rl_bounded):
                    results["all_stable"] = False
                print(f" {status} (trials={config_result['trial_count_median']})")

    n_ok = sum(1 for c in results["configurations"]
               if c["all_finite"] and c["rl_bounded"])
    print(f"\n  {n_ok}/{total} configurations stable")

    return results


def main():
    results_a = analysis_a_jacobian()
    results_b = analysis_b_numerical()

    results = {
        "analysis_a_jacobian": results_a,
        "analysis_b_numerical": results_b,
        "summary": {
            "jacobian_all_match": results_a["all_match_arousal"],
            "numerical_all_stable": results_b["all_stable"],
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), '..',
                             'results', 'p4_stability.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    print("=== P4 Step 1: Triple Composite Stability ===\n")
    main()
