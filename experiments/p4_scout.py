"""P4 Step 3: Scout — coarse grid over (s_kappa, s_value).

Vectorized: all grid nodes × replicates packed into a single batch axis.
3x3 grid, s_alpha=2 fixed. n_rep=30 per node, T=3600 (1 model hour).
record_behavior=True for trial-level observables.

Results: results/p4_scout.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpsych.core.composite_setup import make_grid_integrator
from cpsych.observables.composite import (
    accuracy, mean_rt, win_stay_rate, trial_count,
)


def run_grid(s_kappa_vals, s_value_vals, s_alpha=2.0,
             n_rep=30, T=3600.0, seed=42):
    """Run entire grid in a single vectorized batch.

    Returns list of result dicts, one per grid node.
    """
    nodes = [(sk, sv) for sk in s_kappa_vals for sv in s_value_vals]
    n_nodes = len(nodes)
    B = n_nodes * n_rep

    s_kappa_arr = np.repeat([sk for sk, sv in nodes], n_rep)
    s_alpha_arr = np.full(B, s_alpha)
    s_value_arr = np.repeat([sv for sk, sv in nodes], n_rep)

    integ = make_grid_integrator(
        s_kappa_arr, s_alpha_arr, s_value_arr,
        seed=seed, obs_dt=T, record_behavior=True,
    )
    state = integ.run(T=T)

    # Per-node observables
    beh = state.behavior
    acc = accuracy(beh).reshape(n_nodes, n_rep)
    mrt = mean_rt(beh).reshape(n_nodes, n_rep)
    ws = win_stay_rate(beh).reshape(n_nodes, n_rep)
    tc = trial_count(beh).reshape(n_nodes, n_rep)

    # Per-node finiteness
    finite_flat = np.ones(B, dtype=bool)
    for name in ["arousal", "ddm", "rl_mood"]:
        finite_flat &= np.all(np.isfinite(state.x[name]), axis=1)
    finite = finite_flat.reshape(n_nodes, n_rep)

    results = []
    for i, (sk, sv) in enumerate(nodes):
        valid = ~np.isnan(acc[i])
        n_valid = int(np.sum(valid))
        r = {
            "s_kappa": sk,
            "s_value": sv,
            "s_alpha": s_alpha,
            "all_finite": bool(np.all(finite[i])),
            "n_valid": n_valid,
        }
        if n_valid > 0:
            r.update({
                "accuracy_mean": round(float(np.nanmean(acc[i])), 4),
                "accuracy_std": round(float(np.nanstd(acc[i])), 4),
                "mean_rt_mean": round(float(np.nanmean(mrt[i])), 4),
                "mean_rt_std": round(float(np.nanstd(mrt[i])), 4),
                "win_stay_mean": round(float(np.nanmean(ws[i])), 4),
                "win_stay_std": round(float(np.nanstd(ws[i])), 4),
                "trial_count_mean": round(float(np.nanmean(tc[i])), 1),
                "trial_count_std": round(float(np.nanstd(tc[i])), 1),
            })
        results.append(r)

    return results


def main():
    s_kappa_vals = [0.0, 2.0, 4.0]
    s_value_vals = [0.0, 3.0, 10.0]

    print(f"Grid: {len(s_kappa_vals)}x{len(s_value_vals)}, "
          f"B={len(s_kappa_vals)*len(s_value_vals)*30}")

    t0 = time.perf_counter()
    grid = run_grid(s_kappa_vals, s_value_vals)
    t1 = time.perf_counter()

    results = {
        "grid": grid,
        "wall_s": round(t1 - t0, 1),
        "description":
            "Coarse 3x3 scout (vectorized): (s_kappa, s_value), s_alpha=2. "
            "n_rep=30, T=3600 (1h). Single batch B=270.",
    }

    print(f"\nWall: {t1-t0:.1f}s")
    print(f"\n{'s_kappa':>8} {'s_value':>8} {'acc':>8} {'rt':>8} "
          f"{'ws':>8} {'trials':>8} {'finite':>6}")
    print("-" * 60)

    for r in grid:
        acc_s = f"{r.get('accuracy_mean', 'N/A'):>8}"
        rt_s = f"{r.get('mean_rt_mean', 'N/A'):>8}"
        ws_s = f"{r.get('win_stay_mean', 'N/A'):>8}"
        tc_s = f"{r.get('trial_count_mean', 'N/A'):>8}"
        print(f"{r['s_kappa']:>8.1f} {r['s_value']:>8.1f} {acc_s} {rt_s} "
              f"{ws_s} {tc_s} {str(r['all_finite']):>6}")

    out_path = os.path.join(os.path.dirname(__file__), '..',
                            'results', 'p4_scout.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    print("=== P4 Step 3: Scout (vectorized) ===\n")
    main()
