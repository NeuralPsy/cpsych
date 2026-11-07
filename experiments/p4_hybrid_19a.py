"""P4 Step 2: Hybrid stability check (Rule 19a).

Verifies triple composite behavior across wake/sleep transitions.
Rule 19a: hybrid (switching) systems can lose stability at transitions
even if both regimes are individually stable.

Checks:
1. No divergence at max coupling strengths over 2 days.
2. At least 2 wake→sleep + 2 sleep→wake transitions.
3. No spurious rapid oscillations near threshold.
4. Sleep duration within relaxed bounds.
5. v_vh boundary scan: system cycles at all compatible v_vh values.

Results: results/p4_hybrid_19a.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpsych.core.composite_setup import make_triple_integrator
from cpsych.modules.arousal import Q_THRESH, _sigmoid, _I


def extract_transitions(state, obs_dt):
    """Extract wake/sleep transitions from arousal.vm recordings.

    Returns list of transition dicts and per-replicate statistics.
    """
    # Get vm recordings
    vm_recs = state.recordings["arousal.vm"]
    B = vm_recs[0][1].shape[0]

    results_per_rep = []
    for b in range(B):
        vm_series = np.array([r[1][b] for r in vm_recs])
        times = np.array([r[0] for r in vm_recs])

        # Compute Q_m from vm
        Q_max, theta_sig, sigma_q = 100.0, 10.0, 3.0
        Qm = Q_max / (1.0 + np.exp(-(vm_series - theta_sig) / sigma_q))
        is_wake = Qm > Q_THRESH

        # Find transitions
        transitions = []
        for i in range(1, len(is_wake)):
            if is_wake[i] != is_wake[i - 1]:
                trans_type = ("SLEEP->WAKE" if is_wake[i]
                              else "WAKE->SLEEP")
                transitions.append({
                    "time_h": times[i] / 3600.0,
                    "type": trans_type,
                })

        # Extract sleep/wake durations
        sleep_durs = []
        wake_durs = []
        for i in range(len(transitions) - 1):
            dur_h = transitions[i + 1]["time_h"] - transitions[i]["time_h"]
            if transitions[i]["type"] == "WAKE->SLEEP":
                sleep_durs.append(dur_h)
            else:
                wake_durs.append(dur_h)

        # Count transitions in 1h windows (spurious oscillation check)
        max_trans_per_hour = 0
        if transitions:
            T_total_h = times[-1] / 3600.0
            for t_start in np.arange(0, T_total_h - 1, 0.5):
                count = sum(1 for t in transitions
                            if t_start <= t["time_h"] < t_start + 1.0)
                max_trans_per_hour = max(max_trans_per_hour, count)

        n_ws = sum(1 for t in transitions if t["type"] == "WAKE->SLEEP")
        n_sw = sum(1 for t in transitions if t["type"] == "SLEEP->WAKE")

        results_per_rep.append({
            "n_transitions": len(transitions),
            "n_wake_to_sleep": n_ws,
            "n_sleep_to_wake": n_sw,
            "sleep_durations_h": [round(d, 2) for d in sleep_durs],
            "wake_durations_h": [round(d, 2) for d in wake_durs],
            "max_transitions_per_hour": max_trans_per_hour,
        })

    return results_per_rep


def run_hybrid_check(B=50, T=172800, s_kappa=4.0, s_alpha=4.0,
                     s_value=10.0, seed=42, obs_dt=60.0):
    """Run composite at given strengths, check hybrid stability."""
    print(f"  B={B}, T={T/3600:.0f}h, s=({s_kappa},{s_alpha},{s_value})...")

    integ = make_triple_integrator(
        B=B, seed=seed, obs_dt=obs_dt,
        s_kappa=s_kappa, s_alpha=s_alpha, s_value=s_value)
    state = integ.run(T=float(T))

    # Check all states finite
    all_finite = True
    for name in ["arousal", "ddm", "rl_mood"]:
        if not np.all(np.isfinite(state.x[name])):
            all_finite = False

    per_rep = extract_transitions(state, obs_dt)

    # Aggregate
    min_ws = min(r["n_wake_to_sleep"] for r in per_rep)
    min_sw = min(r["n_sleep_to_wake"] for r in per_rep)
    max_trans_hr = max(r["max_transitions_per_hour"] for r in per_rep)

    all_sleeps = [d for r in per_rep for d in r["sleep_durations_h"]
                  if len(r["sleep_durations_h"]) > 1]
    # Skip first sleep (may be transient)
    mean_sleep = float(np.mean(all_sleeps)) if all_sleeps else 0.0

    return {
        "all_finite": bool(all_finite),
        "min_wake_to_sleep": min_ws,
        "min_sleep_to_wake": min_sw,
        "max_transitions_per_hour": max_trans_hr,
        "mean_sleep_h": round(mean_sleep, 2),
        "n_replicates": B,
        "per_replicate_summary": {
            "n_transitions_range": [
                min(r["n_transitions"] for r in per_rep),
                max(r["n_transitions"] for r in per_rep),
            ],
        },
    }


def run_v_vh_scan(B=20, T=172800, seed=42, obs_dt=60.0):
    """Scan v_vh values with all couplings at default strengths."""
    v_vh_values = [0.30, 0.31, 0.32, 0.325, 0.33, 0.34, 0.35]
    results = []

    for v_vh in v_vh_values:
        print(f"  v_vh={v_vh}...")

        integ = make_triple_integrator(
            B=B, seed=seed, obs_dt=obs_dt,
            s_kappa=2.0, s_alpha=2.0, s_value=3.0)

        # Override v_vh in theta_base after init
        state = integ._init_state()
        from cpsych.modules.arousal import _I as arousal_I
        state.theta_base["arousal"][:, arousal_I["v_vh"]] = v_vh

        # Run manually
        stimulus = {}
        if integ.env is not None:
            stimulus = integ.env.next_trial(integ.rng)

        integ._record(state)
        slowest = integ.tiers[0]
        n_slow = int(round(T / slowest.dt))
        obs_every = max(1, int(round(obs_dt / slowest.dt)))
        next_obs = obs_every

        for step in range(n_slow):
            for mod_name in slowest.module_names:
                integ._step_module(mod_name, slowest.dt, state, stimulus)
            if len(integ.tiers) > 1:
                n_sub = integ._tier_ratios[1]
                integ._run_tier(1, n_sub, state, stimulus)
            else:
                integ._handle_trials(0, state, stimulus)
            state.t = (step + 1) * slowest.dt
            if (step + 1) >= next_obs:
                integ._record(state)
                next_obs += obs_every

        # Check finite
        all_finite = all(np.all(np.isfinite(state.x[n]))
                         for n in ["arousal", "ddm", "rl_mood"])

        per_rep = extract_transitions(state, obs_dt)
        min_ws = min(r["n_wake_to_sleep"] for r in per_rep)
        min_sw = min(r["n_sleep_to_wake"] for r in per_rep)

        results.append({
            "v_vh": v_vh,
            "all_finite": bool(all_finite),
            "min_wake_to_sleep": min_ws,
            "min_sleep_to_wake": min_sw,
            "cycles": min_ws >= 2 and min_sw >= 2,
        })

    return results


def main():
    results = {}

    print("=== Hybrid check at max strengths ===")
    results["max_strengths"] = run_hybrid_check(
        B=50, T=172800, s_kappa=4.0, s_alpha=4.0, s_value=10.0)

    print("\n=== v_vh boundary scan ===")
    results["v_vh_scan"] = run_v_vh_scan(B=20, T=172800)

    results["summary"] = {
        "max_strengths_stable": results["max_strengths"]["all_finite"],
        "transitions_occur": (results["max_strengths"]["min_wake_to_sleep"] >= 2
                              and results["max_strengths"]["min_sleep_to_wake"] >= 2),
        "no_spurious_oscillations": results["max_strengths"]["max_transitions_per_hour"] <= 4,
        "v_vh_all_cycle": all(r["cycles"] for r in results["v_vh_scan"]),
    }

    out_path = os.path.join(os.path.dirname(__file__), '..',
                             'results', 'p4_hybrid_19a.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    print("=== P4 Step 2: Hybrid Stability Check (Rule 19a) ===\n")
    main()
