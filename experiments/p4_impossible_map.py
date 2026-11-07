"""P4 Step 4: Impossible map — what the model refuses to produce.

Windowed wake observations from settled arousal state.
Five windows at different circadian phases, stored per-window.
dt_fast=0.001 (D37: scheme not coarsened for budget).

Three maps:
  1. (s_kappa, s_value)  — s_alpha=2 fixed
  2. (s_kappa, s_alpha)  — s_value=3 fixed
  3. (s_alpha, s_value)  — s_kappa=2 fixed

Results: results/p4_impossible_map_{pair}.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpsych.core.composite_setup import make_grid_integrator, wake_equilibrium
from cpsych.modules.arousal import ArousalModule, _sigmoid, _I, Q_THRESH
from cpsych.observables.composite import accuracy, mean_rt, win_stay_rate, trial_count


# ── Window schedule ──────────────────────────────────────────────────
# From settled wake equilibrium. Gain declines over ~11h of wake.
WINDOWS = [
    # (name, offset_hours, description)
    ("peak",       4.0,  "gain≈1.05, coupling boosts κ_eff +22% at s=4"),
    ("decline",    8.5,  "gain≈0.99, coupling nearly neutral — baseline"),
    ("mid_evening", 9.0, "gain≈0.97, moderate coupling effect"),
    ("evening",    9.5,  "gain≈0.94, strong coupling effect"),
    ("transition", 10.3, "gain≈0.86, maximum coupling effect in wake"),
]
WIN_DUR = 600.0  # seconds per window


def precompute_arousal_states(windows):
    """Deterministic arousal pre-run: return (x, gain) at each window start."""
    mod = ArousalModule(q_m_wake_mean_baseline=4.2883)
    weq = wake_equilibrium(mod, T_settle_days=3)
    x = weq["x0"].copy()
    theta = weq["theta0"].copy()
    # Zero noise for deterministic trajectory
    theta[0, _I["sigma_vv"]] = 0.0
    theta[0, _I["sigma_vm"]] = 0.0
    theta[0, _I["sigma_h"]] = 0.0
    u = np.zeros((1, 0))

    max_t = max(t_h for _, t_h, _ in windows)
    max_steps = int(max_t * 3600) + 1

    # Advance to each window, save state
    states = {}
    next_win = 0
    sorted_wins = sorted(enumerate(windows), key=lambda iw: iw[1][1])

    for step in range(max_steps):
        t_h = step / 3600.0
        # Check if we've reached a window start
        while next_win < len(sorted_wins):
            wi, (name, t_target, _) = sorted_wins[next_win]
            if t_h >= t_target:
                Qm = _sigmoid(x[0, 1], 100.0, 10.0, 3.0)
                gain = float(Qm / 4.2883)
                is_wake = Qm > Q_THRESH
                states[name] = {
                    "x": x.copy(),
                    "gain": gain,
                    "is_wake": bool(is_wake),
                    "t_h": t_h,
                }
                next_win += 1
            else:
                break
        dx = mod.drift(x, theta, u)
        x = x + dx * 1.0

    return states


def kappa_eff(s_kappa, gain, kappa_base=14.7242):
    """Effective kappa after coupling modulation."""
    return kappa_base * np.maximum(0.0, 1.0 + s_kappa * (gain - 1.0))


def classify_no_trials(n_trials, kappa_eff_val, was_wake):
    """Classify cause of absent or very low trials.

    Returns None if trials are sufficient, else one of:
      'gate': sleep gate blocked trials (arousal in sleep)
      'kappa_suppression': κ_eff ≈ 0, DDM cannot accumulate evidence
      'timeout': κ_eff > 0 but trials too slow / all timed out
    """
    if n_trials >= 5:
        return None
    if not was_wake:
        return "gate"
    if kappa_eff_val < 0.5:
        return "kappa_suppression"
    return "timeout"


def classify_point(windows_data, n_trials_total, acc_total, acc_std_total):
    """Classify a grid point into region types.

    Returns (label, detail) where label is one of:
      UNSTABLE, NO_DATA_GATE, NO_DATA_KAPPA, NO_DATA_TIMEOUT,
      CHANCE, NORMAL
    """
    # Check finiteness across all windows
    if not all(w["all_finite"] for w in windows_data):
        return "UNSTABLE", "non-finite states"

    # Check trial availability
    if n_trials_total == 0:
        # Determine cause from individual windows
        causes = [w.get("no_trial_cause") for w in windows_data
                  if w.get("no_trial_cause")]
        if "gate" in causes:
            return "NO_DATA_GATE", "sleep gate blocked all windows"
        if "kappa_suppression" in causes:
            return "NO_DATA_KAPPA", "kappa_eff suppressed to ~0"
        return "NO_DATA_TIMEOUT", "all trials timed out"

    if n_trials_total < 10:
        return "NO_DATA_KAPPA", f"only {n_trials_total} trials"

    # CHANCE: binomial test against 0.50
    n_correct = int(round(acc_total * n_trials_total))
    p = binomtest(n_correct, n_trials_total, 0.5, alternative="greater").pvalue
    if p > 0.05:
        return "CHANCE", f"acc={acc_total:.3f}, p={p:.3f}"

    return "NORMAL", None


def detect_degeneracy(grid, n_nodes_x, n_nodes_y, threshold_sem=2.0):
    """Flag nodes whose output is indistinguishable from ALL neighbours.

    A node is INDISTINGUISHABLE only if accuracy, mean_rt, AND win_stay_rate
    are all within threshold_sem * pooled_SEM of every neighbour.
    Returns (B,) bool array. True = degenerate (flat region).
    """
    # Pre-compute SEMs for each observable
    for g in grid:
        n = g["trial_count_total"]
        acc = g.get("accuracy_agg")
        if acc is not None and n > 0:
            g["acc_sem"] = np.sqrt(acc * (1 - acc) / n)
        else:
            g["acc_sem"] = 0.01
        rts = [w["mean_rt"] for w in g["windows"]
               if w["mean_rt"] is not None]
        g["rt_sem"] = (np.std(rts) / np.sqrt(len(rts))
                       if len(rts) > 1 else 0.01)
        ws_vals = [w["win_stay"] for w in g["windows"]
                   if w["win_stay"] is not None]
        g["ws_agg"] = float(np.mean(ws_vals)) if ws_vals else None
        g["ws_sem"] = (np.std(ws_vals) / np.sqrt(len(ws_vals))
                       if len(ws_vals) > 1 else 0.01)

    obs_checks = [
        ("accuracy_agg", "acc_sem"),
        ("mean_rt_agg", "rt_sem"),
        ("ws_agg", "ws_sem"),
    ]
    labels = [g["label"] for g in grid]
    degenerate = np.zeros(len(grid), dtype=bool)

    for i in range(len(grid)):
        if labels[i] != "NORMAL":
            continue
        ix, iy = divmod(i, n_nodes_y)
        neighbours = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < n_nodes_x and 0 <= ny < n_nodes_y:
                j = nx * n_nodes_y + ny
                if labels[j] == "NORMAL":
                    neighbours.append(j)
        if not neighbours:
            continue
        all_obs_flat = True
        for obs_key, sem_key in obs_checks:
            val_i = grid[i].get(obs_key)
            if val_i is None:
                continue
            sem_i = grid[i][sem_key]
            for j in neighbours:
                val_j = grid[j].get(obs_key)
                if val_j is None:
                    continue
                sem_j = grid[j][sem_key]
                pooled = np.sqrt((sem_i**2 + sem_j**2) / 2)
                if abs(val_i - val_j) >= threshold_sem * max(pooled, 1e-6):
                    all_obs_flat = False
                    break
            if not all_obs_flat:
                break
        if all_obs_flat:
            degenerate[i] = True

    return degenerate


def run_windowed_map(pair_name, s_kappa_vals, s_alpha_vals, s_value_vals,
                     fixed_name, fixed_val, seed=42):
    """Run 2D grid with windowed wake observations."""
    # Build node list
    nodes = []
    for sk in s_kappa_vals:
        for sa in s_alpha_vals:
            for sv in s_value_vals:
                nodes.append((sk, sa, sv))
    n_nodes = len(nodes)
    B = n_nodes  # n_rep=1

    s_kappa_arr = np.array([sk for sk, sa, sv in nodes])
    s_alpha_arr = np.array([sa for sk, sa, sv in nodes])
    s_value_arr = np.array([sv for sk, sa, sv in nodes])

    # Pre-compute arousal states at each window
    arousal_states = precompute_arousal_states(WINDOWS)

    # Run each window
    all_window_data = {name: {} for name, _, _ in WINDOWS}
    for win_idx, (win_name, t_h, win_desc) in enumerate(WINDOWS):
        astate = arousal_states[win_name]
        print(f"  Window '{win_name}' (t={t_h:.1f}h, gain={astate['gain']:.4f})...",
              end="", flush=True)

        integ = make_grid_integrator(
            s_kappa_arr, s_alpha_arr, s_value_arr,
            seed=seed + win_idx, obs_dt=WIN_DUR,
            record_behavior=True,
        )
        state = integ._init_state()
        # Set arousal initial state (same for all nodes — arousal autonomous)
        for b in range(B):
            state.x["arousal"][b] = astate["x"][0]
        state = integ.run(T=WIN_DUR, state=state)

        beh = state.behavior
        acc = accuracy(beh)
        mrt = mean_rt(beh)
        ws = win_stay_rate(beh)
        tc = trial_count(beh)

        finite_arr = np.ones(B, dtype=bool)
        for mname in ["arousal", "ddm", "rl_mood"]:
            finite_arr &= np.all(np.isfinite(state.x[mname]), axis=1)

        # Per-node kappa_eff at this window's gain
        ke = kappa_eff(s_kappa_arr, astate["gain"])

        all_window_data[win_name] = {
            "gain": astate["gain"],
            "t_h": astate["t_h"],
            "is_wake": astate["is_wake"],
            "acc": acc,
            "mrt": mrt,
            "ws": ws,
            "tc": tc,
            "finite": finite_arr,
            "kappa_eff": ke,
        }
        print(f" done (trials={np.nanmean(tc):.0f})")

    # Assemble per-node results
    grid = []
    for i, (sk, sa, sv) in enumerate(nodes):
        node = {
            "s_kappa": round(sk, 4),
            "s_alpha": round(sa, 4),
            "s_value": round(sv, 4),
        }
        # Per-window observables
        windows_out = []
        total_correct = 0
        total_trials = 0
        total_rt_sum = 0
        total_ws_num = 0
        total_ws_den = 0

        for win_name, _, _ in WINDOWS:
            wd = all_window_data[win_name]
            tc_i = float(wd["tc"][i])
            acc_i = float(wd["acc"][i]) if not np.isnan(wd["acc"][i]) else None
            mrt_i = float(wd["mrt"][i]) if not np.isnan(wd["mrt"][i]) else None
            ws_i = float(wd["ws"][i]) if not np.isnan(wd["ws"][i]) else None
            fin_i = bool(wd["finite"][i])
            ke_i = float(wd["kappa_eff"][i])

            no_cause = classify_no_trials(tc_i, ke_i, wd["is_wake"])

            w = {
                "window": win_name,
                "gain": round(wd["gain"], 4),
                "trial_count": int(tc_i),
                "accuracy": round(acc_i, 4) if acc_i is not None else None,
                "mean_rt": round(mrt_i, 4) if mrt_i is not None else None,
                "win_stay": round(ws_i, 4) if ws_i is not None else None,
                "kappa_eff": round(ke_i, 2),
                "all_finite": fin_i,
                "no_trial_cause": no_cause,
            }
            windows_out.append(w)

            if tc_i > 0 and acc_i is not None:
                total_correct += acc_i * tc_i
                total_trials += tc_i
            if tc_i > 0 and mrt_i is not None:
                total_rt_sum += mrt_i * tc_i

        node["windows"] = windows_out

        # Aggregate (trial-weighted across windows)
        if total_trials > 0:
            acc_agg = total_correct / total_trials
            rt_agg = total_rt_sum / total_trials
        else:
            acc_agg = np.nan
            rt_agg = np.nan

        # Per-window accuracy for std computation
        per_win_acc = [w["accuracy"] for w in windows_out
                       if w["accuracy"] is not None]
        acc_std = float(np.std(per_win_acc)) if len(per_win_acc) > 1 else 0.0

        node["accuracy_agg"] = round(float(acc_agg), 4) if not np.isnan(acc_agg) else None
        node["accuracy_std_agg"] = round(acc_std, 4)
        node["mean_rt_agg"] = round(float(rt_agg), 4) if not np.isnan(rt_agg) else None
        node["trial_count_total"] = int(total_trials)

        # Classify
        label, detail = classify_point(
            windows_out, int(total_trials),
            float(acc_agg) if not np.isnan(acc_agg) else 0.0,
            acc_std)
        node["label"] = label
        if detail:
            node["detail"] = detail

        grid.append(node)

    # Degeneracy detection (third region type)
    n_x = len(s_kappa_vals) if len(s_kappa_vals) > 1 else len(s_alpha_vals)
    n_y = len(s_value_vals) if len(s_value_vals) > 1 else len(s_alpha_vals)
    degen = detect_degeneracy(grid, n_x, n_y)
    for i, g in enumerate(grid):
        if degen[i] and g["label"] == "NORMAL":
            g["label"] = "INDISTINGUISHABLE"
            g["detail"] = "output flat relative to neighbours"

    counts = {}
    for g in grid:
        counts[g["label"]] = counts.get(g["label"], 0) + 1

    return {
        "pair": pair_name,
        "fixed": {fixed_name: fixed_val},
        "windows": [{"name": n, "t_h": t, "desc": d} for n, t, d in WINDOWS],
        "win_dur_s": WIN_DUR,
        "n_points": n_nodes,
        "n_rep": 1,
        "dt_fast": 0.001,
        "counts": counts,
        "grid": grid,
    }


def main():
    N = 15
    s_kappa_range = np.linspace(0, 4, N).tolist()
    s_alpha_range = np.linspace(0, 4, N).tolist()
    s_value_range = np.linspace(0, 20, N).tolist()

    maps = [
        ("kappa_value", s_kappa_range, [2.0], s_value_range,
         "s_alpha", 2.0),
        ("kappa_alpha", s_kappa_range, s_alpha_range, [3.0],
         "s_value", 3.0),
        ("alpha_value", [2.0], s_alpha_range, s_value_range,
         "s_kappa", 2.0),
    ]

    for pair_name, sk_v, sa_v, sv_v, fixed_name, fixed_val in maps:
        print(f"\n=== Map: {pair_name} ({fixed_name}={fixed_val}) ===")
        n_nodes = len(sk_v) * len(sa_v) * len(sv_v)
        print(f"  {n_nodes} nodes, {len(WINDOWS)} windows x {WIN_DUR:.0f}s")

        t0 = time.perf_counter()
        result = run_windowed_map(
            pair_name, sk_v, sa_v, sv_v, fixed_name, fixed_val)
        t1 = time.perf_counter()

        result["wall_s"] = round(t1 - t0, 1)

        out_path = os.path.join(
            os.path.dirname(__file__), '..',
            'results', f'p4_impossible_map_{pair_name}.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Wall: {t1-t0:.1f}s = {(t1-t0)/60:.1f} min")
        print(f"  Saved to {out_path}")
        for label, cnt in sorted(result["counts"].items()):
            print(f"  {label}: {cnt}")

    # Cross-map interaction check: s_kappa reduces acc, s_value compensates
    print("\n=== Interaction check (kappa_value map) ===")
    kv_path = os.path.join(os.path.dirname(__file__), '..',
                           'results', 'p4_impossible_map_kappa_value.json')
    with open(kv_path) as f:
        kv = json.load(f)
    print(f"{'s_kappa':>8} {'s_value':>8} {'acc':>8} {'trials':>8} {'label':>15}")
    print("-" * 55)
    for g in kv["grid"]:
        if g["s_kappa"] in [0.0, 2.0, 4.0] and g["s_value"] in [0.0, 3.0, 10.0]:
            acc = g.get("accuracy_agg", "N/A")
            acc_s = f"{acc:.4f}" if isinstance(acc, float) else acc
            print(f"{g['s_kappa']:>8.1f} {g['s_value']:>8.1f} {acc_s:>8} "
                  f"{g['trial_count_total']:>8} {g['label']:>15}")


if __name__ == "__main__":
    print("=== P4: Impossible Map (windowed, dt=0.001) ===")
    main()
