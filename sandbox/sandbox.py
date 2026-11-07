"""P5: Interactive sandbox for computational psychiatry platform.

Spec: specs/sandbox.md v2.
Three response tiers:
  1. Instant (< 100ms): analytical DDM formulas, λ_crit, map position, warnings.
  2. On-demand (~2-3 min): stochastic simulation with B=30 for confidence intervals.
  3. Pre-computed: impossible map from P4 JSON.

Analytical results labeled "DDM без связей, идеализация".
Simulation results show both analytic and simulated values side by side.

Usage: jupyter notebook, then run this file or import and call create_sandbox().
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# ── Analytical DDM (instant) ─────────────────────────────────────────

def ddm_accuracy(kappa: float, a: float, coh: float) -> float:
    """Analytical DDM accuracy. Exact for standalone DDM without couplings."""
    p = kappa * a * coh
    if abs(p) < 1e-12:
        return 0.5
    return 1.0 / (1.0 + np.exp(-p))


def ddm_mean_rt(kappa: float, a: float, Ter: float, coh: float) -> float:
    """Analytical DDM mean RT = mean DT + Ter."""
    p = kappa * a * coh
    if abs(p) < 1e-12:
        dt = a**2 / 4  # limit: a^2 * 0.25
    else:
        dt = a**2 * np.tanh(p / 2) / (2 * p)
    return dt + Ter


def lambda_crit(tau_m: float, r_mean: float = 0.60) -> float:
    """Critical λ for mood stability. D20."""
    gamma = np.exp(-1.0 / tau_m)
    return (1.0 - gamma) / (gamma * r_mean)


# ── Map lookup ───────────────────────────────────────────────────────

def load_map(pair: str) -> dict | None:
    """Load pre-computed impossible map."""
    path = RESULTS / f"p4_impossible_map_{pair}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def find_nearest_node(grid: list, s1_key: str, s1_val: float,
                      s2_key: str, s2_val: float) -> dict | None:
    """Find grid node closest to given coupling strengths."""
    best = None
    best_dist = float("inf")
    for g in grid:
        d = (g[s1_key] - s1_val)**2 + (g[s2_key] - s2_val)**2
        if d < best_dist:
            best_dist = d
            best = g
    return best


# ── Simulation runner ────────────────────────────────────────────────

def run_simulation(params: dict, callback=None) -> dict:
    """Run windowed simulation with B=30 for confidence intervals.

    params: dict with keys kappa, a, Ter, beta, alpha, tau_m, lam,
            s_kappa, s_alpha, s_value, coherence, p_reward, volatility.
    callback: called with (progress_pct, message) during run.

    Returns dict with per-window and aggregate results + CI.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cpsych.core.composite_setup import wake_equilibrium
    from cpsych.core.integrator import Integrator, Tier
    from cpsych.core.coupling import Coupling
    from cpsych.modules.arousal import ArousalModule
    from cpsych.modules.ddm import DDMModule
    from cpsych.modules.rl_mood import RLMoodModule
    from cpsych.environments.composite_task import CompositeTask
    from cpsych.observables.composite import (
        accuracy, mean_rt, win_stay_rate, trial_count)

    B = 30
    WINDOWS = [
        ("peak", 4.0), ("decline", 8.5), ("mid_evening", 9.0),
        ("evening", 9.5), ("transition", 10.3),
    ]
    WIN_DUR = 600.0

    weq = wake_equilibrium(ArousalModule(q_m_wake_mean_baseline=4.2883))

    all_acc = np.zeros((len(WINDOWS), B))
    all_rt = np.zeros((len(WINDOWS), B))
    all_ws = np.zeros((len(WINDOWS), B))
    all_tc = np.zeros((len(WINDOWS), B))

    for wi, (wname, wt) in enumerate(WINDOWS):
        if callback:
            callback(wi * 20, f"Window {wname} ({wi+1}/5)...")

        modules = {
            "arousal": ArousalModule(q_m_wake_mean_baseline=4.2883),
            "ddm": DDMModule(),
            "rl_mood": RLMoodModule(),
        }
        env = CompositeTask(
            coherence=params["coherence"],
            p_reward=params["p_reward"],
            volatility=params["volatility"])
        couplings = [
            Coupling(
                source_module="arousal", source_output="gain",
                target_module="ddm", target_param="kappa",
                h=lambda g, k: k * (g - 1),
                strength=params["s_kappa"], max_strength=4.0, param_min=0.0),
            Coupling(
                source_module="arousal", source_output="gain",
                target_module="rl_mood", target_param="alpha",
                h=lambda g, a: a * (g - 1),
                strength=params["s_alpha"], max_strength=4.0,
                param_min=0.0, param_max=1.0),
            Coupling(
                source_module="rl_mood", source_output="value_diff",
                target_module="ddm", target_param="drift_bias",
                strength=params["s_value"], max_strength=20.0),
        ]

        integ = Integrator(
            modules=modules, environment=env, couplings=couplings,
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 0.001, ["ddm", "rl_mood"]),
            ],
            trial_driver="ddm", gate_module="arousal",
            gate_output="is_wake",
            B=B, seed=42 + wi, obs_dt=WIN_DUR,
            record_behavior=True)

        state = integ._init_state()

        # Override module parameters
        state.theta_base["ddm"][:, 0] = params["kappa"]
        state.theta_base["ddm"][:, 1] = params["a"]
        state.theta_base["ddm"][:, 2] = params["Ter"]
        state.theta_base["rl_mood"][:, 0] = params["alpha"]
        state.theta_base["rl_mood"][:, 1] = params["beta"]
        state.theta_base["rl_mood"][:, 2] = params["tau_m"]
        state.theta_base["rl_mood"][:, 3] = params["lam"]

        for b in range(B):
            state.x["arousal"][b] = weq["x0"][0]

        state = integ.run(T=WIN_DUR, state=state)

        beh = state.behavior
        all_acc[wi] = accuracy(beh)
        all_rt[wi] = mean_rt(beh)
        all_ws[wi] = win_stay_rate(beh)
        all_tc[wi] = trial_count(beh)

    if callback:
        callback(100, "Done")

    # Aggregate: trial-weighted mean per replicate across windows
    valid = all_tc > 0
    tc_total = np.sum(all_tc, axis=0)  # (B,)

    acc_agg = np.zeros(B)
    rt_agg = np.zeros(B)
    for b in range(B):
        if tc_total[b] > 0:
            w = all_tc[:, b]
            acc_agg[b] = np.sum(all_acc[:, b] * w) / tc_total[b]
            rts = np.where(np.isnan(all_rt[:, b]), 0, all_rt[:, b])
            rt_agg[b] = np.sum(rts * w) / tc_total[b]

    ws_valid = all_ws[~np.isnan(all_ws)].reshape(-1) if np.any(~np.isnan(all_ws)) else np.array([])
    ws_agg = np.nanmean(all_ws, axis=0)

    def ci(arr):
        """Prediction interval: mean +/- 1.96*SD across replicates.

        Shows the range a single new replicate would fall in, not the
        precision of the mean estimate. User sees 'what one run produces',
        not 'average of infinite runs'.
        """
        m = np.nanmean(arr)
        s = np.nanstd(arr, ddof=1)
        return m, m - 1.96*s, m + 1.96*s

    return {
        "accuracy": ci(acc_agg),
        "mean_rt": ci(rt_agg),
        "win_stay": ci(ws_agg),
        "trial_count": (float(np.mean(tc_total)), float(np.min(tc_total)),
                        float(np.max(tc_total))),
        "per_window": {
            wname: {
                "accuracy": ci(all_acc[wi]),
                "mean_rt": ci(all_rt[wi]),
                "trial_count": float(np.mean(all_tc[wi])),
            }
            for wi, (wname, _) in enumerate(WINDOWS)
        },
        "B": B,
        "n_trials_total": int(np.sum(tc_total)),
    }


# ── Warnings ─────────────────────────────────────────────────────────

def check_warnings(params: dict) -> list[str]:
    """Return list of warning strings for current parameter values."""
    warnings = []

    lam = params.get("lam", 0.49)
    tau_m = params.get("tau_m", 2.163)
    lc = lambda_crit(tau_m)
    if lam > lc:
        warnings.append(
            f"Mood loop unstable: lambda={lam:.2f} > lambda_crit={lc:.2f}")

    v_vh = params.get("v_vh", 0.325)
    if v_vh < 0.325:
        warnings.append(
            f"v_vh={v_vh:.3f} below compatibility window [0.325, 0.35]"
            " (no margin below)")
    elif v_vh > 0.35:
        warnings.append(
            f"v_vh={v_vh:.3f} above compatibility window [0.325, 0.35]")

    s_value = params.get("s_value", 3.0)
    if s_value > 10:
        warnings.append(
            f"s_value={s_value:.1f} > 10: accuracy degrades (D29)")

    return warnings


# ── Widget interface ─────────────────────────────────────────────────

def create_sandbox():
    """Create and display the interactive sandbox widget."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt
    except ImportError:
        print("Requires: pip install ipywidgets matplotlib")
        print("Run in Jupyter notebook.")
        return

    # ── Defaults ──
    DEFAULTS = {
        "kappa": 14.724, "a": 1.492, "Ter": 0.318,
        "beta": 2.50, "alpha": 0.15,
        "tau_m": 2.163, "lam": 0.49,
        "s_kappa": 2.0, "s_alpha": 2.0, "s_value": 3.0,
        "coherence": 0.128, "p_reward": 0.75, "volatility": 0.3,
        "v_vh": 0.325,
    }

    # ── Resolution steps ──
    MDD = {
        "kappa": 0.5, "a": 0.05, "Ter": 0.01,
        "beta": 0.4, "alpha": 0.06,
    }

    # ── Sliders ──
    style = {"description_width": "120px"}

    def fitted_slider(name, val, lo, hi, step, desc):
        return widgets.FloatSlider(
            value=val, min=lo, max=hi, step=step,
            description=f"● {desc}", style=style,
            layout=widgets.Layout(width="400px"),
            readout_format=f".{max(0, -int(np.floor(np.log10(step))))}f")

    def exploratory_slider(name, val, lo, hi, step, desc):
        return widgets.FloatSlider(
            value=val, min=lo, max=hi, step=step,
            description=f"○ {desc}", style=style,
            layout=widgets.Layout(width="400px"),
            readout_format=".3f")

    # Fitted (●)
    sl_kappa = fitted_slider("kappa", 14.724, 5.0, 25.0, 0.5, "κ (drift)")
    sl_a = fitted_slider("a", 1.492, 0.5, 3.0, 0.05, "a (threshold)")
    sl_Ter = fitted_slider("Ter", 0.318, 0.1, 0.6, 0.01, "Ter (non-dec)")
    sl_beta = fitted_slider("beta", 2.50, 0.5, 5.0, 0.4, "β (inv.temp)")
    sl_alpha = fitted_slider("alpha", 0.15, 0.03, 0.50, 0.06, "α (learn)")

    # Exploratory (○)
    sl_tau_m = exploratory_slider("tau_m", 2.163, 0.5, 10.0, 0.1, "τ_m (mood)")
    sl_lam = exploratory_slider("lam", 0.49, 0.0, 3.0, 0.05, "λ (mood coup)")
    sl_v_vh = exploratory_slider("v_vh", 0.325, 0.20, 0.45, 0.005, "v_vh (H sens)")

    # Couplings
    sl_s_kappa = widgets.FloatSlider(
        value=2.0, min=0, max=4, step=0.2,
        description="s_kappa", style=style,
        layout=widgets.Layout(width="400px"))
    sl_s_alpha = widgets.FloatSlider(
        value=2.0, min=0, max=4, step=0.2,
        description="s_alpha", style=style,
        layout=widgets.Layout(width="400px"))
    sl_s_value = widgets.FloatSlider(
        value=3.0, min=0, max=20, step=0.5,
        description="s_value", style=style,
        layout=widgets.Layout(width="400px"))

    # Task
    sl_coh = widgets.Dropdown(
        options=[0.0, 0.032, 0.064, 0.128, 0.256, 0.512],
        value=0.128, description="coherence", style=style)

    # Output areas
    out_analytic = widgets.Output()
    out_sim = widgets.Output()
    out_warnings = widgets.Output()
    out_map = widgets.Output()
    progress = widgets.FloatProgress(
        value=0, min=0, max=100, description="Simulate:",
        layout=widgets.Layout(width="400px", visibility="hidden"))
    btn_sim = widgets.Button(description="Simulate (~3 min)",
                             button_style="primary")

    def get_params():
        return {
            "kappa": sl_kappa.value, "a": sl_a.value, "Ter": sl_Ter.value,
            "beta": sl_beta.value, "alpha": sl_alpha.value,
            "tau_m": sl_tau_m.value, "lam": sl_lam.value,
            "s_kappa": sl_s_kappa.value, "s_alpha": sl_s_alpha.value,
            "s_value": sl_s_value.value,
            "coherence": sl_coh.value,
            "p_reward": 0.75, "volatility": 0.3,
            "v_vh": sl_v_vh.value,
        }

    sim_result = [None]  # mutable container for thread result

    def update_analytic(_=None):
        """Instant: analytical DDM + warnings."""
        p = get_params()
        coh = p["coherence"]

        with out_analytic:
            clear_output(wait=True)
            print("═══ DDM без связей (идеализация) ═══")
            if coh > 0:
                acc = ddm_accuracy(p["kappa"], p["a"], coh)
                rt = ddm_mean_rt(p["kappa"], p["a"], p["Ter"], coh)
                print(f"  accuracy = {acc:.4f}")
                print(f"  mean RT  = {rt:.4f} s")
            else:
                print(f"  accuracy = 0.500 (coh=0, chance)")
                rt = p["a"]**2 / 4 + p["Ter"]
                print(f"  mean RT  = {rt:.4f} s")

            lc = lambda_crit(p["tau_m"])
            print(f"\n  λ_crit = {lc:.3f} (current λ = {p['lam']:.2f})")

            # α class
            # Estimate trials per window at this coherence
            drift = p["kappa"] * coh if coh > 0 else 0
            if drift > 0:
                mean_dt = p["a"]**2 * np.tanh(drift * p["a"] / 2) / (2 * drift * p["a"])
            else:
                mean_dt = p["a"]**2 / 4
            trials_per_window = int(600.0 / (mean_dt + p["Ter"]))
            total_trials = trials_per_window * 5
            alpha_class = "fitted" if total_trials >= 1000 else "exploratory"
            print(f"\n  α class: {alpha_class} (~{total_trials} trials"
                  f" at coh={coh})")
            if alpha_class == "exploratory":
                print(f"    r=0.67 at N=500. Fitted at N≥1000.")

            if sim_result[0]:
                sr = sim_result[0]
                print(f"\n═══ Simulation (B={sr['B']}) ═══")
                for obs_name in ["accuracy", "mean_rt", "win_stay"]:
                    m, lo, hi = sr[obs_name]
                    print(f"  {obs_name:12s} = {m:.4f} [{lo:.4f}, {hi:.4f}]")
                tc_m, tc_lo, tc_hi = sr["trial_count"]
                print(f"  {'trials':12s} = {tc_m:.0f} [{tc_lo:.0f}, {tc_hi:.0f}]")

        with out_warnings:
            clear_output(wait=True)
            ws = check_warnings(p)
            if ws:
                print("⚠ Warnings:")
                for w in ws:
                    print(f"  - {w}")

    def on_simulate(_):
        """Run full simulation in background thread."""
        p = get_params()
        progress.layout.visibility = "visible"
        progress.value = 0
        btn_sim.disabled = True

        def cb(pct, msg):
            progress.value = pct
            progress.description = msg

        def worker():
            result = run_simulation(p, callback=cb)
            sim_result[0] = result
            progress.layout.visibility = "hidden"
            btn_sim.disabled = False
            update_analytic()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    btn_sim.on_click(on_simulate)

    # Observe all sliders
    for sl in [sl_kappa, sl_a, sl_Ter, sl_beta, sl_alpha,
               sl_tau_m, sl_lam, sl_v_vh,
               sl_s_kappa, sl_s_alpha, sl_s_value, sl_coh]:
        sl.observe(update_analytic, names="value")

    # Layout
    fitted_box = widgets.VBox([
        widgets.HTML("<b>● Fitted</b> (step = MDD)"),
        sl_kappa, sl_a, sl_Ter, sl_beta, sl_alpha,
    ])
    exploratory_box = widgets.VBox([
        widgets.HTML("<b>○ Exploratory</b> (recovery < 0.7)"),
        sl_tau_m, sl_lam, sl_v_vh,
    ])
    coupling_box = widgets.VBox([
        widgets.HTML("<b>Coupling strengths</b>"),
        sl_s_kappa, sl_s_alpha, sl_s_value,
    ])
    task_box = widgets.VBox([
        widgets.HTML("<b>Task</b>"),
        sl_coh,
    ])

    left = widgets.VBox([fitted_box, exploratory_box, coupling_box, task_box])
    right = widgets.VBox([
        out_analytic,
        widgets.HBox([btn_sim, progress]),
        out_warnings,
    ])

    ui = widgets.HBox([left, right],
                       layout=widgets.Layout(width="100%"))

    update_analytic()
    display(ui)


if __name__ == "__main__":
    print("Run in Jupyter notebook:")
    print("  from sandbox.sandbox import create_sandbox")
    print("  create_sandbox()")
