"""Shared composite assembly helpers.

Used by experiments/p4_*.py and tests/test_p4_*.py to avoid copy-paste
of module construction, coupling setup, and integrator assembly.
"""
from __future__ import annotations

import numpy as np

from .coupling import Coupling
from .integrator import Integrator, Tier
from .stability import composite_jacobian
from ..modules.arousal import ArousalModule, _sigmoid, _I, Q_THRESH
from ..modules.ddm import DDMModule
from ..modules.rl_mood import RLMoodModule
from ..modules.baa_attention import BAAModule
from ..environments.composite_task import CompositeTask


def make_triple_modules():
    """Return dict of 3 modules with calibrated defaults."""
    return {
        "arousal": ArousalModule(q_m_wake_mean_baseline=4.2883),
        "ddm": DDMModule(),
        "rl_mood": RLMoodModule(),
    }


def make_triple_couplings(s_kappa=2.0, s_alpha=2.0, s_value=3.0):
    """Return list of 3 couplings with given strengths."""
    return [
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="ddm", target_param="kappa",
            h=lambda gain, kappa_base: kappa_base * (gain - 1),
            strength=s_kappa, max_strength=4.0, param_min=0.0,
            doc="gain->kappa",
        ),
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="rl_mood", target_param="alpha",
            h=lambda gain, alpha_base: alpha_base * (gain - 1),
            strength=s_alpha, max_strength=4.0,
            param_min=0.0, param_max=1.0,
            doc="gain->alpha",
        ),
        Coupling(
            source_module="rl_mood", source_output="value_diff",
            target_module="ddm", target_param="drift_bias",
            strength=s_value, max_strength=20.0,
            doc="value_diff->drift_bias",
        ),
    ]


def make_triple_integrator(
    B=1, seed=42, obs_dt=86.4,
    s_kappa=2.0, s_alpha=2.0, s_value=3.0,
    coherence=0.128, p_reward=0.75, volatility=0.3,
    record_behavior=False,
):
    """Full integrator with all modules + couplings + env.

    Parameters
    ----------
    B : batch size
    seed : random seed
    obs_dt : observation interval
    s_kappa, s_alpha, s_value : coupling strengths
    coherence, p_reward, volatility : CompositeTask params
    record_behavior : if True, track trial-level behavioral counters
    """
    modules = make_triple_modules()
    env = CompositeTask(
        coherence=coherence, p_reward=p_reward, volatility=volatility)
    couplings = make_triple_couplings(s_kappa, s_alpha, s_value)

    kwargs = {}
    if record_behavior:
        kwargs["record_behavior"] = True

    integ = Integrator(
        modules=modules,
        environment=env,
        couplings=couplings,
        tiers=[
            Tier("slow", 1.0, ["arousal"]),
            Tier("fast", 0.001, ["ddm", "rl_mood"]),
        ],
        trial_driver="ddm",
        gate_module="arousal",
        gate_output="is_wake",
        B=B,
        seed=seed,
        obs_dt=obs_dt,
        **kwargs,
    )
    return integ


def make_grid_integrator(
    s_kappa_arr: np.ndarray,
    s_alpha_arr: np.ndarray,
    s_value_arr: np.ndarray,
    seed=42,
    obs_dt=86.4,
    coherence=0.128,
    p_reward=0.75,
    volatility=0.3,
    record_behavior=False,
):
    """Vectorized grid integrator: nodes × replicates packed into batch axis.

    Parameters
    ----------
    s_kappa_arr, s_alpha_arr, s_value_arr : (B,) per-replicate coupling strengths
        All must have the same length. Grid nodes × n_rep packed flat.
    """
    B = len(s_kappa_arr)
    assert len(s_alpha_arr) == B and len(s_value_arr) == B

    modules = make_triple_modules()
    env = CompositeTask(
        coherence=coherence, p_reward=p_reward, volatility=volatility)
    couplings = [
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="ddm", target_param="kappa",
            h=lambda gain, kappa_base: kappa_base * (gain - 1),
            strength=np.asarray(s_kappa_arr, dtype=float),
            max_strength=4.0, param_min=0.0,
            doc="gain->kappa",
        ),
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="rl_mood", target_param="alpha",
            h=lambda gain, alpha_base: alpha_base * (gain - 1),
            strength=np.asarray(s_alpha_arr, dtype=float),
            max_strength=4.0,
            param_min=0.0, param_max=1.0,
            doc="gain->alpha",
        ),
        Coupling(
            source_module="rl_mood", source_output="value_diff",
            target_module="ddm", target_param="drift_bias",
            strength=np.asarray(s_value_arr, dtype=float),
            max_strength=20.0,
            doc="value_diff->drift_bias",
        ),
    ]

    kwargs = {}
    if record_behavior:
        kwargs["record_behavior"] = True

    return Integrator(
        modules=modules,
        environment=env,
        couplings=couplings,
        tiers=[
            Tier("slow", 1.0, ["arousal"]),
            Tier("fast", 0.001, ["ddm", "rl_mood"]),
        ],
        trial_driver="ddm",
        gate_module="arousal",
        gate_output="is_wake",
        B=B,
        seed=seed,
        obs_dt=obs_dt,
        **kwargs,
    )


def wake_equilibrium(mod_arousal=None, T_settle_days=3):
    """Pre-run arousal deterministically, return mid-wake state.

    Runs for T_settle_days, then advances to the middle of the next
    wake period (peak gain). Returns a solidly-awake linearization point.

    Returns dict with:
        x0: (1, 4) wake equilibrium state
        theta0: (1, n_params) default params (noise zeroed)
        gain: float, gain at equilibrium
        Qm: float, firing rate at equilibrium
    """
    if mod_arousal is None:
        mod_arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)

    x = mod_arousal.default_state(1)
    theta = mod_arousal.default_params(1)
    # Zero noise for deterministic settling
    theta[0, _I["sigma_vv"]] = 0.0
    theta[0, _I["sigma_vm"]] = 0.0
    theta[0, _I["sigma_h"]] = 0.0
    u = np.zeros((1, 0))

    dt = 1.0
    T = int(T_settle_days * 86400)
    for _ in range(T):
        dx = mod_arousal.drift(x, theta, u)
        x = x + dx * dt

    # Find wake onset then advance to mid-wake for a solid linearization point.
    # Step 1: ensure we're in a wake period (step through any remaining sleep).
    was_wake = False
    for _ in range(86400):
        dx = mod_arousal.drift(x, theta, u)
        x = x + dx * dt
        Qm = _sigmoid(x[0, 1], theta[0, _I["Q_max"]],
                       theta[0, _I["theta_sig"]],
                       theta[0, _I["sigma_q"]])
        is_wake = Qm > Q_THRESH
        if not was_wake and is_wake:
            # Just entered wake — now advance ~4 hours into wake
            break
        was_wake = is_wake

    # Step 2: advance 4h (14400s) into wake for mid-wake plateau
    best_gain = 0.0
    best_x = x.copy()
    for _ in range(14400):
        dx = mod_arousal.drift(x, theta, u)
        x = x + dx * dt
        Qm = _sigmoid(x[0, 1], theta[0, _I["Q_max"]],
                       theta[0, _I["theta_sig"]],
                       theta[0, _I["sigma_q"]])
        g = Qm / mod_arousal.q_m_wake_mean_baseline
        if g > best_gain:
            best_gain = g
            best_x = x.copy()
            best_Qm = float(Qm)

    return {
        "x0": best_x,
        "theta0": theta.copy(),
        "gain": float(best_gain),
        "Qm": best_Qm,
    }


def triple_linearization_point(wake_eq=None):
    """Linearization point for the triple composite at wake equilibrium.

    Returns (modules, x0, theta0, u0) suitable for composite_jacobian().
    """
    modules = make_triple_modules()

    if wake_eq is None:
        wake_eq = wake_equilibrium(modules["arousal"])

    x0 = {
        "arousal": wake_eq["x0"],
        "ddm": modules["ddm"].default_state(1),
        "rl_mood": np.array([[0.5, 0.5, 0.0]]),  # V0=0.5, V1=0.5, m=0
    }
    theta0 = {
        "arousal": wake_eq["theta0"],
        "ddm": modules["ddm"].default_params(1),
        "rl_mood": modules["rl_mood"].default_params(1),
    }
    u0 = {
        "arousal": np.zeros((1, 0)),
        "ddm": np.zeros((1, 2)),
        "rl_mood": np.zeros((1, 0)),
    }
    return modules, x0, theta0, u0


def make_quad_integrator(
    B=1, seed=42, obs_dt=86.4,
    s_kappa=2.0, s_alpha=2.0, s_value=3.0, s_baa=2.0,
    coherence=0.128, p_reward=0.75, volatility=0.3,
    record_behavior=False,
):
    """Quad integrator: triple composite + BAA as 4th module (D33).

    BAA on slow tier alongside arousal (timescale ~10s, dt=1.0 sufficient).
    Coupling: arousal.gain → BAA.gain, multiplicative, same form as gain→κ.
    BAA.gain class: untested (D33).
    """
    modules = make_triple_modules()
    modules["baa"] = BAAModule(nc=5)

    env = CompositeTask(
        coherence=coherence, p_reward=p_reward, volatility=volatility)

    couplings = make_triple_couplings(s_kappa, s_alpha, s_value)
    couplings.append(Coupling(
        source_module="arousal", source_output="gain",
        target_module="baa", target_param="gain",
        h=lambda gain, gain_base: gain_base * (gain - 1),
        strength=s_baa, max_strength=4.0, param_min=0.0,
        doc="arousal gain -> BAA gain (D33, untested)",
    ))

    kwargs = {}
    if record_behavior:
        kwargs["record_behavior"] = True

    return Integrator(
        modules=modules,
        environment=env,
        couplings=couplings,
        tiers=[
            Tier("slow", 1.0, ["arousal", "baa"]),
            Tier("fast", 0.001, ["ddm", "rl_mood"]),
        ],
        trial_driver="ddm",
        gate_module="arousal",
        gate_output="is_wake",
        B=B,
        seed=seed,
        obs_dt=obs_dt,
        **kwargs,
    )
