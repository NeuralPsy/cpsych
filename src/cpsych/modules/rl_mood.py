"""Rescorla-Wagner RL with momentary affect (Rutledge happiness).

Momentary affective state (m) modulates reward sensitivity via
rho_eff = 1 + lambda * m (Eldar 2016 mechanism). Internal feedback loop:
mood -> RPE -> mood, with decay gamma per trial.

m is NOT clinical mood (hours/days). tau_m ~ 2 trials = seconds.
Clinical mood -> future variant B (D19).

References:
  Rescorla & Wagner (1972) Classical Conditioning II, pp 64-99.
  Rutledge et al. (2014) PNAS 111(33):12252-12257.
  Eldar et al. (2016) Trends Cogn Sci 20(1):15-24.
  Behrens et al. (2007) Nat Neurosci 10(9):1214-1221.

Spec: specs/rl_mood.md v1.3.
Decisions: D19 (variant A), D20 (lambda_crit), D21 (decay in trial_update).
"""
from __future__ import annotations

import numpy as np

from ..core.module import SDEModule
from ..core.specs import (
    ModuleSpec, StateVar, Param,
    Temporality, Exposure,
)

# Defaults TBD after calibration. Placeholders from spec structure.
# tau_m from Rutledge gamma=0.63: -1/ln(0.63) = 2.163.
_TAU_M_DEFAULT = 2.163

_RL_MOOD_SPEC = ModuleSpec(
    name="rl_mood",
    states=(
        StateVar("V0", Temporality.CONTINUOUS, 0.5,
                 doc="Value estimate arm 0"),
        StateVar("V1", Temporality.CONTINUOUS, 0.5,
                 doc="Value estimate arm 1"),
        StateVar("m", Temporality.CONTINUOUS, 0.0,
                 doc="Momentary affect (Rutledge happiness, eta=1 units)"),
    ),
    params=(
        Param("alpha", 0.15, 0.01, 0.50,
              exposure=Exposure.EXPOSED,
              doc="Learning rate. Calibrated: recovery ~10. r=0.67 exploratory."),
        Param("beta", 2.50, 0.5, 10.0,
              exposure=Exposure.EXPOSED,
              doc="Inverse temperature. Calibrated: win-stay ~0.70. r=0.90 fitted."),
        Param("tau_m", _TAU_M_DEFAULT, 1.0, 50.0,
              exposure=Exposure.INTERNAL,
              doc="Mood decay time in trials. gamma=exp(-1/tau_m)"),
        Param("lam", 0.0, 0.0, 1.0,
              exposure=Exposure.INTERNAL,
              doc="Mood->reward coupling. Capped at lambda_crit(tau_m)"),
        Param("sigma_m", 0.0, 0.0, 1.0,
              exposure=Exposure.INTERNAL,
              doc="Mood noise (between-trial fluctuations)"),
    ),
    input_names=(),
    output_names=("choice_prob", "mood", "rho_eff", "value_diff"),
    timescale=1.0,
    doc="RW RL + momentary affect (Rutledge 2014). "
        "eta=1, rho=1 fixed. Decay in trial_update (D21).",
)

# Parameter indices for readability.
_I_ALPHA = 0
_I_BETA = 1
_I_TAU_M = 2
_I_LAM = 3
_I_SIGMA_M = 4


def lambda_crit(tau_m: np.ndarray | float,
                r_mean: float = 0.60) -> np.ndarray | float:
    """Critical lambda for stability: lambda_crit = (1-gamma)/(gamma*r_mean).

    D20: eigenvalue = gamma*(1 + lam*r_mean). Critical at eigenvalue=1.
    Formula conservative (~20% below numerical threshold).
    """
    gamma = np.exp(-1.0 / tau_m)
    return (1.0 - gamma) / (gamma * r_mean)


class RLMoodModule(SDEModule):
    """Rescorla-Wagner RL with momentary affect.

    States: V0 (CONTINUOUS), V1 (CONTINUOUS), m (CONTINUOUS).
    Params: alpha, beta, tau_m, lam, sigma_m.
    Fixed: eta=1 (scale convention), rho=1 (degenerate with beta), m_eq=0.

    drift(m) = 0 (D21). All dynamics in trial_update:
      V[chosen] += alpha * delta
      m <- gamma * (m + delta)
    """

    @property
    def spec(self) -> ModuleSpec:
        return _RL_MOOD_SPEC

    def drift(self, x: np.ndarray, theta: np.ndarray,
              u: np.ndarray) -> np.ndarray:
        # All states: no continuous drift. V updates discrete, m decay in
        # trial_update (D21: trial-paced, invariant to trial duration).
        return np.zeros_like(x)

    def diffusion(self, x: np.ndarray,
                  theta: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        sigma_m = theta[:, _I_SIGMA_M:_I_SIGMA_M + 1]
        return np.hstack([
            np.zeros((B, 1)),   # V0: no noise
            np.zeros((B, 1)),   # V1: no noise
            sigma_m,            # m: mood noise
        ])

    def outputs(self, x: np.ndarray,
                theta: np.ndarray) -> dict[str, np.ndarray]:
        V0 = x[:, 0]
        V1 = x[:, 1]
        m = x[:, 2]
        beta = theta[:, _I_BETA]
        lam = theta[:, _I_LAM]

        # Softmax choice probabilities.
        logits = np.column_stack([beta * V0, beta * V1])
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        choice_prob = probs[:, 1]  # P(action=1)

        rho_eff = 1.0 + lam * m

        return {
            "choice_prob": choice_prob,
            "mood": m,
            "rho_eff": rho_eff,
            "value_diff": V1 - V0,
        }

    def trial_done(self, x: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
        # Every step is a trial boundary in standalone (bandit: one-shot).
        # In composite (P3): DDM trial_done gates the trial loop.
        return np.ones(x.shape[0], dtype=bool)

    def trial_update(self, x: np.ndarray, theta: np.ndarray,
                     mask: np.ndarray,
                     env_result: dict[str, np.ndarray]) -> np.ndarray:
        """Discrete RL update + mood decay at trial boundary.

        Order (D20, D21):
          1. Compute RPE: delta = rho_eff * r - V_chosen
          2. Update V: V[chosen] += alpha * delta  (skip on timeout, D28)
          3. Update m: m <- gamma * (m + delta)
             (add delta first, then decay — eigenvalue gamma*(1+lam*r_mean))

        Timeout (D28): truncated trials carry no value information.
        V is not updated, but mood still receives the RPE from the truncated
        outcome (the agent did experience reward/no-reward).
        """
        x_new = x.copy()
        r = env_result["r"]
        choice = env_result["choice"]
        is_timeout = env_result.get(
            "is_timeout", np.zeros(x.shape[0]))

        alpha = theta[:, _I_ALPHA]
        tau_m = theta[:, _I_TAU_M]
        lam = theta[:, _I_LAM]
        gamma = np.exp(-1.0 / tau_m)

        m = x_new[:, 2]
        rho_eff = 1.0 + lam * m

        V0 = x_new[:, 0]
        V1 = x_new[:, 1]
        V_chosen = np.where(choice == 0, V0, V1)
        delta = rho_eff * r - V_chosen

        # Value update (only chosen arm, skip timeouts D28).
        update_v = mask & (is_timeout < 0.5)
        dV = alpha * delta
        mask0 = update_v & (choice == 0)
        mask1 = update_v & (choice == 1)
        x_new[mask0, 0] += dV[mask0]
        x_new[mask1, 1] += dV[mask1]

        # Mood update: m <- gamma * (m + delta).
        # Decay AFTER adding delta (D20: eigenvalue = gamma*(1 + lam*r_mean)).
        # Mood updates on ALL trials including timeouts.
        x_new[mask, 2] = (gamma * (m + delta))[mask]

        return x_new
