"""Drift-Diffusion Model for perceptual decisions.

Canonical DDM (Ratcliff 1978) without between-trial variability.
  dx = kappa * coherence * direction * dt + dW   (sigma=1 fixed)
  trial ends when |evidence| >= a/2

References:
  Ratcliff (1978) Psychological Review 85(2):59-108.
  Ratcliff & McKoon (2008) Neural Computation 20(4):873-922.
  Bogacz et al. (2006) Psychological Review 113(4):700-765.

Data: Roitman & Shadlen (2002) J Neurosci 22(21):9475-9489.
Spec: specs/ddm.md v1.2.
"""
from __future__ import annotations

import numpy as np

from ..core.module import SDEModule
from ..core.specs import (
    ModuleSpec, StateVar, Param,
    Temporality, Exposure,
)

# Calibrated from monkey N data (Roitman & Shadlen 2002).
# Procedure: closed-form (specs/ddm.md v1.2).
# results/ddm_calibration.json has full details.
_KAPPA_DEFAULT = 14.7242
_A_DEFAULT = 1.4916
_TER_DEFAULT = 0.318

_DDM_SPEC = ModuleSpec(
    name="ddm",
    states=(
        StateVar("evidence", Temporality.EPISODIC, 0.0,
                 doc="Accumulated evidence"),
        StateVar("trial_clock", Temporality.EPISODIC, 0.0,
                 doc="Time since trial onset (drift=1, diffusion=0)"),
    ),
    params=(
        Param("kappa", _KAPPA_DEFAULT, 5.0, 30.0,
              exposure=Exposure.INTERNAL,
              doc="Drift sensitivity: v = kappa * coherence * direction"),
        Param("a", _A_DEFAULT, 0.5, 3.0,
              exposure=Exposure.INTERNAL,
              doc="Boundary separation: trial ends at |evidence| >= a/2"),
        Param("Ter", _TER_DEFAULT, 0.1, 0.5,
              exposure=Exposure.INTERNAL,
              doc="Non-decision time: RT = decision_time + Ter"),
        Param("drift_bias", 0.0, -50.0, 50.0,
              exposure=Exposure.INTERNAL,
              doc="Value-based drift component. Modulated by RL.value_diff coupling."),
    ),
    input_names=("coherence", "direction"),
    output_names=("evidence", "choice", "rt", "is_timeout"),
    timescale=0.5,
    doc="Drift-diffusion model (Ratcliff 1978). sigma=1 fixed.",
)

# Computational safeguard — not a behavioral deadline.
# Set to 5*(a_max/2)^2 where a_max is the upper bound of param range.
# At a_max=5.0: T_MAX = 5*(2.5)^2 = 31.25 MTU.
# P(DT > T_MAX | coh=0, a=5.0) < 0.001 (random walk exit time tail).
T_MAX_TRIAL = 5.0 * (_DDM_SPEC.params[1].high / 2) ** 2


class DDMModule(SDEModule):
    """Drift-Diffusion Model for two-choice perceptual decisions.

    States: evidence (EPISODIC), trial_clock (EPISODIC).
    Params: kappa (sensitivity), a (boundary), Ter (non-decision time),
            drift_bias (value-based component, D27).
    Fixed: sigma=1 (identifiability), z=0 (no bias).
    Inputs: coherence, direction from PerceptualTask (D15).
    """

    @property
    def spec(self) -> ModuleSpec:
        return _DDM_SPEC

    def drift(self, x: np.ndarray, theta: np.ndarray,
              u: np.ndarray) -> np.ndarray:
        kappa = theta[:, 0:1]
        coherence = u[:, 0:1]
        direction = u[:, 1:2]
        drift_bias = theta[:, 3:4]
        v = kappa * coherence * direction + drift_bias
        return np.hstack([v, np.ones_like(v)])

    def diffusion(self, x: np.ndarray,
                  theta: np.ndarray) -> np.ndarray:
        B = x.shape[0]
        return np.hstack([np.ones((B, 1)), np.zeros((B, 1))])

    def outputs(self, x: np.ndarray,
                theta: np.ndarray) -> dict[str, np.ndarray]:
        evidence = x[:, 0]
        a = theta[:, 1]
        half_a = a / 2
        choice = np.where(
            evidence >= half_a, 1.0,
            np.where(evidence <= -half_a, -1.0, 0.0))
        trial_clock = x[:, 1]
        Ter = theta[:, 2]
        rt = trial_clock + Ter
        is_timeout = (trial_clock >= T_MAX_TRIAL).astype(float)
        return {
            "evidence": evidence, "choice": choice,
            "rt": rt, "is_timeout": is_timeout,
        }

    def trial_done(self, x: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
        a = theta[:, 1]
        half_a = a / 2
        evidence_done = np.abs(x[:, 0]) >= half_a
        timeout = x[:, 1] >= T_MAX_TRIAL
        return evidence_done | timeout

    def reset_state(self, x: np.ndarray, theta: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
        x_new = x.copy()
        x_new[mask, 0] = 0.0  # evidence -> 0 (z=0, no bias)
        x_new[mask, 1] = 0.0  # trial_clock -> 0
        return x_new
