"""Coupling objects: output of one module modulates parameter of another.

Semantics:  θ_target[j] = θ_base[j] + clamp(strength, 0, max_strength) · h(source_output, θ_base[j])

h receives both source output and base parameter value, enabling multiplicative
couplings like h(gain, κ_base) = κ_base * (max(0, 1+s*(gain-1)) - 1).
Default h ignores base and returns source (additive).
Linear in strength (Jacobian-friendly).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _identity(x: np.ndarray, _base: np.ndarray) -> np.ndarray:
    return x


@dataclass
class Coupling:
    """A directional coupling: source output → target parameter modulation.

    Optional param_min/param_max clamp the effective parameter after
    modulation. Use param_min=0.0 for parameters that must stay
    non-negative (e.g., kappa, beta).
    """
    source_module: str
    source_output: str
    target_module: str
    target_param: str
    h: Callable[[np.ndarray, np.ndarray], np.ndarray] = field(
        default=_identity, repr=False)
    strength: float = 0.0
    max_strength: float = 1.0
    param_min: float | None = None
    param_max: float | None = None
    doc: str = ""

    def modulate(self, theta_target: np.ndarray, param_idx: int,
                 source_value: np.ndarray) -> np.ndarray:
        """Apply modulation. Returns copy with modified param column.

        Parameters
        ----------
        theta_target : (B, n_params)
        param_idx : column index of target parameter
        source_value : (B,) output from source module

        Notes
        -----
        strength may be scalar or array of shape (B,) for vectorized
        grid sweeps where different replicates use different coupling
        strengths. np.clip handles both cases.
        """
        theta_out = theta_target.copy()
        s = np.clip(self.strength, 0.0, self.max_strength)
        val = (
            theta_target[:, param_idx]
            + s * self.h(source_value, theta_target[:, param_idx])
        )
        if self.param_min is not None:
            val = np.maximum(val, self.param_min)
        if self.param_max is not None:
            val = np.minimum(val, self.param_max)
        theta_out[:, param_idx] = val
        return theta_out
