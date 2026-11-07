"""Abstract base class for SDE modules.

A module is a dynamical system  dx = drift(x, θ, u) dt + diffusion(x, θ) dW
with batch axis: all arrays have leading dimension B (number of replicates).

The module does NOT own state — state and parameters are held by the
integrator as arrays of shape (B, n_state) and (B, n_params). This makes
drift/diffusion pure functions (Rule 20).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .specs import ModuleSpec, Temporality


class SDEModule(ABC):
    """Base class for stochastic differential equation modules."""

    @property
    @abstractmethod
    def spec(self) -> ModuleSpec:
        ...

    @property
    def n_state(self) -> int:
        return len(self.spec.states)

    @property
    def n_params(self) -> int:
        return len(self.spec.params)

    @property
    def n_inputs(self) -> int:
        return len(self.spec.input_names)

    @property
    def n_outputs(self) -> int:
        return len(self.spec.output_names)

    def default_state(self, B: int) -> np.ndarray:
        """Initial state array, shape (B, n_state)."""
        x0 = np.array([s.initial for s in self.spec.states])
        return np.broadcast_to(x0, (B, self.n_state)).copy()

    def default_params(self, B: int) -> np.ndarray:
        """Default parameter array, shape (B, n_params)."""
        p0 = np.array([p.default for p in self.spec.params])
        return np.broadcast_to(p0, (B, self.n_params)).copy()

    def state_index(self, name: str) -> int:
        return self.spec.state_index(name)

    def param_index(self, name: str) -> int:
        return self.spec.param_index(name)

    @abstractmethod
    def drift(self, x: np.ndarray, theta: np.ndarray,
              u: np.ndarray) -> np.ndarray:
        """Deterministic part.  (B, n_state) everywhere."""
        ...

    @abstractmethod
    def diffusion(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Noise coefficient (diagonal).  (B, n_state) -> (B, n_state)."""
        ...

    @abstractmethod
    def outputs(self, x: np.ndarray,
                theta: np.ndarray) -> dict[str, np.ndarray]:
        """Named outputs from state.  {name: (B,)}."""
        ...

    def trial_done(self, x: np.ndarray,
                   theta: np.ndarray) -> np.ndarray:
        """Which replicates finished the current trial. Default: none.

        D14: theta passed so modules like DDM can check threshold.
        """
        return np.zeros(x.shape[0], dtype=bool)

    def trial_update(self, x: np.ndarray, theta: np.ndarray,
                     mask: np.ndarray,
                     env_result: dict[str, np.ndarray]) -> np.ndarray:
        """Discrete update at trial boundary (e.g., RL). Default: no-op."""
        return x

    def project_state(self, x: np.ndarray,
                      theta: np.ndarray) -> np.ndarray:
        """Project state back into feasible set after EM step.

        Default: identity (no projection). Override for reflecting
        boundaries (e.g., non-negative activations, bounded energy).
        """
        return x

    def reset_state(self, x: np.ndarray, theta: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
        """Reset episodic variables where mask is True.

        D14: theta passed so modules can use parameter-dependent reset
        (e.g., starting point z that depends on threshold a).
        """
        x_new = x.copy()
        for i, sv in enumerate(self.spec.states):
            if sv.temporality == Temporality.EPISODIC:
                x_new[mask, i] = sv.initial
        return x_new
