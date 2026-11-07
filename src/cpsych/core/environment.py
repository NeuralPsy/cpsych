"""Abstract base class for task environments.

Not an SDE module: no drift/diffusion. Trial-based interface:
next_trial() generates stimuli, reward() processes actions.
All methods batch-compatible: B replicates run independently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .specs import EnvSpec


class Environment(ABC):
    """Base class for task environments (stimulus generators)."""

    @property
    @abstractmethod
    def spec(self) -> EnvSpec:
        ...

    @abstractmethod
    def next_trial(self, rng: np.random.Generator,
                   mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Generate stimulus for B replicates. {name: (B,)}.

        Parameters
        ----------
        mask : (B,) bool or None
            If provided, only update internal state (counters, correct_arm)
            for replicates where mask is True.  Stimulus arrays are still
            returned for all B (caller selects via np.where).
            None means all replicates (default, backward-compatible).
        """
        ...

    @abstractmethod
    def reward(self, action: np.ndarray,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Process action (B,) and return reward. {name: (B,)}."""
        ...

    @abstractmethod
    def reset(self, B: int, rng: np.random.Generator) -> None:
        """Initialize internal state for B replicates."""
        ...
