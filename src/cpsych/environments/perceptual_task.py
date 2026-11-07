"""Perceptual decision task: random dot motion with graded coherence.

Protocol: Roitman & Shadlen (2002), J Neurosci 22(21):9475-9489.
DOI: 10.1523/JNEUROSCI.22-21-09475.2002

Six coherence levels matching the original experiment, enabling
per-coherence quantile comparison with raw trial-level data.
"""
from __future__ import annotations

import numpy as np

from ..core.environment import Environment
from ..core.specs import EnvSpec

COHERENCE_LEVELS = np.array([0.0, 0.032, 0.064, 0.128, 0.256, 0.512])

_SPEC = EnvSpec(
    name="PerceptualTask",
    stimulus_names=("coherence", "direction"),
    reward_names=("correct", "r"),
    doc="Random dot motion with 6 coherence levels (Roitman & Shadlen 2002).",
)


class PerceptualTask(Environment):
    """Perceptual decision task with graded coherence.

    next_trial: samples coherence uniformly from 6 levels, direction from {-1, +1}.
    reward: deterministic — correct iff action == direction.
    """

    def __init__(self) -> None:
        self._current_direction: np.ndarray = np.array([])
        self._B: int = 0

    @property
    def spec(self) -> EnvSpec:
        return _SPEC

    def reset(self, B: int, rng: np.random.Generator) -> None:
        self._B = B
        self._current_direction = np.zeros(B)

    def next_trial(self, rng: np.random.Generator,
                   mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        B = self._B
        idx = rng.integers(0, len(COHERENCE_LEVELS), size=B)
        coherence = COHERENCE_LEVELS[idx]
        direction = rng.choice(np.array([-1, 1]), size=B)
        if mask is not None:
            self._current_direction = np.where(
                mask, direction, self._current_direction)
        else:
            self._current_direction = direction
        return {"coherence": coherence, "direction": direction}

    def reward(self, action: np.ndarray,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
        correct = (action == self._current_direction).astype(float)
        return {"correct": correct, "r": correct}
