"""Composite task: perceptual-guided bandit.

DDM decides direction based on coherence; RL learns from reward.
ReversalBandit determines correct_arm. Coherence from PerceptualTask.

Mapping: DDM choice=+1 → arm 1, DDM choice=-1 → arm 0.
Direction = +1 if correct_arm==1, -1 if correct_arm==0.

Spec: specs/p3_couplings.md, pair 3.
Decision: D4 (composition in P3), D23 (no mood→DDM).
"""
from __future__ import annotations

import numpy as np

from ..core.environment import Environment
from ..core.specs import EnvSpec


_SPEC = EnvSpec(
    name="CompositeTask",
    stimulus_names=("coherence", "direction", "trial_index"),
    reward_names=("r", "choice"),
    doc="Perceptual-guided bandit: DDM decides, RL learns.",
)


class CompositeTask(Environment):
    """Perceptual-guided bandit combining coherence and reversal.

    Parameters
    ----------
    coherence : float
        Fixed coherence for P3. In P4, could be varied per trial.
    p_reward : float
        P(reward | correct arm). Range [0.5, 1.0].
    volatility : float
        Reversal frequency (0 = no reversals).
    """

    def __init__(self, coherence: float = 0.128,
                 p_reward: float = 0.75,
                 volatility: float = 0.0) -> None:
        if not 0.0 <= coherence <= 1.0:
            raise ValueError(f"coherence must be in [0, 1], got {coherence}")
        if not 0.5 <= p_reward <= 1.0:
            raise ValueError(f"p_reward must be in [0.5, 1], got {p_reward}")
        self.coherence = coherence
        self.p_reward = p_reward
        self.volatility = volatility
        self._B: int = 0
        self._correct_arm: np.ndarray = np.array([])
        self._trials_in_block: np.ndarray = np.array([])
        self._trial_counter: np.ndarray = np.array([])

    @property
    def spec(self) -> EnvSpec:
        return _SPEC

    def reset(self, B: int, rng: np.random.Generator) -> None:
        self._B = B
        self._correct_arm = rng.integers(0, 2, size=B).astype(float)
        self._trials_in_block = np.zeros(B, dtype=int)
        self._trial_counter = np.zeros(B, dtype=int)

    def next_trial(self, rng: np.random.Generator,
                   mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        B = self._B
        if mask is None:
            mask = np.ones(B, dtype=bool)

        self._trials_in_block += mask.astype(int)
        self._trial_counter += mask.astype(int)

        # Reversals only for replicates that completed a trial
        if self.volatility > 0:
            p_rev = self.volatility / 20.0
            reverse = mask & (rng.random(B) < p_rev)
            self._correct_arm = np.where(
                reverse, 1.0 - self._correct_arm, self._correct_arm)
            self._trials_in_block = np.where(reverse, 0, self._trials_in_block)

        # Direction: correct_arm → DDM direction
        # correct_arm=1 → direction=+1, correct_arm=0 → direction=-1
        direction = 2.0 * self._correct_arm - 1.0
        coherence = np.full(B, self.coherence)

        return {
            "coherence": coherence,
            "direction": direction,
            "trial_index": self._trial_counter.astype(float),
        }

    def reward(self, action: np.ndarray,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Process DDM action (±1) and return reward.

        Mapping: DDM choice=+1 → arm 1, DDM choice=-1 → arm 0.
        RL needs arm index (0/1), so we pass it as 'choice' in result.
        """
        # Map DDM choice to arm index
        arm = np.where(action >= 0, 1.0, 0.0)

        # Reward from bandit logic
        chose_correct = (arm == self._correct_arm)
        p = np.where(chose_correct, self.p_reward, 1.0 - self.p_reward)
        r = (rng.random(self._B) < p).astype(float)

        return {"r": r, "choice": arm,
                "chose_correct": chose_correct.astype(float)}
