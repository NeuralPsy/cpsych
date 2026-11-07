"""Two-armed reversal bandit with two reversal modes.

Mode 'geom' (default): Behrens et al. (2007) structure.
  Stochastic reversals via Geom(volatility/20). D9 deviations.
  DOI: 10.1038/nn1954

Mode 'criterion': Eckstein et al. (2022) structure.
  Reversal after collecting criterion_range[0]..criterion_range[1]
  rewards (uniform random per block). Reversal only after rewarded trial.
  p_incorrect separate from p_reward.
  DOI: 10.1016/j.dcn.2022.101106
"""
from __future__ import annotations

import numpy as np

from ..core.environment import Environment
from ..core.specs import EnvSpec

_SPEC = EnvSpec(
    name="ReversalBandit",
    stimulus_names=("trial_index",),
    reward_names=("r",),
    doc="Two-armed bandit with reversals (geom or criterion mode).",
)


class ReversalBandit(Environment):
    """Two-armed reversal bandit.

    Parameters
    ----------
    p_reward : float
        P(reward | correct arm). Range [0.5, 1.0]. Default 0.75.
    p_incorrect : float or None
        P(reward | incorrect arm). If None, uses 1-p_reward (geom default).
        Set to 0.0 for Eckstein protocol. Range [0.0, 0.5].
    volatility : float
        Geom mode only. Controls reversal frequency.
        Formula: mean_block = 20 / volatility. Range [0.0, 1.0].
    reversal_mode : str
        'geom' (default): stochastic Geom(volatility/20) reversals.
        'criterion': reversal after collecting N rewards (Eckstein).
    criterion_range : tuple[int, int]
        Criterion mode only. Uniform[lo, hi] rewards to collect before
        reversal becomes possible. Default (7, 15) per Eckstein.
    """

    def __init__(self, p_reward: float = 0.75,
                 p_incorrect: float | None = None,
                 volatility: float = 0.0,
                 reversal_mode: str = "geom",
                 criterion_range: tuple[int, int] = (7, 15),
                 ) -> None:
        if not 0.5 <= p_reward <= 1.0:
            raise ValueError(f"p_reward must be in [0.5, 1.0], got {p_reward}")
        if reversal_mode not in ("geom", "criterion"):
            raise ValueError(
                f"reversal_mode must be 'geom' or 'criterion', "
                f"got '{reversal_mode}'")
        if reversal_mode == "geom" and not 0.0 <= volatility <= 1.0:
            raise ValueError(
                f"volatility must be in [0.0, 1.0], got {volatility}")

        self.p_reward = p_reward
        self.p_incorrect = p_incorrect if p_incorrect is not None else 1.0 - p_reward
        self.volatility = volatility
        self.reversal_mode = reversal_mode
        self.criterion_range = criterion_range

        self._B: int = 0
        self._correct_arm: np.ndarray = np.array([])
        self._trials_in_block: np.ndarray = np.array([])
        self._trial_counter: np.ndarray = np.array([])
        self._rewards_in_block: np.ndarray = np.array([])
        self._criterion: np.ndarray = np.array([])
        self._last_rewarded: np.ndarray = np.array([])

    @property
    def spec(self) -> EnvSpec:
        return _SPEC

    def reset(self, B: int, rng: np.random.Generator) -> None:
        self._B = B
        self._correct_arm = rng.integers(0, 2, size=B).astype(float)
        self._trials_in_block = np.zeros(B, dtype=int)
        self._trial_counter = np.zeros(B, dtype=int)
        self._rewards_in_block = np.zeros(B, dtype=int)
        self._last_rewarded = np.zeros(B, dtype=bool)
        if self.reversal_mode == "criterion":
            lo, hi = self.criterion_range
            self._criterion = rng.integers(lo, hi + 1, size=B)

    def next_trial(self, rng: np.random.Generator,
                   mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
        B = self._B
        if mask is None:
            mask = np.ones(B, dtype=bool)

        self._trials_in_block += mask.astype(int)
        self._trial_counter += mask.astype(int)

        if self.reversal_mode == "geom":
            if self.volatility > 0:
                p_rev = self.volatility / 20.0
                reverse = mask & (rng.random(B) < p_rev)
                self._correct_arm = np.where(
                    reverse, 1.0 - self._correct_arm, self._correct_arm)
                self._trials_in_block = np.where(
                    reverse, 0, self._trials_in_block)

        elif self.reversal_mode == "criterion":
            # Reverse when: criterion met AND last trial was rewarded
            met = self._rewards_in_block >= self._criterion
            reverse = mask & met & self._last_rewarded
            self._correct_arm = np.where(
                reverse, 1.0 - self._correct_arm, self._correct_arm)
            # Reset block counters and draw new criterion
            if np.any(reverse):
                lo, hi = self.criterion_range
                new_crit = rng.integers(lo, hi + 1, size=B)
                self._criterion = np.where(reverse, new_crit, self._criterion)
                self._rewards_in_block = np.where(
                    reverse, 0, self._rewards_in_block)
                self._trials_in_block = np.where(
                    reverse, 0, self._trials_in_block)

        return {"trial_index": self._trial_counter.astype(float)}

    def reward(self, action: np.ndarray,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
        chose_correct = (action == self._correct_arm)
        p = np.where(chose_correct, self.p_reward, self.p_incorrect)
        r = (rng.random(self._B) < p).astype(float)
        # Track rewards for criterion mode
        self._rewards_in_block += r.astype(int)
        self._last_rewarded = r > 0.5
        return {"r": r}
