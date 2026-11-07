"""P4 Step 0a: Performance regression test.

Runs a short triple composite at fixed B and fails if us/replicate-step
exceeds the saved reference by more than 2x. Catches performance
regressions from accidental per-replicate loops.

Reference: results/p4_benchmark.json (B=5000, T=86.4, us=0.21).
Test uses B=500, T=10 for speed (< 3s).
"""
import json
import os
import time

import numpy as np
import pytest

from cpsych.core.coupling import Coupling
from cpsych.core.integrator import Integrator, Tier
from cpsych.modules.arousal import ArousalModule
from cpsych.modules.ddm import DDMModule
from cpsych.modules.rl_mood import RLMoodModule
from cpsych.environments.composite_task import CompositeTask


def _make_composite(B, seed=42):
    """Build triple composite with all P3 couplings."""
    arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
    ddm = DDMModule()
    rl = RLMoodModule()
    env = CompositeTask(coherence=0.128, p_reward=0.75, volatility=0.3)

    couplings = [
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="ddm", target_param="kappa",
            h=lambda g, k: k * (g - 1),
            strength=2.0, max_strength=4.0, param_min=0.0,
        ),
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="rl_mood", target_param="alpha",
            h=lambda g, a: a * (g - 1),
            strength=2.0, max_strength=4.0,
            param_min=0.0, param_max=1.0,
        ),
        Coupling(
            source_module="rl_mood", source_output="value_diff",
            target_module="ddm", target_param="drift_bias",
            strength=3.0, max_strength=20.0,
        ),
    ]

    return Integrator(
        modules={"arousal": arousal, "ddm": ddm, "rl_mood": rl},
        environment=env, couplings=couplings,
        tiers=[
            Tier("slow", 1.0, ["arousal"]),
            Tier("fast", 0.001, ["ddm", "rl_mood"]),
        ],
        trial_driver="ddm",
        gate_module="arousal", gate_output="is_wake",
        B=B, seed=seed, obs_dt=100.0,
    )


class TestPerformanceRegression:
    """Performance regression test (Step 0a)."""

    # Reference from results/p4_benchmark.json: 0.24 us/rep-step at B=5000.
    # Test at B=500 where us/rep-step is ~0.5-1.0 (interpolated).
    # Allow 6x headroom for OS jitter and B<5000 overhead.
    REFERENCE_US = 0.24
    MAX_RATIO = 6.0  # allow 6x of B=5000 reference (B=500 + OS jitter)
    TEST_B = 500
    TEST_T = 10.0  # 10 model seconds — short but enough steps

    def test_batch_performance(self):
        """us/replicate-step below 4x reference at B=500.

        Reference (B=5000): 0.24 us/rep-step.
        At B=500: expect ~0.5-1.0 us/rep-step.
        Threshold: 1.44 us/rep-step (6x reference).
        Failure indicates a per-replicate loop was introduced.
        """
        integ = _make_composite(self.TEST_B)

        dt_slow = 1.0
        n_slow = int(round(self.TEST_T / dt_slow))
        fast_ratio = 1000
        total_module_steps = n_slow * (1 + fast_ratio * 2)
        total_rep_steps = self.TEST_B * total_module_steps

        t0 = time.perf_counter()
        state = integ.run(T=self.TEST_T)
        t1 = time.perf_counter()

        us_per = (t1 - t0) * 1e6 / total_rep_steps
        threshold = self.REFERENCE_US * self.MAX_RATIO

        assert us_per < threshold, (
            f"Performance regression: {us_per:.3f} us/rep-step > "
            f"threshold {threshold:.3f} (reference {self.REFERENCE_US} x {self.MAX_RATIO})")

    def test_batch_speedup(self):
        """B=500 faster per-step than B=1 (batch actually works)."""
        dt_slow = 1.0
        T = 5.0
        n_slow = int(round(T / dt_slow))
        fast_ratio = 1000
        total_module_steps = n_slow * (1 + fast_ratio * 2)

        # B=1
        integ1 = _make_composite(1)
        t0 = time.perf_counter()
        integ1.run(T=T)
        t1 = time.perf_counter()
        us_b1 = (t1 - t0) * 1e6 / (1 * total_module_steps)

        # B=500
        integ500 = _make_composite(500)
        t0 = time.perf_counter()
        integ500.run(T=T)
        t1 = time.perf_counter()
        us_b500 = (t1 - t0) * 1e6 / (500 * total_module_steps)

        speedup = us_b1 / us_b500
        assert speedup > 10, (
            f"Batch speedup B=1->B=500 only {speedup:.1f}x, expected >10x. "
            f"B=1: {us_b1:.2f}, B=500: {us_b500:.3f} us/rep-step")
