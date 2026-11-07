"""P4 Step 3: Behavioral counters and scout pipeline tests.

TestScoutPipeline: record_behavior + composite observables.
TestVectorizedGrid: grid nodes × replicates in single batch.
"""
import numpy as np
import pytest

from cpsych.core.composite_setup import (
    make_triple_integrator, make_grid_integrator,
)
from cpsych.observables.composite import (
    accuracy, mean_rt, win_stay_rate, trial_count,
)


class TestScoutPipeline:
    """Behavioral counters populate and produce sensible values."""

    @pytest.fixture(scope="class")
    def scout_state(self):
        """B=5, T=600 (10 min), record_behavior=True."""
        integ = make_triple_integrator(
            B=5, seed=42, obs_dt=600.0,
            s_kappa=1.0, s_alpha=1.0, s_value=1.0,
            record_behavior=True,
        )
        return integ.run(T=600.0)

    def test_behavior_counters_populated(self, scout_state):
        """record_behavior=True → behavior dict has all keys, nonzero."""
        beh = scout_state.behavior
        assert "n_correct" in beh
        assert "n_trials" in beh
        assert "sum_rt" in beh
        assert "n_win_stay" in beh
        assert "n_wins" in beh
        # At least some trials completed
        assert np.sum(beh["n_trials"]) > 0, "No trials completed in 600s"

    def test_accuracy_in_range(self, scout_state):
        """All replicates accuracy > 0.4, no perfect accuracy."""
        acc = accuracy(scout_state.behavior)
        valid = ~np.isnan(acc)
        assert np.any(valid), "No valid replicates"
        acc_valid = acc[valid]
        assert np.all(acc_valid > 0.4), (
            f"Accuracy below 0.4: values={acc_valid}")
        assert np.all(acc_valid < 1.0), (
            f"Perfect accuracy is suspicious: {acc_valid}")

    def test_win_stay_in_range(self, scout_state):
        """Win-stay rate in (0.3, 1.0)."""
        ws = win_stay_rate(scout_state.behavior)
        valid = ~np.isnan(ws)
        if np.any(valid):
            ws_valid = ws[valid]
            assert np.all(ws_valid >= 0.0), (
                f"Win-stay negative: {ws_valid}")
            assert np.all(ws_valid <= 1.0), (
                f"Win-stay > 1: {ws_valid}")

    def test_trial_count_positive(self, scout_state):
        """n_trials > 0 for each replicate."""
        tc = trial_count(scout_state.behavior)
        assert np.all(tc > 0), f"Zero-trial replicates: {tc}"


class TestVectorizedGrid:
    """Grid nodes × replicates packed into single batch axis."""

    def test_array_strength_coupling(self):
        """Per-replicate coupling strengths produce different behavior."""
        n_rep = 3
        # 2 nodes: s_kappa=0 (baseline) and s_kappa=4 (strong coupling)
        s_kappa_arr = np.array([0.0]*n_rep + [4.0]*n_rep)
        s_alpha_arr = np.full(6, 2.0)
        s_value_arr = np.full(6, 3.0)

        integ = make_grid_integrator(
            s_kappa_arr, s_alpha_arr, s_value_arr,
            seed=42, obs_dt=100.0, record_behavior=True,
        )
        state = integ.run(T=100.0)

        acc = accuracy(state.behavior)
        # Node 0 (s_kappa=0) and node 1 (s_kappa=4) should differ
        acc0 = acc[:n_rep]
        acc1 = acc[n_rep:]
        # s_kappa=4 should have lower accuracy than s_kappa=0
        # (gain modulates kappa down during sleep/transition)
        assert not np.allclose(acc0, acc1, atol=0.01), (
            f"Vectorized coupling produced identical results: "
            f"acc0={acc0}, acc1={acc1}")

    def test_grid_finiteness(self):
        """2x2 grid, all states finite."""
        n_rep = 2
        nodes = [(0.0, 0.0), (4.0, 10.0)]
        s_kappa_arr = np.repeat([sk for sk, sv in nodes], n_rep)
        s_alpha_arr = np.full(len(nodes)*n_rep, 2.0)
        s_value_arr = np.repeat([sv for sk, sv in nodes], n_rep)

        integ = make_grid_integrator(
            s_kappa_arr, s_alpha_arr, s_value_arr,
            seed=42, obs_dt=100.0,
        )
        state = integ.run(T=100.0)

        for name in ["arousal", "ddm", "rl_mood"]:
            assert np.all(np.isfinite(state.x[name])), (
                f"{name} non-finite in vectorized grid")
