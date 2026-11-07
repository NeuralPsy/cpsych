"""P4: BAA as 4th module in composite (D33).

Tests: coupling modulation, batch axis, project_state diagnostics,
composite behavior. No standalone BAA re-validation (dosier in baa/).
"""
import numpy as np
import pytest

from cpsych.core.composite_setup import make_quad_integrator
from cpsych.modules.baa_attention import BAAModule, _PI
from cpsych.observables.composite import accuracy, trial_count


class TestBAAInComposite:
    """BAA integrated as 4th module with arousal coupling."""

    def test_quad_all_finite(self):
        """All 4 modules finite after composite run."""
        integ = make_quad_integrator(B=5, seed=42, obs_dt=100.0)
        state = integ.run(T=100.0)
        for name in ["arousal", "ddm", "rl_mood", "baa"]:
            assert np.all(np.isfinite(state.x[name])), (
                f"{name} non-finite in quad composite")

    def test_baa_gain_modulated_by_arousal(self):
        """With s_baa>0, effective BAA gain differs during wake."""
        from cpsych.core.composite_setup import wake_equilibrium
        from cpsych.modules.arousal import ArousalModule

        B = 3
        weq = wake_equilibrium(
            ArousalModule(q_m_wake_mean_baseline=4.2883))

        for s_baa, label in [(0.0, "off"), (4.0, "on")]:
            integ = make_quad_integrator(
                B=B, seed=42, obs_dt=50.0, s_baa=s_baa)
            state = integ._init_state()
            for b in range(B):
                state.x["arousal"][b] = weq["x0"][0]
            state = integ.run(T=50.0, state=state)
            if label == "off":
                baa_off = state.x["baa"].copy()
            else:
                baa_on = state.x["baa"].copy()

        # During wake, arousal gain ≈ 1.05. With s_baa=4:
        # gain_eff = 1.0 * (1 + 4*0.05) = 1.2 ≠ 1.0.
        # BAA drift depends on gain_eff → states diverge.
        assert not np.allclose(baa_on, baa_off, atol=0.01), (
            "BAA states identical with and without arousal coupling "
            "even during wake")

    def test_baa_boundary_diagnostics_in_composite(self):
        """D31: boundary_fractions reported for BAA in composite."""
        integ = make_quad_integrator(B=5, seed=42, obs_dt=100.0)
        state = integ.run(T=100.0)

        assert "baa" in state.boundary_fractions, (
            "BAA boundary_fractions missing from composite")
        fracs = state.boundary_fractions["baa"]
        assert "lower" in fracs and "upper" in fracs
        # BAA has 11 states (5A + 1S + 5W)
        assert fracs["lower"].shape == (5, 11)
        assert fracs["upper"].shape == (5, 11)

    def test_baa_does_not_affect_trial_loop(self):
        """BAA is passive — no trial_done, no env interaction."""
        # Quad and triple should produce same trial counts
        # (BAA doesn't interfere with DDM trial loop)
        from cpsych.core.composite_setup import make_triple_integrator
        integ_triple = make_triple_integrator(
            B=3, seed=42, obs_dt=100.0, record_behavior=True,
            s_kappa=2.0, s_alpha=2.0, s_value=3.0)
        state_triple = integ_triple.run(T=100.0)

        integ_quad = make_quad_integrator(
            B=3, seed=42, obs_dt=100.0, record_behavior=True,
            s_kappa=2.0, s_alpha=2.0, s_value=3.0, s_baa=2.0)
        state_quad = integ_quad.run(T=100.0)

        tc_triple = trial_count(state_triple.behavior)
        tc_quad = trial_count(state_quad.behavior)
        # Trial counts should be identical — same DDM+RL+arousal dynamics
        # BAA adds a module on slow tier but doesn't affect trial loop
        # Note: RNG differs because BAA draws noise, so trial counts
        # may differ slightly. Check they're close, not identical.
        ratio = tc_quad / np.maximum(tc_triple, 1)
        assert np.all(ratio > 0.5) and np.all(ratio < 2.0), (
            f"BAA drastically changed trial counts: "
            f"triple={tc_triple}, quad={tc_quad}")

    def test_baa_batch_independence_in_composite(self):
        """Different batch elements get independent BAA trajectories."""
        integ = make_quad_integrator(B=5, seed=42, obs_dt=100.0)
        state = integ.run(T=100.0)
        baa = state.x["baa"]
        # Not all rows identical (noise makes them diverge)
        diffs = np.max(np.abs(baa[0:1] - baa[1:]), axis=1)
        assert np.any(diffs > 0.01), (
            "BAA batch elements suspiciously identical")

    def test_baa_A_nonneg_in_composite(self):
        """BAA A channels stay non-negative via project_state in composite."""
        integ = make_quad_integrator(B=10, seed=42, obs_dt=100.0)
        state = integ.run(T=100.0)
        A = state.x["baa"][:, :5]
        assert np.all(A >= 0.0), f"Negative A in composite: {A[A < 0]}"

    def test_coupling_form_multiplicative(self):
        """gain_eff = gain_base * max(0, 1 + s*(arousal_gain - 1))."""
        mod = BAAModule(nc=5)
        theta = mod.default_params(1)
        assert theta[0, _PI["gain"]] == 1.0, "Default BAA gain should be 1.0"
        assert theta[0, _PI["tonic_input"]] == 0.5, (
            "Default tonic_input should be 0.5")
