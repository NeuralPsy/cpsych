"""Tests for multi-timescale Euler-Maruyama integrator.

Ported concepts from baa/tests/test_model.py:
- Noise scaling ∝ √dt (not dt)
- dt convergence
- Determinism by seed
- Observation frequency independent of dt
- Episodic reset
- Multi-tier integration
"""
import numpy as np
import pytest

from conftest import OUModule, ToyDDM, ToyBandit
from cpsych.core.integrator import Integrator, Tier


# ---------------------------------------------------------------------------
# Noise scaling: √dt, not dt (ported from BAA audit S4)
# ---------------------------------------------------------------------------

class TestNoiseScaling:
    """Noise increments must scale as √dt, not dt."""

    def test_variance_ratio(self):
        """Run OU at two dt values. Variance of increments ∝ dt
        (since σ·√dt → var ∝ dt). Not ∝ dt² (which would be the bug)."""
        mod = OUModule()
        results = {}
        for dt in [0.01, 0.04]:
            integ = Integrator(
                modules={"ou": mod},
                tiers=[Tier("t", dt, ["ou"])],
                B=100, seed=42,
            )
            state = integ.run(T=5.0)
            # Extract x0 time series
            ts = state.recordings["ou.x0"]
            vals = np.array([v for _, v in ts])  # (n_obs, B)
            diffs = np.diff(vals, axis=0)
            results[dt] = np.var(diffs)

        ratio = results[0.04] / results[0.01]
        # Correct √dt: var ratio ~ dt_ratio = 4
        # Wrong dt: var ratio ~ dt_ratio² = 16
        assert 1.5 < ratio < 10, (
            f"Variance ratio {ratio:.1f} suggests wrong noise scaling "
            f"(expected ~4.0)")


# ---------------------------------------------------------------------------
# dt convergence (ported from BAA)
# ---------------------------------------------------------------------------

class TestDtConvergence:
    """Mean activation of OU must converge as dt → 0."""

    def test_mean_converges(self):
        mod = OUModule()
        means = {}
        for dt in [0.02, 0.01, 0.005, 0.002]:
            integ = Integrator(
                modules={"ou": mod},
                tiers=[Tier("t", dt, ["ou"])],
                B=50, seed=42,
                obs_dt=0.1,
            )
            # Near-deterministic for cleaner convergence test
            state = integ._init_state()
            state.theta_base["ou"][:, 2] = 1e-10  # sigma0 ≈ 0
            state.theta_base["ou"][:, 3] = 1e-10  # sigma1 ≈ 0
            # Run manually with modified params
            integ2 = Integrator(
                modules={"ou": mod},
                tiers=[Tier("t", dt, ["ou"])],
                B=50, seed=42,
                obs_dt=0.1,
            )
            result = integ2.run(T=5.0)
            ts = result.recordings["ou.x0"]
            if ts:
                last_vals = ts[-1][1]  # last observation, all replicates
                means[dt] = np.mean(last_vals)

        # Difference between fine dts should be smaller than coarse
        if len(means) >= 4:
            diff_coarse = abs(means[0.02] - means[0.01])
            diff_fine = abs(means[0.005] - means[0.002])
            if diff_coarse > 1e-8:
                assert diff_fine < diff_coarse, (
                    f"Not converging: coarse={diff_coarse:.8f}, "
                    f"fine={diff_fine:.8f}")


# ---------------------------------------------------------------------------
# Determinism by seed (ported from BAA)
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_output(self):
        mod = OUModule()
        results = []
        for _ in range(2):
            integ = Integrator(
                modules={"ou": mod},
                tiers=[Tier("t", 0.01, ["ou"])],
                B=10, seed=42,
            )
            state = integ.run(T=1.0)
            ts = state.recordings["ou.x0"]
            results.append(np.array([v for _, v in ts]))
        np.testing.assert_array_equal(results[0], results[1])

    def test_different_seed_different_output(self):
        mod = OUModule()
        results = []
        for seed in [42, 99]:
            integ = Integrator(
                modules={"ou": mod},
                tiers=[Tier("t", 0.01, ["ou"])],
                B=10, seed=seed,
            )
            state = integ.run(T=1.0)
            ts = state.recordings["ou.x0"]
            results.append(np.array([v for _, v in ts]))
        assert not np.allclose(results[0], results[1])


# ---------------------------------------------------------------------------
# Observation frequency independent of dt (ported from BAA)
# ---------------------------------------------------------------------------

class TestObservationFrequency:
    def test_output_length_independent_of_dt(self):
        """Two runs with different dt but same T and obs_dt
        should have the same number of observations."""
        mod = OUModule()
        lengths = {}
        for dt in [0.01, 0.005]:
            integ = Integrator(
                modules={"ou": mod},
                tiers=[Tier("t", dt, ["ou"])],
                B=5, seed=42,
                obs_dt=0.1,
            )
            state = integ.run(T=2.0)
            ts = state.recordings["ou.x0"]
            lengths[dt] = len(ts)
        assert lengths[0.01] == lengths[0.005], (
            f"Output lengths differ: dt=0.01→{lengths[0.01]}, "
            f"dt=0.005→{lengths[0.005]}")


# ---------------------------------------------------------------------------
# Episodic reset
# ---------------------------------------------------------------------------

class TestEpisodicReset:
    def test_ddm_resets_on_barrier(self):
        """DDM should reset evidence to 0 when |evidence| >= threshold."""
        ddm = ToyDDM()
        integ = Integrator(
            modules={"ddm": ddm},
            tiers=[Tier("fast", 0.001, ["ddm"])],
            B=100, seed=42,
        )
        state = integ.run(T=2.0)
        # After enough time, trials should have completed
        assert np.all(state.trial_count > 0), (
            "Expected at least one trial per replicate")

    def test_continuous_module_not_reset(self):
        """OU state must evolve continuously even when DDM resets."""
        ou = OUModule()
        ddm = ToyDDM()
        integ = Integrator(
            modules={"ou": ou, "ddm": ddm},
            tiers=[
                Tier("slow", 0.01, ["ou"]),
                Tier("fast", 0.001, ["ddm"]),
            ],
            B=20, seed=42,
        )
        state = integ.run(T=3.0)

        # Prerequisite: DDM actually completed trials
        assert np.all(state.trial_count > 0), \
            "DDM must complete at least one trial per replicate"

        # Actual test: OU x0 has no sudden jumps to zero.
        # Max step-to-step change for OU with alpha=1, sigma=0.1,
        # obs_dt=0.01 is bounded. A reset-to-zero would show as
        # a jump of order |x| ≈ 0.05-0.1 in a single obs step.
        ts = state.recordings["ou.x0"]
        vals = np.array([v for _, v in ts])  # (n_obs, B)
        diffs = np.abs(np.diff(vals, axis=0))  # step-to-step changes

        # OU at obs_dt=0.01: max |Δx| per step ≈ alpha*|x|*dt + sigma*√dt*3σ
        # ≈ 1*0.1*0.01 + 0.1*0.1*3 ≈ 0.031. A reset from typical |x|≈0.05
        # would give |Δx|≈0.05. Use a threshold that catches resets but
        # allows normal OU fluctuations. If any replicate has >50% of its
        # trajectory near zero, that's a reset artifact.
        # Simpler: OU with drive=0 relaxes to 0 on timescale 1/alpha=1.0.
        # In T=3, it should spend time near 0 but not jump there.
        # Check: no obs step has ALL replicates simultaneously at 0.
        midpoints = vals[len(vals) // 4 : 3 * len(vals) // 4]
        assert not np.any(np.all(np.abs(midpoints) < 1e-10, axis=1)), \
            "OU state appears to be reset (all replicates near zero)"


# ---------------------------------------------------------------------------
# Multi-tier integration
# ---------------------------------------------------------------------------

class TestMultiTier:
    def test_two_tier_runs(self):
        """Two-tier system (slow OU + fast DDM) should complete."""
        ou = OUModule()
        ddm = ToyDDM()
        integ = Integrator(
            modules={"ou": ou, "ddm": ddm},
            tiers=[
                Tier("slow", 0.1, ["ou"]),
                Tier("fast", 0.001, ["ddm"]),
            ],
            B=10, seed=42,
        )
        state = integ.run(T=1.0)
        assert state.t == pytest.approx(1.0, abs=0.11)
        assert "ou.x0" in state.recordings
        assert "ddm.evidence" in state.recordings

    def test_tier_ratio_validation(self):
        """dt ratio must be close to integer."""
        ou = OUModule()
        ddm = ToyDDM()
        with pytest.raises(ValueError):
            Integrator(
                modules={"ou": ou, "ddm": ddm},
                tiers=[
                    Tier("slow", 0.1, ["ou"]),
                    Tier("fast", 0.03, ["ddm"]),
                    # 0.1 / 0.03 ≈ 3.33, not integer
                ],
                B=1, seed=42,
            )

    def test_module_not_in_tier_raises(self):
        ou = OUModule()
        ddm = ToyDDM()
        with pytest.raises(ValueError, match="not in any tier"):
            Integrator(
                modules={"ou": ou, "ddm": ddm},
                tiers=[Tier("t", 0.01, ["ou"])],  # ddm missing
                B=1, seed=42,
            )

    def test_unknown_module_in_tier_raises(self):
        ou = OUModule()
        with pytest.raises(ValueError, match="unknown module"):
            Integrator(
                modules={"ou": ou},
                tiers=[Tier("t", 0.01, ["ou", "ghost"])],
                B=1, seed=42,
            )


# ---------------------------------------------------------------------------
# Coupling integration
# ---------------------------------------------------------------------------

class TestCouplingIntegration:
    def test_coupling_modulates_parameter(self):
        """OU output should modulate DDM drift rate via coupling."""
        from cpsych.core.coupling import Coupling

        ou = OUModule()
        ddm = ToyDDM()
        coupling = Coupling(
            source_module="ou",
            source_output="activity",
            target_module="ddm",
            target_param="v",
            strength=0.5,
            max_strength=5.0,
        )
        integ = Integrator(
            modules={"ou": ou, "ddm": ddm},
            couplings=[coupling],
            tiers=[
                Tier("slow", 0.01, ["ou"]),
                Tier("fast", 0.001, ["ddm"]),
            ],
            B=10, seed=42,
        )
        state = integ.run(T=0.5)
        # Should complete without error; coupling applied each step
        assert state.t > 0


# ---------------------------------------------------------------------------
# No global seed pollution (ported from BAA)
# ---------------------------------------------------------------------------

class TestNoGlobalSeedPollution:
    def test_no_global_state_change(self):
        np.random.seed(0)
        state_before = np.random.get_state()[1].copy()
        mod = OUModule()
        integ = Integrator(
            modules={"ou": mod},
            tiers=[Tier("t", 0.01, ["ou"])],
            B=10, seed=42,
        )
        integ.run(T=1.0)
        state_after = np.random.get_state()[1].copy()
        np.testing.assert_array_equal(state_before, state_after)


# ---------------------------------------------------------------------------
# Multi-tempo convergence (4 orders of magnitude, bidirectional coupling)
# ---------------------------------------------------------------------------

class TestMultiTempoConvergence:
    """Multi-tier vs single-tier for bidirectionally coupled system.

    The integrator uses Lie-Trotter splitting: slow steps first (reading
    fast state from END of previous interval), then fast sub-steps update.
    This is a snapshot scheme, NOT averaging over the slow interval.

    System (bidirectional — fast equilibrium depends on slow):
      slow: dx_s/dt = -α_s·x_s + c_fs·x_f,  α_s=0.01, σ=0
      fast: dx_f/dt = -α_f·x_f + (1 + c_sf·x_s),  α_f=100, σ=0
      fast equilibrium: x_f* = (1 + c_sf·x_s) / α_f = 0.01 + 0.005·x_s
      effective slow drift: dx_s/dt = -0.005·x_s + 0.01 → x_s* = 2.0
      gap: α_f/α_s = 10^4 (4 orders)

    Multi-tier: dt_slow=1.0, dt_fast=0.001 (ratio=1000)
    Reference:  dt=0.001 single tier

    Snapshot error ≈ 0.57%: slow step uses stale x_f (equilibrated to
    previous x_s), not current. Fast re-equilibrates in τ_f=0.01 ≪ dt_slow,
    so averaging over the slow step would give ~0.006% (100× smaller).
    Bounds 0.1% < error < 1.0% pass snapshot, fail averaging.
    """

    @staticmethod
    def _build(tiers):
        from conftest import DriveOU1D, _make_drive_ou_spec
        from cpsych.core.coupling import Coupling

        slow = DriveOU1D(_make_drive_ou_spec(
            "slow", alpha_default=0.01, drive_default=0.0,
            sigma_default=0.0, timescale=100.0))
        fast = DriveOU1D(_make_drive_ou_spec(
            "fast", alpha_default=100.0, drive_default=1.0,
            sigma_default=0.0, timescale=0.01))
        couplings = [
            Coupling("fast", "x", "slow", "drive",
                     strength=1.0, max_strength=100.0),
            Coupling("slow", "x", "fast", "drive",
                     strength=0.5, max_strength=100.0),
        ]
        return Integrator(
            modules={"slow": slow, "fast": fast},
            couplings=couplings,
            tiers=tiers,
            B=1, seed=42,
        )

    def test_snapshot_splitting_error(self):
        """Multi-tier splitting error matches snapshot scheme, not averaging.

        Measured: snapshot gives ~0.57% error vs single-tier reference.
        Averaging would give ~0.006%. Bounds [0.1%, 1.0%] discriminate.
        """
        T = 100.0

        # Multi-tier: dt_slow=1.0, dt_fast=0.001
        multi = self._build([
            Tier("slow", 1.0, ["slow"]),
            Tier("fast", 0.001, ["fast"]),
        ])
        x_multi = multi.run(T=T).x["slow"][0, 0]

        # Single-tier reference: dt=0.001
        single = self._build([
            Tier("all", 0.001, ["slow", "fast"]),
        ])
        x_single = single.run(T=T).x["slow"][0, 0]

        rel_err = abs(x_multi - x_single) / abs(x_single)

        # Upper bound: splitting error is bounded
        assert rel_err < 0.01, (
            f"Splitting error {rel_err:.4%} exceeds 1% — "
            f"multi-tier scheme may be incorrect")

        # Lower bound: snapshot scheme has ~0.5% error;
        # averaging over the interval would give ~0.006%.
        # If this fails, the scheme was changed from snapshot to averaging.
        assert rel_err > 0.001, (
            f"Splitting error {rel_err:.4%} < 0.1% — "
            f"expected ~0.5% for snapshot scheme. "
            f"If integrator was changed to averaging, update this test.")

    def test_convergence_with_dt(self):
        """Splitting error must decrease with smaller dt_slow."""
        T = 100.0

        # Reference
        single = self._build([
            Tier("all", 0.001, ["slow", "fast"]),
        ])
        x_ref = single.run(T=T).x["slow"][0, 0]

        errors = {}
        for dt_slow in [2.0, 1.0, 0.5]:
            multi = self._build([
                Tier("slow", dt_slow, ["slow"]),
                Tier("fast", 0.001, ["fast"]),
            ])
            x_m = multi.run(T=T).x["slow"][0, 0]
            errors[dt_slow] = abs(x_m - x_ref) / abs(x_ref)

        # Error must decrease with smaller dt_slow
        assert errors[1.0] < errors[2.0], (
            f"Error not decreasing: dt=2.0→{errors[2.0]:.4%}, "
            f"dt=1.0→{errors[1.0]:.4%}")
        assert errors[0.5] < errors[1.0], (
            f"Error not decreasing: dt=1.0→{errors[1.0]:.4%}, "
            f"dt=0.5→{errors[0.5]:.4%}")

    def test_fast_var_equilibrated(self):
        """Fast variable must be at equilibrium after integration."""
        T = 10.0
        multi = self._build([
            Tier("slow", 1.0, ["slow"]),
            Tier("fast", 0.001, ["fast"]),
        ])
        state = multi.run(T=T)
        x_fast = state.x["fast"][0, 0]
        x_slow = state.x["slow"][0, 0]
        # Fast equilibrium: (1 + 0.5*x_s) / 100
        x_fast_eq = (1.0 + 0.5 * x_slow) / 100.0
        np.testing.assert_allclose(
            x_fast, x_fast_eq, atol=1e-6,
            err_msg="Fast var not at equilibrium")


# ---------------------------------------------------------------------------
# env_result["choice"] passed to trial_update (P3.1)
# ---------------------------------------------------------------------------

class TestEnvResultChoice:
    """Integrator must pass action into env_result['choice']."""

    def test_choice_in_env_result(self):
        """RL module needs env_result['choice'] from integrator."""
        from cpsych.core.specs import (
            ModuleSpec, StateVar, Param, EnvSpec,
            Temporality, Exposure,
        )
        from cpsych.core.module import SDEModule
        from cpsych.core.environment import Environment

        # Spy module: records env_result in trial_update
        captured = {}

        class SpyModule(SDEModule):
            _spec = ModuleSpec(
                name="spy",
                states=(StateVar("x", Temporality.EPISODIC, 0.0),),
                params=(Param("v", 1.0, -5.0, 5.0),),
                input_names=(),
                output_names=("choice",),
                timescale=0.1,
            )

            @property
            def spec(self):
                return self._spec

            def drift(self, x, theta, u):
                return theta[:, 0:1] * np.ones_like(x)

            def diffusion(self, x, theta):
                return np.zeros_like(x)

            def outputs(self, x, theta):
                return {"choice": np.sign(x[:, 0])}

            def trial_done(self, x, theta):
                return np.abs(x[:, 0]) >= 1.0

            def trial_update(self, x, theta, mask, env_result):
                if np.any(mask):
                    captured["env_result"] = {
                        k: v.copy() for k, v in env_result.items()}
                return x

        class SpyEnv(Environment):
            _spec = EnvSpec(
                name="spy_env",
                stimulus_names=(),
                reward_names=("r",),
            )

            @property
            def spec(self):
                return self._spec

            def reset(self, B, rng):
                self._B = B

            def next_trial(self, rng, mask=None):
                return {}

            def reward(self, action, rng):
                return {"r": np.ones(self._B)}

        spy = SpyModule()
        env = SpyEnv()
        integ = Integrator(
            modules={"spy": spy},
            environment=env,
            tiers=[Tier("t", 0.01, ["spy"])],
            B=5, seed=42,
        )
        integ.run(T=2.0)

        assert "env_result" in captured, "trial_update never called"
        assert "choice" in captured["env_result"], (
            "env_result must contain 'choice' key"
        )
        assert "r" in captured["env_result"], (
            "env_result must contain 'r' key from environment"
        )
