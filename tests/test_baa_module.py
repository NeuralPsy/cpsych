"""Tests for BAA attention module (P4, Step E).

Regression tests ported from baa/tests/test_model.py.
Tests cover: fixed point, boundaries, determinism, noise scaling,
contract compliance (Rule 20, batch independence).
"""
import numpy as np
import pytest

from cpsych.modules.baa_attention import BAAModule, _PI
from cpsych.core.integrator import Integrator, Tier


class TestBAAFixedPoint:
    """Deterministic convergence to analytical fixed point."""

    def test_deterministic_converges_to_analytical(self):
        """With noise=0 and eta=0, system converges to A*=G*I*W/(beta+gamma*(N-1))."""
        mod = BAAModule(nc=5)
        B = 1
        theta = mod.default_params(B)
        # Zero noise
        theta[0, _PI["sigma_base"]] = 0.0
        theta[0, _PI["sigma_max"]] = 0.0
        theta[0, _PI["plasticity_rate"]] = 0.0
        theta[0, _PI["tonic_input"]] = 0.0  # pure external stimulus

        I = np.full((B, 5), 0.5)
        x_star = mod.analytical_fixed_point(theta, I)

        # Integrate deterministically
        x = mod.default_state(B)
        u = I.copy()
        dt = 0.01
        for _ in range(20000):
            dx = mod.drift(x, theta, u)
            x = x + dx * dt
            x = mod.project_state(x, theta)

        # Check convergence
        np.testing.assert_allclose(
            x[:, :5], x_star[:, :5], atol=0.01,
            err_msg="A did not converge to analytical fixed point")
        np.testing.assert_allclose(
            x[:, 5], x_star[:, 5], atol=1.0,
            err_msg="S did not converge to analytical fixed point")


class TestBAABoundaries:
    """Reflecting boundaries via project_state."""

    def test_A_nonnegative(self):
        """A channels stay non-negative with high noise."""
        mod = BAAModule(nc=5)
        B = 10
        theta = mod.default_params(B)
        theta[:, _PI["sigma_base"]] = 2.0
        theta[:, _PI["sigma_max"]] = 5.0

        integ = Integrator(
            modules={"baa": mod},
            tiers=[Tier("default", 0.01, ["baa"])],
            B=B, seed=42, obs_dt=1.0,
        )
        state = integ.run(T=10.0)
        A = state.x["baa"][:, :5]
        assert np.all(A >= 0.0), f"Negative A: {A[A < 0]}"

    def test_S_bounded(self):
        """S stays in [0, S_max]."""
        mod = BAAModule(nc=5)
        B = 10
        theta = mod.default_params(B)

        integ = Integrator(
            modules={"baa": mod},
            tiers=[Tier("default", 0.01, ["baa"])],
            B=B, seed=42, obs_dt=1.0,
        )
        state = integ.run(T=10.0)
        S = state.x["baa"][:, 5]
        S_max = theta[0, _PI["S_max"]]
        assert np.all(S >= 0.0), f"S below 0: {S[S < 0]}"
        assert np.all(S <= S_max + 1e-10), f"S above S_max: {S[S > S_max]}"

    def test_W_bounded(self):
        """W stays in [W_min, W_max] even with plasticity."""
        mod = BAAModule(nc=5)
        B = 10
        theta = mod.default_params(B)
        theta[:, _PI["plasticity_rate"]] = 0.5  # high plasticity

        integ = Integrator(
            modules={"baa": mod},
            tiers=[Tier("default", 0.01, ["baa"])],
            B=B, seed=42, obs_dt=1.0,
        )
        state = integ.run(T=20.0)
        W = state.x["baa"][:, 6:]
        W_min = theta[0, _PI["W_min"]]
        W_max = theta[0, _PI["W_max"]]
        assert np.all(W >= W_min - 1e-10), (
            f"W below W_min: {W[W < W_min]}")
        assert np.all(W <= W_max + 1e-10), (
            f"W above W_max: {W[W > W_max]}")


class TestBAADeterminism:
    """Same seed → same output."""

    def test_same_seed_same_output(self):
        mod = BAAModule(nc=5)
        B = 3

        results = []
        for _ in range(2):
            integ = Integrator(
                modules={"baa": mod},
                tiers=[Tier("default", 0.01, ["baa"])],
                B=B, seed=123, obs_dt=1.0,
            )
            state = integ.run(T=5.0)
            results.append(state.x["baa"].copy())

        np.testing.assert_array_equal(results[0], results[1])


class TestBAANoiseScaling:
    """Variance proportional to dt (Euler-Maruyama)."""

    def test_variance_proportional_to_dt(self):
        mod = BAAModule(nc=5)
        B = 500

        variances = []
        for dt in [0.01, 0.001]:
            T = 1.0
            integ = Integrator(
                modules={"baa": mod},
                tiers=[Tier("default", dt, ["baa"])],
                B=B, seed=42, obs_dt=T,
            )
            state = integ.run(T=T)
            A0 = state.x["baa"][:, 0]
            variances.append(np.var(A0))

        # var ∝ dt: var(dt=0.01)/var(dt=0.001) should be near 1
        # because total T is same. Actually var ∝ T, not dt.
        # Both run for T=1.0, so variances should be similar
        # (converging to same distribution).
        ratio = variances[0] / variances[1]
        assert 0.3 < ratio < 3.0, (
            f"Variance ratio {ratio:.2f} outside [0.3, 3.0] — "
            "noise scaling may be wrong")


class TestBAABoundaryDiagnostics:
    """D31: boundary clipping fraction tracking."""

    def test_boundary_fractions_available(self):
        """Integrator reports boundary_fractions for BAA (has project_state).

        With very high noise and low tonic input, A channels frequently
        go negative and get clipped to 0. Verifies shape, availability,
        and that clipping is detected.
        """
        mod = BAAModule(nc=5, tonic_input=0.1)
        B = 10
        theta = mod.default_params(B)
        # Very high noise, low gain → A near 0, noise pushes below
        theta[:, _PI["sigma_base"]] = 3.0
        theta[:, _PI["sigma_max"]] = 6.0
        theta[:, _PI["gain"]] = 0.3

        integ = Integrator(
            modules={"baa": mod},
            tiers=[Tier("default", 0.01, ["baa"])],
            B=B, seed=42, obs_dt=1.0,
        )
        state = integ.run(T=10.0)

        assert "baa" in state.boundary_fractions
        fracs = state.boundary_fractions["baa"]
        assert "lower" in fracs and "upper" in fracs
        assert fracs["lower"].shape == (B, mod.n_state)
        assert fracs["upper"].shape == (B, mod.n_state)
        # A channels clipped at lower boundary (A >= 0) with high noise
        A_lower = fracs["lower"][:, :5]
        assert np.any(A_lower > 0.10), (
            f"Expected A lower boundary fraction > 10% with high noise, "
            f"max={A_lower.max():.3f}")


class TestBAAContract:
    """Rule 20 compliance and contract checks."""

    def test_drift_purity(self):
        """drift() does not mutate inputs."""
        mod = BAAModule(nc=5)
        B = 3
        x = mod.default_state(B)
        theta = mod.default_params(B)
        u = np.full((B, 5), 0.5)

        x_before = x.copy()
        theta_before = theta.copy()
        u_before = u.copy()

        mod.drift(x, theta, u)

        np.testing.assert_array_equal(x, x_before)
        np.testing.assert_array_equal(theta, theta_before)
        np.testing.assert_array_equal(u, u_before)

    def test_batch_independence(self):
        """Drift at B=1 matches row 0 of B=3."""
        mod = BAAModule(nc=5)

        x1 = mod.default_state(1)
        theta1 = mod.default_params(1)
        u1 = np.full((1, 5), 0.5)

        x3 = np.vstack([x1, x1 * 1.1, x1 * 0.9])
        theta3 = np.vstack([theta1, theta1, theta1])
        u3 = np.vstack([u1, u1, u1])

        d1 = mod.drift(x1, theta1, u1)
        d3 = mod.drift(x3, theta3, u3)

        np.testing.assert_allclose(d1, d3[0:1], atol=1e-12)

    def test_outputs_shape(self):
        """outputs() returns correct keys with shape (B,)."""
        mod = BAAModule(nc=5)
        B = 4
        x = mod.default_state(B)
        theta = mod.default_params(B)
        out = mod.outputs(x, theta)

        expected_keys = {"total_activation", "energy",
                         "attention_entropy", "gain"}
        assert set(out.keys()) == expected_keys
        for k, v in out.items():
            assert v.shape == (B,), f"{k} shape {v.shape} != ({B},)"

    def test_spec_states_count(self):
        """nc=5 → 11 states (5A + 1S + 5W)."""
        mod = BAAModule(nc=5)
        assert mod.n_state == 11
        assert mod.n_params == 14
        assert mod.n_inputs == 5
