"""P4 Step 1: Triple composite stability tests.

TestTripleJacobian: 9x9 structure, eigenvalue verification.
TestNumericalStability: large-batch finiteness (B=200, T=600).
"""
import numpy as np
import pytest

from cpsych.core.composite_setup import (
    make_triple_modules, make_triple_couplings,
    make_triple_integrator, wake_equilibrium,
    triple_linearization_point,
)
from cpsych.core.stability import (
    composite_jacobian, numerical_jacobian, max_real_eigenvalue,
)


@pytest.fixture(scope="module")
def wake_eq():
    """Wake equilibrium, computed once for all tests."""
    modules = make_triple_modules()
    return wake_equilibrium(modules["arousal"], T_settle_days=2)


@pytest.fixture(scope="module")
def lin_point(wake_eq):
    """Linearization point at wake equilibrium."""
    return triple_linearization_point(wake_eq)


@pytest.fixture(scope="module")
def arousal_eigs(wake_eq):
    """Arousal-only eigenvalues at wake equilibrium."""
    modules = make_triple_modules()
    J = numerical_jacobian(
        modules["arousal"], wake_eq["x0"], wake_eq["theta0"],
        np.zeros((1, 0)))
    return np.sort(np.real(np.linalg.eigvals(J)))


class TestTripleJacobian:
    """Composite Jacobian structure verification."""

    def test_shape_9x9(self, lin_point):
        """Composite Jacobian is 9x9 (4+2+3 states)."""
        modules, x0, theta0, u0 = lin_point
        couplings = make_triple_couplings()
        J = composite_jacobian(modules, x0, theta0, u0, couplings)
        assert J.shape == (9, 9)

    def test_all_finite(self, lin_point):
        """No NaN/Inf in Jacobian."""
        modules, x0, theta0, u0 = lin_point
        couplings = make_triple_couplings()
        J = composite_jacobian(modules, x0, theta0, u0, couplings)
        assert np.all(np.isfinite(J))

    def test_eigenvalues_are_arousal_plus_zeros(self, lin_point, arousal_eigs):
        """Composite eigenvalues = arousal eigenvalues ∪ {0,0,0,0,0}.

        Block-triangular structure: arousal autonomous, DDM ∂drift/∂evidence=0,
        RL drift=0. Cross terms don't affect eigenvalues.
        """
        modules, x0, theta0, u0 = lin_point
        couplings = make_triple_couplings(s_kappa=2.0, s_alpha=2.0, s_value=3.0)
        J = composite_jacobian(modules, x0, theta0, u0, couplings)
        full_eigs = np.sort(np.real(np.linalg.eigvals(J)))

        # Expect 5 zeros (from DDM: 2, RL: 3) and 4 arousal eigenvalues
        expected = np.sort(np.concatenate([arousal_eigs, np.zeros(5)]))
        np.testing.assert_allclose(full_eigs, expected, atol=0.01,
                                   err_msg="Eigenvalues differ from arousal∪{0⁵}")

    def test_stable_at_default_strengths(self, lin_point):
        """Non-positive eigenvalues at default strengths."""
        modules, x0, theta0, u0 = lin_point
        couplings = make_triple_couplings(s_kappa=2.0, s_alpha=2.0, s_value=3.0)
        J = composite_jacobian(modules, x0, theta0, u0, couplings)
        max_re = max_real_eigenvalue(J)
        assert max_re <= 1e-6, f"Max Re(eigenvalue) = {max_re:.6f}, expected ≤ 0"

    def test_stable_at_max_strengths(self, lin_point):
        """Non-positive eigenvalues at max strengths."""
        modules, x0, theta0, u0 = lin_point
        couplings = make_triple_couplings(s_kappa=4.0, s_alpha=4.0, s_value=10.0)
        J = composite_jacobian(modules, x0, theta0, u0, couplings)
        max_re = max_real_eigenvalue(J)
        assert max_re <= 1e-6, f"Max Re(eigenvalue) = {max_re:.6f}, expected ≤ 0"

    def test_dissipative_eigenvalues_negative(self, lin_point, arousal_eigs):
        """The 3 non-zero arousal eigenvalues are strictly negative."""
        nonzero = arousal_eigs[np.abs(arousal_eigs) > 1e-8]
        assert len(nonzero) == 3, f"Expected 3 non-zero arousal eigs, got {len(nonzero)}"
        assert np.all(nonzero < -1e-6), (
            f"Non-zero arousal eigenvalues should be negative: {nonzero}")


class TestNumericalStability:
    """Numerical stability: large-batch finiteness.

    B=200, T=600 (10 min). 200 independent replicates catch divergence
    statistically. Each replicate traverses default→sleep transition
    within first 7s, then remains in sleep for the rest of T.
    """

    @pytest.fixture(scope="class")
    def default_run(self):
        """B=200, T=600, default coupling strengths."""
        integ = make_triple_integrator(
            B=200, seed=42, obs_dt=600.0,
            s_kappa=2.0, s_alpha=2.0, s_value=3.0)
        return integ.run(T=600.0)

    @pytest.fixture(scope="class")
    def high_s_run(self):
        """B=200, T=600, s_value=10."""
        integ = make_triple_integrator(
            B=200, seed=43, obs_dt=600.0,
            s_kappa=2.0, s_alpha=2.0, s_value=10.0)
        return integ.run(T=600.0)

    def test_all_states_finite(self, default_run):
        """All 200 replicates finite after 10 min at default strengths."""
        for name in ["arousal", "ddm", "rl_mood"]:
            assert np.all(np.isfinite(default_run.x[name])), (
                f"{name} has non-finite states")

    def test_no_divergence_at_high_s_value(self, high_s_run):
        """s_value=10, all 200 replicates finite after 10 min."""
        for name in ["arousal", "ddm", "rl_mood"]:
            assert np.all(np.isfinite(high_s_run.x[name])), (
                f"{name} has non-finite states at s_value=10")

    def test_rl_values_bounded(self, default_run):
        """V0, V1 ∈ [-2, 2], m bounded, across 200 replicates."""
        V0 = default_run.x["rl_mood"][:, 0]
        V1 = default_run.x["rl_mood"][:, 1]
        m = default_run.x["rl_mood"][:, 2]
        assert np.all(np.abs(V0) < 2.0), f"V0 out of bounds: {V0}"
        assert np.all(np.abs(V1) < 2.0), f"V1 out of bounds: {V1}"
        assert np.all(np.abs(m) < 10.0), f"mood out of bounds: {m}"
