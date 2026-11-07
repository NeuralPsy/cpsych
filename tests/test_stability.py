"""Tests for stability utilities.

1. Jacobian vs analytical for linear OU (original)
2. Jacobian on nonlinear system — central differences must give ~1e-10 accuracy
3. Composite Jacobian for coupled system
4. Critical coupling strength vs analytical closed form (Rule 19)
"""
import numpy as np
import pytest

from conftest import OUModule, NonlinearModule, DriveOU1D, _make_drive_ou_spec
from cpsych.core.stability import (
    numerical_jacobian, composite_jacobian,
    find_critical_strength, max_real_eigenvalue,
)
from cpsych.core.coupling import Coupling


class TestNumericalJacobian:
    def test_ou_jacobian_vs_analytical(self):
        """OU drift: dx0 = -alpha*x0 + drive, dx1 = -beta*x1.
        Analytical Jacobian: [[-alpha, 0], [0, -beta]]."""
        mod = OUModule()
        alpha, beta = 1.5, 3.0
        x0 = np.array([[0.5, 0.3]])
        theta0 = np.array([[alpha, beta, 0.1, 0.1]])
        u0 = np.array([[0.0]])

        J = numerical_jacobian(mod, x0, theta0, u0)
        J_expected = np.array([[-alpha, 0.0], [0.0, -beta]])
        np.testing.assert_allclose(J, J_expected, atol=1e-8)

    def test_ou_jacobian_at_different_point(self):
        """Jacobian should be independent of x0 for linear OU."""
        mod = OUModule()
        alpha, beta = 2.0, 0.5
        theta0 = np.array([[alpha, beta, 0.1, 0.1]])
        u0 = np.array([[1.0]])

        J1 = numerical_jacobian(mod, np.array([[0.0, 0.0]]), theta0, u0)
        J2 = numerical_jacobian(mod, np.array([[5.0, -3.0]]), theta0, u0)
        np.testing.assert_allclose(J1, J2, atol=1e-8)

    def test_eigenvalues_stable(self):
        """OU is stable: eigenvalues are -alpha, -beta (both negative)."""
        mod = OUModule()
        theta0 = np.array([[1.0, 2.0, 0.1, 0.1]])
        J = numerical_jacobian(mod, np.zeros((1, 2)), theta0, np.zeros((1, 1)))
        max_re = max_real_eigenvalue(J)
        assert max_re < 0, f"Expected stable, got max Re(eig) = {max_re}"

    def test_eigenvalues_values(self):
        """Eigenvalues should be -alpha, -beta."""
        mod = OUModule()
        alpha, beta = 1.5, 3.0
        theta0 = np.array([[alpha, beta, 0.1, 0.1]])
        J = numerical_jacobian(mod, np.zeros((1, 2)), theta0, np.zeros((1, 1)))
        eigs = sorted(np.real(np.linalg.eigvals(J)))
        np.testing.assert_allclose(eigs, [-beta, -alpha], atol=1e-8)


class TestNonlinearJacobian:
    """Central differences on a nonlinear system.

    dx0/dt = -x0³ + x1      →  ∂f0/∂x0 = -3x0², ∂f0/∂x1 = 1
    dx1/dt = -x0 - x1²      →  ∂f1/∂x0 = -1,    ∂f1/∂x1 = -2x1

    On linear OU the Jacobian is constant, so ANY finite-difference scheme
    (forward, backward, wrong eps) gives the correct answer. This test
    requires central differences and ~1e-10 accuracy.
    """

    def test_nonlinear_jacobian_accuracy(self):
        """Central differences at eps=1e-5 give O(eps²) ≈ 1e-10 error."""
        mod = NonlinearModule()
        a, b = 1.7, -0.4
        x0 = np.array([[a, b]])
        theta0 = np.zeros((1, 0))
        u0 = np.zeros((1, 0))

        J = numerical_jacobian(mod, x0, theta0, u0)
        J_expected = np.array([
            [-3 * a ** 2, 1.0],
            [-1.0, -2 * b],
        ])
        np.testing.assert_allclose(J, J_expected, atol=1e-10)

    def test_nonlinear_jacobian_varies_with_point(self):
        """Unlike OU, nonlinear Jacobian depends on x0."""
        mod = NonlinearModule()
        theta0 = np.zeros((1, 0))
        u0 = np.zeros((1, 0))

        J1 = numerical_jacobian(mod, np.array([[1.0, 0.5]]),
                                theta0, u0)
        J2 = numerical_jacobian(mod, np.array([[2.0, -1.0]]),
                                theta0, u0)
        # Jacobians must differ
        assert not np.allclose(J1, J2), \
            "Nonlinear Jacobian should depend on linearization point"

    def test_nonlinear_jacobian_multiple_points(self):
        """Verify accuracy at several linearization points.

        For cubic f=-x³, central differences give exactly -3x²-ε².
        At x=0 the error IS the truncation term ε²=1e-10, so atol
        must exceed ε² to accommodate the zero-derivative case.
        """
        mod = NonlinearModule()
        theta0 = np.zeros((1, 0))
        u0 = np.zeros((1, 0))

        for a, b in [(0.0, 0.0), (2.5, 1.3), (-0.8, 3.1), (0.1, -2.0)]:
            x0 = np.array([[a, b]])
            J = numerical_jacobian(mod, x0, theta0, u0)
            J_expected = np.array([
                [-3 * a ** 2, 1.0],
                [-1.0, -2 * b],
            ])
            # atol=2e-10: central diff on cubic has error exactly ε²=1e-10
            np.testing.assert_allclose(
                J, J_expected, atol=2e-10,
                err_msg=f"Failed at point ({a}, {b})")


class TestCriticalStrength:
    """Two coupled 1D OU modules with mutual excitatory coupling.

    Module a: dx_a/dt = -α_a·x_a + drive_a,  output = x_a
    Module b: dx_b/dt = -α_b·x_b + drive_b,  output = x_b

    Coupling a→b: drive_b += c·x_a
    Coupling b→a: drive_a += c·x_b

    Composite Jacobian:
        J = [[-α_a, c], [c, -α_b]]

    Eigenvalues: λ = -(α_a+α_b)/2 ± √((α_a-α_b)²/4 + c²)
    Critical: c_crit = √(α_a·α_b)
    """

    @staticmethod
    def _make_system(alpha_a, alpha_b):
        spec_a = _make_drive_ou_spec("mod_a", alpha_a, 0.0, 0.0, 1.0 / alpha_a)
        spec_b = _make_drive_ou_spec("mod_b", alpha_b, 0.0, 0.0, 1.0 / alpha_b)
        modules = {"a": DriveOU1D(spec_a), "b": DriveOU1D(spec_b)}
        x0 = {"a": np.zeros((1, 1)), "b": np.zeros((1, 1))}
        theta0 = {
            "a": modules["a"].default_params(1),
            "b": modules["b"].default_params(1),
        }
        u0 = {"a": np.zeros((1, 0)), "b": np.zeros((1, 0))}
        return modules, x0, theta0, u0

    @staticmethod
    def _make_couplings(c):
        return [
            Coupling("a", "x", "b", "drive",
                     strength=c, max_strength=100.0),
            Coupling("b", "x", "a", "drive",
                     strength=c, max_strength=100.0),
        ]

    def test_composite_jacobian_structure(self):
        """Composite Jacobian = [[-α_a, c], [c, -α_b]] at x=0."""
        alpha_a, alpha_b = 1.0, 4.0
        c = 1.5
        modules, x0, theta0, u0 = self._make_system(alpha_a, alpha_b)

        J = composite_jacobian(modules, x0, theta0, u0,
                               self._make_couplings(c))
        J_expected = np.array([[-alpha_a, c], [c, -alpha_b]])
        np.testing.assert_allclose(J, J_expected, atol=1e-10)

    def test_critical_strength_analytical(self):
        """c_crit = √(α_a·α_b). For α_a=1, α_b=4: c_crit = 2."""
        alpha_a, alpha_b = 1.0, 4.0
        c_analytical = np.sqrt(alpha_a * alpha_b)  # 2.0
        modules, x0, theta0, u0 = self._make_system(alpha_a, alpha_b)

        c_numerical = find_critical_strength(
            modules, x0, theta0, u0,
            make_couplings=self._make_couplings,
            s_min=0.0, s_max=5.0,
        )
        np.testing.assert_allclose(c_numerical, c_analytical, atol=1e-4)

    def test_critical_strength_asymmetric(self):
        """Verify with different α values: α_a=2, α_b=8 → c_crit=4."""
        alpha_a, alpha_b = 2.0, 8.0
        c_analytical = np.sqrt(alpha_a * alpha_b)  # 4.0
        modules, x0, theta0, u0 = self._make_system(alpha_a, alpha_b)

        c_numerical = find_critical_strength(
            modules, x0, theta0, u0,
            make_couplings=self._make_couplings,
            s_min=0.0, s_max=10.0,
        )
        np.testing.assert_allclose(c_numerical, c_analytical, atol=1e-4)

    def test_below_critical_stable(self):
        """Below c_crit, system must be stable."""
        alpha_a, alpha_b = 1.0, 4.0
        c = 1.5  # < c_crit = 2.0
        modules, x0, theta0, u0 = self._make_system(alpha_a, alpha_b)

        J = composite_jacobian(modules, x0, theta0, u0,
                               self._make_couplings(c))
        assert max_real_eigenvalue(J) < 0

    def test_above_critical_unstable(self):
        """Above c_crit, system must be unstable."""
        alpha_a, alpha_b = 1.0, 4.0
        c = 2.5  # > c_crit = 2.0
        modules, x0, theta0, u0 = self._make_system(alpha_a, alpha_b)

        J = composite_jacobian(modules, x0, theta0, u0,
                               self._make_couplings(c))
        assert max_real_eigenvalue(J) > 0
