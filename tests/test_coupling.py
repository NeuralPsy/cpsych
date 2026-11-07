"""Tests for Coupling contract."""
import numpy as np
import pytest

from cpsych.core.coupling import Coupling


class TestCouplingModulate:
    def test_zero_strength_no_effect(self):
        c = Coupling("src", "out", "tgt", "param", strength=0.0)
        theta = np.ones((5, 3))
        source_val = np.ones(5) * 10.0
        theta_new = c.modulate(theta, param_idx=1, source_value=source_val)
        np.testing.assert_array_equal(theta_new, theta)

    def test_additive_modulation(self):
        c = Coupling("src", "out", "tgt", "param",
                     strength=0.5, max_strength=1.0)
        theta = np.ones((3, 2))  # base = 1.0
        source_val = np.array([2.0, 4.0, 6.0])
        theta_new = c.modulate(theta, param_idx=0, source_value=source_val)
        expected = 1.0 + 0.5 * source_val
        np.testing.assert_allclose(theta_new[:, 0], expected)
        # Other columns unchanged
        np.testing.assert_array_equal(theta_new[:, 1], 1.0)

    def test_clamping_strength(self):
        """strength > max_strength → clamped to max_strength."""
        c = Coupling("src", "out", "tgt", "param",
                     strength=5.0, max_strength=2.0)
        theta = np.zeros((2, 1))
        source_val = np.array([1.0, 1.0])
        theta_new = c.modulate(theta, param_idx=0, source_value=source_val)
        np.testing.assert_allclose(theta_new[:, 0], 2.0)  # 0 + 2.0 * 1.0

    def test_negative_strength_clamped_to_zero(self):
        c = Coupling("src", "out", "tgt", "param",
                     strength=-1.0, max_strength=1.0)
        theta = np.ones((3, 1))
        source_val = np.ones(3) * 10.0
        theta_new = c.modulate(theta, param_idx=0, source_value=source_val)
        np.testing.assert_array_equal(theta_new, theta)  # no effect

    def test_custom_h(self):
        """Custom transform function h (receives source and base)."""
        c = Coupling("src", "out", "tgt", "param",
                     h=lambda x, _base: x ** 2,
                     strength=1.0, max_strength=5.0)
        theta = np.zeros((3, 1))
        source_val = np.array([2.0, 3.0, 4.0])
        theta_new = c.modulate(theta, param_idx=0, source_value=source_val)
        np.testing.assert_allclose(theta_new[:, 0], [4.0, 9.0, 16.0])

    def test_no_mutation(self):
        """modulate must not mutate theta_target."""
        c = Coupling("src", "out", "tgt", "param", strength=1.0)
        theta = np.ones((5, 2))
        theta_copy = theta.copy()
        c.modulate(theta, 0, np.ones(5))
        np.testing.assert_array_equal(theta, theta_copy)

    def test_batch_independence(self):
        """Different replicates get different modulations."""
        c = Coupling("src", "out", "tgt", "param", strength=1.0, max_strength=5.0)
        theta = np.zeros((4, 1))
        source_val = np.array([0.0, 1.0, 2.0, 3.0])
        theta_new = c.modulate(theta, 0, source_val)
        np.testing.assert_allclose(theta_new[:, 0], [0.0, 1.0, 2.0, 3.0])

    def test_param_min_clamp(self):
        """param_min clamps modulated parameter from below."""
        c = Coupling("src", "out", "tgt", "param",
                     strength=2.0, max_strength=5.0, param_min=0.0)
        theta = np.array([[1.0]])
        source_val = np.array([-5.0])
        theta_new = c.modulate(theta, 0, source_val)
        # Without clamp: 1.0 + 2.0*(-5.0) = -9.0
        # With param_min=0.0: 0.0
        assert theta_new[0, 0] == 0.0

    def test_param_max_clamp(self):
        """param_max clamps modulated parameter from above."""
        c = Coupling("src", "out", "tgt", "param",
                     strength=1.0, max_strength=5.0, param_max=3.0)
        theta = np.array([[1.0]])
        source_val = np.array([10.0])
        theta_new = c.modulate(theta, 0, source_val)
        # Without clamp: 1.0 + 1.0*10.0 = 11.0
        # With param_max=3.0: 3.0
        assert theta_new[0, 0] == 3.0
