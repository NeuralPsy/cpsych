"""Tests for SDEModule contract.

Uses toy OUModule and ToyDDM from conftest.
"""
import numpy as np
import pytest

from conftest import OUModule, ToyDDM


class TestSDEModulePurity:
    """Rule 20: drift and diffusion are pure functions."""

    def test_drift_no_mutation(self):
        """drift must not mutate its arguments."""
        mod = OUModule()
        B = 10
        x = mod.default_state(B)
        theta = mod.default_params(B)
        u = np.ones((B, mod.n_inputs))
        x_copy = x.copy()
        theta_copy = theta.copy()
        u_copy = u.copy()
        mod.drift(x, theta, u)
        np.testing.assert_array_equal(x, x_copy)
        np.testing.assert_array_equal(theta, theta_copy)
        np.testing.assert_array_equal(u, u_copy)

    def test_diffusion_no_mutation(self):
        mod = OUModule()
        B = 10
        x = mod.default_state(B)
        theta = mod.default_params(B)
        x_copy = x.copy()
        mod.diffusion(x, theta)
        np.testing.assert_array_equal(x, x_copy)

    def test_drift_deterministic(self):
        """Same inputs → same outputs."""
        mod = OUModule()
        B = 5
        x = np.ones((B, mod.n_state)) * 0.5
        theta = mod.default_params(B)
        u = np.ones((B, mod.n_inputs)) * 0.3
        dx1 = mod.drift(x, theta, u)
        dx2 = mod.drift(x, theta, u)
        np.testing.assert_array_equal(dx1, dx2)


class TestDefaultState:
    def test_shape(self):
        mod = OUModule()
        x = mod.default_state(100)
        assert x.shape == (100, 2)

    def test_initial_values(self):
        mod = OUModule()
        x = mod.default_state(5)
        assert np.all(x[:, 0] == 0.0)  # x0 initial = 0
        assert np.all(x[:, 1] == 0.0)  # x1 initial = 0

    def test_independent_copies(self):
        """Modifying returned array should not affect next call."""
        mod = OUModule()
        x1 = mod.default_state(3)
        x1[0, 0] = 999.0
        x2 = mod.default_state(3)
        assert x2[0, 0] == 0.0


class TestDefaultParams:
    def test_shape(self):
        mod = OUModule()
        theta = mod.default_params(10)
        assert theta.shape == (10, 4)

    def test_values(self):
        mod = OUModule()
        theta = mod.default_params(1)
        assert theta[0, 0] == 1.0   # alpha
        assert theta[0, 1] == 2.0   # beta
        assert theta[0, 2] == 0.1   # sigma0
        assert theta[0, 3] == 0.1   # sigma1


class TestBatchAxis:
    def test_drift_shape(self):
        mod = OUModule()
        B = 50
        x = mod.default_state(B)
        theta = mod.default_params(B)
        u = np.zeros((B, mod.n_inputs))
        dx = mod.drift(x, theta, u)
        assert dx.shape == (B, mod.n_state)

    def test_diffusion_shape(self):
        mod = OUModule()
        B = 50
        x = mod.default_state(B)
        theta = mod.default_params(B)
        sigma = mod.diffusion(x, theta)
        assert sigma.shape == (B, mod.n_state)

    def test_outputs_shape(self):
        mod = OUModule()
        B = 50
        x = mod.default_state(B)
        theta = mod.default_params(B)
        out = mod.outputs(x, theta)
        assert out["activity"].shape == (B,)

    def test_per_replicate_independence(self):
        """Different params per replicate → different drift."""
        mod = OUModule()
        B = 3
        x = np.ones((B, 2)) * 0.5
        theta = mod.default_params(B)
        theta[0, 0] = 0.1   # alpha = 0.1 for replicate 0
        theta[1, 0] = 10.0  # alpha = 10.0 for replicate 1
        u = np.zeros((B, 1))
        dx = mod.drift(x, theta, u)
        # dx0 = -alpha * x0; different alpha → different dx
        assert dx[0, 0] != dx[1, 0]


class TestEpisodicReset:
    def test_continuous_module_no_reset(self):
        """OUModule is continuous — trial_done always False."""
        mod = OUModule()
        x = np.ones((10, 2))
        theta = mod.default_params(10)
        done = mod.trial_done(x, theta)
        assert not np.any(done)

    def test_episodic_trial_done(self):
        """ToyDDM fires trial_done when |evidence| >= threshold (D14)."""
        mod = ToyDDM()
        x = np.array([[0.5], [1.5], [-1.2], [0.0]])
        theta = mod.default_params(4)  # threshold=1.0
        done = mod.trial_done(x, theta)
        np.testing.assert_array_equal(done, [False, True, True, False])

    def test_trial_done_uses_theta_threshold(self):
        """D14: trial_done respects per-replicate threshold from theta."""
        mod = ToyDDM()
        x = np.array([[0.8], [0.8]])  # same evidence
        theta = mod.default_params(2)
        theta[0, 1] = 0.5   # low threshold → done
        theta[1, 1] = 1.0   # high threshold → not done
        done = mod.trial_done(x, theta)
        np.testing.assert_array_equal(done, [True, False])

    def test_reset_only_episodic_vars(self):
        """reset_state resets EPISODIC variables, keeps CONTINUOUS."""
        mod = ToyDDM()
        x = np.array([[1.5], [-1.2], [0.3]])
        theta = mod.default_params(3)
        mask = np.array([True, True, False])
        x_new = mod.reset_state(x, theta, mask)
        # evidence is EPISODIC → reset to 0 where mask
        assert x_new[0, 0] == 0.0
        assert x_new[1, 0] == 0.0
        assert x_new[2, 0] == 0.3  # not reset

    def test_reset_preserves_continuous(self):
        """For a module with both CONTINUOUS and EPISODIC states,
        reset_state only touches EPISODIC ones."""
        # OUModule has only CONTINUOUS states — reset should change nothing
        mod = OUModule()
        x = np.ones((5, 2)) * 3.0
        theta = mod.default_params(5)
        mask = np.ones(5, dtype=bool)
        x_new = mod.reset_state(x, theta, mask)
        np.testing.assert_array_equal(x_new, x)

    def test_trial_update_default_noop(self):
        """Default trial_update returns x unchanged."""
        mod = OUModule()
        x = np.ones((3, 2))
        theta = mod.default_params(3)
        mask = np.ones(3, dtype=bool)
        x_new = mod.trial_update(x, theta, mask, {})
        np.testing.assert_array_equal(x_new, x)


class TestIndexHelpers:
    def test_state_index(self):
        mod = OUModule()
        assert mod.state_index("x0") == 0
        assert mod.state_index("x1") == 1

    def test_param_index(self):
        mod = OUModule()
        assert mod.param_index("alpha") == 0
        assert mod.param_index("sigma1") == 3

    def test_n_properties(self):
        mod = OUModule()
        assert mod.n_state == 2
        assert mod.n_params == 4
        assert mod.n_inputs == 1
        assert mod.n_outputs == 1
