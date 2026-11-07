"""Tests for DDMModule.

Standalone validation per specs/ddm.md v1.2.
"""
import numpy as np
import pytest

from cpsych.modules.ddm import DDMModule, T_MAX_TRIAL
from cpsych.core.integrator import Integrator, Tier
from cpsych.environments.perceptual_task import PerceptualTask


@pytest.fixture
def mod():
    return DDMModule()


# ---- Contract tests ----

class TestContract:
    def test_ddm1_state_count(self, mod):
        """DDM1: 2 state variables (evidence, trial_clock)."""
        assert mod.n_state == 2

    def test_ddm2_param_count(self, mod):
        """DDM2: 4 parameters (kappa, a, Ter, drift_bias)."""
        assert mod.n_params == 4

    def test_ddm3_input_names(self, mod):
        """DDM3: inputs are coherence and direction (D15)."""
        assert mod.spec.input_names == ("coherence", "direction")

    def test_ddm4_output_names(self, mod):
        """DDM4: outputs are evidence, choice, rt, is_timeout."""
        assert mod.spec.output_names == ("evidence", "choice", "rt", "is_timeout")

    def test_ddm5_default_state_shape(self, mod):
        """DDM5: default state is (B, 2) zeros."""
        x = mod.default_state(10)
        assert x.shape == (10, 2)
        np.testing.assert_array_equal(x, 0.0)

    def test_ddm6_default_params_shape(self, mod):
        """DDM6: default params is (B, 4)."""
        theta = mod.default_params(5)
        assert theta.shape == (5, 4)


# ---- Drift purity (Rule 20) ----

class TestPurity:
    def test_ddm7_drift_pure(self, mod):
        """DDM7: drift does not mutate inputs."""
        B = 4
        x = np.random.default_rng(0).standard_normal((B, 2))
        theta = mod.default_params(B)
        u = np.random.default_rng(1).standard_normal((B, 2))
        x_copy, theta_copy, u_copy = x.copy(), theta.copy(), u.copy()
        mod.drift(x, theta, u)
        np.testing.assert_array_equal(x, x_copy)
        np.testing.assert_array_equal(theta, theta_copy)
        np.testing.assert_array_equal(u, u_copy)

    def test_ddm8_diffusion_pure(self, mod):
        """DDM8: diffusion does not mutate inputs."""
        B = 4
        x = np.random.default_rng(0).standard_normal((B, 2))
        theta = mod.default_params(B)
        x_copy = x.copy()
        mod.diffusion(x, theta)
        np.testing.assert_array_equal(x, x_copy)


# ---- Drift correctness ----

class TestDrift:
    def test_ddm9_drift_at_zero_coherence(self, mod):
        """DDM9: v=0 at zero coherence, clock drift=1."""
        B = 3
        x = mod.default_state(B)
        theta = mod.default_params(B)
        u = np.zeros((B, 2))  # coherence=0, direction=0
        dx = mod.drift(x, theta, u)
        np.testing.assert_allclose(dx[:, 0], 0.0)
        np.testing.assert_allclose(dx[:, 1], 1.0)

    def test_ddm10_drift_proportional_to_coherence(self, mod):
        """DDM10: v = kappa * coherence * direction."""
        B = 2
        x = mod.default_state(B)
        theta = mod.default_params(B)
        kappa = theta[0, 0]
        u = np.array([[0.512, 1.0], [0.128, -1.0]])
        dx = mod.drift(x, theta, u)
        np.testing.assert_allclose(dx[0, 0], kappa * 0.512 * 1.0)
        np.testing.assert_allclose(dx[1, 0], kappa * 0.128 * (-1.0))

    def test_ddm10b_drift_with_drift_bias(self, mod):
        """DDM10b: v = kappa*coh*dir + drift_bias (D27)."""
        B = 2
        x = mod.default_state(B)
        theta = mod.default_params(B)
        theta[:, 3] = 5.0  # drift_bias = 5.0
        u = np.array([[0.0, 1.0], [0.128, 1.0]])
        dx = mod.drift(x, theta, u)
        kappa = theta[0, 0]
        # At coh=0: drift = 0 + 5.0 = 5.0
        np.testing.assert_allclose(dx[0, 0], 5.0)
        # At coh=0.128: drift = kappa*0.128*1 + 5.0
        np.testing.assert_allclose(dx[1, 0], kappa * 0.128 + 5.0)

    def test_ddm11_diffusion_sigma_one(self, mod):
        """DDM11: sigma=1 on evidence, 0 on clock."""
        B = 5
        x = mod.default_state(B)
        theta = mod.default_params(B)
        s = mod.diffusion(x, theta)
        np.testing.assert_allclose(s[:, 0], 1.0)
        np.testing.assert_allclose(s[:, 1], 0.0)


# ---- Trial mechanics ----

class TestTrialMechanics:
    def test_ddm12_trial_done_at_boundary(self, mod):
        """DDM12: trial_done fires when |evidence| >= a/2."""
        B = 4
        theta = mod.default_params(B)
        a = theta[0, 1]
        half_a = a / 2
        x = mod.default_state(B)
        x[0, 0] = half_a + 0.01   # above upper
        x[1, 0] = -half_a - 0.01  # below lower
        x[2, 0] = half_a - 0.01   # just inside
        x[3, 0] = 0.0             # center
        done = mod.trial_done(x, theta)
        np.testing.assert_array_equal(done, [True, True, False, False])

    def test_ddm13_trial_done_per_replicate_threshold(self, mod):
        """DDM13: different thresholds per replicate (D14)."""
        B = 2
        theta = mod.default_params(B)
        theta[0, 1] = 1.0   # a=1, boundary at 0.5
        theta[1, 1] = 3.0   # a=3, boundary at 1.5
        x = mod.default_state(B)
        x[0, 0] = 0.6   # done for a=1, not for a=3
        x[1, 0] = 0.6
        done = mod.trial_done(x, theta)
        np.testing.assert_array_equal(done, [True, False])

    def test_ddm14_timeout(self, mod):
        """DDM14: trial_done fires on timeout."""
        B = 2
        theta = mod.default_params(B)
        x = mod.default_state(B)
        x[0, 1] = T_MAX_TRIAL + 0.1  # past timeout
        x[1, 1] = T_MAX_TRIAL - 0.1  # before timeout
        x[:, 0] = 0.0  # not at boundary
        done = mod.trial_done(x, theta)
        np.testing.assert_array_equal(done, [True, False])

    def test_ddm15_reset_state(self, mod):
        """DDM15: reset_state zeros evidence and clock."""
        B = 3
        theta = mod.default_params(B)
        x = mod.default_state(B)
        x[:, 0] = [1.5, -0.8, 0.3]
        x[:, 1] = [0.5, 0.3, 0.1]
        mask = np.array([True, True, False])
        x_new = mod.reset_state(x, theta, mask)
        assert x_new[0, 0] == 0.0
        assert x_new[0, 1] == 0.0
        assert x_new[1, 0] == 0.0
        assert x_new[1, 1] == 0.0
        assert x_new[2, 0] == 0.3
        assert x_new[2, 1] == 0.1

    def test_ddm16_choice_sign(self, mod):
        """DDM16: choice = sign(evidence) at boundary."""
        B = 3
        theta = mod.default_params(B)
        a = theta[0, 1]
        x = mod.default_state(B)
        x[0, 0] = a / 2 + 0.1    # upper boundary → +1
        x[1, 0] = -a / 2 - 0.1   # lower boundary → -1
        x[2, 0] = 0.0             # center → 0
        out = mod.outputs(x, theta)
        np.testing.assert_array_equal(out["choice"], [1.0, -1.0, 0.0])

    def test_ddm16b_is_timeout_output(self, mod):
        """DDM16b: is_timeout = 1 when trial_clock >= T_MAX_TRIAL."""
        B = 3
        theta = mod.default_params(B)
        x = mod.default_state(B)
        x[0, 1] = T_MAX_TRIAL + 0.1  # past timeout
        x[1, 1] = T_MAX_TRIAL - 0.1  # before timeout
        x[2, 1] = 0.0                 # start
        out = mod.outputs(x, theta)
        np.testing.assert_array_equal(
            out["is_timeout"], [1.0, 0.0, 0.0])


# ---- Integration with PerceptualTask (D15) ----

class TestIntegration:
    def test_ddm17_stimulus_reaches_drift(self):
        """DDM17: coherence and direction from env reach DDM drift (D15)."""
        mod = DDMModule()
        env = PerceptualTask()
        B = 50
        integrator = Integrator(
            modules={"ddm": mod},
            environment=env,
            tiers=[Tier("fast", 0.001, ["ddm"])],
            B=B,
            seed=42,
        )
        # Run long enough for trials to complete.
        # At 0% coherence, mean DT = (a/2)^2 = 1.0 MTU. At T=5, even
        # unlucky replicates should have completed >= 1 trial.
        state = integrator.run(T=5.0)
        assert np.all(state.trial_count > 0), \
            f"Not all replicates completed: min={np.min(state.trial_count)}"

    def test_ddm18_many_trials_run(self):
        """DDM18: enough trials complete in reasonable time."""
        mod = DDMModule()
        env = PerceptualTask()
        B = 20
        integrator = Integrator(
            modules={"ddm": mod},
            environment=env,
            tiers=[Tier("fast", 0.001, ["ddm"])],
            B=B,
            seed=123,
        )
        state = integrator.run(T=5.0)
        # With mixed coherences, expect ~5-10 trials per replicate in 5 MTU
        assert np.median(state.trial_count) >= 3
