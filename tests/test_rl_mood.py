"""Tests for RL+Mood module (specs/rl_mood.md v1.3).

Covers: purity, batch, defaults, trial_update mechanics,
softmax outputs, trial_done, trial-duration invariance (D21),
stability below/above lambda_crit (D20), E11 behavioral
distinguishability.
"""
import numpy as np
import pytest

from cpsych.modules.rl_mood import RLMoodModule, lambda_crit


@pytest.fixture
def mod():
    return RLMoodModule()


@pytest.fixture
def B():
    return 50


class TestContract:
    def test_state_count(self, mod):
        assert mod.n_state == 3

    def test_param_count(self, mod):
        assert mod.n_params == 5

    def test_state_names(self, mod):
        names = [s.name for s in mod.spec.states]
        assert names == ["V0", "V1", "m"]

    def test_param_names(self, mod):
        names = [p.name for p in mod.spec.params]
        assert names == ["alpha", "beta", "tau_m", "lam", "sigma_m"]

    def test_output_names(self, mod):
        assert set(mod.spec.output_names) == {"choice_prob", "mood", "rho_eff", "value_diff"}

    def test_no_inputs(self, mod):
        assert mod.spec.input_names == ()

    def test_default_state(self, mod, B):
        x = mod.default_state(B)
        assert x.shape == (B, 3)
        np.testing.assert_array_equal(x[:, 0], 0.5)  # V0
        np.testing.assert_array_equal(x[:, 1], 0.5)  # V1
        np.testing.assert_array_equal(x[:, 2], 0.0)  # m

    def test_default_params(self, mod, B):
        theta = mod.default_params(B)
        assert theta.shape == (B, 5)
        assert theta[0, 0] == pytest.approx(0.15)   # alpha
        assert theta[0, 1] == pytest.approx(2.5)    # beta
        assert theta[0, 2] == pytest.approx(2.163)  # tau_m
        assert theta[0, 3] == pytest.approx(0.0)    # lam
        assert theta[0, 4] == pytest.approx(0.0)    # sigma_m


class TestPurity:
    """drift and diffusion are pure functions (Rule 20)."""

    def test_drift_no_mutation(self, mod, B):
        x = mod.default_state(B)
        theta = mod.default_params(B)
        u = np.zeros((B, 0))
        x_before = x.copy()
        theta_before = theta.copy()
        mod.drift(x, theta, u)
        np.testing.assert_array_equal(x, x_before)
        np.testing.assert_array_equal(theta, theta_before)

    def test_diffusion_no_mutation(self, mod, B):
        x = mod.default_state(B)
        theta = mod.default_params(B)
        x_before = x.copy()
        mod.diffusion(x, theta)
        np.testing.assert_array_equal(x, x_before)

    def test_trial_update_no_mutation(self, mod, B):
        x = mod.default_state(B)
        theta = mod.default_params(B)
        mask = np.ones(B, dtype=bool)
        env_result = {"r": np.ones(B), "choice": np.zeros(B)}
        x_before = x.copy()
        mod.trial_update(x, theta, mask, env_result)
        np.testing.assert_array_equal(x, x_before)


class TestDrift:
    """drift(m) = 0 (D21: decay in trial_update, not drift)."""

    def test_drift_all_zero(self, mod, B):
        x = mod.default_state(B)
        x[:, 2] = np.random.default_rng(0).normal(0, 1, B)  # nonzero m
        theta = mod.default_params(B)
        u = np.zeros((B, 0))
        dx = mod.drift(x, theta, u)
        np.testing.assert_array_equal(dx, 0.0)

    def test_diffusion_only_mood(self, mod, B):
        x = mod.default_state(B)
        theta = mod.default_params(B)
        theta[:, 4] = 0.1  # sigma_m
        diff = mod.diffusion(x, theta)
        assert diff.shape == (B, 3)
        np.testing.assert_array_equal(diff[:, 0], 0.0)  # V0
        np.testing.assert_array_equal(diff[:, 1], 0.0)  # V1
        np.testing.assert_array_equal(diff[:, 2], 0.1)  # m


class TestTrialUpdate:
    """Discrete RL + mood update at trial boundary."""

    def test_value_update_chosen_arm_only(self, mod):
        B = 4
        x = mod.default_state(B)
        theta = mod.default_params(B)
        theta[:, 3] = 0.0  # lam=0, no mood feedback
        mask = np.ones(B, dtype=bool)
        # All choose arm 0, all get reward
        env = {"r": np.ones(B), "choice": np.zeros(B)}
        x_new = mod.trial_update(x, theta, mask, env)
        # V0 should increase (delta = 1*1 - 0.5 = 0.5, dV = 0.15*0.5 = 0.075)
        assert np.all(x_new[:, 0] > 0.5)
        # V1 should not change
        np.testing.assert_array_equal(x_new[:, 1], 0.5)

    def test_mood_decay_order(self, mod):
        """m <- gamma * (m + delta), not gamma*m + delta."""
        B = 1
        x = mod.default_state(B)
        x[0, 2] = 1.0  # initial m=1
        theta = mod.default_params(B)
        theta[0, 3] = 0.0  # lam=0
        mask = np.ones(B, dtype=bool)
        # r=0, choice=0 -> delta = rho_eff*0 - V0 = 0 - 0.5 = -0.5
        env = {"r": np.zeros(B), "choice": np.zeros(B)}
        x_new = mod.trial_update(x, theta, mask, env)

        tau_m = theta[0, 2]
        gamma = np.exp(-1.0 / tau_m)
        delta = -0.5  # rho_eff=1 (lam=0), r=0, V=0.5
        expected_m = gamma * (1.0 + delta)  # gamma * (m + delta)
        assert x_new[0, 2] == pytest.approx(expected_m, abs=1e-10)

    def test_mood_with_feedback(self, mod):
        """With lam>0, rho_eff modulates RPE."""
        B = 1
        x = mod.default_state(B)
        x[0, 2] = 0.5  # m=0.5
        theta = mod.default_params(B)
        theta[0, 3] = 0.2  # lam=0.2
        mask = np.ones(B, dtype=bool)
        env = {"r": np.ones(B), "choice": np.zeros(B)}
        x_new = mod.trial_update(x, theta, mask, env)

        tau_m = theta[0, 2]
        gamma = np.exp(-1.0 / tau_m)
        rho_eff = 1.0 + 0.2 * 0.5  # 1.1
        delta = rho_eff * 1.0 - 0.5  # 0.6
        expected_m = gamma * (0.5 + delta)  # gamma * 1.1
        assert x_new[0, 2] == pytest.approx(expected_m, abs=1e-10)

    def test_mask_respected(self, mod):
        """Only masked replicates are updated."""
        B = 4
        x = mod.default_state(B)
        theta = mod.default_params(B)
        mask = np.array([True, True, False, False])
        env = {"r": np.ones(B), "choice": np.zeros(B)}
        x_new = mod.trial_update(x, theta, mask, env)
        # Masked: updated
        assert x_new[0, 0] != 0.5
        # Unmasked: unchanged
        assert x_new[2, 0] == 0.5
        assert x_new[2, 2] == 0.0


class TestOutputs:
    def test_softmax_equal_values(self, mod, B):
        """V0=V1 -> choice_prob = 0.5."""
        x = mod.default_state(B)
        theta = mod.default_params(B)
        out = mod.outputs(x, theta)
        np.testing.assert_allclose(out["choice_prob"], 0.5, atol=1e-10)

    def test_softmax_prefers_higher_value(self, mod, B):
        """V1 > V0 -> choice_prob > 0.5."""
        x = mod.default_state(B)
        x[:, 1] = 0.8  # V1 > V0
        theta = mod.default_params(B)
        out = mod.outputs(x, theta)
        assert np.all(out["choice_prob"] > 0.5)

    def test_softmax_beta_sharpens(self, mod):
        """Higher beta -> more extreme probabilities."""
        B = 10
        x = mod.default_state(B)
        x[:, 1] = 0.6  # V1 slightly > V0
        theta_low = mod.default_params(B)
        theta_low[:, 1] = 1.0  # low beta
        theta_high = mod.default_params(B)
        theta_high[:, 1] = 10.0  # high beta
        p_low = mod.outputs(x, theta_low)["choice_prob"]
        p_high = mod.outputs(x, theta_high)["choice_prob"]
        assert np.all(p_high > p_low)

    def test_mood_output(self, mod, B):
        x = mod.default_state(B)
        x[:, 2] = 0.42
        theta = mod.default_params(B)
        out = mod.outputs(x, theta)
        np.testing.assert_array_equal(out["mood"], 0.42)

    def test_rho_eff_with_lambda(self, mod, B):
        x = mod.default_state(B)
        x[:, 2] = 0.5
        theta = mod.default_params(B)
        theta[:, 3] = 0.3  # lam
        out = mod.outputs(x, theta)
        np.testing.assert_allclose(out["rho_eff"], 1.0 + 0.3 * 0.5)


class TestTrialDone:
    def test_always_true(self, mod, B):
        x = mod.default_state(B)
        theta = mod.default_params(B)
        done = mod.trial_done(x, theta)
        assert done.shape == (B,)
        assert np.all(done)


class TestTrialDurationInvariance:
    """D21: m update is trial-paced, not time-paced.
    Same reward sequence -> same m trajectory regardless of dt between trials."""

    def test_same_trajectory_different_dt(self):
        """Run 50 trials with dt=1 and dt=0.3. Same rewards -> same m."""
        mod = RLMoodModule()
        B = 1
        n_trials = 50
        rng = np.random.default_rng(42)
        rewards = rng.integers(0, 2, size=n_trials).astype(float)
        choices = rng.integers(0, 2, size=n_trials).astype(float)

        def run_with_dt(dt, n_substeps):
            x = mod.default_state(B)
            theta = mod.default_params(B)
            theta[0, 4] = 0.0  # no noise, deterministic comparison
            mask = np.ones(B, dtype=bool)
            m_trajectory = []
            for t in range(n_trials):
                # Substeps of drift (which is zero, but let's be explicit)
                for _ in range(n_substeps):
                    dx = mod.drift(x, theta, np.zeros((B, 0)))
                    x = x + dx * dt  # Euler step (dx=0, so no-op)
                # Trial update
                env = {"r": rewards[t:t+1], "choice": choices[t:t+1]}
                x = mod.trial_update(x, theta, mask, env)
                m_trajectory.append(x[0, 2])
            return np.array(m_trajectory)

        m_dt1 = run_with_dt(dt=1.0, n_substeps=1)
        m_dt03 = run_with_dt(dt=0.1, n_substeps=10)  # 10x more substeps

        np.testing.assert_allclose(m_dt1, m_dt03, atol=1e-12,
                                   err_msg="m trajectory depends on dt")


class TestLambdaCrit:
    def test_formula_baseline(self):
        """lambda_crit at tau_m=2.163, r_mean=0.60."""
        tau_m = 2.163
        gamma = np.exp(-1.0 / tau_m)
        lc = lambda_crit(tau_m, r_mean=0.60)
        expected = (1.0 - gamma) / (gamma * 0.60)
        assert lc == pytest.approx(expected, rel=1e-6)

    def test_increases_with_smaller_tau_m(self):
        """Faster decay (smaller tau_m) -> higher lambda_crit."""
        lc_fast = lambda_crit(1.0)
        lc_slow = lambda_crit(10.0)
        assert lc_fast > lc_slow

    def test_vectorized(self):
        tau_m = np.array([1.0, 2.163, 5.0, 10.0])
        lc = lambda_crit(tau_m)
        assert lc.shape == (4,)
        assert np.all(np.diff(lc) < 0)  # monotonically decreasing

    def test_stability_below_crit(self):
        """At lambda < lambda_crit, mood stays bounded over 500 trials."""
        mod = RLMoodModule()
        B = 100
        tau_m = 2.163
        lam = 0.5 * lambda_crit(tau_m, r_mean=0.60)
        rng = np.random.default_rng(42)

        x = mod.default_state(B)
        theta = mod.default_params(B)
        theta[:, 2] = tau_m
        theta[:, 3] = lam
        theta[:, 4] = 0.0  # no noise
        mask = np.ones(B, dtype=bool)

        max_m = np.zeros(B)
        for _ in range(500):
            choice = rng.integers(0, 2, size=B).astype(float)
            # Simulate reward with p=0.75 for correct arm
            correct = rng.integers(0, 2, size=B).astype(float)
            p = np.where(choice == correct, 0.75, 0.25)
            r = (rng.random(B) < p).astype(float)
            env = {"r": r, "choice": choice}
            x = mod.trial_update(x, theta, mask, env)
            max_m = np.maximum(max_m, np.abs(x[:, 2]))

        assert np.mean(max_m < 50) > 0.90, \
            f"Only {np.mean(max_m < 50):.0%} stable at lam={lam:.3f}"

    def test_instability_above_crit(self):
        """At lambda > lambda_crit, mood diverges in most replicates."""
        mod = RLMoodModule()
        B = 100
        tau_m = 2.163
        lam = 1.5 * lambda_crit(tau_m, r_mean=0.60)
        rng = np.random.default_rng(42)

        x = mod.default_state(B)
        theta = mod.default_params(B)
        theta[:, 2] = tau_m
        theta[:, 3] = lam
        theta[:, 4] = 0.0
        mask = np.ones(B, dtype=bool)

        max_m = np.zeros(B)
        for _ in range(500):
            choice = rng.integers(0, 2, size=B).astype(float)
            correct = rng.integers(0, 2, size=B).astype(float)
            p = np.where(choice == correct, 0.75, 0.25)
            r = (rng.random(B) < p).astype(float)
            env = {"r": r, "choice": choice}
            x = mod.trial_update(x, theta, mask, env)
            max_m = np.maximum(max_m, np.abs(x[:, 2]))

        assert np.mean(max_m > 50) > 0.50, \
            f"Only {np.mean(max_m > 50):.0%} diverged at lam={lam:.3f}"


class TestBatchIndependence:
    """Different replicates with different params should not interfere."""

    def test_different_alpha(self, mod):
        B = 2
        x = mod.default_state(B)
        theta = mod.default_params(B)
        theta[0, 0] = 0.05  # slow learner
        theta[1, 0] = 0.40  # fast learner
        mask = np.ones(B, dtype=bool)
        env = {"r": np.ones(B), "choice": np.zeros(B)}
        x_new = mod.trial_update(x, theta, mask, env)
        # Fast learner should have larger V0 update
        assert x_new[1, 0] > x_new[0, 0]
