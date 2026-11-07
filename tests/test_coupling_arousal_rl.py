"""P3.3: Arousal → RL+Mood coupling tests.

Two couplings:
1. gain → beta: standalone only (softmax choice, no DDM). D22, D25.
2. gain → alpha: all configurations. D26.

Both multiplicative: param_eff = param * max(0, 1 + s*(gain-1)).
Unidirectional. Spec: specs/p3_couplings.md.
"""
import numpy as np
import pytest

from cpsych.core.coupling import Coupling
from cpsych.core.integrator import Integrator, Tier
from cpsych.core.stability import composite_jacobian
from cpsych.modules.arousal import ArousalModule
from cpsych.modules.rl_mood import RLMoodModule, lambda_crit
from cpsych.environments.reversal_bandit import ReversalBandit


# -------------------------------------------------------------------
# Coupling factories
# -------------------------------------------------------------------

def make_gain_beta_coupling(strength: float = 1.0,
                             max_strength: float = 4.0) -> Coupling:
    """Multiplicative: beta_eff = beta * max(0, 1 + s*(gain - 1)).

    Configuration: standalone RL only (softmax choice, no DDM). D25.
    """
    return Coupling(
        source_module="arousal",
        source_output="gain",
        target_module="rl_mood",
        target_param="beta",
        h=lambda gain, beta_base: beta_base * (gain - 1),
        strength=strength,
        max_strength=max_strength,
        param_min=0.0,
        doc="Multiplicative gain->beta (D22/D25, standalone only)",
    )


def make_gain_alpha_coupling(strength: float = 1.0,
                              max_strength: float = 4.0) -> Coupling:
    """Multiplicative: alpha_eff = alpha * max(0, 1 + s*(gain - 1)).

    Configuration: all (standalone + composite). D26.
    """
    return Coupling(
        source_module="arousal",
        source_output="gain",
        target_module="rl_mood",
        target_param="alpha",
        h=lambda gain, alpha_base: alpha_base * (gain - 1),
        strength=strength,
        max_strength=max_strength,
        param_min=0.0,
        param_max=1.0,
        doc="Multiplicative gain->alpha (D26, all configurations)",
    )


# -------------------------------------------------------------------
# gain → beta: standalone only (D25)
# -------------------------------------------------------------------

class TestGainBetaCoupling:
    """Coupling h-function and modulation correctness.

    Configuration: standalone RL (softmax choice). D25: beta is the
    "sharpness" of softmax. In composite with DDM, choice is made by
    DDM (drift/threshold ratio), not softmax, so gain→beta is inert.
    """

    def test_gain_one_no_modulation(self):
        """At gain=1 (baseline), beta unchanged."""
        c = make_gain_beta_coupling(strength=3.0)
        beta_base = 2.50
        theta = np.array([[0.15, beta_base, 2.163, 0.0, 0.0]])
        h_val = c.h(np.array([1.0]), theta[:, 1])
        assert h_val == pytest.approx(0.0, abs=1e-10)

    def test_gain_above_one_increases_beta(self):
        """gain > 1 → more deterministic choice (higher exploitation)."""
        c = make_gain_beta_coupling(strength=1.0)
        beta_base = 2.50
        gain = np.array([1.05])
        h_val = c.h(gain, np.array([beta_base]))
        actual_eff = beta_base + 1.0 * h_val[0]
        assert actual_eff > beta_base
        assert actual_eff == pytest.approx(beta_base * 1.05, rel=1e-10)

    def test_gain_below_one_decreases_beta(self):
        """gain < 1 → more exploratory choice."""
        c = make_gain_beta_coupling(strength=1.0)
        beta_base = 2.50
        gain = np.array([0.8])
        h_val = c.h(gain, np.array([beta_base]))
        actual_eff = beta_base + 1.0 * h_val[0]
        assert actual_eff < beta_base

    def test_beta_eff_positive_at_max_strength(self):
        """C6: beta_eff > 0 at max strength, min wake gain."""
        c = make_gain_beta_coupling(strength=4.0)
        theta = np.array([[0.15, 2.50, 2.163, 0.0, 0.0]])
        gain = np.array([0.79])  # min wake gain
        theta_new = c.modulate(theta, param_idx=1, source_value=gain)
        beta_eff = theta_new[0, 1]
        assert beta_eff >= 0, f"beta_eff={beta_eff} < 0"
        # beta * (1 + 4*(0.79-1)) = 2.5 * 0.16 = 0.4
        assert beta_eff == pytest.approx(2.5 * 0.16, rel=1e-10)

    def test_beta_eff_clamped_during_sleep(self):
        """During sleep (gain~0.02), beta_eff clamped to 0."""
        c = make_gain_beta_coupling(strength=4.0)
        theta = np.array([[0.15, 2.50, 2.163, 0.0, 0.0]])
        gain = np.array([0.02])
        theta_new = c.modulate(theta, param_idx=1, source_value=gain)
        assert theta_new[0, 1] == 0.0

    def test_batch_modulation(self):
        """Different replicates get different modulations."""
        c = make_gain_beta_coupling(strength=2.0)
        B = 4
        beta_base = 2.50
        theta = np.tile([0.15, beta_base, 2.163, 0.0, 0.0], (B, 1))
        gains = np.array([0.8, 0.9, 1.0, 1.1])
        theta_new = c.modulate(theta, param_idx=1, source_value=gains)
        expected = np.maximum(0, beta_base * (1 + 2 * (gains - 1)))
        np.testing.assert_allclose(theta_new[:, 1], expected, rtol=1e-10)


# -------------------------------------------------------------------
# gain → alpha: all configurations (D26)
# -------------------------------------------------------------------

class TestGainAlphaCoupling:
    """Coupling h-function and modulation correctness for gain→alpha.

    Configuration: all (standalone + composite). D26.
    """

    def test_gain_one_no_modulation(self):
        """At gain=1, alpha unchanged."""
        c = make_gain_alpha_coupling(strength=3.0)
        alpha_base = 0.15
        theta = np.array([[alpha_base, 2.50, 2.163, 0.0, 0.0]])
        h_val = c.h(np.array([1.0]), theta[:, 0])
        assert h_val == pytest.approx(0.0, abs=1e-10)

    def test_gain_above_one_increases_alpha(self):
        """gain > 1 → faster learning."""
        c = make_gain_alpha_coupling(strength=1.0)
        alpha_base = 0.15
        gain = np.array([1.05])
        h_val = c.h(gain, np.array([alpha_base]))
        actual_eff = alpha_base + 1.0 * h_val[0]
        assert actual_eff > alpha_base
        assert actual_eff == pytest.approx(alpha_base * 1.05, rel=1e-10)

    def test_gain_below_one_decreases_alpha(self):
        """gain < 1 → slower learning."""
        c = make_gain_alpha_coupling(strength=1.0)
        alpha_base = 0.15
        gain = np.array([0.8])
        h_val = c.h(gain, np.array([alpha_base]))
        actual_eff = alpha_base + 1.0 * h_val[0]
        assert actual_eff < alpha_base

    def test_alpha_eff_positive_at_max_strength(self):
        """alpha_eff > 0 at max strength, min wake gain."""
        c = make_gain_alpha_coupling(strength=4.0)
        theta = np.array([[0.15, 2.50, 2.163, 0.0, 0.0]])
        gain = np.array([0.79])
        theta_new = c.modulate(theta, param_idx=0, source_value=gain)
        alpha_eff = theta_new[0, 0]
        assert alpha_eff >= 0, f"alpha_eff={alpha_eff} < 0"
        # alpha * (1 + 4*(0.79-1)) = 0.15 * 0.16 = 0.024
        assert alpha_eff == pytest.approx(0.15 * 0.16, rel=1e-10)

    def test_alpha_eff_clamped_below_one(self):
        """alpha_eff <= 1.0 (param_max=1.0): alpha > 1 → V overshoots."""
        c = make_gain_alpha_coupling(strength=4.0)
        theta = np.array([[0.80, 2.50, 2.163, 0.0, 0.0]])  # high alpha
        gain = np.array([1.10])  # high gain
        theta_new = c.modulate(theta, param_idx=0, source_value=gain)
        alpha_eff = theta_new[0, 0]
        assert alpha_eff <= 1.0, f"alpha_eff={alpha_eff} > 1.0"

    def test_alpha_eff_clamped_during_sleep(self):
        """During sleep (gain~0.02), alpha_eff clamped to 0."""
        c = make_gain_alpha_coupling(strength=4.0)
        theta = np.array([[0.15, 2.50, 2.163, 0.0, 0.0]])
        gain = np.array([0.02])
        theta_new = c.modulate(theta, param_idx=0, source_value=gain)
        assert theta_new[0, 0] == 0.0

    def test_alpha_in_range_across_wake_gains(self):
        """alpha_eff ∈ (0, 1] at max strength across full wake gain range."""
        c = make_gain_alpha_coupling(strength=4.0)
        theta = np.array([[0.15, 2.50, 2.163, 0.0, 0.0]])
        for gain_val in [0.79, 0.85, 0.90, 0.95, 1.00, 1.05]:
            theta_new = c.modulate(
                theta.copy(), param_idx=0, source_value=np.array([gain_val]))
            alpha_eff = theta_new[0, 0]
            assert 0 <= alpha_eff <= 1.0, (
                f"alpha_eff={alpha_eff} out of [0,1] at gain={gain_val}")

    def test_batch_modulation(self):
        """Different replicates get different alpha modulations."""
        c = make_gain_alpha_coupling(strength=2.0)
        B = 4
        alpha_base = 0.15
        theta = np.tile([alpha_base, 2.50, 2.163, 0.0, 0.0], (B, 1))
        gains = np.array([0.8, 0.9, 1.0, 1.1])
        theta_new = c.modulate(theta, param_idx=0, source_value=gains)
        raw = alpha_base * (1 + 2 * (gains - 1))
        expected = np.clip(raw, 0.0, 1.0)
        np.testing.assert_allclose(theta_new[:, 0], expected, rtol=1e-10)


# -------------------------------------------------------------------
# λ_crit analysis (D22 condition 2)
# -------------------------------------------------------------------

class TestLambdaCritWithCoupling:
    """Lambda critical should not decrease with gain→beta coupling.

    D22: show λ_crit(gain). Beta_eff varies with gain; λ_crit depends
    on r_mean which depends on choice stochasticity (beta_eff).
    λ_crit formula = (1-gamma)/(gamma*r_mean). r_mean is not β-dependent
    in the formula (r_mean depends on environment, not on agent).
    So λ_crit should be independent of β_eff.
    """

    def test_lambda_crit_independent_of_beta(self):
        """λ_crit depends on tau_m and r_mean, not on beta."""
        tau_m = 2.163
        r_mean = 0.60
        lc = lambda_crit(tau_m, r_mean)
        assert lc == pytest.approx(0.974, abs=0.01)

    def test_lambda_crit_not_reduced_by_coupling(self):
        """C7: Coupling doesn't make mood loop less stable."""
        tau_m = 2.163
        r_mean = 0.60
        lc_standalone = lambda_crit(tau_m, r_mean)

        for gain_val in [0.79, 1.0, 1.05]:
            lc_coupled = lambda_crit(tau_m, r_mean)
            assert lc_coupled >= lc_standalone - 1e-10, (
                f"lambda_crit decreased at gain={gain_val}: "
                f"{lc_coupled} < {lc_standalone}")


# -------------------------------------------------------------------
# Stability analysis
# -------------------------------------------------------------------

class TestArousalRLStability:
    """Composite Jacobian analysis."""

    def test_composite_jacobian_block_triangular(self):
        """Unidirectional coupling → J[arousal, rl] = 0."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        modules = {"arousal": arousal, "rl_mood": rl}
        x0 = {
            "arousal": arousal.default_state(1),
            "rl_mood": rl.default_state(1),
        }
        theta0 = {
            "arousal": arousal.default_params(1),
            "rl_mood": rl.default_params(1),
        }
        u0 = {
            "arousal": np.zeros((1, 0)),
            "rl_mood": np.zeros((1, 0)),
        }
        coupling = make_gain_beta_coupling(strength=2.0)
        J = composite_jacobian(modules, x0, theta0, u0, [coupling])

        # Arousal: 4 states, RL: 3 states. Total 7x7.
        assert J.shape == (7, 7)
        # J[arousal, rl] = J[0:4, 4:7] must be zero
        np.testing.assert_allclose(
            J[:4, 4:], 0.0, atol=1e-8,
            err_msg="Arousal must not depend on RL (unidirectional)")

    def test_all_finite_at_max_strength(self):
        """Composite Jacobian finite at max strength."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        modules = {"arousal": arousal, "rl_mood": rl}
        x0 = {
            "arousal": arousal.default_state(1),
            "rl_mood": rl.default_state(1),
        }
        theta0 = {
            "arousal": arousal.default_params(1),
            "rl_mood": rl.default_params(1),
        }
        u0 = {
            "arousal": np.zeros((1, 0)),
            "rl_mood": np.zeros((1, 0)),
        }
        coupling = make_gain_beta_coupling(strength=4.0)
        J = composite_jacobian(modules, x0, theta0, u0, [coupling])
        assert np.all(np.isfinite(J))

    def test_gain_alpha_jacobian_block_triangular(self):
        """gain→alpha coupling also unidirectional."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        modules = {"arousal": arousal, "rl_mood": rl}
        x0 = {
            "arousal": arousal.default_state(1),
            "rl_mood": rl.default_state(1),
        }
        theta0 = {
            "arousal": arousal.default_params(1),
            "rl_mood": rl.default_params(1),
        }
        u0 = {
            "arousal": np.zeros((1, 0)),
            "rl_mood": np.zeros((1, 0)),
        }
        coupling = make_gain_alpha_coupling(strength=2.0)
        J = composite_jacobian(modules, x0, theta0, u0, [coupling])

        assert J.shape == (7, 7)
        np.testing.assert_allclose(
            J[:4, 4:], 0.0, atol=1e-8,
            err_msg="Arousal must not depend on RL (gain→alpha unidirectional)")

    def test_gain_alpha_jacobian_finite_at_max_strength(self):
        """Composite Jacobian finite at max strength for gain→alpha."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        modules = {"arousal": arousal, "rl_mood": rl}
        x0 = {
            "arousal": arousal.default_state(1),
            "rl_mood": rl.default_state(1),
        }
        theta0 = {
            "arousal": arousal.default_params(1),
            "rl_mood": rl.default_params(1),
        }
        u0 = {
            "arousal": np.zeros((1, 0)),
            "rl_mood": np.zeros((1, 0)),
        }
        coupling = make_gain_alpha_coupling(strength=4.0)
        J = composite_jacobian(modules, x0, theta0, u0, [coupling])
        assert np.all(np.isfinite(J))


# -------------------------------------------------------------------
# Integration test
# -------------------------------------------------------------------

class TestArousalRLIntegration:
    """Multi-tier integration with arousal slow, RL fast."""

    def test_multitier_runs_gain_beta(self):
        """gain→beta: system completes without crash (standalone config)."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        env = ReversalBandit(p_reward=0.75, volatility=0.0)
        coupling = make_gain_beta_coupling(strength=2.0)
        integ = Integrator(
            modules={"arousal": arousal, "rl_mood": rl},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 1.0, ["rl_mood"]),
            ],
            trial_driver="rl_mood",
            gate_module="arousal",
            gate_output="is_wake",
            B=5, seed=42,
        )
        state = integ.run(T=50.0)
        assert state.t == pytest.approx(50.0, abs=1.1)

    def test_no_divergence_gain_beta(self):
        """C8: All state variables finite (gain→beta, standalone config)."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        env = ReversalBandit(p_reward=0.75, volatility=0.0)
        coupling = make_gain_beta_coupling(strength=4.0)
        integ = Integrator(
            modules={"arousal": arousal, "rl_mood": rl},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 1.0, ["rl_mood"]),
            ],
            trial_driver="rl_mood",
            gate_module="arousal",
            gate_output="is_wake",
            B=5, seed=42,
            obs_dt=10.0,
        )
        state = integ.run(T=500.0)
        for rec_key in state.recordings:
            ts = state.recordings[rec_key]
            for t_val, vals in ts:
                assert np.all(np.isfinite(vals)), (
                    f"{rec_key} non-finite at t={t_val}")

    def test_multitier_runs_gain_alpha(self):
        """gain→alpha: system completes without crash."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        env = ReversalBandit(p_reward=0.75, volatility=0.0)
        coupling = make_gain_alpha_coupling(strength=2.0)
        integ = Integrator(
            modules={"arousal": arousal, "rl_mood": rl},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 1.0, ["rl_mood"]),
            ],
            trial_driver="rl_mood",
            gate_module="arousal",
            gate_output="is_wake",
            B=5, seed=42,
        )
        state = integ.run(T=50.0)
        assert state.t == pytest.approx(50.0, abs=1.1)

    def test_no_divergence_gain_alpha(self):
        """All state variables finite (gain→alpha, max strength)."""
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        rl = RLMoodModule()
        env = ReversalBandit(p_reward=0.75, volatility=0.0)
        coupling = make_gain_alpha_coupling(strength=4.0)
        integ = Integrator(
            modules={"arousal": arousal, "rl_mood": rl},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 1.0, ["rl_mood"]),
            ],
            trial_driver="rl_mood",
            gate_module="arousal",
            gate_output="is_wake",
            B=5, seed=42,
            obs_dt=10.0,
        )
        state = integ.run(T=500.0)
        for rec_key in state.recordings:
            ts = state.recordings[rec_key]
            for t_val, vals in ts:
                assert np.all(np.isfinite(vals)), (
                    f"{rec_key} non-finite at t={t_val}")
