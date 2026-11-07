"""P3.2: Arousal → DDM coupling tests (gain → kappa).

Coupling: kappa_eff = kappa_base * max(0, 1 + s*(gain-1))
Unidirectional. Composite Jacobian block-triangular.
Spec: specs/p3_couplings.md. Decision: D22 (related), A6.
"""
import numpy as np
import pytest

from cpsych.core.coupling import Coupling
from cpsych.core.integrator import Integrator, Tier
from cpsych.core.stability import composite_jacobian, max_real_eigenvalue
from cpsych.modules.arousal import ArousalModule
from cpsych.modules.ddm import DDMModule
from cpsych.environments.perceptual_task import PerceptualTask


# -------------------------------------------------------------------
# Coupling factory
# -------------------------------------------------------------------

def make_gain_kappa_coupling(strength: float = 1.0,
                              max_strength: float = 4.0) -> Coupling:
    """Multiplicative: kappa_eff = kappa * max(0, 1 + s*(gain - 1))."""
    return Coupling(
        source_module="arousal",
        source_output="gain",
        target_module="ddm",
        target_param="kappa",
        h=lambda gain, kappa_base: kappa_base * (gain - 1),
        strength=strength,
        max_strength=max_strength,
        param_min=0.0,
        doc="Multiplicative gain->kappa (A6, specs/p3_couplings.md)",
    )


# -------------------------------------------------------------------
# Unit tests: coupling mechanics
# -------------------------------------------------------------------

class TestGainKappaCoupling:
    """Coupling h-function and modulation correctness."""

    def test_gain_one_no_modulation(self):
        """At gain=1 (baseline), kappa unchanged regardless of strength."""
        c = make_gain_kappa_coupling(strength=3.0)
        theta = np.array([[14.7242, 1.5, 0.318]])  # kappa, a, Ter
        h_val = c.h(np.array([1.0]), theta[:, 0])
        assert h_val == pytest.approx(0.0, abs=1e-10)

    def test_gain_above_one_increases_kappa(self):
        """gain > 1 → kappa_eff > kappa_base."""
        c = make_gain_kappa_coupling(strength=1.0)
        kappa_base = 14.7242
        gain = np.array([1.05])
        h_val = c.h(gain, np.array([kappa_base]))
        # kappa_eff = kappa + 1.0 * kappa*(1.05-1) = kappa*1.05
        expected_eff = kappa_base * 1.05
        actual_eff = kappa_base + 1.0 * h_val[0]
        assert actual_eff == pytest.approx(expected_eff, rel=1e-10)

    def test_gain_below_one_decreases_kappa(self):
        """gain < 1 → kappa_eff < kappa_base."""
        c = make_gain_kappa_coupling(strength=1.0)
        kappa_base = 14.7242
        gain = np.array([0.8])
        h_val = c.h(gain, np.array([kappa_base]))
        actual_eff = kappa_base + 1.0 * h_val[0]
        assert actual_eff < kappa_base
        # kappa_eff = kappa*(1 + 1*(0.8-1)) = kappa*0.8
        assert actual_eff == pytest.approx(kappa_base * 0.8, rel=1e-10)

    def test_kappa_eff_positive_at_max_strength(self):
        """C1: kappa_eff > 0 at max strength, min wake gain (0.79)."""
        c = make_gain_kappa_coupling(strength=4.0)
        kappa_base = np.array([[14.7242, 1.5, 0.318]])
        gain = np.array([0.79])  # min wake gain
        theta_new = c.modulate(kappa_base, param_idx=0, source_value=gain)
        kappa_eff = theta_new[0, 0]
        # kappa_eff = kappa*(1 + 4*(0.79-1)) = kappa*(1-0.84) = kappa*0.16
        assert kappa_eff > 0, f"kappa_eff={kappa_eff} <= 0 at max strength"

    def test_batch_modulation(self):
        """Different replicates get different modulations."""
        c = make_gain_kappa_coupling(strength=2.0)
        B = 4
        kappa_base = 14.7242
        theta = np.tile([kappa_base, 1.5, 0.318], (B, 1))
        gains = np.array([0.8, 0.9, 1.0, 1.1])
        theta_new = c.modulate(theta, param_idx=0, source_value=gains)
        # kappa_eff = kappa * (1 + 2*(gain-1))
        expected = kappa_base * (1 + 2 * (gains - 1))
        np.testing.assert_allclose(theta_new[:, 0], expected, rtol=1e-10)


# -------------------------------------------------------------------
# Stability analysis
# -------------------------------------------------------------------

class TestArousalDDMStability:
    """Composite Jacobian and eigenvalue analysis."""

    @pytest.fixture
    def arousal(self):
        return ArousalModule(q_m_wake_mean_baseline=4.2883)

    @pytest.fixture
    def ddm(self):
        return DDMModule()

    def test_composite_jacobian_block_triangular(self, arousal, ddm):
        """Unidirectional coupling → block-triangular Jacobian.

        Off-diagonal block J[ddm, arousal] may be nonzero (coupling).
        Off-diagonal block J[arousal, ddm] must be zero (no feedback).
        """
        modules = {"arousal": arousal, "ddm": ddm}
        x0 = {
            "arousal": arousal.default_state(1),
            "ddm": ddm.default_state(1),
        }
        theta0 = {
            "arousal": arousal.default_params(1),
            "ddm": ddm.default_params(1),
        }
        u0 = {
            "arousal": np.zeros((1, 0)),
            "ddm": np.array([[0.128, 1.0]]),  # coherence, direction
        }
        coupling = make_gain_kappa_coupling(strength=2.0)
        J = composite_jacobian(modules, x0, theta0, u0, [coupling])

        # Arousal has 4 states, DDM has 2. Total 6x6.
        assert J.shape == (6, 6)

        # J[arousal block, ddm block] = J[0:4, 4:6] must be zero
        np.testing.assert_allclose(
            J[:4, 4:], 0.0, atol=1e-8,
            err_msg="Arousal must not depend on DDM (unidirectional)")

    def test_eigenvalues_independent_of_strength(self, arousal, ddm):
        """Eigenvalues should not change with coupling strength.

        DDM drift ∂/∂evidence = 0 → eigenvalue contribution is 0.
        Arousal eigenvalues unchanged (unidirectional).
        """
        modules = {"arousal": arousal, "ddm": ddm}
        x0 = {
            "arousal": arousal.default_state(1),
            "ddm": ddm.default_state(1),
        }
        theta0 = {
            "arousal": arousal.default_params(1),
            "ddm": ddm.default_params(1),
        }
        u0 = {
            "arousal": np.zeros((1, 0)),
            "ddm": np.array([[0.128, 1.0]]),
        }

        eigs = {}
        for s in [0.0, 1.0, 4.0]:
            coupling = make_gain_kappa_coupling(strength=s)
            J = composite_jacobian(modules, x0, theta0, u0, [coupling])
            eigs[s] = sorted(np.real(np.linalg.eigvals(J)))

        # Compare eigenvalues at s=0 vs s=4
        np.testing.assert_allclose(
            eigs[0.0], eigs[4.0], atol=1e-6,
            err_msg="Eigenvalues should not depend on strength "
                    "(block-triangular, DDM contribution is 0)")


# -------------------------------------------------------------------
# Integration tests (multi-tier)
# -------------------------------------------------------------------

class TestArousalDDMIntegration:
    """Multi-tier integration with arousal slow, DDM fast."""

    @pytest.fixture
    def setup(self):
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        ddm = DDMModule()
        env = PerceptualTask()
        coupling = make_gain_kappa_coupling(strength=2.0)
        return arousal, ddm, env, coupling

    def test_multitier_runs(self, setup):
        """System completes without crash."""
        arousal, ddm, env, coupling = setup
        integ = Integrator(
            modules={"arousal": arousal, "ddm": ddm},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 0.01, ["ddm"]),
            ],
            trial_driver="ddm",
            gate_module="arousal",
            gate_output="is_wake",
            B=5, seed=42,
        )
        state = integ.run(T=100.0)  # ~100 seconds
        assert state.t == pytest.approx(100.0, abs=1.1)

    def test_trials_complete_during_wake(self, setup):
        """C4: DDM completes trials during wake."""
        arousal, ddm, env, coupling = setup
        integ = Integrator(
            modules={"arousal": arousal, "ddm": ddm},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 0.01, ["ddm"]),
            ],
            trial_driver="ddm",
            gate_module="arousal",
            gate_output="is_wake",
            B=10, seed=42,
        )
        # Run 100 seconds (well within first wake period ~16h)
        state = integ.run(T=100.0)
        assert np.all(state.trial_count > 0), (
            f"Expected trials during wake. counts={state.trial_count}")

    def test_no_trials_during_sleep(self, setup):
        """C5: Gate blocks trials during sleep.

        Start from a sleeping state (V_m low) and verify no trials complete
        in a short window.
        """
        arousal, ddm, env, coupling = setup
        integ = Integrator(
            modules={"arousal": arousal, "ddm": ddm},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 0.01, ["ddm"]),
            ],
            trial_driver="ddm",
            gate_module="arousal",
            gate_output="is_wake",
            B=5, seed=42,
        )
        state = integ._init_state()
        # Force sleep state: set V_m to very low (VLPO winning)
        state.x["arousal"][:, 1] = -10.0  # V_m deep in sleep regime
        # Manually step for 50 seconds (should stay asleep)
        stimulus = env.next_trial(integ.rng)
        for _ in range(50):
            integ._step_module("arousal", 1.0, state, stimulus)
            for _ in range(100):
                integ._step_module("ddm", 0.01, state, stimulus)
            integ._handle_trials(1, state, stimulus)
            state.t += 1.0
        assert np.all(state.trial_count == 0), (
            f"Expected no trials during sleep. counts={state.trial_count}")

    def test_accuracy_increases_with_gain(self, setup):
        """C2: Higher gain → higher kappa_eff → better accuracy.

        Uses batch DDM with fixed kappa_eff at three gain levels.
        Higher kappa → stronger drift → better accuracy.
        """
        ddm = DDMModule()

        accuracies = {}
        for gain_val in [0.85, 1.0, 1.1]:
            rng = np.random.default_rng(42)
            B = 200
            kappa_base = 14.7242
            strength = 2.0
            kappa_eff = kappa_base * (1 + strength * (gain_val - 1))

            theta = ddm.default_params(B)
            theta[:, 0] = kappa_eff

            n_correct = 0
            n_total = 0
            for trial in range(50):
                coh = rng.choice([0.064, 0.128, 0.256, 0.512], size=B)
                direction = rng.choice([-1, 1], size=B).astype(float)
                u = np.column_stack([coh, direction])

                x = ddm.default_state(B)
                for step in range(500):
                    dx = ddm.drift(x, theta, u)
                    dW = rng.standard_normal(x.shape)
                    x = x + dx * 0.001 + np.sqrt(0.001) * dW
                    done = ddm.trial_done(x, theta)
                    if np.all(done):
                        break

                choice = np.sign(x[:, 0])
                correct = (choice == direction)
                n_correct += np.sum(correct)
                n_total += B

            accuracies[gain_val] = n_correct / n_total

        assert accuracies[1.0] > accuracies[0.85], (
            f"acc(gain=1.0)={accuracies[1.0]:.3f} <= "
            f"acc(gain=0.85)={accuracies[0.85]:.3f}")
        assert accuracies[1.1] > accuracies[1.0], (
            f"acc(gain=1.1)={accuracies[1.1]:.3f} <= "
            f"acc(gain=1.0)={accuracies[1.0]:.3f}")


# -------------------------------------------------------------------
# Rule 19a: hybrid system transition check
# -------------------------------------------------------------------

class TestHybridTransition:
    """No divergence at wake/sleep transitions (rule 19a)."""

    def test_no_divergence_at_transition(self):
        """C3: All state variables finite through simulation.

        With param_min=0 on kappa coupling, kappa_eff >= 0. During sleep,
        kappa_eff = 0, evidence is pure random walk (bounded in probability,
        not pathologically divergent). trial_clock grows linearly during
        sleep (gate blocks resets) — expected, not divergence.
        """
        arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
        ddm = DDMModule()
        env = PerceptualTask()
        coupling = make_gain_kappa_coupling(strength=4.0)  # max strength

        integ = Integrator(
            modules={"arousal": arousal, "ddm": ddm},
            environment=env,
            couplings=[coupling],
            tiers=[
                Tier("slow", 1.0, ["arousal"]),
                Tier("fast", 0.01, ["ddm"]),
            ],
            trial_driver="ddm",
            gate_module="arousal",
            gate_output="is_wake",
            B=5, seed=42,
            obs_dt=100.0,
        )
        state = integ.run(T=3000.0)

        # Check all state variables are finite (no NaN, no Inf)
        for rec_key in state.recordings:
            ts = state.recordings[rec_key]
            for t_val, vals in ts:
                assert np.all(np.isfinite(vals)), (
                    f"{rec_key} has non-finite values at t={t_val}")

    def test_kappa_eff_clamped_to_zero_during_sleep(self):
        """Without param_min, sleep gain~0.02 at s=4 gives kappa<0.
        With param_min=0, kappa_eff is clamped to 0."""
        c = make_gain_kappa_coupling(strength=4.0)
        kappa_base = np.array([[14.7242, 1.5, 0.318]])
        gain_sleep = np.array([0.02])  # deep sleep
        theta_new = c.modulate(kappa_base, param_idx=0, source_value=gain_sleep)
        # Without clamp: 14.7242*(1+4*(0.02-1)) = 14.7242*(-2.92) < 0
        # With param_min=0: clamped to 0
        assert theta_new[0, 0] == 0.0
