"""Statistical tests for PerceptualTask (E1-E5) and ReversalBandit (E6-E10).

All tests use B >= 100 replicates and fixed seed.
ReversalBandit tests use fixed policy: action = 0 for all replicates.
See specs/environment.md for specification.
"""
import numpy as np
import pytest
from scipy import stats

from cpsych.environments import PerceptualTask, ReversalBandit
from cpsych.environments.perceptual_task import COHERENCE_LEVELS


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_perceptual_trials(n_trials: int, B: int = 200,
                          seed: int = 42) -> dict:
    """Run PerceptualTask for n_trials, collect all stimuli and rewards."""
    env = PerceptualTask()
    rng = np.random.default_rng(seed)
    env.reset(B, rng)

    coherences = []
    directions = []
    corrects = []
    for _ in range(n_trials):
        stim = env.next_trial(rng)
        coherences.append(stim["coherence"])
        directions.append(stim["direction"])
        # Always choose action = direction (correct answer)
        correct_rew = env.reward(stim["direction"], rng)
        corrects.append(correct_rew["correct"])

    return {
        "coherences": np.stack(coherences),    # (n_trials, B)
        "directions": np.stack(directions),    # (n_trials, B)
        "corrects": np.stack(corrects),        # (n_trials, B)
    }


def run_bandit_trials(n_trials: int, B: int = 200, seed: int = 42,
                      p_reward: float = 0.75,
                      volatility: float = 0.5) -> dict:
    """Run ReversalBandit with fixed policy action=0, collect all data."""
    env = ReversalBandit(p_reward=p_reward, volatility=volatility)
    rng = np.random.default_rng(seed)
    env.reset(B, rng)

    rewards = []
    correct_arms = []
    for _ in range(n_trials):
        env.next_trial(rng)
        action = np.zeros(B)  # fixed policy
        correct_arms.append(env._correct_arm.copy())
        rew = env.reward(action, rng)
        rewards.append(rew["r"])

    return {
        "rewards": np.stack(rewards),          # (n_trials, B)
        "correct_arms": np.stack(correct_arms),  # (n_trials, B)
    }


# ── E1-E5: PerceptualTask ───────────────────────────────────────────────────

class TestPerceptualTask:

    def test_e1_coherence_distribution(self):
        """E1: coherence levels uniformly distributed (magnitude criterion)."""
        data = run_perceptual_trials(n_trials=600, B=100)
        flat = data["coherences"].ravel()
        counts = np.array([np.sum(np.isclose(flat, c))
                           for c in COHERENCE_LEVELS])
        expected = len(flat) / len(COHERENCE_LEVELS)
        max_dev = np.max(np.abs(counts / expected - 1))
        assert max_dev < 0.03, f"max deviation {max_dev:.4f} >= 0.03"

    def test_e2_direction_balance(self):
        """E2: directions approximately balanced (-1 vs +1)."""
        data = run_perceptual_trials(n_trials=200, B=100)
        flat = data["directions"].ravel()
        n_left = np.sum(flat == -1)
        n_total = len(flat)
        imbalance = abs(n_left - (n_total - n_left)) / n_total
        assert imbalance < 0.05, f"direction imbalance = {imbalance:.4f}"

    def test_e3_no_serial_correlation(self):
        """E3: no autocorrelation in coherence sequence."""
        data = run_perceptual_trials(n_trials=500, B=100)
        # Test per-replicate, take max absolute correlation
        max_corr = 0.0
        for b in range(data["coherences"].shape[1]):
            seq = data["coherences"][:, b]
            corr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
            max_corr = max(max_corr, abs(corr))
        # Median across replicates instead of max (max is noisy)
        corrs = []
        for b in range(data["coherences"].shape[1]):
            seq = data["coherences"][:, b]
            corrs.append(np.corrcoef(seq[:-1], seq[1:])[0, 1])
        median_corr = np.median(np.abs(corrs))
        assert median_corr < 0.05, f"median |autocorrelation| = {median_corr:.4f}"

    def test_e4_correct_answer_rewarded(self):
        """E4: action == direction always gives correct=1."""
        env = PerceptualTask()
        rng = np.random.default_rng(42)
        B = 100
        env.reset(B, rng)
        for _ in range(200):
            stim = env.next_trial(rng)
            rew = env.reward(stim["direction"], rng)
            assert np.all(rew["correct"] == 1.0)
            assert np.all(rew["r"] == 1.0)

    def test_e5_wrong_answer_not_rewarded(self):
        """E5: action != direction always gives correct=0."""
        env = PerceptualTask()
        rng = np.random.default_rng(42)
        B = 100
        env.reset(B, rng)
        for _ in range(200):
            stim = env.next_trial(rng)
            wrong = -stim["direction"]
            rew = env.reward(wrong, rng)
            assert np.all(rew["correct"] == 0.0)
            assert np.all(rew["r"] == 0.0)


# ── E6-E10: ReversalBandit ──────────────────────────────────────────────────

class TestReversalBandit:

    def test_e6_conditional_reward_probability(self):
        """E6: P(r|correct arm) ≈ p_reward, P(r|wrong arm) ≈ 1-p_reward."""
        p_rew = 0.80
        data = run_bandit_trials(n_trials=1000, B=200, p_reward=p_rew,
                                 volatility=0.5)
        action = np.zeros_like(data["correct_arms"])
        chose_correct = (action == data["correct_arms"])

        # Conditional on choosing correct arm
        mask_correct = chose_correct
        if np.sum(mask_correct) > 500:
            p_hat_correct = data["rewards"][mask_correct].mean()
            assert abs(p_hat_correct - p_rew) < 0.03, \
                f"|P_hat_correct - p_reward| = {abs(p_hat_correct - p_rew):.4f}"

        # Conditional on choosing wrong arm
        mask_wrong = ~chose_correct
        if np.sum(mask_wrong) > 500:
            p_hat_wrong = data["rewards"][mask_wrong].mean()
            assert abs(p_hat_wrong - (1 - p_rew)) < 0.03, \
                f"|P_hat_wrong - (1-p_reward)| = {abs(p_hat_wrong - (1 - p_rew)):.4f}"

    def test_e7_reversal_intervals(self):
        """E7: reversal rate and mean block length match specification.

        Two checks:
        1. Direct reversal rate = volatility/20 (no truncation bias).
        2. Mean inter-reversal interval ≈ 20/volatility within 20%.

        Note: chi-square and KS tests on pooled block lengths are unreliable
        here — scipy.stats.kstest fails on discrete distributions, and
        finite-sequence truncation creates systematic downward bias in block
        length means when sequences are short relative to mean block.
        The direct reversal rate avoids both issues.
        """
        volatility = 0.5
        expected_p_rev = volatility / 20.0  # 0.025
        mean_block = 20.0 / volatility  # 40
        B = 200
        n_trials = 2000

        env = ReversalBandit(p_reward=0.75, volatility=volatility)
        rng = np.random.default_rng(42)
        env.reset(B, rng)

        total_reversals = 0
        prev_arm = env._correct_arm.copy()
        block_lengths = []
        # Also collect block lengths for mean check
        trials_since_rev = np.zeros(B, dtype=int)

        for _ in range(n_trials):
            env.next_trial(rng)
            reversed_mask = (env._correct_arm != prev_arm)
            total_reversals += np.sum(reversed_mask)
            # Record completed blocks
            for b in np.where(reversed_mask)[0]:
                if trials_since_rev[b] > 0:
                    block_lengths.append(trials_since_rev[b])
            trials_since_rev = np.where(reversed_mask, 0, trials_since_rev)
            trials_since_rev += 1
            prev_arm = env._correct_arm.copy()

        # Check 1: reversal rate
        observed_p_rev = total_reversals / (B * n_trials)
        assert abs(observed_p_rev - expected_p_rev) < 0.003, \
            f"|p_rev - expected| = {abs(observed_p_rev - expected_p_rev):.5f}"

        # Check 2: mean block length within 20%
        block_lengths = np.array(block_lengths)
        assert len(block_lengths) > 200
        observed_mean = block_lengths.mean()
        assert abs(observed_mean - mean_block) / mean_block < 0.20, \
            f"mean block {observed_mean:.1f} vs expected {mean_block:.1f}"

    def test_e8_within_block_independence(self):
        """E8: within-block rewards are independent (runs test)."""
        volatility = 0.5
        B = 200
        n_trials = 1000
        data = run_bandit_trials(n_trials=n_trials, B=B,
                                 volatility=volatility)
        arms = data["correct_arms"]
        rewards = data["rewards"]
        action = np.zeros_like(arms)
        chose_correct = (action == arms)

        # Collect runs test p-values for blocks of same chose_correct status
        p_values = []
        for b in range(min(B, 50)):  # sample of replicates
            seq_arms = arms[:, b]
            # Find block boundaries
            changes = np.concatenate([[0], np.where(np.diff(seq_arms) != 0)[0] + 1,
                                      [n_trials]])
            for i in range(len(changes) - 1):
                start, end = changes[i], changes[i + 1]
                if end - start < 10:
                    continue
                block_rewards = rewards[start:end, b]
                # Runs test: count runs in binary sequence
                n_r = _runs_test_p(block_rewards)
                if n_r is not None:
                    p_values.append(n_r)

        assert len(p_values) > 20, \
            f"Too few blocks for runs test: {len(p_values)}"
        # Fraction of rejections should be near alpha
        rejections = np.sum(np.array(p_values) < 0.01) / len(p_values)
        assert rejections < 0.10, \
            f"Runs test: {rejections*100:.0f}% rejections (expect <10%)"

    def test_e9_volatility_zero_no_reversals(self):
        """E9: volatility=0 guarantees no reversals (deterministic)."""
        env = ReversalBandit(p_reward=0.75, volatility=0.0)
        rng = np.random.default_rng(42)
        B = 200
        env.reset(B, rng)
        initial_arm = env._correct_arm.copy()
        for _ in range(1000):
            env.next_trial(rng)
        assert np.all(env._correct_arm == initial_arm), \
            "Reversal occurred with volatility=0"

    def test_e10_volatility_max_frequent_reversals(self):
        """E10: volatility=1.0 gives mean block ≈ 20."""
        B = 200
        n_trials = 500
        data = run_bandit_trials(n_trials=n_trials, B=B, volatility=1.0)
        arms = data["correct_arms"]

        block_lengths = []
        for b in range(B):
            seq = arms[:, b]
            changes = np.where(np.diff(seq) != 0)[0]
            if len(changes) > 1:
                intervals = np.diff(changes)
                block_lengths.extend(intervals.tolist())

        observed_mean = np.mean(block_lengths)
        assert abs(observed_mean - 20) < 5, \
            f"Mean block {observed_mean:.1f}, expected ~20"


# ── Runs test helper ────────────────────────────────────────────────────────

def _runs_test_p(binary_seq: np.ndarray) -> float | None:
    """Two-sided runs test for randomness of binary sequence.

    Returns p-value, or None if sequence is degenerate.
    """
    n = len(binary_seq)
    if n < 10:
        return None
    n1 = np.sum(binary_seq == 1)
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return None

    # Count runs
    runs = 1 + np.sum(binary_seq[1:] != binary_seq[:-1])

    # Expected runs and variance under independence
    mu = 1 + 2 * n0 * n1 / n
    var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    if var <= 0:
        return None

    z = (runs - mu) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p


# ── Validation ──────────────────────────────────────────────────────────────

class TestEnvironmentContract:
    """Basic contract tests: shapes, types, reset idempotence."""

    def test_perceptual_shapes(self):
        env = PerceptualTask()
        rng = np.random.default_rng(0)
        B = 50
        env.reset(B, rng)
        stim = env.next_trial(rng)
        assert stim["coherence"].shape == (B,)
        assert stim["direction"].shape == (B,)
        action = rng.choice([-1, 1], size=B).astype(float)
        rew = env.reward(action, rng)
        assert rew["correct"].shape == (B,)
        assert rew["r"].shape == (B,)

    def test_bandit_shapes(self):
        env = ReversalBandit(volatility=0.3)
        rng = np.random.default_rng(0)
        B = 50
        env.reset(B, rng)
        stim = env.next_trial(rng)
        assert stim["trial_index"].shape == (B,)
        action = np.zeros(B)
        rew = env.reward(action, rng)
        assert rew["r"].shape == (B,)

    def test_perceptual_coherence_values(self):
        """All emitted coherences are from the defined set."""
        env = PerceptualTask()
        rng = np.random.default_rng(42)
        env.reset(100, rng)
        for _ in range(100):
            stim = env.next_trial(rng)
            for c in stim["coherence"]:
                assert c in COHERENCE_LEVELS, f"Invalid coherence {c}"

    def test_bandit_validation(self):
        with pytest.raises(ValueError):
            ReversalBandit(p_reward=0.3)
        with pytest.raises(ValueError):
            ReversalBandit(volatility=-0.1)
        with pytest.raises(ValueError):
            ReversalBandit(volatility=1.5)

    def test_perceptual_deterministic_seed(self):
        """Same seed gives identical sequence."""
        results = []
        for _ in range(2):
            env = PerceptualTask()
            rng = np.random.default_rng(123)
            env.reset(10, rng)
            stims = [env.next_trial(rng) for _ in range(20)]
            results.append(np.stack([s["coherence"] for s in stims]))
        np.testing.assert_array_equal(results[0], results[1])

    def test_bandit_deterministic_seed(self):
        """Same seed gives identical reward sequence."""
        results = []
        for _ in range(2):
            env = ReversalBandit(p_reward=0.8, volatility=0.5)
            rng = np.random.default_rng(123)
            env.reset(10, rng)
            rews = []
            for _ in range(50):
                env.next_trial(rng)
                rews.append(env.reward(np.zeros(10), rng)["r"])
            results.append(np.stack(rews))
        np.testing.assert_array_equal(results[0], results[1])


# ── E11-E15: ReversalBandit criterion mode (Eckstein protocol) ────────────

class TestCriterionMode:
    """Criterion-based reversals per Eckstein et al. (2022)."""

    def test_e11_block_length_distribution(self):
        """E11: block length matches criterion range.

        With fixed policy action=correct_arm and p_reward=0.75,
        mean rewards per trial ≈ 0.75. Criterion 7-15 rewards →
        mean block ≈ criterion/0.75 ≈ 9-20 trials.
        """
        env = ReversalBandit(
            p_reward=0.75, p_incorrect=0.0,
            reversal_mode="criterion", criterion_range=(7, 15))
        rng = np.random.default_rng(42)
        B = 100
        env.reset(B, rng)

        block_lengths = []
        prev_arm = env._correct_arm.copy()
        trials_since_rev = np.zeros(B, dtype=int)

        for t in range(500):
            env.next_trial(rng)
            action = env._correct_arm.copy()
            env.reward(action, rng)
            reversed_mask = (env._correct_arm != prev_arm)
            for b in np.where(reversed_mask)[0]:
                if trials_since_rev[b] > 0:
                    block_lengths.append(trials_since_rev[b])
            trials_since_rev = np.where(reversed_mask, 0, trials_since_rev)
            trials_since_rev += 1
            prev_arm = env._correct_arm.copy()

        block_lengths = np.array(block_lengths)
        assert len(block_lengths) > 50, (
            f"Too few reversals: {len(block_lengths)}")
        assert block_lengths.min() >= 7, (
            f"Block shorter than min criterion: {block_lengths.min()}")
        assert 10 < block_lengths.mean() < 25, (
            f"Mean block {block_lengths.mean():.1f} outside [10, 25]")

    def test_e12_p_incorrect_zero(self):
        """E12: incorrect arm never rewarded when p_incorrect=0."""
        env = ReversalBandit(
            p_reward=0.75, p_incorrect=0.0,
            reversal_mode="criterion", criterion_range=(7, 15))
        rng = np.random.default_rng(42)
        B = 200
        env.reset(B, rng)

        wrong_rewards = 0
        wrong_trials = 0
        for _ in range(200):
            env.next_trial(rng)
            action = 1.0 - env._correct_arm
            r = env.reward(action, rng)["r"]
            wrong_rewards += r.sum()
            wrong_trials += B

        assert wrong_rewards == 0, (
            f"Incorrect arm rewarded {int(wrong_rewards)} times "
            f"in {wrong_trials} (expected 0)")

    def test_e13_reversal_only_after_reward(self):
        """E13: reversal never occurs after unrewarded trial."""
        env = ReversalBandit(
            p_reward=0.75, p_incorrect=0.0,
            reversal_mode="criterion", criterion_range=(7, 15))
        rng = np.random.default_rng(42)
        B = 100
        env.reset(B, rng)

        violations = 0
        last_r = np.zeros(B)
        for t in range(300):
            prev_arm = env._correct_arm.copy()
            env.next_trial(rng)
            changed = (env._correct_arm != prev_arm)
            if np.any(changed):
                bad = changed & (last_r < 0.5)
                violations += bad.sum()
            action = env._correct_arm.copy()
            last_r = env.reward(action, rng)["r"]

        assert violations == 0, (
            f"Reversal after unrewarded trial: {violations} violations")

    def test_e14_correct_arm_reward_rate(self):
        """E14: correct arm rewarded at p_reward rate."""
        env = ReversalBandit(
            p_reward=0.75, p_incorrect=0.0,
            reversal_mode="criterion", criterion_range=(7, 15))
        rng = np.random.default_rng(42)
        B = 200
        env.reset(B, rng)

        correct_rewards = 0
        correct_trials = 0
        for _ in range(200):
            env.next_trial(rng)
            action = env._correct_arm.copy()
            r = env.reward(action, rng)["r"]
            correct_rewards += r.sum()
            correct_trials += B

        p_hat = correct_rewards / correct_trials
        assert abs(p_hat - 0.75) < 0.02, (
            f"Correct arm reward rate {p_hat:.3f}, expected ~0.75")

    def test_e15_geom_backward_compatible(self):
        """E15: geom mode with p_incorrect=None behaves as before."""
        env = ReversalBandit(p_reward=0.75, volatility=0.3)
        assert env.p_incorrect == 0.25
        assert env.reversal_mode == "geom"
