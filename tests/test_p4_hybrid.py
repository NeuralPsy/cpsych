"""P4 Step 2: Hybrid transition tests (Rule 19a).

TestHybridFiniteness: large-batch finiteness at extreme couplings (fast).
TestHybridTransition: sleep-wake cycling verification (slow, shared fixture).
"""
import numpy as np
import pytest

from cpsych.core.composite_setup import make_triple_integrator
from cpsych.modules.arousal import Q_THRESH, _I as arousal_I


def _extract_is_wake_series(state):
    """Extract per-replicate wake/sleep from arousal.vm recordings."""
    vm_recs = state.recordings["arousal.vm"]
    B = vm_recs[0][1].shape[0]

    Q_max, theta_sig, sigma_q = 100.0, 10.0, 3.0
    results = []
    for b in range(B):
        vm_series = np.array([r[1][b] for r in vm_recs])
        times = np.array([r[0] for r in vm_recs])
        Qm = Q_max / (1.0 + np.exp(-(vm_series - theta_sig) / sigma_q))
        is_wake = Qm > Q_THRESH
        results.append((times, is_wake))
    return results


def _count_transitions(is_wake):
    """Count wake→sleep and sleep→wake transitions."""
    changes = np.diff(is_wake.astype(int))
    n_ws = int(np.sum(changes == -1))
    n_sw = int(np.sum(changes == 1))
    return n_ws, n_sw


def _max_transitions_per_hour(times, is_wake):
    """Maximum transitions in any 1-hour window."""
    changes = np.diff(is_wake.astype(int))
    trans_times = times[1:][changes != 0]
    if len(trans_times) == 0:
        return 0
    max_count = 0
    T_total_h = times[-1] / 3600.0
    for t_start in np.arange(0, max(T_total_h - 1, 0.1), 0.5):
        count = np.sum((trans_times / 3600.0 >= t_start) &
                       (trans_times / 3600.0 < t_start + 1.0))
        max_count = max(max_count, count)
    return int(max_count)


class TestHybridFiniteness:
    """Fast finiteness checks: large batch, short T."""

    def test_max_coupling_finite(self):
        """B=200, T=3600, max strengths — all finite."""
        integ = make_triple_integrator(
            B=200, seed=42, obs_dt=600.0,
            s_kappa=4.0, s_alpha=4.0, s_value=10.0)
        state = integ.run(T=600.0)
        for name in ["arousal", "ddm", "rl_mood"]:
            assert np.all(np.isfinite(state.x[name])), (
                f"{name} has non-finite states at max coupling")

    def test_extreme_v_vh_still_finite(self):
        """v_vh at window boundaries (0.30, 0.34), B=100 — no divergence."""
        B_each = 50
        B = B_each * 2
        integ = make_triple_integrator(
            B=B, seed=42, obs_dt=600.0,
            s_kappa=2.0, s_alpha=2.0, s_value=3.0)
        # Override v_vh: first half at 0.30, second half at 0.34
        state = integ._init_state()
        state.theta_base["arousal"][:B_each, arousal_I["v_vh"]] = 0.30
        state.theta_base["arousal"][B_each:, arousal_I["v_vh"]] = 0.34
        state = integ.run(T=600.0, state=state)
        for name in ["arousal", "ddm", "rl_mood"]:
            assert np.all(np.isfinite(state.x[name])), (
                f"{name} non-finite at extreme v_vh")


@pytest.mark.slow
class TestHybridTransition:
    """Hybrid stability: wake/sleep cycling with all couplings.

    Shared fixture: single T=172800 (2-day) simulation.
    Run separately: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def two_day_state(self):
        """B=10, T=172800 (2 days), moderate couplings. Shared by all tests."""
        integ = make_triple_integrator(
            B=10, seed=42, obs_dt=60.0,
            s_kappa=2.0, s_alpha=2.0, s_value=3.0)
        return integ.run(T=172800.0)

    def test_all_states_finite(self, two_day_state):
        """All states finite after 2 days."""
        for name in ["arousal", "ddm", "rl_mood"]:
            assert np.all(np.isfinite(two_day_state.x[name])), (
                f"{name} has non-finite states after 2 days")

    def test_transitions_occur(self, two_day_state):
        """At least 2 wake→sleep and 2 sleep→wake per replicate."""
        for b, (times, is_wake) in enumerate(
                _extract_is_wake_series(two_day_state)):
            n_ws, n_sw = _count_transitions(is_wake)
            assert n_ws >= 2, (
                f"Replicate {b}: only {n_ws} wake→sleep (need ≥2)")
            assert n_sw >= 2, (
                f"Replicate {b}: only {n_sw} sleep→wake (need ≥2)")

    def test_no_spurious_oscillations(self, two_day_state):
        """No >4 transitions in any 1-hour window."""
        for b, (times, is_wake) in enumerate(
                _extract_is_wake_series(two_day_state)):
            mt = _max_transitions_per_hour(times, is_wake)
            assert mt <= 4, (
                f"Replicate {b}: {mt} transitions in 1h (max 4)")

    def test_sleep_duration_in_range(self, two_day_state):
        """Sleep duration 5-12h (relaxed for coupled system)."""
        all_sleep_durs = []
        for b, (times, is_wake) in enumerate(
                _extract_is_wake_series(two_day_state)):
            changes = np.diff(is_wake.astype(int))
            ws_times = times[1:][changes == -1]
            sw_times = times[1:][changes == 1]
            for ws_t in ws_times:
                next_sw = sw_times[sw_times > ws_t]
                if len(next_sw) > 0:
                    dur_h = (next_sw[0] - ws_t) / 3600.0
                    all_sleep_durs.append(dur_h)

        if len(all_sleep_durs) > 2:
            settled_durs = all_sleep_durs[1:]
            mean_dur = np.mean(settled_durs)
            assert 5.0 < mean_dur < 12.0, (
                f"Mean sleep duration {mean_dur:.1f}h outside [5, 12]")
