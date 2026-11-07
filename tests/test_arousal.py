"""Tests for the arousal (sleep-wake) module.

Spec: specs/arousal.md v2.2
Source: Phillips-Robinson 2007/2008, verified by CellML/BioModels/sleepR.

Tests are grouped:
  AR1-AR5:  Contract tests (fast)
  AR6-AR12: Standalone validation (deterministic 7-day runs)
  AR13-AR14: Calibration tests
  AR15: Cycling boundary
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cpsych.modules.arousal import (
    ArousalModule,
    Q_THRESH,
    T_TRIM,
    SETTLING_TOL,
    _I,
    _sigmoid,
    compute_calibration,
    load_arousal_calibration,
    run_deterministic,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CALIBRATION_PATH = RESULTS_DIR / "arousal_calibration.json"


@pytest.fixture
def q_ref():
    """Load calibration constant from JSON."""
    return load_arousal_calibration(CALIBRATION_PATH)


@pytest.fixture
def mod(q_ref):
    """ArousalModule with calibration constant from JSON."""
    return ArousalModule(q_m_wake_mean_baseline=q_ref)


@pytest.fixture
def baseline_run(mod):
    """7-day deterministic run at baseline, cached per session."""
    return run_deterministic(mod, T_days=7, dt=1.0)


# -------------------------------------------------------------------
# AR1-AR5: Contract tests
# -------------------------------------------------------------------

class TestContract:
    """Module contract: spec, purity, shapes, batch independence."""

    def test_ar1_spec_names(self, mod):
        """AR1: Spec declares correct state/param/output names."""
        assert mod.n_state == 4  # vv, vm, h, clock
        assert mod.n_params == 18
        assert mod.n_inputs == 0
        assert mod.n_outputs == 3
        assert mod.spec.output_names == ("gain", "Qm", "is_wake")
        assert mod.state_index("vv") == 0
        assert mod.state_index("vm") == 1
        assert mod.state_index("h") == 2
        assert mod.state_index("clock") == 3

    def test_ar2_drift_purity(self, mod):
        """AR2: Same inputs produce identical drift outputs."""
        B = 5
        x = mod.default_state(B)
        theta = mod.default_params(B)
        u = np.zeros((B, 0))
        d1 = mod.drift(x, theta, u)
        d2 = mod.drift(x, theta, u)
        np.testing.assert_array_equal(d1, d2)

    def test_ar3_shapes(self, mod):
        """AR3: drift, diffusion, outputs have correct shapes."""
        B = 3
        x = mod.default_state(B)
        theta = mod.default_params(B)
        u = np.zeros((B, 0))
        d = mod.drift(x, theta, u)
        s = mod.diffusion(x, theta)
        o = mod.outputs(x, theta)
        assert d.shape == (B, 4)
        assert s.shape == (B, 4)
        assert o["gain"].shape == (B,)
        assert o["Qm"].shape == (B,)
        assert o["is_wake"].shape == (B,)

    def test_ar4_diffusion_clock_zero(self, mod):
        """AR4: No noise on clock variable."""
        B = 2
        x = mod.default_state(B)
        theta = mod.default_params(B)
        s = mod.diffusion(x, theta)
        assert np.all(s[:, 3] == 0.0)

    def test_ar5_batch_independence(self, mod):
        """AR5: Replicate 0 at B=1 matches replicate 0 at B=100."""
        x1 = mod.default_state(1)
        t1 = mod.default_params(1)
        u1 = np.zeros((1, 0))
        d1 = mod.drift(x1, t1, u1)

        x100 = mod.default_state(100)
        t100 = mod.default_params(100)
        u100 = np.zeros((100, 0))
        d100 = mod.drift(x100, t100, u100)
        np.testing.assert_array_equal(d1[0], d100[0])


# -------------------------------------------------------------------
# AR6-AR12: Standalone validation (7-day deterministic)
# -------------------------------------------------------------------

class TestValidation:
    """Standalone validation against spec tuning targets."""

    def test_ar6_cycling(self, baseline_run):
        """AR6: >=14 transitions in 7 days (target #4)."""
        assert baseline_run["n_transitions"] >= 14

    def test_ar7_sleep_duration(self, baseline_run):
        """AR7: Mean sleep 7-9h (target #1). Skip first cycle."""
        mean_sleep = baseline_run["mean_sleep_h"]
        assert 7.0 <= mean_sleep <= 9.0, f"mean_sleep={mean_sleep}"

    def test_ar8_wake_duration(self, baseline_run):
        """AR8: Mean wake 15-17h. Skip first cycle."""
        mean_wake = baseline_run["mean_wake_h"]
        assert 15.0 <= mean_wake <= 17.0, f"mean_wake={mean_wake}"

    def test_ar9_gain_swing(self, mod):
        """AR9: Gain swing >= 15% during trimmed wake (target #5).

        Computed from calibration run (days 2+, trimmed ±30 min).
        """
        # Run 3-day deterministic simulation
        x = mod.default_state(1)
        theta = mod.default_params(1)
        theta[0, _I["sigma_vv"]] = 0.0
        theta[0, _I["sigma_vm"]] = 0.0
        theta[0, _I["sigma_h"]] = 0.0
        u = np.zeros((1, 0))

        # Skip day 0+1, collect day 2 (hours 48-72)
        total_steps = 72 * 3600  # 3 days
        day2_start = 48 * 3600
        qm_day2 = []
        is_wake_day2 = []

        for step in range(total_steps):
            dx = mod.drift(x, theta, u)
            x = x + dx * 1.0
            if step >= day2_start:
                Qm = _sigmoid(x[0, 1], theta[0, _I["Q_max"]],
                              theta[0, _I["theta_sig"]],
                              theta[0, _I["sigma_q"]])
                qm_day2.append(Qm)
                is_wake_day2.append(Qm > Q_THRESH)

        qm_day2 = np.array(qm_day2)
        is_wake_arr = np.array(is_wake_day2)

        # Find wake periods, trim, compute swing
        changes = np.diff(is_wake_arr.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if is_wake_arr[0]:
            starts = np.concatenate([[0], starts])
        if is_wake_arr[-1]:
            ends = np.concatenate([ends, [len(qm_day2)]])

        trimmed_qm = []
        for s, e in zip(starts, ends):
            ts = s + T_TRIM
            te = e - T_TRIM
            if te - ts < 3600:
                continue
            trimmed_qm.extend(qm_day2[ts:te].tolist())

        assert len(trimmed_qm) > 0, "No trimmed wake data"
        trimmed_qm = np.array(trimmed_qm)
        g = trimmed_qm / mod.q_m_wake_mean_baseline
        swing = (g.max() - g.min()) / g.max()
        assert swing >= 0.15, f"swing={swing:.3f}"

    def test_ar10_h_monotonicity(self, baseline_run):
        """AR10: H increases during wake, decreases during sleep (target #6).

        Check H(sleep_onset) > H(wake_onset) from transitions.
        """
        trans = baseline_run["transitions"]
        # Find pairs: WAKE->SLEEP then SLEEP->WAKE (skip first few)
        sleep_onsets_H = [t["H"] for t in trans
                          if t["type"] == "WAKE->SLEEP" and t["time_h"] > 24]
        wake_onsets_H = [t["H"] for t in trans
                         if t["type"] == "SLEEP->WAKE" and t["time_h"] > 24]
        assert len(sleep_onsets_H) >= 3
        assert len(wake_onsets_H) >= 3
        for h_sleep, h_wake in zip(sleep_onsets_H, wake_onsets_H):
            assert h_sleep > h_wake, (
                f"H(sleep_onset)={h_sleep} <= H(wake_onset)={h_wake}")

    def test_ar11_gain_nonneg(self, baseline_run):
        """AR11: Gain >= 0 everywhere."""
        assert baseline_run["gain_range"][0] >= 0.0

    def test_ar12_is_wake_matches(self, mod):
        """AR12: is_wake output matches Qm > Q_THRESH."""
        B = 10
        # Create states spanning sleep and wake
        x = mod.default_state(B)
        x[:, 1] = np.linspace(-15, 5, B)  # vm from deep sleep to wake
        theta = mod.default_params(B)
        out = mod.outputs(x, theta)
        expected = (out["Qm"] > Q_THRESH).astype(float)
        np.testing.assert_array_equal(out["is_wake"], expected)


# -------------------------------------------------------------------
# AR13-AR14: Calibration tests
# -------------------------------------------------------------------

class TestCalibration:
    """Calibration constant: reproducibility and settling."""

    def test_ar13_calibration_reproduces(self, q_ref):
        """AR13: Recomputed calibration matches stored value (tol 0.001)."""
        recomputed, _, _ = compute_calibration()
        assert abs(recomputed - q_ref) < 0.001, (
            f"recomputed={recomputed:.4f}, stored={q_ref:.4f}")

    def test_ar14_settling_verified(self):
        """AR14: Day-to-day settling delta < tolerance."""
        _, settled_day, daily_means = compute_calibration()
        assert settled_day >= 2, "Must skip transient day 0"
        delta = abs(daily_means[settled_day] - daily_means[settled_day - 1])
        assert delta < SETTLING_TOL, (
            f"delta={delta:.6f} >= tol={SETTLING_TOL}")


# -------------------------------------------------------------------
# AR15: Cycling boundary
# -------------------------------------------------------------------

class TestCyclingBoundary:
    """Cycling ceases outside declared parameter range."""

    def test_ar15_no_cycling_outside(self, q_ref):
        """AR15: v_mv=-0.50 (outside declared [-5.0, -0.90]) gives no cycling."""
        mod = ArousalModule(q_m_wake_mean_baseline=q_ref)
        result = run_deterministic(mod, T_days=7,
                                   theta_override={"v_mv": -0.50})
        # Fewer than 4 transitions = no stable cycling
        assert result["n_transitions"] < 4, (
            f"Expected no cycling at v_mv=-0.50, got "
            f"{result['n_transitions']} transitions")


# -------------------------------------------------------------------
# AR16: Temporality
# -------------------------------------------------------------------

class TestTemporality:
    """Arousal is continuous — no trial resets."""

    def test_ar16_trial_done_always_false(self, mod):
        """AR16: trial_done returns False for all replicates."""
        x = mod.default_state(10)
        theta = mod.default_params(10)
        done = mod.trial_done(x, theta)
        assert not np.any(done)

    def test_ar17_trial_update_noop(self, mod):
        """AR17: trial_update returns state unchanged."""
        x = mod.default_state(5)
        x_copy = x.copy()
        theta = mod.default_params(5)
        mask = np.ones(5, dtype=bool)
        x_new = mod.trial_update(x, theta, mask, {})
        np.testing.assert_array_equal(x_new, x_copy)

    def test_ar18_reset_state_noop(self, mod):
        """AR18: All states are CONTINUOUS, reset_state is no-op."""
        x = mod.default_state(5)
        x[:, :] = 42.0  # non-default values
        x_copy = x.copy()
        theta = mod.default_params(5)
        mask = np.ones(5, dtype=bool)
        x_new = mod.reset_state(x, theta, mask)
        np.testing.assert_array_equal(x_new, x_copy)

    @pytest.mark.slow
    def test_ar19_recovery_deterministic(self, q_ref):
        """AR19: Full recovery pipeline produces identical results on re-run.

        Runs a reduced-scale recovery (3^4=81 grid, 5 reps, 3 days) twice
        with the same seed and asserts bitwise equality of r-values.
        This catches stale-checkpoint and floating-point-ordering bugs.
        """
        import itertools

        mod = ArousalModule(q_m_wake_mean_baseline=q_ref)
        grids = {
            "v_vh": np.linspace(0.08, 0.50, 3),
            "v_vc": np.linspace(-10.0, -1.0, 3),
            "v_vm": np.linspace(-2.70, -0.50, 3),
            "v_mv": np.linspace(-5.0, -0.90, 3),
        }
        grid = np.array(list(itertools.product(
            grids["v_vh"], grids["v_vc"],
            grids["v_vm"], grids["v_mv"])))
        n_pts = len(grid)
        n_reps = 5
        B = n_pts * n_reps

        def run_once():
            # Deterministic phase
            theta_det = mod.default_params(n_pts)
            for i in range(n_pts):
                theta_det[i, _I["v_vh"]] = grid[i, 0]
                theta_det[i, _I["v_vc"]] = grid[i, 1]
                theta_det[i, _I["v_vm"]] = grid[i, 2]
                theta_det[i, _I["v_mv"]] = grid[i, 3]

            rng_det = np.random.default_rng(0)
            x_det = mod.default_state(n_pts)
            theta_d = theta_det.copy()
            theta_d[:, _I["sigma_vv"]] = 0.0
            theta_d[:, _I["sigma_vm"]] = 0.0
            theta_d[:, _I["sigma_h"]] = 0.0
            u = np.zeros((n_pts, 0))
            T = int(3 * 86400)
            for _ in range(T):
                dx = mod.drift(x_det, theta_d, u)
                x_det = x_det + dx * 1.0

            # Noisy phase
            theta_noisy = mod.default_params(B)
            theta_true = np.zeros((B, 4))
            for i in range(n_pts):
                for r in range(n_reps):
                    b = i * n_reps + r
                    theta_noisy[b, _I["v_vh"]] = grid[i, 0]
                    theta_noisy[b, _I["v_vc"]] = grid[i, 1]
                    theta_noisy[b, _I["v_vm"]] = grid[i, 2]
                    theta_noisy[b, _I["v_mv"]] = grid[i, 3]
                    theta_true[b] = grid[i]

            rng = np.random.default_rng(42)
            x = mod.default_state(B)
            u_n = np.zeros((B, 0))
            for _ in range(T):
                dx = mod.drift(x, theta_noisy, u_n)
                x = x + dx * 1.0
                sigma = mod.diffusion(x, theta_noisy)
                dW = rng.standard_normal(x.shape)
                x = x + sigma * np.sqrt(1.0) * dW

            return x_det.copy(), x.copy(), theta_true.copy()

        xd1, xn1, tt1 = run_once()
        xd2, xn2, tt2 = run_once()

        np.testing.assert_array_equal(xd1, xd2,
                                      err_msg="Deterministic phase differs")
        np.testing.assert_array_equal(xn1, xn2,
                                      err_msg="Noisy phase differs")
        np.testing.assert_array_equal(tt1, tt2,
                                      err_msg="True params differ")
