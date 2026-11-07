"""Behavioral observables from trial-level behavior counters.

Requires Integrator(record_behavior=True).
"""
import numpy as np


def accuracy(behavior: dict) -> np.ndarray:
    """Fraction correct across trials. (B,), NaN where n_trials=0."""
    n = behavior["n_trials"]
    return np.where(n > 0, behavior["n_correct"] / n, np.nan)


def mean_rt(behavior: dict) -> np.ndarray:
    """Mean RT across trials. (B,), NaN where n_trials=0."""
    n = behavior["n_trials"]
    return np.where(n > 0, behavior["sum_rt"] / n, np.nan)


def win_stay_rate(behavior: dict) -> np.ndarray:
    """P(stay | previous win). (B,), NaN where n_wins=0."""
    n = behavior["n_wins"]
    return np.where(n > 0, behavior["n_win_stay"] / n, np.nan)


def trial_count(behavior: dict) -> np.ndarray:
    """Number of completed trials per replicate. (B,)."""
    return behavior["n_trials"]


def sleep_wake_from_recordings(recordings, q_thresh,
                               Q_max=100.0, theta_sig=10.0,
                               sigma_q=3.0):
    """Extract per-replicate wake/sleep series from arousal.vm recordings.

    Parameters
    ----------
    recordings : dict from SimState.recordings
    q_thresh : Q_m threshold for wake (use arousal.Q_THRESH)
    Q_max, theta_sig, sigma_q : sigmoid params (arousal defaults)

    Returns
    -------
    list of (times, is_wake) tuples, one per replicate.
    """
    vm_recs = recordings["arousal.vm"]
    B = vm_recs[0][1].shape[0]
    results = []
    for b in range(B):
        vm_series = np.array([r[1][b] for r in vm_recs])
        times = np.array([r[0] for r in vm_recs])
        Qm = Q_max / (1.0 + np.exp(-(vm_series - theta_sig) / sigma_q))
        is_wake = Qm > q_thresh
        results.append((times, is_wake))
    return results


def gain_range_wake(recordings, q_thresh, q_ref,
                    Q_max=100.0, theta_sig=10.0, sigma_q=3.0):
    """Gain range during wake periods, per replicate.

    Returns (B,) array of gain_max - gain_min during wake.
    NaN if no wake samples.
    """
    vm_recs = recordings["arousal.vm"]
    B = vm_recs[0][1].shape[0]
    ranges = np.full(B, np.nan)
    for b in range(B):
        vm_series = np.array([r[1][b] for r in vm_recs])
        Qm = Q_max / (1.0 + np.exp(-(vm_series - theta_sig) / sigma_q))
        is_wake = Qm > q_thresh
        if np.any(is_wake):
            gain_wake = Qm[is_wake] / q_ref
            ranges[b] = np.max(gain_wake) - np.min(gain_wake)
    return ranges
