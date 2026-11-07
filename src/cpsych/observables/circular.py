"""Circular statistics for phase-like observables.

Used for observables declared in ModuleSpec.circular_observables
(e.g. sleep onset time-of-day, onset_advance).
"""
import numpy as np


def circular_mean(values, period=24.0):
    """Circular mean of values with given period.

    Returns mean in [0, period).
    """
    theta = values * 2 * np.pi / period
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    mean_angle = np.arctan2(S, C)
    return (mean_angle * period / (2 * np.pi)) % period


def circular_sd(values, period=24.0):
    """Circular SD (Mardia & Jupp) of values with given period.

    Returns SD in same units as period.
    """
    theta = values * 2 * np.pi / period
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C ** 2 + S ** 2)
    if R < 1e-10:
        return float("inf")
    return np.sqrt(-2 * np.log(R)) * period / (2 * np.pi)


def circular_diff(a, b, period=24.0):
    """Signed shortest-arc difference a - b on circle of given period.

    Returns values in [-period/2, period/2).
    """
    d = (np.asarray(a) - np.asarray(b)) % period
    return np.where(d >= period / 2, d - period, d)
