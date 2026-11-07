"""MLE fitting for DDM with 3 parameters (kappa, a, Ter).

z = 0 fixed (symmetric start). Boundaries at ±a/2 (our convention).
Sigma = 1 (absorbed into kappa*coherence).

Density: Navarro & Fuss (2009) approximation for first-passage time
of Wiener process with drift. For speed, uses the series truncation
with automatic term selection.

Reference: Navarro & Fuss (2009), J Math Psych 53:222-230.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def wiener_pdf(t: np.ndarray, v: float, a: float, w: float = 0.5,
               eps: float = 1e-29) -> np.ndarray:
    """Density of first-passage time for Wiener with drift.

    Parameters
    ----------
    t : decision times (RT - Ter), must be > 0
    v : drift rate
    a : boundary separation (full distance, 0 to a)
    w : relative start point (0 to 1). w=0.5 = unbiased.
    eps : minimum density to avoid log(0)

    Returns
    -------
    f : density at each t, summed over upper and lower boundary.

    Convention: our model has boundaries at ±a/2 with sigma=1.
    Mapping to standard DDM: a_std = a (full separation = our a),
    w = 0.5 (z=0 → midpoint), v = kappa*coh*dir.
    """
    # Small-time series (Navarro & Fuss eq. 5)
    # f(t|v,a,w) = (a^-2) * exp(-v*a*w - v^2*t/2) * f_tilde(t/a^2 | w)
    # where f_tilde(tau|w) = sum_k (w+2k) * exp(-(w+2k)^2 / (2*tau)) / sqrt(2*pi*tau^3)

    t = np.maximum(t, 1e-10)
    tau = t / (a * a)

    # Number of terms needed for convergence
    # Use enough terms for the sum to converge
    K = max(4, int(np.ceil(1.0 / (2 * np.sqrt(2 * np.pi * tau.min())) + 0.5)))
    K = min(K, 50)

    f_tilde = np.zeros_like(t)
    sqrt_2pi_tau3 = np.sqrt(2 * np.pi * tau**3)

    for k in range(-K, K + 1):
        wk = w + 2 * k
        f_tilde += wk * np.exp(-wk**2 / (2 * tau)) / sqrt_2pi_tau3

    # Full density for upper boundary (correct response)
    f_upper = (1.0 / (a * a)) * np.exp(-v * a * w - v**2 * t / 2) * f_tilde

    # Lower boundary (error): drift is -v, start is 1-w
    f_tilde_lower = np.zeros_like(t)
    w_lower = 1.0 - w
    for k in range(-K, K + 1):
        wk = w_lower + 2 * k
        f_tilde_lower += wk * np.exp(-wk**2 / (2 * tau)) / sqrt_2pi_tau3

    f_lower = (1.0 / (a * a)) * np.exp(
        (-v) * a * w_lower - v**2 * t / 2) * f_tilde_lower

    return np.maximum(f_upper + f_lower, eps)


def neg_log_likelihood(params: np.ndarray, rt: np.ndarray,
                       correct: np.ndarray,
                       coherence: np.ndarray | None = None,
                       ) -> float:
    """Negative log-likelihood for DDM with 3 parameters.

    Parameters
    ----------
    params : [kappa, a, Ter]
    rt : response times (seconds)
    correct : 1 if correct, 0 if error
    coherence : stimulus difficulty per trial. If None, all trials
                have the same (unknown) difficulty → estimate v directly.
    """
    kappa, a, Ter = params

    if kappa < 0.01 or a < 0.1 or Ter < 0.01:
        return 1e12
    if Ter > np.min(rt) - 0.01:
        return 1e12

    dt = rt - Ter
    if np.any(dt <= 0):
        return 1e12

    if coherence is not None:
        # Multi-condition: v = kappa * coherence * direction
        # For correct trials, direction aligns with drift → v > 0
        # For error trials, they chose wrong → v still > 0 but they
        # crossed the wrong boundary
        v = kappa * coherence
    else:
        # Single-condition: v = kappa (coherence absorbed)
        v = np.full_like(dt, kappa)

    # Compute per-trial log-likelihood
    # Correct trials: upper boundary density
    # Error trials: lower boundary density
    # Since we sum upper+lower in wiener_pdf, we need to separate them.

    # Actually, for the joint density:
    # P(RT=t, correct) = f_upper(t | v, a, w=0.5)
    # P(RT=t, error) = f_lower(t | v, a, w=0.5)
    # where f_upper uses drift v, f_lower uses drift -v (symmetric start)

    a_full = a  # Our a is already full separation (boundaries ±a/2 → distance a)
    w = 0.5  # z = 0 → midpoint

    tau = dt / (a_full * a_full)
    K = max(4, int(np.ceil(1.0 / (2 * np.sqrt(2 * np.pi * tau.min())) + 0.5)))
    K = min(K, 50)

    sqrt_2pi_tau3 = np.sqrt(2 * np.pi * tau**3)
    eps = 1e-29

    # Pre-compute f_tilde for w=0.5 (same for upper at w and lower at 1-w)
    f_tilde = np.zeros_like(dt)
    for k in range(-K, K + 1):
        wk = w + 2 * k
        f_tilde += wk * np.exp(-wk**2 / (2 * tau)) / sqrt_2pi_tau3

    # Upper boundary (correct): drift = +v
    log_f_upper = (np.log(np.maximum(f_tilde, eps))
                   - 2 * np.log(a_full)
                   - v * a_full * w
                   - v**2 * dt / 2)

    # Lower boundary (error): drift = -v, w_lower = 1-w = 0.5 = w
    # When w = 0.5: f_tilde_lower = f_tilde (symmetric!)
    log_f_lower = (np.log(np.maximum(f_tilde, eps))
                   - 2 * np.log(a_full)
                   - (-v) * a_full * w  # note: -v * a * w = +v*a*w... wait
                   - v**2 * dt / 2)

    # Wait — for lower boundary with drift -v:
    # exp(-(-v)*a*w_lower - v^2*t/2) = exp(v*a*(1-w) - v^2*t/2)
    # Since w=0.5: exp(v*a*0.5 - v^2*t/2)
    # For upper: exp(-v*a*0.5 - v^2*t/2)
    # These are different! Upper has -v*a*0.5, lower has +v*a*0.5.

    # Navarro & Fuss (2009): lower boundary density has exp(-v*a*w),
    # upper boundary (using -v, 1-w) has exp(+v*a*(1-w)).
    # For w=0.5: upper = exp(+v*a/2), lower = exp(-v*a/2).
    log_f_upper = (np.log(np.maximum(f_tilde, eps))
                   - 2 * np.log(a_full)
                   + v * a_full * 0.5    # correct: +v*a/2
                   - v**2 * dt / 2)

    log_f_lower = (np.log(np.maximum(f_tilde, eps))
                   - 2 * np.log(a_full)
                   - v * a_full * 0.5    # error: -v*a/2
                   - v**2 * dt / 2)

    # Select correct or error
    log_lik = np.where(correct > 0.5, log_f_upper, log_f_lower)

    return -np.sum(log_lik)


def fit_ddm(rt: np.ndarray, correct: np.ndarray,
            coherence: np.ndarray | None = None,
            init: tuple = (5.0, 1.5, 0.2),
            ) -> dict:
    """Fit DDM to trial-level data via MLE.

    Parameters
    ----------
    rt : (N,) response times in seconds
    correct : (N,) 1=correct, 0=error
    coherence : (N,) stimulus difficulty. None for single-condition.
    init : (kappa, a, Ter) initial guess

    Returns
    -------
    dict with keys: kappa, a, Ter, nll, success, n_trials
    """
    # Bounds
    bounds = [(0.1, 200.0),   # kappa
              (0.1, 10.0),    # a
              (0.01, min(np.min(rt) - 0.01, 2.0))]  # Ter < min(RT)

    result = minimize(
        neg_log_likelihood, x0=init, args=(rt, correct, coherence),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8})

    kappa, a, Ter = result.x
    return {
        "kappa": float(kappa),
        "a": float(a),
        "Ter": float(Ter),
        "nll": float(result.fun),
        "success": bool(result.success),
        "n_trials": len(rt),
    }
