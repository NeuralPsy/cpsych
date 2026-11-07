"""Stability analysis utilities.

Jacobian by central finite differences (decision D3: no SymPy, no JAX).
Composite Jacobian for coupled systems, critical strength bisection (Rule 19).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .module import SDEModule
from .coupling import Coupling


def numerical_jacobian(
    mod: SDEModule,
    x0: np.ndarray,
    theta0: np.ndarray,
    u0: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Jacobian of drift by central finite differences.

    Parameters
    ----------
    mod : SDEModule
    x0 : (1, n_state) — linearization point (single replicate)
    theta0 : (1, n_params)
    u0 : (1, n_inputs)
    eps : perturbation size

    Returns
    -------
    J : (n_state, n_state)
    """
    n = x0.shape[1]
    J = np.zeros((n, n))
    for j in range(n):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[0, j] += eps
        x_minus[0, j] -= eps
        f_plus = mod.drift(x_plus, theta0, u0)[0]
        f_minus = mod.drift(x_minus, theta0, u0)[0]
        J[:, j] = (f_plus - f_minus) / (2 * eps)
    return J


def composite_jacobian(
    modules: dict[str, SDEModule],
    x0: dict[str, np.ndarray],
    theta0: dict[str, np.ndarray],
    u0: dict[str, np.ndarray],
    couplings: list[Coupling],
    eps: float = 1e-5,
) -> np.ndarray:
    """Jacobian of the full coupled system by central finite differences.

    Parameters
    ----------
    modules : {name: SDEModule}
    x0 : {name: (1, n_state_i)} — linearization point per module
    theta0 : {name: (1, n_params_i)} — base parameters per module
    u0 : {name: (1, n_inputs_i)} — inputs per module
    couplings : coupling objects linking modules
    eps : perturbation size

    Returns
    -------
    J : (N, N) where N = sum of all state dimensions
    """
    mod_names = list(modules.keys())
    mod_sizes = {name: x0[name].shape[1] for name in mod_names}
    total_n = sum(mod_sizes.values())

    def _composite_drift(x_flat: np.ndarray) -> np.ndarray:
        # Split flat vector into per-module states
        offset = 0
        x_dict = {}
        for name in mod_names:
            n = mod_sizes[name]
            x_dict[name] = x_flat[offset:offset + n].reshape(1, n)
            offset += n

        # Compute outputs for all modules
        out = {}
        for name in mod_names:
            out[name] = modules[name].outputs(x_dict[name], theta0[name])

        # Apply couplings to get effective theta
        theta_eff = {name: theta0[name].copy() for name in mod_names}
        for c in couplings:
            src_out = out[c.source_module][c.source_output]
            pidx = modules[c.target_module].param_index(c.target_param)
            theta_eff[c.target_module] = c.modulate(
                theta_eff[c.target_module], pidx, src_out)

        # Compute drifts
        drifts = []
        for name in mod_names:
            f = modules[name].drift(
                x_dict[name], theta_eff[name], u0[name])
            drifts.append(f[0])
        return np.concatenate(drifts)

    # Flatten x0
    x_flat = np.concatenate([x0[name][0] for name in mod_names])

    # Central differences
    J = np.zeros((total_n, total_n))
    for j in range(total_n):
        x_plus = x_flat.copy()
        x_minus = x_flat.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        J[:, j] = (_composite_drift(x_plus)
                    - _composite_drift(x_minus)) / (2 * eps)
    return J


def find_critical_strength(
    modules: dict[str, SDEModule],
    x0: dict[str, np.ndarray],
    theta0: dict[str, np.ndarray],
    u0: dict[str, np.ndarray],
    make_couplings: Callable[[float], list[Coupling]],
    s_min: float = 0.0,
    s_max: float = 10.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Find critical coupling strength by bisection.

    Searches for the strength at which max Re(eigenvalue) of the
    composite Jacobian crosses zero. Assumes stable at s_min,
    unstable at s_max.

    Parameters
    ----------
    modules, x0, theta0, u0 : system specification
    make_couplings : callable(strength) -> list[Coupling]
        Factory that creates coupling list for a given strength value.
        No mutation — a new list is created each call.
    s_min, s_max : search bracket
    tol : convergence tolerance on max eigenvalue
    max_iter : maximum bisection iterations
    """
    for _ in range(max_iter):
        s_mid = (s_min + s_max) / 2
        couplings = make_couplings(s_mid)
        J = composite_jacobian(modules, x0, theta0, u0, couplings)
        lam = max_real_eigenvalue(J)
        if abs(lam) < tol:
            break
        if lam < 0:
            s_min = s_mid
        else:
            s_max = s_mid
    return s_mid


def max_real_eigenvalue(J: np.ndarray) -> float:
    """Maximum real part of eigenvalues of J."""
    return float(np.max(np.real(np.linalg.eigvals(J))))
