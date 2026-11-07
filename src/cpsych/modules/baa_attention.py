"""BAA (Boundary-Attention Architecture) as SDEModule.

Wraps the BAA model from baa/src/baa/model.py into the platform contract.
Equations:
    dAi = [gain*Ii*Wi - (beta-gamma)*Ai - gamma*sum(A)] dt + sigma(S)*dBi
    dS  = [R_rec*(S_max - S) - k_cost*sum(A)] dt
    dWi = eta*(Ai*Ii - Ai^2*Wi) dt

Reflecting boundaries (project_state):
    Ai >= 0, 0 <= S <= S_max, W_min <= W <= W_max

Noise: sigma(S) = sigma_base + (sigma_max - sigma_base)/(1+exp(0.2*(S-S_crit)))
       Noise on A channels only; S and W are deterministic.

Fixed nc=5 (D32). Coupling: arousal.gain -> BAA.gain (D33).
"""
from __future__ import annotations

import numpy as np

from ..core.module import SDEModule
from ..core.specs import ModuleSpec, StateVar, Param, Temporality, Exposure


def _make_baa_spec(nc: int = 5, tonic_input: float = 0.5) -> ModuleSpec:
    """Build spec for nc channels: A(nc) + S(1) + W(nc) = 2*nc+1 states."""
    states = []
    for i in range(nc):
        states.append(StateVar(
            f"A{i}", Temporality.CONTINUOUS, initial=0.0,
            doc=f"Activation channel {i}"))
    states.append(StateVar(
        "S", Temporality.CONTINUOUS, initial=100.0,
        doc="Somatic energy"))
    for i in range(nc):
        states.append(StateVar(
            f"W{i}", Temporality.CONTINUOUS, initial=1.0,
            doc=f"Hebbian weight channel {i}"))

    params = (
        Param("gain", 1.0, 0.1, 5.0, Exposure.EXPOSED,
              doc="Attentional gain"),
        Param("beta", 0.5, 0.01, 2.0, Exposure.EXPOSED,
              doc="Self-inhibition"),
        Param("gamma", 0.2, 0.01, 1.0, Exposure.EXPOSED,
              doc="Lateral inhibition"),
        Param("sigma_base", 0.1, 0.01, 2.0, Exposure.INTERNAL,
              doc="Baseline noise"),
        Param("sigma_max", 2.0, 0.1, 5.0, Exposure.INTERNAL,
              doc="Max noise at low energy"),
        Param("plasticity_rate", 0.0, 0.0, 1.0, Exposure.EXPOSED,
              doc="Oja learning rate (eta)"),
        Param("R_rec", 0.05, 0.001, 0.5, Exposure.INTERNAL,
              doc="Energy recovery rate"),
        Param("k_cost", 0.1, 0.001, 1.0, Exposure.INTERNAL,
              doc="Energy cost per activation"),
        Param("S_max", 100.0, 10.0, 500.0, Exposure.INTERNAL,
              doc="Max somatic energy"),
        Param("S_crit", 30.0, 5.0, 100.0, Exposure.INTERNAL,
              doc="Energy threshold for noise sigmoid"),
        Param("theta_hebbian", 1.0, 0.0, 5.0, Exposure.INTERNAL,
              doc="Hebbian threshold (unused in Oja)"),
        Param("W_min", 0.5, 0.0, 2.0, Exposure.INTERNAL,
              doc="Min Hebbian weight"),
        Param("W_max", 4.0, 1.0, 10.0, Exposure.INTERNAL,
              doc="Max Hebbian weight"),
        Param("tonic_input", tonic_input, 0.0, 5.0, Exposure.INTERNAL,
              doc="Tonic input when no external stimulus"),
    )

    # Inputs: external drive per channel (from environment stimulus)
    input_names = tuple(f"I{i}" for i in range(nc))

    output_names = ("total_activation", "energy",
                    "attention_entropy", "gain")

    return ModuleSpec(
        name="baa_attention",
        states=tuple(states),
        params=params,
        input_names=input_names,
        output_names=output_names,
        timescale=10.0,
        doc="Boundary-Attention Architecture (BAA), nc channels",
    )


# Parameter indices
_PI = {
    "gain": 0, "beta": 1, "gamma": 2,
    "sigma_base": 3, "sigma_max": 4,
    "plasticity_rate": 5, "R_rec": 6, "k_cost": 7,
    "S_max": 8, "S_crit": 9,
    "theta_hebbian": 10, "W_min": 11, "W_max": 12,
    "tonic_input": 13,
}


class BAAModule(SDEModule):
    """BAA attention model as SDEModule.

    Parameters
    ----------
    nc : number of attention channels (default 5, D32).
    tonic_input : default tonic input level per channel.
    """

    def __init__(self, nc: int = 5, tonic_input: float = 0.5):
        self._nc = nc
        self._tonic_input = tonic_input
        self._spec = _make_baa_spec(nc, tonic_input)

    @property
    def spec(self) -> ModuleSpec:
        return self._spec

    @property
    def nc(self) -> int:
        return self._nc

    def drift(self, x: np.ndarray, theta: np.ndarray,
              u: np.ndarray) -> np.ndarray:
        nc = self._nc
        B = x.shape[0]

        A = x[:, :nc]
        S = x[:, nc:nc + 1]
        W = x[:, nc + 1:]

        # Input = tonic baseline (from theta) + external stimulus (from u).
        # When no stimulus is routed to BAA (u = zeros), I = tonic_input.
        # When external stimulus is provided, it adds on top.
        tonic = theta[:, _PI["tonic_input"]:_PI["tonic_input"] + 1]
        if u.shape[1] >= nc:
            I = tonic + u[:, :nc]
        else:
            I = np.broadcast_to(tonic, (B, nc))

        gain = theta[:, _PI["gain"]:_PI["gain"] + 1]
        beta = theta[:, _PI["beta"]:_PI["beta"] + 1]
        gamma = theta[:, _PI["gamma"]:_PI["gamma"] + 1]
        eta = theta[:, _PI["plasticity_rate"]:_PI["plasticity_rate"] + 1]
        R_rec = theta[:, _PI["R_rec"]:_PI["R_rec"] + 1]
        k_cost = theta[:, _PI["k_cost"]:_PI["k_cost"] + 1]
        S_max = theta[:, _PI["S_max"]:_PI["S_max"] + 1]

        tot_A = np.sum(A, axis=1, keepdims=True)

        # dA = gain*I*W - (beta-gamma)*A - gamma*sum(A)
        dA = gain * I * W - (beta - gamma) * A - gamma * tot_A

        # dS = R_rec*(S_max - S) - k_cost*sum(A)
        dS = R_rec * (S_max - S) - k_cost * tot_A

        # dW = eta*(A*I - A^2*W)  (Oja plasticity)
        dW = eta * (A * I - A**2 * W)

        return np.hstack([dA, dS, dW])

    def diffusion(self, x: np.ndarray,
                  theta: np.ndarray) -> np.ndarray:
        nc = self._nc
        S = x[:, nc]
        sigma_base = theta[:, _PI["sigma_base"]]
        sigma_max = theta[:, _PI["sigma_max"]]
        S_crit = theta[:, _PI["S_crit"]]

        sig = sigma_base + (sigma_max - sigma_base) / (
            1.0 + np.exp(0.2 * (S - S_crit)))

        noise = np.zeros_like(x)
        noise[:, :nc] = sig[:, None]  # noise on A channels only
        return noise

    def project_state(self, x: np.ndarray,
                      theta: np.ndarray) -> np.ndarray:
        """Reflecting boundaries: A>=0, 0<=S<=S_max, W_min<=W<=W_max."""
        nc = self._nc
        x_new = x.copy()

        # A >= 0
        x_new[:, :nc] = np.maximum(x_new[:, :nc], 0.0)

        # 0 <= S <= S_max
        S_max = theta[:, _PI["S_max"]:_PI["S_max"] + 1]
        x_new[:, nc:nc + 1] = np.clip(x_new[:, nc:nc + 1], 0.0, S_max)

        # W_min <= W <= W_max
        W_min = theta[:, _PI["W_min"]:_PI["W_min"] + 1]
        W_max = theta[:, _PI["W_max"]:_PI["W_max"] + 1]
        x_new[:, nc + 1:] = np.clip(x_new[:, nc + 1:], W_min, W_max)

        return x_new

    def outputs(self, x: np.ndarray,
                theta: np.ndarray) -> dict[str, np.ndarray]:
        nc = self._nc
        A = x[:, :nc]
        S = x[:, nc]

        total_act = np.sum(A, axis=1)

        # Attention entropy: -sum(p*log(p)) where p = A/sum(A)
        eps = 1e-12
        A_safe = np.maximum(A, eps)
        sum_A = np.sum(A_safe, axis=1, keepdims=True)
        p = A_safe / np.maximum(sum_A, eps)
        entropy = -np.sum(p * np.log(p + eps), axis=1)

        return {
            "total_activation": total_act,
            "energy": S,
            "attention_entropy": entropy,
            "gain": theta[:, _PI["gain"]],
        }

    def analytical_fixed_point(self, theta: np.ndarray,
                               I: np.ndarray) -> np.ndarray:
        """Analytical fixed point for deterministic system (eta=0).

        A* = gain*I*W / (beta + gamma*(nc-1))
        S* = S_max - k_cost*sum(A*) / R_rec
        W* = W_init (unchanged when eta=0)

        Parameters
        ----------
        theta : (B, n_params)
        I : (B, nc) or (nc,) input vector

        Returns
        -------
        x_star : (B, n_state) fixed point
        """
        nc = self._nc
        B = theta.shape[0]

        gain = theta[:, _PI["gain"]:_PI["gain"] + 1]
        beta = theta[:, _PI["beta"]:_PI["beta"] + 1]
        gamma = theta[:, _PI["gamma"]:_PI["gamma"] + 1]
        R_rec = theta[:, _PI["R_rec"]:_PI["R_rec"] + 1]
        k_cost = theta[:, _PI["k_cost"]:_PI["k_cost"] + 1]
        S_max = theta[:, _PI["S_max"]:_PI["S_max"] + 1]

        tonic = theta[:, _PI["tonic_input"]:_PI["tonic_input"] + 1]
        I = np.broadcast_to(I, (B, nc)) + tonic
        W = np.ones((B, nc))

        # A* = gain*I*W / (beta + gamma*(nc-1))
        denom = beta + gamma * (nc - 1)
        A_star = gain * I * W / denom

        # S* = S_max - k_cost*sum(A*)/R_rec
        tot_A = np.sum(A_star, axis=1, keepdims=True)
        S_star = S_max - k_cost * tot_A / R_rec

        x_star = np.hstack([A_star, S_star, W])
        return x_star
