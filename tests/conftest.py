def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running tests (T>86400)")


"""Toy modules for testing the core platform.

OUModule: 2-state Ornstein-Uhlenbeck (continuous, analytical solution).
  dx0/dt = -alpha * x0 + u0
  dx1/dt = -beta * x1
  diffusion = [sigma0, sigma1]

ToyDDM: 1-state drift-diffusion (episodic, resets at barrier).
  dx/dt = v  (drift rate from input)
  diffusion = sigma
  trial_done when |x| >= threshold

NonlinearModule: 2-state nonlinear system with analytical Jacobian.
  dx0/dt = -x0³ + x1
  dx1/dt = -x0 - x1²
  J(a,b) = [[-3a², 1], [-1, -2b]]

DriveOU1D: 1-state OU with drive as parameter (for coupling tests).
  dx/dt = -alpha * x + drive
"""
import numpy as np

from cpsych.core.specs import (
    ModuleSpec, StateVar, Param, EnvSpec,
    Temporality, Exposure,
)
from cpsych.core.module import SDEModule
from cpsych.core.environment import Environment


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck (continuous, 2 states)
# ---------------------------------------------------------------------------

_OU_SPEC = ModuleSpec(
    name="ou",
    states=(
        StateVar("x0", Temporality.CONTINUOUS, 0.0),
        StateVar("x1", Temporality.CONTINUOUS, 0.0),
    ),
    params=(
        Param("alpha", 1.0, 0.01, 10.0, doc="decay rate x0"),
        Param("beta", 2.0, 0.01, 10.0, doc="decay rate x1"),
        Param("sigma0", 0.1, 0.0, 5.0, doc="noise x0"),
        Param("sigma1", 0.1, 0.0, 5.0, doc="noise x1"),
    ),
    input_names=("drive",),
    output_names=("activity",),
    timescale=1.0,
    doc="2-state OU for testing. Analytical steady state: x0* = drive/alpha.",
)


class OUModule(SDEModule):
    """2-state Ornstein-Uhlenbeck process."""

    @property
    def spec(self) -> ModuleSpec:
        return _OU_SPEC

    def drift(self, x, theta, u):
        alpha = theta[:, 0:1]
        beta = theta[:, 1:2]
        drive = u[:, 0:1]
        dx0 = -alpha * x[:, 0:1] + drive
        dx1 = -beta * x[:, 1:2]
        return np.hstack([dx0, dx1])

    def diffusion(self, x, theta):
        sigma0 = theta[:, 2:3]
        sigma1 = theta[:, 3:4]
        return np.hstack([sigma0, sigma1])

    def outputs(self, x, theta):
        return {"activity": x[:, 0] + x[:, 1]}


# ---------------------------------------------------------------------------
# Toy DDM (episodic, 1 state)
# ---------------------------------------------------------------------------

_DDM_SPEC = ModuleSpec(
    name="ddm",
    states=(
        StateVar("evidence", Temporality.EPISODIC, 0.0),
    ),
    params=(
        Param("v", 1.0, -5.0, 5.0, exposure=Exposure.EXPOSED, doc="drift rate"),
        Param("threshold", 1.0, 0.1, 5.0, doc="decision barrier"),
        Param("sigma", 0.1, 0.0, 5.0, doc="noise"),
    ),
    input_names=("drift_modulation",),
    output_names=("evidence", "choice"),
    timescale=0.3,
    doc="1D DDM for testing. Episodic: resets at barrier.",
)


class ToyDDM(SDEModule):
    """Toy drift-diffusion model."""

    @property
    def spec(self) -> ModuleSpec:
        return _DDM_SPEC

    def drift(self, x, theta, u):
        v = theta[:, 0:1]
        drift_mod = u[:, 0:1]
        return v + drift_mod

    def diffusion(self, x, theta):
        sigma = theta[:, 2:3]
        return sigma

    def outputs(self, x, theta):
        evidence = x[:, 0]
        threshold = theta[:, 1]
        choice = np.where(evidence >= threshold, 1.0,
                          np.where(evidence <= -threshold, -1.0, 0.0))
        return {"evidence": evidence, "choice": choice}

    def trial_done(self, x, theta):
        # D14: threshold from theta (param index 1)
        threshold = theta[:, 1]
        return np.abs(x[:, 0]) >= threshold


# ---------------------------------------------------------------------------
# Toy Environment (simple bandit)
# ---------------------------------------------------------------------------

_BANDIT_SPEC = EnvSpec(
    name="bandit",
    stimulus_names=("drift_diff",),
    reward_names=("r",),
    doc="2-armed bandit for testing.",
)


class ToyBandit(Environment):
    """Simple 2-armed bandit with fixed reward probabilities."""

    def __init__(self, p_reward: float = 0.7):
        self._p = p_reward
        self._B = 0

    @property
    def spec(self) -> EnvSpec:
        return _BANDIT_SPEC

    def reset(self, B, rng):
        self._B = B

    def next_trial(self, rng, mask=None):
        return {"drift_diff": rng.standard_normal(self._B) * 0.1}

    def reward(self, action, rng):
        r = (rng.random(self._B) < self._p).astype(float)
        return {"r": r}


# ---------------------------------------------------------------------------
# Nonlinear 2D (for Jacobian accuracy testing)
# ---------------------------------------------------------------------------

_NONLINEAR_SPEC = ModuleSpec(
    name="nonlinear",
    states=(
        StateVar("x0", Temporality.CONTINUOUS, 0.0),
        StateVar("x1", Temporality.CONTINUOUS, 0.0),
    ),
    params=(),
    input_names=(),
    output_names=(),
    timescale=1.0,
    doc="dx0=-x0³+x1, dx1=-x0-x1². J(a,b)=[[-3a²,1],[-1,-2b]].",
)


class NonlinearModule(SDEModule):
    """Nonlinear 2-state system with known analytical Jacobian."""

    @property
    def spec(self) -> ModuleSpec:
        return _NONLINEAR_SPEC

    def drift(self, x, theta, u):
        x0, x1 = x[:, 0:1], x[:, 1:2]
        return np.hstack([-x0 ** 3 + x1, -x0 - x1 ** 2])

    def diffusion(self, x, theta):
        return np.zeros_like(x)

    def outputs(self, x, theta):
        return {}


# ---------------------------------------------------------------------------
# 1-state OU with drive parameter (for coupling / multi-tempo tests)
# ---------------------------------------------------------------------------

def _make_drive_ou_spec(name, alpha_default, drive_default, sigma_default,
                        timescale):
    return ModuleSpec(
        name=name,
        states=(StateVar("x", Temporality.CONTINUOUS, 0.0),),
        params=(
            Param("alpha", alpha_default, 0.0001, 1000.0,
                  doc="decay rate"),
            Param("drive", drive_default, -100.0, 100.0,
                  exposure=Exposure.EXPOSED, doc="additive drive"),
            Param("sigma", sigma_default, 0.0, 5.0, doc="noise"),
        ),
        input_names=(),
        output_names=("x",),
        timescale=timescale,
    )


_DRIVE_OU_SPEC = _make_drive_ou_spec("drive_ou", 1.0, 0.0, 0.0, 1.0)


class DriveOU1D(SDEModule):
    """1-state OU with drive as parameter: dx/dt = -alpha*x + drive."""

    def __init__(self, spec=None):
        self._spec = spec or _DRIVE_OU_SPEC

    @property
    def spec(self) -> ModuleSpec:
        return self._spec

    def drift(self, x, theta, u):
        return -theta[:, 0:1] * x + theta[:, 1:2]

    def diffusion(self, x, theta):
        return theta[:, 2:3]

    def outputs(self, x, theta):
        return {"x": x[:, 0]}
