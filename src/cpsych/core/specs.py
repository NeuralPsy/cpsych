"""Static specifications for modules and environments.

Frozen dataclasses describing what a module declares about itself:
state variables, parameters, inputs, outputs, timescale.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Temporality(Enum):
    """Whether a state variable resets between trials."""
    CONTINUOUS = "continuous"
    EPISODIC = "episodic"


class Exposure(Enum):
    """Whether a parameter is user-facing or internal."""
    EXPOSED = "exposed"
    INTERNAL = "internal"


@dataclass(frozen=True)
class StateVar:
    """One variable in the module's state vector."""
    name: str
    temporality: Temporality
    initial: float = 0.0
    doc: str = ""


@dataclass(frozen=True)
class Param:
    """One parameter in the module's parameter vector."""
    name: str
    default: float
    low: float
    high: float
    exposure: Exposure = Exposure.INTERNAL
    unit: str = ""
    doc: str = ""


@dataclass(frozen=True)
class ModuleSpec:
    """Complete static specification of an SDE module."""
    name: str
    states: tuple[StateVar, ...]
    params: tuple[Param, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    timescale: float
    doc: str = ""
    circular_observables: tuple[tuple[str, float], ...] = ()
    """Observables (direct or derived) that wrap with a period.
    Each entry: (name, period). For these: use circular mean/SD and
    shortest-arc differences. Prevents wrapping artifacts in aggregation."""

    def state_index(self, name: str) -> int:
        for i, s in enumerate(self.states):
            if s.name == name:
                return i
        raise KeyError(f"No state '{name}' in {self.name}")

    def param_index(self, name: str) -> int:
        for i, p in enumerate(self.params):
            if p.name == name:
                return i
        raise KeyError(f"No param '{name}' in {self.name}")


@dataclass(frozen=True)
class EnvSpec:
    """Specification for an environment (task) module."""
    name: str
    stimulus_names: tuple[str, ...]
    reward_names: tuple[str, ...]
    doc: str = ""
