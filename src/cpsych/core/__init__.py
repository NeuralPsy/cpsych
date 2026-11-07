"""Core platform: module contract, coupling, integrator, stability."""
from .specs import (
    StateVar, Param, ModuleSpec, EnvSpec,
    Temporality, Exposure,
)
from .module import SDEModule
from .environment import Environment
from .coupling import Coupling
from .integrator import Integrator, Tier, SimState
from .stability import (
    numerical_jacobian, composite_jacobian,
    find_critical_strength, max_real_eigenvalue,
)
