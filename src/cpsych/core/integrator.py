"""Multi-timescale Euler-Maruyama integrator with trial loop.

Handles:
- Multiple timescale tiers (fast/slow modules with different dt)
- Episodic resets (DDM barrier crossing, per-replicate mask)
- Discrete trial updates (RL value learning)
- Batch axis (B replicates)
- Observation recording at fixed frequency independent of dt

Splitting scheme: Lie-Trotter (first-order). Each slow step reads the fast
state from the END of the previous slow interval (snapshot), then fast
sub-steps update. This is NOT averaging over the slow interval. The splitting
error is O(dt_slow) per step. For bidirectionally coupled systems this gives
a characteristic error proportional to (c_fs · c_sf / α_f) · dt_slow.
See TestMultiTempoConvergence for empirical validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .module import SDEModule
from .environment import Environment
from .coupling import Coupling


@dataclass
class Tier:
    """A group of modules sharing the same integration timestep."""
    name: str
    dt: float
    module_names: list[str]


@dataclass
class SimState:
    """Mutable container for all simulation state."""
    x: dict[str, np.ndarray] = field(default_factory=dict)
    theta_base: dict[str, np.ndarray] = field(default_factory=dict)
    t: float = 0.0
    trial_count: np.ndarray = field(default_factory=lambda: np.array([]))
    recordings: dict[str, list] = field(default_factory=dict)
    behavior: dict[str, np.ndarray] = field(default_factory=dict)
    boundary_counts: dict[str, dict] = field(default_factory=dict)
    boundary_fractions: dict[str, dict] = field(default_factory=dict)


class Integrator:
    """Multi-timescale Euler-Maruyama integrator.

    Parameters
    ----------
    modules : dict of name -> SDEModule
    environment : optional Environment for trial-based tasks
    couplings : list of Coupling objects
    tiers : list of Tier objects, sorted slowest-first internally
    B : number of replicates (batch size)
    seed : random seed
    obs_dt : observation interval (independent of integration dt)
    """

    def __init__(
        self,
        modules: dict[str, SDEModule],
        environment: Environment | None = None,
        couplings: list[Coupling] | None = None,
        tiers: list[Tier] | None = None,
        B: int = 1,
        seed: int = 42,
        obs_dt: float | None = None,
        trial_driver: str | None = None,
        gate_module: str | None = None,
        gate_output: str | None = None,
        record_behavior: bool = False,
    ):
        self.modules = modules
        self.env = environment
        self.couplings = couplings or []
        self.B = B
        self.rng = np.random.default_rng(seed)
        self.trial_driver = trial_driver
        self.gate_module = gate_module
        self.gate_output = gate_output
        self.record_behavior = record_behavior

        if trial_driver is not None and trial_driver not in modules:
            raise ValueError(
                f"trial_driver '{trial_driver}' not in modules")
        if gate_module is not None:
            if gate_module not in modules:
                raise ValueError(
                    f"gate_module '{gate_module}' not in modules")
            if gate_output is None:
                raise ValueError(
                    "gate_output required when gate_module is set")

        # Default single tier if not specified
        if tiers is None:
            tiers = [Tier("default", 0.01,
                          list(modules.keys()))]
        self.tiers = sorted(tiers, key=lambda t: t.dt, reverse=True)

        self.obs_dt = obs_dt

        # Validate: every module in exactly one tier
        assigned = set()
        for tier in self.tiers:
            for mn in tier.module_names:
                if mn not in modules:
                    raise ValueError(f"Tier '{tier.name}' references "
                                     f"unknown module '{mn}'")
                if mn in assigned:
                    raise ValueError(f"Module '{mn}' in multiple tiers")
                assigned.add(mn)
        for mn in modules:
            if mn not in assigned:
                raise ValueError(f"Module '{mn}' not in any tier")

        # Precompute coupling map: target_module -> [(coupling, param_idx)]
        self._coupling_map: dict[str, list[tuple[Coupling, int]]] = {}
        for c in self.couplings:
            target_mod = modules[c.target_module]
            pidx = target_mod.param_index(c.target_param)
            self._coupling_map.setdefault(
                c.target_module, []).append((c, pidx))

        # Which tier each module belongs to (for output caching)
        self._module_tier: dict[str, int] = {}
        for ti, tier in enumerate(self.tiers):
            for mn in tier.module_names:
                self._module_tier[mn] = ti

        # Precompute tier nesting ratios
        self._tier_ratios: dict[int, int] = {}
        for i in range(1, len(self.tiers)):
            slow_dt = self.tiers[i - 1].dt
            fast_dt = self.tiers[i].dt
            raw_ratio = slow_dt / fast_dt
            ratio = round(raw_ratio)
            if ratio < 1:
                raise ValueError(
                    f"Tiers not ordered by dt: "
                    f"{self.tiers[i-1].name}={slow_dt} < "
                    f"{self.tiers[i].name}={fast_dt}")
            if abs(raw_ratio - ratio) > 0.01:
                raise ValueError(
                    f"dt ratio {self.tiers[i-1].name}/{self.tiers[i].name}"
                    f" = {raw_ratio:.4f}, not close to integer")
            self._tier_ratios[i] = ratio

    def _init_state(self) -> SimState:
        state = SimState()
        state.trial_count = np.zeros(self.B, dtype=int)
        for name, mod in self.modules.items():
            state.x[name] = mod.default_state(self.B)
            state.theta_base[name] = mod.default_params(self.B)
        state.recordings = {
            f"{mn}.{s.name}": []
            for mn, mod in self.modules.items()
            for s in mod.spec.states
        }
        if self.record_behavior:
            state.behavior = {
                "n_correct": np.zeros(self.B),
                "n_trials": np.zeros(self.B),
                "sum_rt": np.zeros(self.B),
                "n_win_stay": np.zeros(self.B),
                "n_wins": np.zeros(self.B),
                "last_action": np.full(self.B, -1.0),
                "last_reward": np.zeros(self.B),
            }
        # Boundary tracking for modules with project_state != identity (D31)
        for name, mod in self.modules.items():
            if type(mod).project_state is not SDEModule.project_state:
                state.boundary_counts[name] = {
                    'lower': np.zeros((self.B, mod.n_state), dtype=int),
                    'upper': np.zeros((self.B, mod.n_state), dtype=int),
                    'total': 0,
                }

        if self.env is not None:
            self.env.reset(self.B, self.rng)
        return state

    def _effective_theta(self, target_name: str,
                         state: SimState,
                         precomputed_outputs: dict | None = None,
                         ) -> np.ndarray:
        """Base theta + coupling modulations.

        Parameters
        ----------
        precomputed_outputs : {module_name: outputs_dict}
            Pre-computed outputs for slow-tier modules. If a coupling
            source is in this dict, its outputs are used directly
            instead of recomputing. This is the Lie-Trotter invariant:
            slow-tier state is constant during fast substeps.
        """
        theta = state.theta_base[target_name].copy()
        if target_name not in self._coupling_map:
            return theta
        for coupling, pidx in self._coupling_map[target_name]:
            src_name = coupling.source_module
            if precomputed_outputs and src_name in precomputed_outputs:
                src_out = precomputed_outputs[src_name]
            else:
                src_mod = self.modules[src_name]
                src_out = src_mod.outputs(
                    state.x[src_name], state.theta_base[src_name])
            theta = coupling.modulate(
                theta, pidx, src_out[coupling.source_output])
        return theta

    def _step_module(self, name: str, dt: float,
                     state: SimState,
                     stimulus: dict | None = None,
                     precomputed_outputs: dict | None = None,
                     ) -> None:
        """One Euler-Maruyama step. Mutates state.x[name].

        D15: stimulus dict fills u for modules that declare input_names.
        """
        mod = self.modules[name]
        x = state.x[name]
        theta = self._effective_theta(name, state, precomputed_outputs)
        u = np.zeros((self.B, mod.n_inputs))
        if stimulus and mod.spec.input_names:
            for i, inp in enumerate(mod.spec.input_names):
                if inp in stimulus:
                    u[:, i] = stimulus[inp]

        dx = mod.drift(x, theta, u)
        sigma = mod.diffusion(x, theta)
        dW = self.rng.standard_normal(x.shape)
        x_new = x + dx * dt + sigma * np.sqrt(dt) * dW
        state.x[name] = mod.project_state(x_new, theta)
        # Boundary clipping diagnostics (D31)
        if name in state.boundary_counts:
            clipped_lower = (state.x[name] > x_new)
            clipped_upper = (state.x[name] < x_new)
            state.boundary_counts[name]['lower'] += clipped_lower.astype(int)
            state.boundary_counts[name]['upper'] += clipped_upper.astype(int)
            state.boundary_counts[name]['total'] += 1

    def _handle_trials(self, tier_idx: int, state: SimState,
                       stimulus: dict,
                       precomputed_outputs: dict | None = None,
                       ) -> None:
        """Check trial_done for driver module(s) and handle.

        D22: trial_driver restricts which module is checked for trial_done.
        Gate (D10): gate output ≤ 0.5 freezes trial loop per-replicate.
        All modules receive trial_update when a trial completes.
        """
        tier = self.tiers[tier_idx]

        # Determine which modules to check for trial_done
        if self.trial_driver is not None:
            # Only check the designated driver
            check_names = [self.trial_driver] if self.trial_driver in tier.module_names else []
        else:
            check_names = tier.module_names

        for mod_name in check_names:
            mod = self.modules[mod_name]
            theta_eff = self._effective_theta(
                mod_name, state, precomputed_outputs)
            done = mod.trial_done(state.x[mod_name], theta_eff)

            # Apply gate: mask out replicates where gate output ≤ 0.5
            if self.gate_module is not None:
                gate_mod = self.modules[self.gate_module]
                gate_theta = self._effective_theta(
                    self.gate_module, state, precomputed_outputs)
                gate_out = gate_mod.outputs(
                    state.x[self.gate_module], gate_theta)
                gate_open = gate_out[self.gate_output] > 0.5
                done = done & gate_open

            if not np.any(done):
                continue

            # Get decision outputs
            decision = mod.outputs(state.x[mod_name], theta_eff)

            # Environment interaction
            env_result: dict[str, np.ndarray] = {}
            if self.env is not None:
                action = decision.get("choice", np.zeros(self.B))
                env_result = self.env.reward(action, self.rng)
                env_result["choice"] = action
                # Pass through is_timeout from decision (D28)
                env_result["is_timeout"] = decision.get(
                    "is_timeout", np.zeros(self.B))

            # Record behavioral counters
            if self.record_behavior and self.env is not None:
                rt = decision.get("rt", np.zeros(self.B))
                arm = env_result.get(
                    "choice", np.where(action >= 0, 1.0, 0.0))
                r = env_result["r"]
                # Use chose_correct if available (CompositeTask),
                # else fall back to reward-based (PerceptualTask)
                if "chose_correct" in env_result:
                    correct = env_result["chose_correct"]
                else:
                    correct = (r > 0.5).astype(float)

                state.behavior["n_correct"] += done * correct
                state.behavior["n_trials"] += done.astype(float)
                state.behavior["sum_rt"] += done * rt

                last_a = state.behavior["last_action"]
                last_r = state.behavior["last_reward"]
                had_prev = (last_a >= 0)
                won_prev = had_prev & (last_r > 0.5)
                stayed = (arm == last_a)
                state.behavior["n_win_stay"] += (
                    done * won_prev * stayed)
                state.behavior["n_wins"] += (
                    done * won_prev.astype(float))

                state.behavior["last_action"] = np.where(
                    done, arm, last_a)
                state.behavior["last_reward"] = np.where(
                    done, r, last_r)

            # Trial update for ALL modules (RL gets reward)
            for other_name, other_mod in self.modules.items():
                state.x[other_name] = other_mod.trial_update(
                    state.x[other_name],
                    self._effective_theta(
                        other_name, state, precomputed_outputs),
                    done, env_result,
                )

            # Reset episodic state (D14: theta passed)
            state.x[mod_name] = mod.reset_state(
                state.x[mod_name], theta_eff, done)
            state.trial_count += done.astype(int)

            # New stimulus for done replicates
            if self.env is not None:
                new_stim = self.env.next_trial(self.rng, mask=done)
                for k in stimulus:
                    stimulus[k] = np.where(
                        done[:, None] if new_stim[k].ndim > 1
                        else done,
                        new_stim[k], stimulus[k])

    def _record(self, state: SimState) -> None:
        """Record current state of all modules."""
        for mn, mod in self.modules.items():
            x = state.x[mn]
            for j, sv in enumerate(mod.spec.states):
                key = f"{mn}.{sv.name}"
                state.recordings[key].append(
                    (state.t, x[:, j].copy()))

    def _run_tier(self, tier_idx: int, n_steps: int,
                  state: SimState, stimulus: dict,
                  precomputed_outputs: dict | None = None,
                  ) -> None:
        """Recursively integrate tier and all faster sub-tiers."""
        tier = self.tiers[tier_idx]

        for _ in range(n_steps):
            # Step modules in this tier
            for mod_name in tier.module_names:
                self._step_module(
                    mod_name, tier.dt, state, stimulus,
                    precomputed_outputs)

            # Recurse into faster tiers
            if tier_idx + 1 < len(self.tiers):
                n_sub = self._tier_ratios[tier_idx + 1]
                self._run_tier(
                    tier_idx + 1, n_sub, state, stimulus,
                    precomputed_outputs)
            else:
                # Leaf tier: handle trials here
                self._handle_trials(
                    tier_idx, state, stimulus, precomputed_outputs)

            # Non-leaf tiers: trials handled by the leaf
            state.t += tier.dt

    def run(self, T: float, state: SimState | None = None) -> SimState:
        """Run simulation for T model time units.

        Parameters
        ----------
        T : total simulation time
        state : optional pre-initialized state (e.g. with overridden theta)
        """
        if state is None:
            state = self._init_state()

        stimulus: dict[str, np.ndarray] = {}
        if self.env is not None:
            stimulus = self.env.next_trial(self.rng)

        # Record initial state
        self._record(state)

        slowest = self.tiers[0]
        n_slow = int(round(T / slowest.dt))

        # Use obs_dt for recording
        obs_interval = self.obs_dt or slowest.dt
        obs_every = max(1, int(round(obs_interval / slowest.dt)))
        next_obs = obs_every

        for step in range(n_slow):
            # Step slowest tier modules
            for mod_name in slowest.module_names:
                self._step_module(mod_name, slowest.dt, state, stimulus)

            # Precompute slow-tier outputs for fast substeps.
            # By Lie-Trotter, slow state is constant during fast interval.
            slow_outputs: dict | None = None
            if len(self.tiers) > 1:
                slow_outputs = {}
                for mn in slowest.module_names:
                    mod = self.modules[mn]
                    theta = self._effective_theta(mn, state)
                    slow_outputs[mn] = mod.outputs(state.x[mn], theta)

            # Recurse into faster tiers
            if len(self.tiers) > 1:
                n_sub = self._tier_ratios[1]
                self._run_tier(1, n_sub, state, stimulus, slow_outputs)
            else:
                # Single tier: handle trials directly
                self._handle_trials(0, state, stimulus)

            state.t = (step + 1) * slowest.dt

            # Record at obs_dt intervals
            if (step + 1) >= next_obs:
                self._record(state)
                next_obs += obs_every

        # Compute boundary fractions (D31)
        for name, counts in state.boundary_counts.items():
            total = counts['total']
            if total > 0:
                state.boundary_fractions[name] = {
                    'lower': counts['lower'] / total,
                    'upper': counts['upper'] / total,
                }

        return state
