# Methods: The platform

## Module contract

Each module is a stochastic dynamical system with an explicit interface:

- **State variables** x ∈ ℝⁿ with names, dimensions, and temporality (continuous or episodic — reset after each trial).
- **Parameters** θ with declared ranges, units, and exposure class (exposed to the sandbox or internal).
- **Drift** f(x, θ, u) and **diffusion** g(x, θ): pure functions without mutation or side effects (rule 20). All randomness enters through an explicit generator argument; state is not stored between calls. This discipline is required for correct finite-difference Jacobians and prepares the codebase for migration to automatic differentiation if needed.
- **Inputs** u from other modules or the environment. **Outputs** computed from (x, θ): observable quantities available to other modules and to the user.
- **Characteristic timescale** τ, declared per module, used to assign modules to integration tiers.

The contract enforces a separation: the module defines its dynamics, the integrator runs them. No module knows about the existence of others.

## Integration

The integrator uses Euler-Maruyama discretization with Lie-Trotter operator splitting for multi-tempo systems. Two tiers: a slow tier (dt = 1.0 model time units) for arousal and attention, and a fast tier (dt = 0.001) for the decision and learning modules. The ratio of timescales (1,000:1) reflects the separation between circadian dynamics (hours) and perceptual accumulation (hundreds of milliseconds).

Within each slow step, the integrator:
1. Advances slow-tier modules by one step.
2. Precomputes their outputs (gain, is_wake) — these are constant during the fast interval by the Lie-Trotter snapshot property.
3. Runs 1,000 fast substeps, using the precomputed slow outputs for coupling calculations rather than recomputing them each substep.

Step 3 is a structural optimization, not a cache: slow-tier outputs are passed down through the call chain as arguments, adding no overhead. This reduced wall time by 2.8× (from 105 s to 45 s per 600-second simulation at B = 1) by eliminating 120,000 redundant sigmoid evaluations per run.

The batch axis is a first-class feature: all arrays have shape (B, n_state), where B is the number of independent replications. Drift, diffusion, coupling, trial logic, and observables operate on the full batch simultaneously. At B = 30, the per-replication cost drops to 1.9 seconds (amortization from Python call overhead), yielding 30 stochastic replications for prediction intervals at negligible marginal cost.

## Coupling

Modules are connected through **parameter modulation**: the output of a source module modifies a parameter of a target module. The effective parameter is

θ_eff = θ_base + s · h(output, θ_base)

where s is the coupling strength and h is a link function. For multiplicative couplings (e.g., arousal gain → DDM drift sensitivity), h(g, κ) = κ(g − 1), so that κ_eff = κ · (1 + s(g − 1)) — unity at gain = 1, linear modulation otherwise. Post-modulation clamping (param_min, param_max) prevents sign reversals during sleep, when gain drops to 0.02.

Direct mixing of state variables between modules is prohibited without specific justification. Parameter modulation is preferred because it preserves the modular structure (each module's drift depends only on its own state and parameters) and because coupling strength can be bounded by stability analysis: the critical strength is computable from the composite Jacobian.

## Trial loop and gating

The drift-diffusion model drives the trial loop: when evidence reaches a boundary (|x| ≥ a/2), the trial ends, the environment delivers reward, all modules receive a trial update, and the DDM resets. The arousal module provides a wake gate: during sleep (Q_m below threshold), the trial loop is suspended — no stimuli are presented and no decisions are made, but continuous dynamics (mood decay, arousal oscillation) continue.

The trial-loop mechanism required a correction discovered during the impossible-map computation (D38). When one replication's trial ended, the environment's `next_trial` method was called for all B replications, causing reversal events in non-finishing replications to desynchronize the correct-arm state from the direction stimulus. The fix: `next_trial` accepts a mask argument and updates only the replications that completed a trial. The bug would have been invisible on real data — estimates would have appeared plausible but been systematically biased — and was caught by the synthetic recovery step that preceded external comparison (D46).

## Modules

### Arousal (Phillips-Robinson 2007)

Three neuronal populations (wake-active monoaminergic, sleep-active ventrolateral preoptic, and a homeostatic drive) form a mutual-inhibition flip-flop with a circadian forcing term. Four state variables: V_v, V_m, H (homeostatic pressure), and a clock variable for the non-autonomous circadian drive. The system produces self-sustained sleep-wake cycles of approximately 8 hours sleep and 16 hours wake at calibrated parameters.

Noise is additive on all three neuronal variables; σ_h was calibrated to produce onset variability of 15–30 minutes circular SD (Klerman and Dijk, 2005 laboratory data). The module outputs gain (firing rate normalized to wake baseline) and is_wake (binary gate).

### Drift-diffusion model (Ratcliff 1978)

Evidence accumulates with drift v = κ · coherence · direction + drift_bias, diffusion σ = 1 (fixed). Boundaries at ±a/2. Three free parameters: κ (drift sensitivity), a (threshold), T_er (non-decision time). All three are fitted with recovery r ≥ 0.945 (MLE on Navarro-Fuss first-passage density; Navarro and Fuss, 2009). The module does not include inter-trial variability parameters (η, s_z, s_t), which are deferred (D16) pending evidence that composite behavior depends on RT distribution shape.

### Reinforcement learner with momentary affect (Rutledge et al. 2014)

A Rescorla-Wagner learner with two value states (V_0, V_1) and a mood variable m. Choice is softmax with inverse temperature β. Mood updates trial-by-trial: m ← γ(m + δ), where δ = ρ · r − V_chosen is the reward prediction error and γ = exp(−1/τ_m). Mood drift is zero (D21): decay is trial-paced, not time-paced, consistent with the calibration source.

The environment supports two reversal protocols: stochastic (Geom(volatility/20), matching Behrens et al. 2007) and criterion-based (reversal after 7–15 collected rewards, only after a rewarded trial, matching Eckstein et al. 2022). The criterion mode was added for external comparison; the model itself is unchanged.

### Boundary-attention architecture

An attention module with five competitive channels, somatic energy resource, and Hebbian plasticity. Connected unidirectionally: arousal gain modulates BAA gain, but BAA has no output coupling to the behavioral loop. In the current configuration it is a passive observer — its internal dynamics respond to arousal but do not influence decisions, learning, or any behavioral observable. It is included to demonstrate that the platform accepts modules developed outside its protocol; it has untested parameters and no external comparison.
