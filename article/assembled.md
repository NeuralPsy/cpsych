# Abstract

Computational psychiatry uses mathematical models to map mental functions onto parameters that can, in principle, be estimated from behavioral data — but whether those parameters are actually recoverable is rarely checked, and what the models fail to produce is described informally if at all. We present a modular platform of stochastic dynamical systems (arousal, drift-diffusion decision, reinforcement learning with momentary affect, and attention) coupled through parameter modulation, together with a protocol that classifies every parameter by recoverability before external comparison, pre-registers agreement criteria, and maps regions of the parameter space where the model refuses to produce data.

Of eleven parameters subjected to recovery testing, five were recoverable (r ≥ 0.7) and six were not; two further parameters remain untested. Five parameter degeneracies — products of a free parameter with a quantity lacking an independent scale — were identified from the model equations and follow a single rule checkable before any simulation. Three composite phenomena absent from any module alone emerged from the coupled system — invisible to standalone validation: non-monotonic accuracy under value transfer, coherence-dependent compensation of perceptual signal by learned values, and mood-loop stabilization through value adaptation.

We compared the platform against three open datasets (Confidence Database, 73 datasets, 4,066 participants; OBF-Psychiatric actigraphy, 32 controls; Eckstein et al. 2022 reversal learning, 114 adults). The decision model, where all three parameters are recoverable (r = 0.945, 0.971, 0.992), produced no positive result on any data subset, against thresholds derived from inter-dataset variability before fitting: single-condition MAE 0.041 above threshold 0.038, coherence-scale MAE 0.086 nearly double threshold 0.044, multi-condition MAE 0.102 at threshold 0.100. The arousal model, where no parameter is recoverable (highest r = 0.42), predicted sleep onset timing within 38–60 minutes at fully fixed parameters on data it was never calibrated on; the originally planned duration comparison was withdrawn because the scoring algorithm could not be reconstructed from the pre-registration. The reinforcement learning model showed systematic underperformance at calibrated parameters (win-stay 0.681 vs 0.952); diagnostic analysis attributed 84% of the gap to the learning rate α — a discrepancy expected when transferring parameters calibrated on one protocol (Behrens) to another (Eckstein), consistent with the finding that learning rates do not transfer between tasks (Eckstein et al., eLife 2022), though not an independent replication since the diagnostic uses their published estimates.

The central finding is that parameter recoverability and predictive accuracy are different properties: the module with no recoverable parameters produced the study's only clean prediction; the module with all parameters recoverable produced none. Four lost analysis scripts, restored from the pre-registration, yielded graded outcomes from exact reproduction through non-reproducibility forcing withdrawal; a fifth failure — stale cached intermediates silently loaded by a fully versioned script — was discovered only by clean end-to-end reproduction, showing that reproducibility degrades with each unspecified or unversioned dependency. Rigorous comparison yielded few confirmations. No other computational psychiatry platform we are aware of reports recovery correlations or classifies parameters by recoverability before comparison with data.

---

# Introduction

Computational psychiatry builds mathematical models of mental functions — decision-making, arousal regulation, reinforcement learning, attention — and connects them to clinical phenomena through parameters that can, in principle, be estimated from behavioral data. The promise is a mechanistic bridge between neuroscience and diagnosis: if a patient's decision threshold or learning rate deviates from a healthy baseline, the deviation localizes the dysfunction and suggests intervention targets.

This promise rests on two assumptions that are rarely tested together. First, that the model's parameters can be recovered from observed behavior — that the data constrain the parameter values tightly enough to distinguish individuals. Second, that the model, once parameterized, correctly predicts the phenomena it was not calibrated on. These are different properties, and they need not co-occur.

Recent work has shown empirically that they often do not. Eckstein, Wilbrecht, and Collins (2022) demonstrated that parameters of reinforcement learning models depend on task context: the same participant yields different parameter estimates on different tasks administered in the same session. Learning rates, in particular, do not transfer between tasks, while decision noise parameters are more stable. This raises a question: can such failures of transferability be anticipated at the design stage, before encountering data?

We present a modular platform for computational psychiatry in which each module is a stochastic dynamical system (SDE) with explicitly declared parameters, and propose a protocol that classifies every parameter by its recoverability before the platform is used for comparison with data. The protocol requires:

1. **Parameter recovery** before any external comparison: generating data from the model with known parameters, estimating them back, and reporting the correlation. Parameters are classified as *fitted* (r ≥ 0.7, suitable for individual estimation), *exploratory* (recovery attempted, r < 0.7, observable in the sandbox but not estimable from data), or *untested*.

2. **Deferred observables**: phenomena declared before implementation that will not be used for calibration and serve as one-time predictions.

3. **Pre-registered comparison criteria**: agreement and disagreement thresholds fixed and committed before data are accessed.

4. **An impossible map**: a systematic survey of the parameter space showing what the model refuses to produce — regions of instability, chance-level performance, and indistinguishable outputs.

5. **Degeneracy identification**: parameters that enter behavior only through a product with another quantity lacking an independent scale are flagged as inherently unidentifiable. Five instances of this pattern were identified in the platform (κ·a, β·ρ, η·λ, s·ρ, and κ·stimulus-units), all predictable from the model equations before any data analysis.

The platform comprises three behavioral modules — a sleep-wake flip-flop (Phillips and Robinson, 2007), a drift-diffusion model of perceptual decisions (Ratcliff, 1978), and a reinforcement learner with momentary affect (Rutledge et al., 2014) — plus an attention module (boundary-attention architecture) connected unidirectionally as a passive receiver of the arousal signal, with no output to the behavioral loop. Modules are coupled through parameter modulation — the output of one module modifies a parameter of another — and the composite is checked for stability before operation. The platform is not a clinical tool; it is a workbench for asking "what if" questions and, more importantly, "what cannot be."

We compare the platform against three open datasets: the Confidence Database (Rahnev et al., 2020; 73 datasets, 4,066 participants) for the decision model, the OBF-Psychiatric actigraphy dataset (Garcia-Ceja et al., 2025; 32 healthy controls) for the arousal model, and the Eckstein et al. (2022) reversal learning dataset (114 adults, 120 trials per participant) for the reinforcement learning model. Criteria were pre-registered and committed before data access.

The principal finding is that recoverability and predictive accuracy are different properties of a model, and conflating them is a methodological error with practical consequences. The arousal module, in which no parameter is recoverable (highest recovery correlation r = 0.42), predicted sleep onset timing within 38–60 minutes of the data (model 23:00, data 23:39–00:01 across three scoring variants), with all parameters fixed at calibrated values. Onset timing was not calibrated to any data in this comparison — the circadian phase was taken from Phillips and Robinson (2008) and not adjusted. This was the only comparison in the study that produced a clean prediction at fully fixed parameters on uncalibrated data. The originally planned comparison on mean sleep duration was withdrawn because the scoring algorithm could not be reconstructed from the pre-registration (D49). On variability, the model showed a structural gap: within-individual night-to-night SD of sleep duration was 2.15–2.18 h in the data versus 0.50 h in the model — a fourfold underestimation reflecting the absence of external variability sources (social schedule, weekends) from the model.

The decision module, in which all three parameters are recoverable (r ≥ 0.945), produced no positive result on any subset of the data. Single-condition datasets: MAE 0.041, above threshold 0.038. Coherence-scale datasets — where the model equation maps directly to the stimulus variable: MAE 0.086, nearly double the threshold 0.044. Multi-condition all scales: MAE 0.102, at threshold 0.100 (indeterminate). The model captures the qualitative accuracy-difficulty relationship but deviates at a level comparable to or exceeding inter-dataset variability.

The reinforcement learning module, tested on a protocol with unambiguous punishment (Eckstein et al., 2022), showed systematic underperformance with parameters calibrated on ambiguous punishment (Behrens et al., 2007): win-stay rate 0.681 versus 0.952 in data, accuracy 0.575 versus 0.731. Diagnostic analysis using the Eckstein et al. published parameter estimates (α = 0.76 instead of our calibrated 0.15) closed most of the gap, consistent with their finding that learning rates depend on task context while decision noise is more stable (β ratio 1.2×, α ratio 5.1×). This is not an independent reproduction — the platform's fixed α was calibrated elsewhere and is too low for this context by construction — but it demonstrates that the protocol's parameter classification would have flagged the non-transferability before the comparison, since α's optimum depends on feedback informativeness, which differs between protocols.

In sum: one positive result (arousal onset, uncalibrated), one structural gap (arousal variability), one withdrawal (arousal duration), three indeterminate or negative DDM results, and one negative RL result with a diagnosed mechanism. These results suggest that the protocol — classifying parameters, pre-registering criteria, mapping what the model cannot produce — yields informative outcomes regardless of whether individual comparisons succeed or fail. The protocol's value is demonstrated precisely by the negative and indeterminate results: each is interpretable (degeneracy, context dependence, structural absence) rather than opaque. Whether this generalizes beyond the specific models tested here remains to be seen.

---

# Methods: The protocol

## Overview

The protocol addresses a specific failure mode of modular computational platforms: a system that produces plausible output for every combination of parameters explains nothing, because it cannot be wrong. The value of a model comes from what it *refuses* to produce. Every instrument described below was introduced to solve a concrete problem encountered during development — not as a precaution, but as a response to a failure that had already occurred or was about to.

## Parameter classification

### The problem

The platform contains eleven parameters across three behavioral modules that were subjected to recovery testing. Without systematic testing, all would appear in the interface identically — as knobs the user can turn and, presumably, estimate from data. Of the eleven tested, five were classified as fitted (recoverable) and six as exploratory (recovery attempted, r < 0.7). A further two parameters are untested — their recoverability is unknown. The distinction matters: exploratory parameters have a definite negative result; untested ones have no result at all.

The arousal module illustrates the issue starkly. Its four candidate parameters (v_vh, v_vc, v_vm, v_mv) govern the sensitivity of sleep-wake switching to homeostatic pressure, circadian drive, and mutual inhibition between wake- and sleep-promoting populations. All four have clear physiological interpretations. Yet parameter recovery — generating data with known parameter values and attempting to estimate them back — yielded correlations of r = 0.42, 0.37, 0.31, and 0.05. The cause is not a defect of the estimation procedure (at zero noise, the same estimator recovers all four parameters with r = 1.0) but a property of the stochastic dynamics: flip-flop switching amplifies timing jitter near the threshold, where the derivative of the state variable is small. We observe this in our implementation; the geometric argument (σ_timing ≈ σ_noise / |dx/dt|) suggests it applies to any flip-flop model with similar dynamics, though we have not tested recovery on other implementations.

### The classification

Every parameter is assigned to one of three classes before external comparison:

- **Fitted** (recovery r ≥ 0.7): the parameter can be estimated from data of the model's own kind. It is displayed at full salience in the interface, with a minimum distinguishable difference (MDD) that sets the knob step size. The drift-diffusion model's three parameters (κ, a, T_er) are all fitted, with recovery correlations of 0.945, 0.971, and 0.992.

- **Exploratory** (recovery attempted, r < 0.7): the parameter can be varied and its effects observed, but it cannot be inferred from behavioral data. It is displayed with reduced salience and an explicit warning. All four arousal parameters are exploratory. The reinforcement learning module's mood-related parameters (τ_m, λ) are exploratory, with r ≈ 0.01.

- **Untested** (recovery not performed): the parameter's recoverability is unknown. It may be reclassified after testing.

The threshold r = 0.7 was chosen by us and fixed before recovery calculations were run. No standard threshold exists in the literature; Wilson and Collins (2019) recommend parameter recovery as a practice but do not specify a cutoff. Our choice reflects a judgment that correlations below 0.7 leave too much variance unexplained for individual-level estimation (r² < 0.49). The threshold was not adjusted after seeing recovery results. The classification is final for exploratory parameters: a failed recovery test cannot be retaken, because the result is a property of the model, not of sample size (except for the learning rate α, where r = 0.67 at N = 500 trials and r = 0.77 at N = 1,000, reflecting the discrete nature of the reversal signal).

### Consequences for comparison with data

When comparing the platform with external datasets, only fitted parameters are estimated from data. Exploratory parameters remain at their calibrated values, and any agreement or disagreement with data reflects the model's prediction, not a fit. This constraint proved consequential: the arousal module, with all parameters fixed, predicted sleep onset timing within 38–60 minutes of the data on data it was never calibrated on (Section Results), though with a fourfold underestimate of night-to-night variability.

## Deferred observables

### The problem

If a model is calibrated on observable X and then shown to reproduce X, the demonstration is circular. Deferred observables break this circularity: phenomena declared before implementation that are not used during calibration and serve as one-time predictions.

### What went wrong

Three complications arose in practice. First, a deferred observable can lose its independence. The arousal module's sleep-onset advance under deprivation was initially deferred, but the parameter v_vh was later adjusted to satisfy deprivation constraints — at which point onset advance became a consistency check, not an independent prediction. The demotion was recorded (D13) and the observable relabeled.

Second, a deferred test can be trivially circular. The mood autocorrelation test checked whether lag-1 autocorrelation of simulated mood matched the value from Rutledge et al. (2014). At λ = 0 (no mood-reward coupling), the measured autocorrelation is identically equal to the decay parameter γ, which was itself calibrated from Rutledge. The test's content reduces to verifying that the coupling does not destroy autocorrelation — informative, but not an independent prediction. It was marked as partially circular.

Third, a criterion can become defective at scale. The DDM's RT-equality test for correct and error responses used a Kolmogorov-Smirnov test, which at N ≈ 5,000 trials detected a Cohen's d = 0.04 difference — real but meaningless. The criterion was revised from statistical significance to effect size (|d| < 0.1), and the revision was recorded as post-factum (D17).

### The rule

Deferred observables are declared and committed before implementation. Once used for calibration, they are demoted to consistency checks and cannot be re-promoted. Revisions to criteria are named as revisions, with the original criterion, the replacement, and the reason recorded.

## Pre-registered comparison criteria

Comparison criteria for external data were fixed and committed to the repository before data were downloaded. The commit hash serves as a timestamp. Three criteria were revised after computation (D17: significance to effect size; D34: test parameter choice; D45: accuracy threshold from single SD to per-level SD). Each revision is recorded as post-factum with the original criterion, the defect, and the replacement. The revisions are listed in the paper alongside the results, not hidden in supplementary material.

Multiple gaps in the pre-registration were discovered during analysis. The sleep-scoring specification did not record the device, the activity count scale, or the algorithm's coefficients — forcing withdrawal of the duration comparison (D49). The dataset-classification specification did not define which columns count as physical difficulty — resolved by formalizing a criterion (D52). Three restored scripts (D48, D51) demonstrated that pre-registration reproducibility is graduated: exact reproduction when every step is specified (resolution script), divergent numbers when one step is unspecified (MLE recovery), and non-reproducibility when multiple steps are unspecified (sleep scoring).

## The impossible map

### The problem

A sweep of coupling strengths produces behavioral observables at each point. Without classification, a region where the model produces chance-level accuracy is visually indistinguishable from a region where it produces meaningful but different accuracy. The user sees a smooth surface and concludes that the model works everywhere.

### What the map shows

The impossible map classifies every grid point into one of seven categories:

1. **Normal**: observables are above chance and distinguishable from neighbors.
2. **Indistinguishable**: all three observables (accuracy, mean RT, win-stay rate) are flat relative to neighbors — different knob settings produce the same output.
3. **Chance**: accuracy is not significantly above 0.50 (binomial test).
4. **Unstable**: non-finite states (divergence).
5–7. **No data** from three distinct causes: sleep gate blocked all windows; κ_eff suppressed to near zero by coupling; all trials timed out.

The distinction between causes of absent data matters: sleep gating is a property of the circadian schedule (not the coupling), κ suppression is a property of the grid point (the coupling reduced drift to near zero), and timeout is a numerical safeguard. Collapsing them into a single "no data" label would misattribute the cause.

### What the map found

Across three 15×15 slices of the coupling-strength space, zero points were unstable and zero were at chance level. The UNSTABLE detector was validated by feeding a configuration with λ above the critical threshold, which produced the expected divergence. Zero instability is genuine: all couplings are unidirectional from arousal, making the Jacobian block-triangular and the eigenvalues independent of coupling strengths. This is an honest limitation of the current architecture, not a strong result — the stability check is nearly tautological when there is no feedback to the arousal module. A genuine test will arise only when bidirectional coupling is introduced.

The dominant finding was indistinguishability: 6–25% of points, highest for the κ–α pair (both modulated by the same source). Accuracy was the least sensitive observable; mean RT provided most of the discriminating power.

## Degeneracy identification

### The pattern

Five parameter pairs were identified where the model's behavior depends only on the product of two quantities, not on either individually:

**Table 1.**

| Product | Consequence |
|---------|------------|
| κ · a | Accuracy depends on κa, not on κ or a separately |
| β · ρ | Softmax depends on βρ |
| η · λ | Inter-trial variability and mood coupling |
| s · ρ | Drift bias depends on the product |
| κ · stimulus units | Drift rate depends on κ times difficulty in arbitrary units |

The first four were identified from the model equations and resolved by fixing one factor (a from calibration, ρ = 1 by convention, η deferred, ρ = 1). The fifth was predicted by the same rule and confirmed by the comparison: coherence-scale datasets form a tighter group (inter-dataset SD = 0.032) than the full multi-condition set (SD = 0.100), confirming that homogeneous stimulus units reduce spread. The model equation contains only the product κ × difficulty, so κ has no fixed scale when difficulty units change. This is not a failure of the model but an inherent limitation of the parameterization, predictable from the equation before any data analysis.

### The rule

A parameter that enters behavior only through a product with a quantity lacking an independent scale is flagged as inherently unidentifiable along that axis. The flag is raised at the design stage, not after fitting.

## Stability of the composite

Modules are coupled through parameter modulation: the output of one module modifies a parameter of another. Before connecting any two modules, the composite Jacobian is computed at a representative operating point, and the critical coupling strength (where the maximum real eigenvalue crosses zero) is found by bisection. The coupling-strength knob is restricted below this threshold.

For hybrid (switching) systems — the sleep-wake gate makes the composite piecewise-smooth — stability within each regime does not guarantee stability at transitions. The protocol requires numerical verification at switching boundaries: a simulation driven through the wake-sleep transition, checked for divergence, latch-up, and parasitic oscillations near the threshold (rule 19a).

In the current platform, this check was nearly tautological. All couplings flow unidirectionally from arousal to the decision and learning modules; no module feeds back to arousal. The Jacobian is block-triangular, and the composite eigenvalues are the union of the subsystem eigenvalues — they do not depend on coupling strengths at all. The only non-trivial stability result is the closed-loop DDM–RL circuit, where the mood-loop critical threshold λ_crit increases from 0.98 (standalone) to 1.1 (closed-loop, a 12% stabilization from value adaptation damping RPE fluctuations). A meaningful stability challenge will arise when bidirectional coupling is introduced — for example, attention feeding back to arousal, or mood modulating sleep latency. Until then, the stability machinery is in place but undertested.

---

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

---

# Results: Platform construction

## Calibration

Each module was calibrated against its canonical dataset using a closed procedure — a deterministic mapping from observables to parameters, without iterative optimization and without free choices (rule 21). Details are in Appendix A; here we report what calibration achieved and where it fell short.

The DDM was calibrated on trial-level reaction time data from monkey N (Roitman and Shadlen, 2002; 3,534 trials across six coherence levels). The closed procedure — logistic MLE on accuracy for κa, RT scaling for a, OLS residual for T_er — yielded κ = 14.72, a = 1.49, T_er = 0.318 s, with RMSE = 0.018 s on mean RT across coherences. Held-out prediction on monkey B passed all six coherence levels (accuracy |diff| ≤ 0.071, RT diff ≤ 14.3%).

The arousal module was calibrated on published parameters from Phillips and Robinson (2007, 2008), with v_vh adjusted from the original 0.19 to 0.325 to bring mean sleep duration into the target range of 7–9 hours (A2). Noise amplitude σ_h = 0.008 was calibrated to produce onset variability of 15–30 minutes circular SD (D18), matching laboratory data from Klerman and Dijk (2005).

The RL module was calibrated on summary statistics from Rutledge et al. (2014): α = 0.15, β = 2.50, producing win-stay rate 0.70 and recovery speed of 10 trials after reversal.

## Recoverability

Of the eleven parameters subjected to recovery testing across three modules, five were classified as fitted (r ≥ 0.7) and six as exploratory:

Two recovery procedures were used. Standalone recovery (generating and estimating from the module's own simulation at calibrated noise) tests identifiability under the module's own dynamics. MLE recovery (generating data and fitting with the Navarro-Fuss likelihood used for external comparison) tests the estimation procedure applied to real data. Both are reported; the MLE values are lower and determine the classification, since they reflect the procedure actually used for comparison.

**Table 2.**

| Module | Parameter | MLE r | Standalone r | Class | Note |
|--------|-----------|:-----:|:------------:|-------|------|
| DDM | κ (drift sensitivity) | 0.945 | 1.00 | **Fitted** | |
| DDM | a (threshold) | 0.971 | 1.00 | **Fitted** | Asymmetric MDD: 6% down, 3% up |
| DDM | T_er (non-decision) | 0.992 | 1.00 | **Fitted** | |
| RL | β (inverse temperature) | — | 0.90 | **Fitted** | Standalone only (softmax, no MLE) |
| RL | α (learning rate) | — | 0.67 / 0.77 | **Fitted at N ≥ 1000** | 0.67 at N = 500 |
| RL | τ_m (mood decay) | — | 0.01 | Exploratory | Not observable in short sequences |
| RL | λ (mood coupling) | — | 0.01 | Exploratory | Second-order effect |
| Arousal | v_vh (H sensitivity) | — | 0.42 | Exploratory | Flip-flop jitter |
| Arousal | v_vc (circadian coupling) | — | 0.37 | Exploratory | |
| Arousal | v_vm (MA→VLPO) | — | 0.31 | Exploratory | Correlated with v_mv (r = 0.62) |
| Arousal | v_mv (VLPO→MA) | — | 0.05 | **Exploratory** | From pair v_vm/v_mv, one exposed |

The arousal module illustrates why recovery matters independently of model correctness. Its four parameters have clear physiological interpretations — sensitivity to homeostatic pressure, circadian drive strength, mutual inhibition weights — and each produces observable effects when varied. Yet none is recoverable. The cause is geometric: near the switching threshold, the state variable's derivative is small, so timing noise is amplified (σ_timing ≈ σ_noise / |dx/dt|). At calibrated noise levels, the signal-to-noise ratio for transition timing is 0.9–3.8. This is not a deficiency of the estimator (at zero noise, all four recover with r = 1.0) but a property of the flip-flop architecture.

The learning rate α depends on observation length: at 500 trials (roughly one experimental session), r = 0.67 (exploratory); at 1,000 trials, r = 0.77 (fitted). The boundary is sharp because α is estimated from post-reversal recovery speed, which requires observing multiple reversals — a discrete signal whose precision improves with the number of reversals, not smoothly with trial count.

## Compromises and compatibility windows

Three calibration targets could not be satisfied simultaneously at a single parameter value, producing compatibility windows — regions where all targets are met — narrower than the full parameter range.

**Sleep architecture vs. deprivation response.** The parameter v_vh controls both sleep duration and the response to sleep deprivation. The compatibility window [0.325, 0.35] spans 6% of the parameter range [0.08, 0.50]. Below 0.325, the deprivation response is too weak (H_excess < 20%); above 0.35, sleep exceeds 9 hours. The lower boundary has no margin; the upper boundary has approximately 8%.

**Accuracy vs. recovery speed under value transfer.** The coupling strength s_value (value_diff → drift_bias) has an optimal accuracy at s ≈ 3 (+1.7% over baseline) but recovery speed degrades from s ≈ 2. A single "good" setting does not exist; the optimal accuracy point already has slower recovery than baseline.

**Onset variability vs. spurious transitions.** At the calibrated σ_h = 0.008, onset variability is 21.5 minutes (target 15–30) with zero spurious transitions. The two constraints do not conflict at this noise level — but this was not known before testing, and the absence of conflict is itself a result.

## Degeneracies

Five parameter products were identified where behavior depends on the product, not on either factor individually:

1. **κ · a**: accuracy = 1/(1 + exp(−κa · coherence)). Only the product κa determines accuracy at any given coherence. Resolved: a estimated independently from RT scaling.

2. **β · ρ**: softmax choice probability depends on β · ρ · ΔV. Resolved: ρ = 1 by convention.

3. **η · λ**: inter-trial variability η and mood coupling λ enter through a product in RPE variance. Resolved: η deferred (D16), λ estimated alone.

4. **s · ρ**: drift_bias = s · ρ · (V₁ − V₀). Resolved: ρ = 1.

5. **κ · stimulus units**: drift = κ · difficulty, where difficulty units (coherence, contrast, SNR) are arbitrary across datasets. Unlike cases 1–4, this cannot be resolved by fixing one factor — stimulus units are determined by the experiment, not the model. The consequence: κ has no fixed scale and is not comparable across datasets that use different difficulty measures.

All five follow the same pattern: a parameter enters the model equation only as a product with another quantity that lacks an independent scale. The pattern is identifiable from the equations before any data analysis or fitting. In cases 1–4, one factor was fixed analytically or by convention; in case 5, the limitation is inherent and was confirmed empirically (Section Results: Comparison).

## Inter-module effects

Three phenomena emerged from the composite that no module produces alone:

**Non-monotonic accuracy (D29).** Accuracy as a function of value-coupling strength s_value peaks at s ≈ 3 (+1.7%) and declines at s ≥ 10. The mechanism: moderate value transfer adds a second evidence signal concordant with the perceptual drift; strong value transfer overrides perception, and after a reversal, the stale value signal drives the DDM to the wrong boundary until values are relearned. Neither DDM alone (no value learning) nor RL alone (no perceptual signal) exhibits this trade-off.

**Coherence-dependent compensation (D39).** The value-transfer benefit grows as perceptual signal weakens: +0.7% at coherence 0.128, +7.0% at 0.064, +13.0% at 0.032. Learned values substitute for perceptual evidence precisely to the extent that evidence is weak. This is a direct consequence of the additive drift structure (drift = κ · coh · dir + s · ΔV), but the quantitative relationship — near-linear substitution — only appears in the composite simulation.

**Mood-loop stabilization (D30).** The critical mood-coupling threshold λ_crit increases from 0.98 (standalone RL) to 1.1 (closed-loop DDM–RL), a 12% stabilization. Value adaptation in the closed loop dampens RPE fluctuations: as values track the correct arm, RPE shrinks, and mood is driven less. This is counterintuitive — adding a feedback loop typically destabilizes — and is a consequence of the negative-feedback character of value learning.

## The impossible map

Three 15×15 slices of the coupling-strength space (s_kappa × s_value at s_alpha = 2; s_kappa × s_alpha at s_value = 3; s_alpha × s_value at s_kappa = 2) were computed using five 600-second windows of wakefulness at dt = 0.001, with behavioral observables (accuracy, mean RT, win-stay rate) recorded per window and aggregated.

Across all three slices (675 grid points total): zero unstable, zero at chance. The UNSTABLE detector was validated with λ above threshold — it produces the expected divergence. Zero instability is genuine but trivial: all couplings are unidirectional from arousal, the Jacobian is block-triangular, and eigenvalues do not depend on coupling strengths.

Indistinguishable points (where all three observables are flat relative to neighbors) ranged from 6% (alpha × value) to 25% (kappa × alpha). The kappa–alpha pair is most degenerate because both are modulated by the same source (arousal gain), producing correlated effects on RT. This is the numerical manifestation of the degeneracy identified algebraically above: when two coupling strengths modulate the same source, their effects are not independently observable. The impossible map does not add a new finding here — it confirms the algebraic prediction on a grid. Its distinct contribution is the classification of *causes* of absent data (gate vs. κ-suppression vs. timeout) and the verification that no instabilities exist, neither of which follows from the degeneracy rule alone.

Accuracy alone was insufficient for discrimination at the calibrated coherence of 0.128 (near ceiling); mean RT provided most of the discriminating power.

The map at lower coherences (0.064, 0.032) confirmed that the value-transfer interaction strengthens with task difficulty: at coherence 0.032, the s_value = 10 boost is +13% (vs +0.7% at 0.128). No CHANCE regions appeared at any coherence tested — even at 0.032 with s_kappa = 4, the residual drift (κ_eff · coh ≈ 0.21) keeps accuracy above 0.50 with thousands of trials. CHANCE would require coherence near zero or coupling strengths beyond the tested range.

---

# Results: Comparison with data

Comparison criteria were pre-registered and committed to the repository before any data were downloaded (commits fff056d, 6aefd62). Three criteria were revised after computation; each revision is noted below alongside the result it affects.

## Decision model: Confidence Database

### Data and procedure

From 97 downloaded datasets (Rahnev et al., 2020), 82 passed the mechanical selection criteria (2AFC perceptual, RT available, ≥ 200 trials per participant). Accuracy coding rules were fixed per dataset in a separate file before any metric was computed; 13 datasets were excluded mechanically (not 2AFC or continuous response) and 2 by judgment (ambiguous readme). Multi/single classification was formalized as a separate decision (D52): a dataset is multi-condition if at least one column encodes physical stimulus difficulty (contrast, coherence, SOA, noise level, dot difference, orientation magnitude); columns encoding task manipulation (cue validity, speed/accuracy instruction, block order) do not qualify. This criterion was applied to 18 ambiguous datasets, with per-dataset decisions recorded in a separate override file. A case-sensitivity bug in the original classification script was also corrected (D52a), affecting four datasets. The resulting 73 datasets with valid accuracy and RT comprised 53 multi-condition and 20 single-condition sets, totaling 4,066 participants.

Parameters κ, a, and T_er were estimated per participant by maximum likelihood on the Navarro-Fuss first-passage density (Navarro and Fuss, 2009), with z = 0 fixed and boundaries at ±a/2. The MLE implementation was validated by synthetic recovery before application to real data: at N = 500 trials, recovery correlations were r = 0.945 (κ), 0.971 (a), and 0.992 (T_er), with cross-correlations below 0.10. A sign error in the density exponent was caught during this recovery step and corrected (D46) — without the synthetic check, it would have produced plausible but systematically biased estimates on real data.

### Thresholds

Agreement thresholds were derived from inter-dataset variability, computed before fitting. The original threshold (1 × SD of per-dataset mean accuracy = 0.047) was revised after discovering that averaging over difficulty levels deflates the spread: datasets with different mixes of easy and hard conditions have similar means regardless of task properties. The revision (D45) replaced the single threshold with three per-level thresholds, computed before any model fitting:

**Table 3.**

| Category | Accuracy SD | Threshold | Quality |
|----------|:-----------:|:---------:|---------|
| Single-condition (20 datasets) | 0.038 | ±0.038 | Informative |
| Multi-condition, easiest level (53) | 0.100 | ±0.100 | Moderate |
| Multi-condition, hardest level (53) | 0.257 | ±0.257 | Uninformative |
| Coherence-scale only (10) | 0.032 | ±0.044 | Informative |

The RT threshold (±0.535 s, 51% of mean) was uniformly weak: datasets differ in apparatus and task design more than the model deviates from any one of them. This threshold is reported for completeness but does not discriminate.

### Results

**Single-condition datasets (20).** MAE between model-predicted and observed accuracy: median 0.041, above threshold 0.038 (ratio 1.08). Eight of twenty datasets fall within threshold, CI [0.2, 0.6]. The result is indeterminate: the median exceeds the threshold marginally, and the confidence interval on the within-threshold fraction spans both sides of 50%.

**Multi-condition, coherence scale (10).** MAE median 0.086, above threshold 0.044 (ratio 1.95). These datasets use the same physical quantity — motion coherence — that the model equation contains. Despite this structural advantage, the model's accuracy deviation is nearly twice the inter-dataset spread. The coherence-scale accuracy SD is 0.032, lower than the full multi-condition spread (0.100), confirming that coherence datasets form a tighter group — but the model does not meet even this tighter standard.

**Multi-condition, all scales (53).** MAE median 0.102, at threshold 0.100 (ratio 1.02). The fraction within threshold: 49%, CI [35.8%, 62.3%]. This is indeterminate: when the median nearly equals the threshold, approximately half the datasets fall on each side by construction.

**No positive DDM result.** None of the three DDM comparisons — single-condition, coherence, or multi-condition — produced a result within threshold. The single-condition comparison was borderline (MAE exceeds threshold by 0.003); the multi-condition comparison was borderline (MAE exceeds threshold by 0.002); the coherence comparison, which in an earlier analysis had been the strongest result, showed a clear deficit (MAE nearly double the threshold). The pattern across all three is consistent with a model that captures the qualitative accuracy-difficulty relationship but deviates at a level comparable to or exceeding inter-dataset variability.

**Scale dependence.** The structural cause remains: the model equation contains only the product κ × difficulty, and difficulty units are arbitrary across datasets (the fifth degeneracy). Coherence-scale datasets form a tighter group (SD = 0.032 vs full SD = 0.100), confirming that homogeneous stimulus units reduce inter-dataset spread, but the model's MAE on coherence datasets (0.086) substantially exceeds this spread. The main variability between datasets comes from populations, apparatus, and instructions, not from the difficulty scale.

## Arousal model: OBF-Psychiatric

### Data and procedure

The OBF-Psychiatric actigraphy dataset (Garcia-Ceja et al., 2025) contains minute-by-minute motor activity recordings from 32 healthy controls, collected with Actiwatch AW4 (Cambridge Neurotechnology) at 32 Hz, aggregated to 1-minute epochs. All model parameters were fixed at calibrated values (all exploratory, no fitting).

### Withdrawn result: mean sleep duration

The comparison of mean sleep duration was withdrawn because the sleep-scoring code used in the original analysis was lost (D48), and the pre-registration did not specify the algorithm with sufficient precision to reconstruct it: it contained two contradicting threshold values (15th and 80th percentile), did not record the actigraph model or activity count scale, and did not specify gap-merging or episode-filtering rules. Attempting to reconstruct the algorithm yielded four variants with durations ranging from 6.3 h to 8.5 h depending on parameters (A15). All published epoch-by-epoch scoring algorithms (Cole-Kripke, Sadeh, Oakley) have coefficients calibrated to specific devices (pyActigraphy documentation: "weights and thresholds are very much device and population dependent"); none are applicable to Actiwatch AW4 counts (scale 0–4245) without recalibration, and the dataset authors did not score sleep.

This is reported as a pre-registration gap: the specification required fixing the scoring algorithm but did not require specifying the device, the activity count scale, or the provenance of the algorithm's coefficients. An algorithm is fully specified only together with the device it was calibrated on; without this, the specification is incomplete (D49).

### Retained results with robustness check

The following results are retained because they are stable across all three attempted scoring variants. The robustness table below shows that while mean duration varies by 0.7 h across variants (confirming the sensitivity of the withdrawn comparison), onset timing and variance decomposition are insensitive to the choice of algorithm.

**Table 4.**

| Scoring variant | Onset (circ. mean) | Onset within-SD | Duration within-SD | Duration between-SD |
|-----------------|:------------------:|:---------------:|:------------------:|:-------------------:|
| 15th pctl. nonzero activity | 23:54 | 103 min | 2.18 h | 1.01 h |
| Fixed threshold 50 | 23:39 | 103 min | 2.18 h | 0.98 h |
| 15th pctl. nighttime nonzero | 00:01 | 104 min | 2.15 h | 1.04 h |
| **Range across variants** | **22 min** | **1 min** | **0.03 h** | **0.06 h** |

**Sleep onset (uncalibrated prediction).** Model: approximately 23:00. Data circular mean: 23:39–00:01 across three scoring variants (a 22-minute range). Model-data difference: 38–60 minutes. Unlike mean duration, where v_vh was adjusted to bring sleep into the 7–9 h range, sleep onset timing was not calibrated to any data in this comparison — the circadian phase α was taken from Phillips and Robinson (2008) and not adjusted. The onset agreement is therefore a genuine prediction at fully fixed parameters, and the retained result is in this sense stronger than the withdrawn one.

**Onset variability (structural gap).** Within-individual onset circular SD: 103–104 minutes across variants. Model: 21.5 minutes (calibrated on laboratory data from Klerman and Dijk, 2005). The 4.8× discrepancy is a structural limitation: the model contains only internal stochasticity; free-living variability is dominated by sources it does not represent (social schedule, weekday/weekend differences, caffeine, alcohol, screen exposure). Increasing σ_h would widen the distribution by the wrong mechanism. The laboratory-vs-free-living comparison has not been tested; a laboratory actigraphy dataset would permit the test.

**Variance decomposition (data property).** Within-individual SD of sleep duration (2.15–2.18 h) exceeds between-individual SD (0.98–1.04 h) by a factor of approximately two across all three scoring variants: a person varies more from night to night than people differ by their averages. The model's SD (0.50 h) is comparable to the between-individual component but captures almost none of the within-individual component. Individual-level parameter fitting from a few nights would be dominated by within-person variability even if the parameters were recoverable. This is an observation about the data, not a model comparison, and it is insensitive to the scoring algorithm.

## Reinforcement learning: Eckstein et al. 2022

### Data and procedure

Trial-level data from 291 participants (Eckstein et al., 2022; ages 8–30) performing a probabilistic reversal learning task: 2AFC, reward probability 0.75 for the correct box and 0.00 for the incorrect box, criterion-based reversals (7–15 collected rewards), 120 trials per participant. Analysis was restricted to the 114 adults (age ≥ 18). All model parameters were fixed at values calibrated on the Behrens et al. (2007) protocol (α = 0.15, β = 2.50, τ_m = 2.163, λ = 0.49); no parameters were estimated from the Eckstein data (120 trials < 500 threshold for β recovery). The environment was configured in criterion mode (p_incorrect = 0.0, criterion range [7, 15]) to match the Eckstein protocol.

### Results

**Table 5.**

| Observable | Data (114 adults) | Model (calibrated) | Difference |
|------------|:-----------------:|:------------------:|:----------:|
| Win-stay | 0.952 | 0.681 | −0.271 |
| Lose-shift | 0.555 | 0.493 | −0.062 |
| Accuracy | 0.731 | 0.575 | −0.156 |
| Recovery (trials) | 6.6 | 14.5 | ×2.2 |

The model systematically underperforms on all observables except lose-shift (which falls within the data IQR). The cause: parameters calibrated on the Behrens protocol (ambiguous punishment, p_incorrect = 0.25) are suboptimal for the Eckstein protocol (unambiguous punishment, p_incorrect = 0.0), where each trial carries more information and faster learning is appropriate.

### Diagnostic

Using the Eckstein et al. published parameter estimates for this task (α = 0.76, β ≈ 3.0, persistence = 0.30) with our model structure, the gap largely closes: WS = 0.937, ACC = 0.726, recovery = 7.5 trials. The dominant contribution is α (accounts for 84% of the WS gap); β and persistence contribute the remaining 16% jointly.

While the ratios between our calibrated values and the Eckstein estimates — α: 5.1×, β: 1.2× — display the same pattern as the Eckstein et al. (eLife, 2022) finding that decision noise transfers across tasks while learning rate does not, this is not an independent replication: the diagnostic uses their published estimates as input, and the negative result on α follows by construction from using a single α calibrated on a different protocol. The consistency arises from a different model (Rescorla-Wagner vs. their four-parameter RL) and a different calibration source (Behrens vs. their multi-task battery), which makes the convergence informative despite the lack of independence. The protocol's parameter classification would have flagged this before the comparison: α's optimal value depends on feedback informativeness (p_incorrect determines how much each trial reveals about the correct arm), which differs between protocols. This is not a degeneracy of the parameterization (unlike κ × difficulty) but a dependence of the optimum on task context.

Note: the Eckstein et al. α is split by valence (α+ = 0.77, α− = 0.62); our model uses a single α. The comparison is approximate.

---

# Discussion

## Recoverability and predictive accuracy are different properties

The central empirical result is a dissociation, sharper than initially expected. The arousal module — where no parameter is recoverable from behavioral data (highest r = 0.42) — predicted sleep onset timing within 38–60 minutes of the data (model 23:00, data 23:39–00:01 across three scoring variants), at fully fixed parameters. Onset timing was not calibrated to any data in this comparison: the circadian phase was taken from Phillips and Robinson (2008) and not adjusted. This is the only comparison in the study that produced a clean prediction at fixed parameters on uncalibrated data.

The DDM — where all three parameters are recoverable (r ≥ 0.945) — produced no positive result on any subset of the data. Single-condition datasets: MAE 0.041, above threshold 0.038. Multi-condition coherence-scale datasets — the subset where the model equation maps directly to the stimulus variable: MAE 0.086, nearly double the threshold 0.044. Multi-condition all scales: MAE 0.102, at threshold 0.100 (indeterminate). The RL module, tested on a different protocol with all parameters fixed, showed systematic underperformance (win-stay 0.681 vs 0.952).

The dissociation is not between "positive" and "indeterminate" results. It is between a module with no recoverable parameters that produced the study's only clean prediction and a module with all parameters recoverable that failed to produce any positive result — even on datasets using the same physical difficulty scale as the calibration source. This is the opposite of what parametric phenotyping assumes: that recoverability is the prerequisite for predictive value.

The practical consequence for parametric phenotyping remains. The arousal model predicts group-level onset timing without allowing individual-level parameter estimation — while simultaneously failing on within-individual variability by a factor of four (model SD 0.50 h vs data 2.15–2.18 h). DDM parameters are individually estimable but do not guarantee accurate prediction on new data. Group adequacy does not entail individual estimability; individual estimability does not entail group adequacy.

The arousal result comes with two caveats. First, the strongest arousal metric — mean sleep duration — was withdrawn (D49) because the sleep-scoring algorithm used in the original analysis was lost and could not be reconstructed from the pre-registration, which did not specify the device, the activity count scale, or the algorithm's coefficients. The retained onset prediction is in one sense stronger (never calibrated), but the overall arousal evidence is narrower than originally planned. Second, v_vh was adjusted within its calibration range to bring sleep duration into the 7–9 h range — the duration agreement that motivated the original comparison was partly a calibration consequence. The onset prediction, being uncalibrated, does not share this caveat.

On variability, the model shows a structural gap: within-individual night-to-night SD is 2.15–2.18 h in data versus 0.50 h in the model (4×), and onset circular SD is 103–104 minutes versus 21.5 minutes (4.8×). This is not a calibration issue — increasing σ_h would widen the distribution by the wrong mechanism. The model contains only internal stochasticity; free-living variability is dominated by sources it does not represent (social schedule, weekday/weekend differences, caffeine, alcohol, screen exposure). The observation that within-individual variability (2.15–2.18 h) exceeds between-individual variability (0.98–1.04 h) by a factor of approximately two further limits the prospects of individual fitting: even if parameters were recoverable, a few nights of data would be dominated by within-person noise.

## Pre-registration specificity is graduated, not binary

Four analysis scripts were lost and restored from the pre-registration (D48, D51). The restorations, together with two further problems — a specification gap (D49) and a stale-cache failure discovered during end-to-end reproduction (D53) — produced four qualitatively different outcomes that constitute a graduated test of reproducibility:

**Table 6.**

| Case | Unspecified or unversioned dependency | Outcome |
|--------|-------------------|---------|
| RL resolution (D51) | None — grid, sample size, power threshold all fixed | Exact reproduction (bitwise match) |
| MLE recovery (D48) | Synthetic data generation method, parameter ranges | Different numbers, same classifications (all three DDM parameters remained fitted) |
| Sleep scoring (D49) | Device, activity count scale, algorithm coefficients, threshold definition | Non-reproducible — forced withdrawal of the strongest arousal metric |
| Arousal recovery (D53) | None unspecified — code, seeds, and package versions all preserved | Silently non-reproducible — stale cached intermediates loaded by the script |

A fifth gap — which dataset columns count as physical difficulty (D52) — was reparable: formalizing the criterion changed dataset counts and verdicts (43/30 → 53/20 multi/single) but preserved the analysis structure.

The fourth case (D53) is qualitatively different from the first three. The arousal recovery script was fully specified, committed, and version-controlled; the seeds were fixed; the package versions matched. Yet running it from scratch produced different r-values (0.42, 0.37, 0.31 vs the committed 0.41, 0.35, 0.29) because the script cached intermediate simulation results in `.npy` files that were not tracked by version control and silently loaded on re-run. The committed values were a hybrid of old cached simulations and current analysis code. The discrepancy was invisible to unit tests (which verified single-trajectory determinism, not full-pipeline reproducibility), to number verification (which checked text against results, not results against a clean recomputation), and to red-team review (which checked classifications, not the raw r-values). It was discovered only by following the reproduction instructions from Appendix H on a clean state — the one check that exercises the full pipeline without any cached state.

The pattern is that pre-registration reproducibility degrades in proportion to the number of unspecified processing steps, not as a binary property — but "unspecified" must include implicit dependencies, not only explicit parameters. A specification that fixes every step reproduces exactly. One that leaves a single step unspecified produces different numbers but preserves conclusions. One that leaves multiple steps unspecified does not reproduce at all. And one that specifies every step but silently depends on unversioned intermediates produces plausible but incorrect numbers — the most dangerous failure mode, because it passes every check except clean reproduction.

This observation extends a principle familiar from the replication-crisis literature ("researcher degrees of freedom," "garden of forking paths") in a specific direction: the same discipline that rule 14 applies to model parameters — report as a function, not at a single value — applies to data-processing pipelines. Every step that affects the output must be specified to the point of unique determination, and every intermediate that persists between runs must either be regenerated or validated against the code that produced it. The four cases show what happens when this standard is met, approximately met, not met, and apparently met but silently violated. We observed this in five instances from a single project; whether the graduated relationship holds generally is an empirical question, but the mechanism — each unspecified or unversioned dependency multiplies the space of defensible outputs — is general.

## Degeneracies as a predictable class

Five instances of the same pattern were identified: κ·a, β·ρ, η·λ, s·ρ, and κ·stimulus-units. In each case, a parameter enters the model equation only through a product with a quantity that lacks an independent scale, and the parameter is therefore unidentifiable along that axis. Four were identified from the equations at the design stage and resolved by fixing one factor. The fifth — κ·stimulus-units — was predicted by the same rule and confirmed by the comparison: coherence-scale datasets, which share the stimulus units of the calibration source, form a tighter group (inter-dataset SD = 0.032) than the full multi-condition set (SD = 0.100). The model still fails on this tighter group (MAE 0.086, nearly double the threshold 0.044), but the structural prediction holds: the main source of inter-dataset variability is populations and apparatus, and κ has no fixed scale when difficulty units change.

The pattern is simple enough to check on paper before writing code. For any proposed model, list every product of a free parameter with another quantity; if the other quantity has no independently measurable scale (because it is another free parameter, a convention, or an external unit), the product is identified but its factors are not. This rule would have predicted the Eckstein et al. (eLife, 2022) non-transferability finding for learning rate: α multiplies reward prediction error, whose scale depends on the reward schedule, which differs between tasks. The rule does not predict the relative stability of β, which multiplies value differences whose scale is set by the same α — the full picture requires considering the information geometry, not just the product structure.

## What the model does not reproduce

Eleven observables available in the data or the literature are absent from the model's output. They fall into three categories that the standard "limitations" section does not distinguish:

**Model limitation** (removable by extending the model): RT quantile profile, right-skew pattern, and slow errors — all absent because the DDM lacks inter-trial variability (η, deferred as D16). The extension is known (Ratcliff and Tuerlinckx, 2002) and deferred because composite behavior does not depend on RT distribution shape in the current architecture.

**Structural absence** (requires new components, not parameter adjustment): confidence ratings (no metacognition module), NREM/REM sleep structure (two-process model by design), free-living sleep variability (no external variability sources). These cannot be addressed by recalibration.

**Calibration-context dependence** (model structure is adequate, but optimal parameter values depend on context): behavior under unambiguous punishment at Behrens-calibrated parameters. The model with Eckstein-estimated parameters reproduces the data; the discrepancy is between calibration contexts, not between model and phenomenon. The protocol's parameter classification flags this in advance: α's optimum depends on feedback informativeness, which differs between the Behrens and Eckstein protocols.

This taxonomy matters because the three types call for different responses. Model limitations have known solutions on a deferred timeline. Structural absences require architectural decisions (rule 8: escalation to the author). Context dependence requires either re-calibration on the target context (destroying the prediction) or acceptance that the parameter is context-specific — which is itself a finding, not a failure.

## No standard threshold for recoverability

The recovery threshold r = 0.7 used throughout this work has no precedent in the literature. Wilson and Collins (2019) recommend parameter recovery as a practice but do not specify a cutoff. No other computational psychiatry platform we are aware of reports recovery correlations systematically or classifies parameters by recoverability before external comparison. We surveyed five platforms that model neural or cognitive dynamics at scales relevant to psychiatry — The Virtual Brain (Sanz Leon et al., 2013), Dynamic Causal Modelling in SPM (Friston et al., 2003), Nengo/Spaun (Eliasmith et al., 2012), Neuroblox.jl (Hofmann et al., 2025), and whole-brain mean-field models (Deco and Kringelbach, 2020) — and found none that report recovery correlations for behavioral parameters or classify parameters by recoverability before external comparison. Individual modeling toolboxes (HDDM: Wiecki et al., 2013; hBayesDM: Ahn et al., 2017) implement parameter estimation but do not include recovery testing as part of the modeling protocol.

This absence supports the paper's framing: if the protocol we describe were standard practice, such a threshold would exist — debated, perhaps, but established. Its absence indicates that the field does not routinely distinguish recoverable from non-recoverable parameters before using them for individual-level inference. The consequence is visible in our own results: six of eleven tested parameters turned out non-recoverable, and without the recovery step, all eleven would have appeared in the interface identically.

## Limitations

**Stability checking is undertested.** All couplings in the current platform flow unidirectionally from arousal. The composite Jacobian is block-triangular, and eigenvalues are the union of subsystem eigenvalues — they do not depend on coupling strengths. The stability machinery is in place but has not been meaningfully challenged. A genuine test requires bidirectional coupling (attention → arousal, mood → sleep latency), which is deferred.

**Calibrations rely partly on published summaries.** The arousal module is calibrated on parameters from Phillips and Robinson (2007, 2008) and noise levels from Klerman and Dijk (2005); the RL module on summary statistics from Rutledge et al. (2014). Only the DDM was calibrated on trial-level data (Roitman and Shadlen, 2002). Calibration on summaries loses information and prevents per-subject fitting even where recovery would permit it.

**The RL comparison is extrapolative by construction.** Parameters calibrated on one protocol (Behrens, ambiguous punishment) are tested on another (Eckstein, unambiguous). The mismatch is predicted by the protocol but does not constitute a test of the model on its native ground. A within-protocol comparison — calibrating and testing on splits of the same dataset — was precluded by the 120-trial length (below the 500-trial threshold for β recovery).

**BAA is connected unilaterally.** The attention module receives arousal gain but produces no output to the behavioral loop. It demonstrates that the platform accepts externally developed modules, but it does not contribute to any behavioral prediction or comparison. It has untested parameters and no external validation.

**Arousal evidence is narrow.** The strongest arousal metric (mean sleep duration) was withdrawn because the scoring algorithm could not be reconstructed from the pre-registration (D49). The retained onset prediction is clean (uncalibrated, stable across scoring variants) but is a single observable. The structural gap in variability (model 0.50 h vs data 2.15–2.18 h within-individual SD) is a known limitation. The overall arousal comparison rests on one positive result and one structural gap, from a module where no parameter is recoverable — a thin empirical basis for the dissociation claim, though the claim concerns the *direction* of the contrast (which module does better), not the *magnitude* of arousal agreement.

**Onset variability prediction not tested.** The model predicts within-individual onset variability of 21.5 minutes (calibrated on laboratory data from Klerman and Dijk, 2005). The data show 103–104 minutes in free-living actigraphy — a 4.8× discrepancy. However, laboratory and free-living conditions differ in kind, not in degree: the prediction has not been tested against data from comparable conditions. A laboratory actigraphy dataset would permit the test.

## What comes next

Three directions are immediate, none requiring new methodology:

First, **extending the DDM with inter-trial variability** (η, and potentially s_z, s_t) as a separate module would address the three "model limitation" items — RT quantiles, right-skew, slow errors — and enable comparison on the shape of RT distributions, not only their means.

Second, **pharmacological parameter transformations** — θ(dose, time) overlaid on module parameters — are the original motivation for the platform. The parameter classification constrains which pharmacological targets are informative: a drug acting on a fitted parameter (κ, a, T_er, β) produces estimable effects; one acting on an exploratory parameter (v_vh, τ_m) produces effects that can be simulated but not recovered from behavioral data. The same drug acting on an untested parameter (BAA gain) has unknown estimability.

Third, **clinical comparisons** — the arousal module against depression or ADHD actigraphy, the RL module against clinical reversal-learning data — are enabled by the OBF-Psychiatric dataset (which includes depressed, schizophrenic, and ADHD groups alongside the controls used here) and by clinical reversal-learning datasets in the literature. The caveat from the RL comparison applies: parameters calibrated on healthy adults may not transfer to clinical populations, and the protocol would flag this before the comparison begins.

---

# Appendices

## Appendix A: Module equations

### A.1 Arousal (Phillips–Robinson two-process flip-flop)

Source: Phillips and Robinson (2007, *J Biol Rhythms* 22:167–179; 2008, *J Theor Biol* 255:413–423). Equations verified against deposited CellML and BioModels implementations (specs/arousal\_sources.md).

**State variables** (all CONTINUOUS):

**Table 7.**

| Variable | Symbol | Initial | Units |
|----------|--------|---------|-------|
| VLPO potential | $V_v$ | 0.0 | mV |
| MA potential | $V_m$ | 0.0 | mV |
| Homeostatic pressure | $H$ | 15.0 | nM |
| Clock | $t_{clock}$ | 0.0 | s |

**Drift:**

$$\frac{dV_v}{dt} = \frac{\nu_{vm}\,Q(V_m) + D(t) - V_v}{\tau_v}$$

$$\frac{dV_m}{dt} = \frac{A + \nu_{mv}\,Q(V_v) - V_m}{\tau_m}$$

$$\frac{dH}{dt} = \frac{\mu\,Q(V_m) - H}{\chi}$$

**Algebraic relations:**

$$D(t) = \nu_{vc}\,C(t) + \nu_{vh}\,H$$

$$C(t) = c_0 + \cos\!\bigl(\omega\,(t - \alpha)\bigr)$$

$$Q(V) = \frac{Q_{\max}}{1 + \exp\!\bigl(-(V - \theta)/\sigma_q\bigr)}$$

$$A = \nu_{ma}\,Q(V_{a0}) \approx 1.0\;\text{mV}$$

**Diffusion:** $\boldsymbol{\sigma} = (\sigma_{vv},\;\sigma_{vm},\;\sigma_h)$, additive, not in the original model.

**Outputs:**

$$\text{gain} = Q(V_m)\,/\,Q_{\text{REF}}, \qquad Q_{\text{REF}} = 4.2883\;\text{s}^{-1}$$

$$\text{is\_wake} = \bigl[Q(V_m) > 1.0\bigr]$$

$Q_{\text{REF}}$ is computed once at baseline parameters (procedure in specs/arousal.md §Calibration constant) and stored in results/arousal\_calibration.json. It is not recomputed when user parameters change.

### A.2 DDM (drift-diffusion model)

Source: Ratcliff (1978, *Psychol Rev* 85:59–108); Ratcliff and McKoon (2008, *Neural Comput* 20:873–922). Analytical formulae from Bogacz et al. (2006, *Psychol Rev* 113:700–765).

**State variables** (all EPISODIC — reset each trial):

**Table 8.**

| Variable | Symbol | Initial |
|----------|--------|---------|
| Evidence | $x$ | 0 |
| Trial clock | $t_{\text{trial}}$ | 0 |

**Drift:**

$$dx = v\,dt + dW, \qquad \sigma = 1\;\text{(fixed)}$$

$$v = \kappa \cdot \text{coherence} \cdot \text{direction}$$

**Boundaries:** absorbing at $\pm a/2$. Trial terminates when $|x| \geq a/2$.

**Composite extension** (P3): $v = \kappa \cdot \text{coh} \cdot \text{dir} + s_{\text{value}} \cdot (V_1 - V_0)$.

**Calibration** (closed-form, no iteration):

1. $\kappa a = 21.96$ from logistic MLE on accuracy × coherence (strictly concave → unique).
2. $a^2 = 2.225$ from OLS on mean RT vs analytical $E[DT]$; $a = 1.4916$, $\kappa = 14.7242$.
3. $T_{er} = 0.318$ s from OLS intercept.

RMSE = 0.018 s. 1 MTU = 1 second. Data: monkey N (Roitman and Shadlen, 2002; 3534 trials).

### A.3 RL+Mood (Rescorla–Wagner + momentary affect)

Sources: Rescorla and Wagner (1972); Rutledge et al. (2014, *PNAS* 111:12252–12257); Eldar et al. (2016, *Trends Cogn Sci* 20:15–24).

**State variables** (all CONTINUOUS — no episodic reset):

**Table 9.**

| Variable | Symbol | Initial |
|----------|--------|---------|
| Value arm 0 | $V_0$ | 0.5 |
| Value arm 1 | $V_1$ | 0.5 |
| Mood | $m$ | 0.0 |

**Drift:** $\dot{V}_0 = \dot{V}_1 = \dot{m} = 0$. All dynamics are discrete (trial\_update).

**Diffusion:** $(0,\;0,\;\sigma_m)$ — additive noise on mood only.

**Trial update** (at each trial boundary):

$$\rho_{\text{eff}} = 1 + \lambda\,m$$

$$\delta = \rho_{\text{eff}} \cdot r - V_{\text{chosen}}$$

$$V_{\text{chosen}} \leftarrow V_{\text{chosen}} + \alpha\,\delta$$

$$m \leftarrow \gamma\,(m + \delta), \qquad \gamma = e^{-1/\tau_m}$$

**Choice rule** (standalone): $P(\text{action}=a) = \exp(\beta\,V_a)\,/\,\Sigma\exp(\beta\,V_i)$.

In composite (P3+), DDM replaces softmax; choice is determined by boundary crossing.

**Stability:** eigenvalue $= \gamma\,(1 + \lambda\,r_{\text{mean}})$. Critical: $\lambda_{\text{crit}} = (1 - \gamma)\,/\,(\gamma\,r_{\text{mean}})$.

**Scale conventions:** $\rho = 1$ (fixed, degenerate with $\beta$), $\eta = 1$ (fixed, degenerate with $\lambda$), $m_{\text{eq}} = 0$.

### A.4 BAA (boundary-attention architecture)

Source: Vasilchenko, internal project (baa/). Integrated as a passive module (D33, D40, D42).

**State variables** ($n_c = 5$ channels):

**Table 10.**

| Variable | Count | Initial |
|----------|-------|---------|
| Activation $A_i$ | $n_c$ | 0.0 |
| Somatic energy $S$ | 1 | 100.0 |
| Hebbian weight $W_i$ | $n_c$ | 1.0 |

**Drift:**

$$\frac{dA_i}{dt} = g\,I_i\,W_i - (\beta_{\text{BAA}} - \gamma_{\text{BAA}})\,A_i - \gamma_{\text{BAA}}\sum_j A_j$$

$$\frac{dS}{dt} = R_{\text{rec}}\,(S_{\max} - S) - k_{\text{cost}}\sum_j A_j$$

$$\frac{dW_i}{dt} = \eta_{\text{BAA}}\,(A_i\,I_i - A_i^2\,W_i)$$

**Diffusion:** $\sigma(S) = \sigma_{\text{base}} + (\sigma_{\max} - \sigma_{\text{base}})\,/\,(1 + e^{0.2(S - S_{\text{crit}})})$, on $A$ channels only.

**Boundaries:** clipping (projection) — $A_i \geq 0$, $0 \leq S \leq S_{\max}$, $W_{\min} \leq W_i \leq W_{\max}$ (D31).

**Input:** $I_i = \text{tonic\_input}$ (D40). No external stimulus in current configuration.

**Coupling:** arousal.gain → BAA.gain, multiplicative. BAA.gain is untested (D33).

**Status:** passive. BAA does not contribute to the behavioral loop (D42).


## Appendix B: Full parameter recovery tables

All recovery results from grid-search estimation on synthetic data generated by the model itself (Wilson and Collins, 2019). Threshold: $r \geq 0.7$ → fitted; $r < 0.7$ with recovery conducted → exploratory; recovery not conducted → untested.

### B.1 Arousal module

Procedure: $5^3 \times 20$ replicates, 7-day simulations (604800 steps, dt = 1 s), 35 observables per replicate (14 transition moments, 14 durations, 7 gain swings). Grid search on $(\nu_{vh}, \nu_{vc}, \nu_{vm})$, $\nu_{mv}$ determined by $\nu_{vm}$–$\nu_{mv}$ pair.

**Table 11.**

| Parameter | Grid | $r$ | RMSE | Max cross-corr. | Class |
|-----------|------|----:|-----:|:----------------:|-------|
| $\nu_{vh}$ | [0.08, 0.50] × 5 | 0.42 | 0.170 | 0.26 ($\nu_{mv}$) | **exploratory** |
| $\nu_{vc}$ | [−10.0, −1.0] × 5 | 0.37 | 4.241 | 0.56 ($\nu_{vm}$) | **exploratory** |
| $\nu_{vm}$ | [−2.70, −0.50] × 5 | 0.31 | 0.864 | 0.56 ($\nu_{vc}$) | **exploratory** |
| $\nu_{mv}$ | [−5.00, −0.90] × 5 | 0.05 | 2.271 | 0.62 ($\nu_{vm}$) | **exploratory** |

Deterministic recovery ($\sigma = 0$): $r = 1.00$ for all four → estimator is correct; non-identifiability is caused by stochastic noise (SNR 0.9–3.8). This is a property of flip-flop models, not only this implementation.

### B.2 DDM module

Procedure: $5^3 \times 20$ replicates, 3500 trials per replicate (~590 per coherence level, matching monkey N). 12 observables (6 accuracy + 6 mean RT).

**Table 12.**

| Parameter | Grid | $r$ | RMSE | Max cross-corr. | Class |
|-----------|------|----:|-----:|:----------------:|-------|
| $\kappa$ | [5.0, 30.0] × 5 | 0.997 | 0.740 | < 0.001 | **fitted** |
| $a$ | [0.5, 3.0] × 5 | 1.000 | 0.013 | < 0.001 | **fitted** |
| $T_{er}$ | [0.1, 0.5] × 5 | 1.000 | 0.002 | < 0.001 | **fitted** |

Resolution (minimum detectable difference at 80% discrimination, 3500 trials):

**Table 13.**

| Parameter | MDD | MDD % | Recommended knob step |
|-----------|----:|------:|----------------------:|
| $\kappa$ | 0.44 | 3.0% | 0.5 |
| $a$ (↑) | 0.045 | 3.0% | 0.05 |
| $a$ (↓) | 0.090 | 6.0% | 0.10 |
| $T_{er}$ | 0.010 | 3.0% | 0.01 |

Asymmetry of $a$: lowering $a$ by 3% produces noisier observables (more errors → higher binomial variance, shorter RT → higher relative variance) → MDD downward is 6%, twice the upward MDD.

### B.3 RL+Mood module

Procedure: $5^4 \times 20$ replicates, 500 trials, $p_{\text{reward}} = 0.75$, volatility = 0.3. 475 of 625 grid nodes stable ($\lambda < \lambda_{\text{crit}}$). 4 observables (win-stay, lose-switch, recovery speed, choice autocorrelation).

**Table 14.**

| Parameter | Grid | $r$ | Max cross-corr. | Class |
|-----------|------|----:|:----------------:|-------|
| $\alpha$ | [0.05, 0.35] × 5 | 0.67 | 0.08 ($\beta$) | **exploratory** (N = 500) |
| $\beta$ | [1.0, 6.0] × 5 | 0.90 | 0.10 ($\alpha$) | **fitted** |
| $\tau_m$ | [1.0, 10.0] × 5 | 0.01 | 0.09 ($\alpha$) | **exploratory** |
| $\lambda$ | [0.0, 0.4] × 5 | 0.005 | 0.12 ($\alpha$) | **exploratory** |

$\alpha$ recovery depends on sequence length:

**Table 15.**

| N trials | $r$ | 95% CI | Class |
|---------:|----:|--------|-------|
| 500 | 0.67 | [0.66, 0.69] | exploratory |
| 1000 | 0.77 | [0.76, 0.78] | **fitted** |
| 2000 | 0.86 | [0.86, 0.87] | **fitted** |

### B.4 MLE recovery (P6)

Navarro–Fuss density, three parameters ($\kappa$, $a$, $T_{er}$), $z = 0$, boundaries $\pm a/2$. Synthetic recovery: 50 parameter points, N = 500 trials each.

**Table 16.**

| Parameter | $r$ |
|-----------|----:|
| $\kappa$ | 0.945 |
| $a$ | 0.971 |
| $T_{er}$ | 0.992 |


## Appendix C: Parameter specifications

### C.1 Arousal parameters

Units: seconds (1 MTU = 1 s). Source: Phillips and Robinson (2007, 2008) except where noted.

**Table 17.**

| Parameter | Symbol | Calibrated | Low | High | Units | Class | Determined by |
|-----------|--------|----------:|----:|-----:|-------|-------|---------------|
| VLPO time constant | $\tau_v$ | 10 | 5 | 50 | s | internal | PR07 |
| MA time constant | $\tau_m$ | 10 | 5 | 50 | s | internal | PR07 |
| MA→VLPO inhibition | $\nu_{vm}$ | −1.9 | −2.70 | −0.50 | mV·s | exploratory | PR07, recovery $r = 0.31$ |
| VLPO→MA inhibition | $\nu_{mv}$ | −1.9 | −5.00 | −0.90 | mV·s | exploratory | PR07, recovery $r = 0.05$; pair $r(\nu_{vm},\nu_{mv}) = 0.62$ |
| Tonic MA input | $A$ | 1.0 | 0.5 | 2.0 | mV | internal | PR07 |
| Circadian amplitude | $\nu_{vc}$ | −3.2 | −10.0 | −1.0 | mV | exploratory | PR08, recovery $r = 0.37$ |
| H sensitivity | $\nu_{vh}$ | 0.325 | 0.08 | 0.50 | mV/nM | exploratory | D13, window [0.325, 0.35] |
| Max firing rate | $Q_{\max}$ | 100 | — | — | s⁻¹ | fixed | PR07 |
| Sigmoid threshold | $\theta$ | 10 | — | — | mV | fixed | PR07 |
| Sigmoid slope | $\sigma_q$ | 3 | 1 | 6 | mV | internal | PR07 |
| Somnogen time const. | $\chi$ | 39600 | 28800 | 50400 | s | internal | PR07 (11 h) |
| Somnogen production | $\mu$ | 3.6 | 1.0 | 10.0 | nM·s | internal | PR07 |
| Circadian offset | $c_0$ | 1.0 | — | — | — | fixed | PR07 |
| Circadian frequency | $\omega$ | $2\pi/86400$ | — | — | s⁻¹ | fixed | PR07 |
| Circadian phase | $\alpha$ | 57600 | 0 | 86400 | s | internal | PR08 (16 h) |
| Noise $V_v$ | $\sigma_{vv}$ | 0.003 | 0 | 2.0 | mV/√s | internal | A5 |
| Noise $V_m$ | $\sigma_{vm}$ | 0.015 | 0 | 5.0 | mV/√s | internal | A5 |
| Noise $H$ | $\sigma_h$ | 0.008 | 0 | 0.5 | nM/√s | internal | D18 |

### C.2 DDM parameters

**Table 18.**

| Parameter | Symbol | Calibrated | Low | High | Class | Determined by |
|-----------|--------|----------:|----:|-----:|-------|---------------|
| Sensitivity | $\kappa$ | 14.724 | 5 | 30 | **fitted** (standalone $r = 1.00$; MLE $r = 0.945$) | $\kappa a$ from accuracy, $a$ from RT |
| Threshold | $a$ | 1.492 | 0.5 | 3.0 | **fitted** (standalone $r = 1.00$; MLE $r = 0.971$) | $a^2$ from RT scale |
| Non-decision time | $T_{er}$ | 0.318 | 0.1 | 0.5 | **fitted** (standalone $r = 1.00$; MLE $r = 0.992$) | OLS intercept |
| Starting point | $z$ | 0 | — | — | fixed | Symmetric task |
| Diffusion noise | $\sigma$ | 1.0 | — | — | fixed | Identifiability convention |

### C.3 RL+Mood parameters

**Table 19.**

| Parameter | Symbol | Calibrated | Low | High | Class | Determined by |
|-----------|--------|----------:|----:|-----:|-------|---------------|
| Learning rate | $\alpha$ | 0.15 | 0.01 | 0.50 | **fitted** (N ≥ 1000) | Recovery speed (Behrens) |
| Inverse temperature | $\beta$ | 2.50 | 0.5 | 10.0 | **fitted** ($r = 0.90$) | Win-stay ≈ 0.70 (Behrens) |
| Mood decay | $\tau_m$ | 2.163 | 1.0 | 50.0 | exploratory ($r = 0.01$) | $\gamma = 0.63$ (Rutledge) |
| Mood→reward coupling | $\lambda$ | 0.490 | 0 | $\lambda_{\text{crit}}$ | exploratory ($r = 0.005$) | Uncalibrated; default = 0.5·$\lambda_{\text{crit}}$ |
| RPE→mood sensitivity | $\eta$ | 1.0 | — | — | fixed | Scale convention |
| Reward sensitivity | $\rho$ | 1.0 | — | — | fixed | Degenerate with $\beta$ |
| Mood equilibrium | $m_{\text{eq}}$ | 0.0 | — | — | fixed | By definition |

### C.4 BAA parameters (passive module)

**Table 20.**

| Parameter | Symbol | Default | Class |
|-----------|--------|--------:|-------|
| Attentional gain | $g$ | 1.0 | **untested** |
| Self-inhibition | $\beta_{\text{BAA}}$ | 0.5 | internal |
| Lateral inhibition | $\gamma_{\text{BAA}}$ | 0.2 | internal |
| Baseline noise | $\sigma_{\text{base}}$ | 0.1 | internal |
| Max noise | $\sigma_{\max}$ | 2.0 | internal |
| Plasticity rate | $\eta_{\text{BAA}}$ | 0.0 | untested |
| Energy recovery | $R_{\text{rec}}$ | 0.05 | internal |
| Energy cost | $k_{\text{cost}}$ | 0.1 | internal |
| Max energy | $S_{\max}$ | 100 | internal |
| Energy threshold | $S_{\text{crit}}$ | 30 | internal |
| Tonic input | $I_{\text{tonic}}$ | 0.5 | internal |
| Min weight | $W_{\min}$ | 0.5 | internal |
| Max weight | $W_{\max}$ | 4.0 | internal |

No parameter recovery was conducted for BAA (D33).


## Appendix D: Coupling specifications

**Table 21.**

| Coupling | Form | Strength range | Constraint |
|----------|------|:--------------:|------------|
| arousal.gain → DDM.$\kappa$ | $\kappa_{\text{eff}} = \kappa\,\max(0,\,1 + s(\text{gain} - 1))$ | $s \in [0, 4]$ | $\kappa_{\text{eff}} \geq 0$ |
| arousal.gain → RL.$\beta$ | $\beta_{\text{eff}} = \beta\,\max(0,\,1 + s(\text{gain} - 1))$ | $s \in [0, 4]$ | standalone only (D25) |
| arousal.gain → RL.$\alpha$ | $\alpha_{\text{eff}} = \alpha\,\max(0,\,1 + s(\text{gain} - 1))$ | $s \in [0, 4]$ | $\alpha_{\text{eff}} \in (0, 1]$ |
| RL.value\_diff → DDM.drift\_bias | $v = \kappa \cdot \text{coh} \cdot \text{dir} + s(V_1 - V_0)$ | $s \in [0, 10]$ | nonmonotonic accuracy (D29) |
| arousal.gain → BAA.gain | multiplicative, same form | $s \in [0, 4]$ | untested (D33) |
| arousal.is\_wake → trial gate | structural | — | binary |

All arousal → X couplings: block-triangular Jacobian, eigenvalues = union of subsystem eigenvalues.

DDM ↔ RL closed loop: $\lambda_{\text{crit}}^{\text{closed}} = 1.1 > \lambda_{\text{crit}}^{\text{standalone}} = 0.98$. Closure stabilizes (+12%, D30).


## Appendix E: Deferred observables and held-out results

### E.1 Summary

**Table 22.**

| Module | # | Observable | Status | Note |
|--------|---|------------|--------|------|
| DDM | 1 | RT quantile profile | FAIL | Q90 +23–34% at low coh. Requires $\eta$ (D16). |
| DDM | 2 | Right-skew vs coherence | FAIL | Model: decreasing; data: flat. |
| DDM | 3 | RT correct ≈ RT error | PASS | Cohen $d < 0.1$. Criterion revised (D17). |
| DDM | HO | Monkey B held-out | PASS | All 6 coh, acc ≤ 0.071, RT ≤ 14.3%. |
| Arousal | 1 | Onset advance (deprivation) | INDET | CI [0.83, 1.77] h. Consistency check (D13). |
| Arousal | 2 | Duration excess (deprivation) | PASS | CI [1.38, 1.81] h. Consistency check (D13). |
| Arousal | 3 | Transition sharpness | PASS | 22 min < 30 min threshold. |
| Arousal | 4 | Spurious transitions | PASS | 0% (B = 100, 7 days). |
| RL+Mood | 1 | Rutledge coefficients | PASS | $w_{\text{RPE}} = 0.63 > w_{\text{CR}} = 0$. |
| RL+Mood | 2 | Volatility invariance | INDET | CI [0.75, 2.25], threshold 2.0. |
| RL+Mood | 3 | Mood autocorrelation | PASS* | $\gamma_{\text{model}} = 0.74$. *Partially circular. |

### E.2 Observables the model does not reproduce

**Table 23.**

| Module | Observable | Category | Cause |
|--------|------------|----------|-------|
| DDM | RT quantile profile | Model limitation | No inter-trial variability $\eta$ (D16) |
| DDM | Right-skew pattern | Model limitation | No $\eta$ |
| DDM | Slow errors | Model limitation | No $\eta$ |
| Arousal | Onset advance | Insufficient resolution | CI spans threshold |
| Arousal | Free-living variability | Structural absence | No external sources |
| RL+Mood | Adaptive $\alpha$ (volatility) | By construction | RW has fixed $\alpha$ |
| RL+Mood | Clinical mood | Structural absence | $\tau_m \approx 2$ trials ≈ seconds |


## Appendix F: Data manifests

All raw data are excluded from the repository (.gitignore). Fetch scripts and manifests are committed. Data are re-downloaded and verified by SHA-256 before any analysis.

### F.1 Confidence Database

- **Source:** https://osf.io/s46pr/
- **Citation:** Rahnev, D. et al. The Confidence Database. *Nat Hum Behav* 4, 317–325 (2020). DOI: 10.1038/s41562-019-0813-1
- **License:** CC-By 4.0
- **Content:** 101 perceptual 2AFC datasets, 97 CSV files fetched
- **Used:** 73 datasets (82 passed initial criteria, 15 excluded — 13 non-2AFC or continuous response, 2 ambiguous readme; 53 multi, 20 single)
- **Fetch script:** data/fetch\_confidence\_db.py (OSF API v2)
- **Manifest:** data/manifest\_confidence\_db.json (per-file SHA-256)

### F.2 OBF-Psychiatric (actigraphy)

- **Source:** https://zenodo.org/records/13754984
- **Citation:** OBF-Psychiatric, a motor activity dataset. *Scientific Data* 12 (2025). DOI: 10.1038/s41597-025-04384-3
- **License:** CC BY 4.0
- **Content:** 175 files (controls, ADHD, depression, schizophrenia). 32 controls used.
- **ZIP SHA-256:** `5a9923ff1a732da0b045730c277d6cc809e1df799bc1ca5f3e052d78d461fae9`
- **ZIP size:** 10,372,134 bytes
- **Fetch script:** data/fetch\_obf\_psychiatric.py (Zenodo)
- **Manifest:** data/manifest\_obf\_psychiatric.json

### F.3 Eckstein et al. 2022 (reversal learning)

- **Source:** OSF projects 7wuh4 and h4qr6
- **Citation:** Eckstein, M.K. et al. Reinforcement learning and Bayesian inference provide complementary models for the unique advantage of adolescents in stochastic reversal. *Dev Cogn Neurosci* 55, 101106 (2022). DOI: 10.1016/j.dcn.2022.101106; eLife: 10.7554/eLife.75474
- **License:** CC-By 4.0
- **Content:** 310 files. 114 adults (18+) used.
- **Key file SHA-256:**
  - `EcksteinEtAl_eLife2022_BUTTERFLY.csv`: `6482beac3820f1fb1f51c58e4a5f4f1b00ba77f0233261cbe15faba4624da5eb`
  - `EcksteinEtAl_eLife2022_RLWM.csv`: `20e1e7f3f37b18391cdcd8e062d06e6fe025905f9b95faf94e6e2b1241c8d20f`
- **Fetch script:** data/fetch\_eckstein2022.py (OSF)
- **Manifest:** data/manifest\_eckstein2022.json (per-file SHA-256)

### F.4 Roitman and Shadlen 2002 (monkey RT)

- **Source:** https://shadlenlab.columbia.edu/resources/RoitmanDataCode.html
- **Content:** Trial-level reaction times for two monkeys, 6 coherence levels.
- **Used:** Monkey N (3534 trials) for calibration; Monkey B for held-out.
- **Committed:** results/roitman\_rts.csv (extracted from MATLAB .mat files).

### F.5 Zou et al. 2022

- **Status:** Unavailable. Raw data files on OSF are 0 bytes (checked 2026-08-05).
- **Replaced by:** Eckstein 2022 (D44).


## Appendix G: Software versions

From requirements.txt (pinned):

**Table 24.**

| Package | Version |
|---------|---------|
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| pandas | 3.0.1 |
| matplotlib | 3.10.9 |
| openpyxl | 3.1.5 |
| pytest | 9.1.1 |
| ipywidgets | ≥ 8.0 |
| Python | ≥ 3.11 |

Platform: Windows 11, Python 3.12. All code is platform-independent (tested on Windows; no platform-specific dependencies).


## Appendix H: Reproduction from a clean environment

### Prerequisites

- Python ≥ 3.11
- Git
- Internet access (for data download)

### Steps

```
# 1. Clone and install
git clone <repo-url>
cd MENTAL_HEALTH_EQUATIONS
pip install -r requirements.txt

# 2. Download data (each script verifies SHA-256)
python data/fetch_obf_psychiatric.py
python data/fetch_confidence_db.py
python data/fetch_eckstein2022.py

# 3. Run tests (expects 256 passed, 5 slow deselected)
PYTHONPATH="src;tests" pytest tests/ -v -m "not slow"

# 4. Reproduce calibrations
PYTHONPATH="src" python experiments/arousal_calibration.py
PYTHONPATH="src" python experiments/ddm_calibration.py
PYTHONPATH="src" python experiments/rl_mood_calibration.py

# 5. Reproduce recovery
PYTHONPATH="src" python experiments/arousal_validation.py
PYTHONPATH="src" python experiments/ddm_recovery.py
PYTHONPATH="src" python experiments/rl_mood_recovery.py
PYTHONPATH="src" python experiments/rl_mood_alpha_recovery_ci.py

# 6. Reproduce P6 comparisons
PYTHONPATH="src" python experiments/p6_ddm_mle_recovery.py
PYTHONPATH="src" python experiments/p6_ddm_comparison.py
PYTHONPATH="src" python experiments/p6_arousal_comparison.py
PYTHONPATH="src" python experiments/p6_rl.py
PYTHONPATH="src" python experiments/rl_mood_resolution.py

# 7. Verify numbers (expects 64/64)
PYTHONPATH="src" python article/verify_numbers.py

# 8. Rebuild provenance
PYTHONPATH="src" python article/build_provenance.py
```

### Expected outputs

Each experiment script writes to `results/`. The `verify_numbers.py` script checks that every number in `article/numbers.json` matches the corresponding value in its source file. If any number has changed, the script exits with a non-zero code and reports the discrepancy.

### Notes

- Step 2 downloads approximately 500 MB of data. Each fetch script computes SHA-256 checksums and compares them against the manifest; the script aborts if checksums do not match.
- Step 3 excludes five slow tests (four hybrid transition at T = 172800 s, one recovery determinism at 3 × 86400 steps). To run them: `pytest -m slow` (adds ~30 minutes).
- Steps 4–6 are computationally intensive. Approximate wall times: arousal recovery ~5 min, DDM recovery ~6 min, RL recovery ~1 min, P6 DDM comparison ~2 min, P6 arousal ~1 min, P6 RL ~1 min.
- All random seeds are fixed and recorded in the experiment scripts. Results are deterministic given the same numpy version.
- On Linux/macOS, replace `PYTHONPATH="src;tests"` with `PYTHONPATH="src:tests"`.
- **Restored scripts (D48, D51).** Three P6 experiment scripts were originally executed in interactive sessions and lost. They were restored from the pre-registered specification: `p6_ddm_mle_recovery.py`, `p6_ddm_comparison.py`, `p6_arousal_comparison.py` (D48), and `rl_mood_resolution.py` (D51). The MLE recovery script produced slightly different numbers from the original (r(κ) = 0.945 vs original 0.889); the resolution script matched exactly. The restored scripts are canonical: they implement the published method description, and results were recomputed from them. See D48 for the rationale for accepting restored values over the original.

---
