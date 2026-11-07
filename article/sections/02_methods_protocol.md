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
