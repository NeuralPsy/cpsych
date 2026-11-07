# Results: Platform construction

## Calibration

Each module was calibrated against its canonical dataset using a closed procedure — a deterministic mapping from observables to parameters, without iterative optimization and without free choices (rule 21). Details are in Appendix A; here we report what calibration achieved and where it fell short.

The DDM was calibrated on trial-level reaction time data from monkey N (Roitman and Shadlen, 2002; 3,534 trials across six coherence levels). The closed procedure — logistic MLE on accuracy for κa, RT scaling for a, OLS residual for T_er — yielded κ = 14.72, a = 1.49, T_er = 0.318 s, with RMSE = 0.018 s on mean RT across coherences. Held-out prediction on monkey B passed all six coherence levels (accuracy |diff| ≤ 0.071, RT diff ≤ 14.3%).

The arousal module was calibrated on published parameters from Phillips and Robinson (2007, 2008), with v_vh adjusted from the original 0.19 to 0.325 to bring mean sleep duration into the target range of 7–9 hours (A2). Noise amplitude σ_h = 0.008 was calibrated to produce onset variability of 15–30 minutes circular SD (D18), matching laboratory data from Klerman and Dijk (2005).

The RL module was calibrated on summary statistics from Rutledge et al. (2014): α = 0.15, β = 2.50, producing win-stay rate 0.70 and recovery speed of 10 trials after reversal.

## Recoverability

Of the eleven parameters subjected to recovery testing across three modules, five were classified as fitted (r ≥ 0.7) and six as exploratory:

Two recovery procedures were used. Standalone recovery (generating and estimating from the module's own simulation at calibrated noise) tests identifiability under the module's own dynamics. MLE recovery (generating data and fitting with the Navarro-Fuss likelihood used for external comparison) tests the estimation procedure applied to real data. Both are reported; the MLE values are lower and determine the classification, since they reflect the procedure actually used for comparison.

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
