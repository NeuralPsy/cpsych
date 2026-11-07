# Results: Comparison with data

Comparison criteria were pre-registered and committed to the repository before any data were downloaded (commits fff056d, 6aefd62). Three criteria were revised after computation; each revision is noted below alongside the result it affects.

## Decision model: Confidence Database

### Data and procedure

From 97 downloaded datasets (Rahnev et al., 2020), 82 passed the mechanical selection criteria (2AFC perceptual, RT available, ≥ 200 trials per participant). Accuracy coding rules were fixed per dataset in a separate file before any metric was computed; 13 datasets were excluded mechanically (not 2AFC or continuous response) and 2 by judgment (ambiguous readme). Multi/single classification was formalized as a separate decision (D52): a dataset is multi-condition if at least one column encodes physical stimulus difficulty (contrast, coherence, SOA, noise level, dot difference, orientation magnitude); columns encoding task manipulation (cue validity, speed/accuracy instruction, block order) do not qualify. This criterion was applied to 18 ambiguous datasets, with per-dataset decisions recorded in a separate override file. A case-sensitivity bug in the original classification script was also corrected (D52a), affecting four datasets. The resulting 73 datasets with valid accuracy and RT comprised 53 multi-condition and 20 single-condition sets, totaling 4,066 participants.

Parameters κ, a, and T_er were estimated per participant by maximum likelihood on the Navarro-Fuss first-passage density (Navarro and Fuss, 2009), with z = 0 fixed and boundaries at ±a/2. The MLE implementation was validated by synthetic recovery before application to real data: at N = 500 trials, recovery correlations were r = 0.945 (κ), 0.971 (a), and 0.992 (T_er), with cross-correlations below 0.10. A sign error in the density exponent was caught during this recovery step and corrected (D46) — without the synthetic check, it would have produced plausible but systematically biased estimates on real data.

### Thresholds

Agreement thresholds were derived from inter-dataset variability, computed before fitting. The original threshold (1 × SD of per-dataset mean accuracy = 0.047) was revised after discovering that averaging over difficulty levels deflates the spread: datasets with different mixes of easy and hard conditions have similar means regardless of task properties. The revision (D45) replaced the single threshold with three per-level thresholds, computed before any model fitting:

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
