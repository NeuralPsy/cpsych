# Appendices

## Appendix A: Module equations

### A.1 Arousal (Phillips–Robinson two-process flip-flop)

Source: Phillips and Robinson (2007, *J Biol Rhythms* 22:167–179; 2008, *J Theor Biol* 255:413–423). Equations verified against deposited CellML and BioModels implementations (specs/arousal\_sources.md).

**State variables** (all CONTINUOUS):

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

| Parameter | Grid | $r$ | RMSE | Max cross-corr. | Class |
|-----------|------|----:|-----:|:----------------:|-------|
| $\nu_{vh}$ | [0.08, 0.50] × 5 | 0.42 | 0.170 | 0.26 ($\nu_{mv}$) | **exploratory** |
| $\nu_{vc}$ | [−10.0, −1.0] × 5 | 0.37 | 4.241 | 0.56 ($\nu_{vm}$) | **exploratory** |
| $\nu_{vm}$ | [−2.70, −0.50] × 5 | 0.31 | 0.864 | 0.56 ($\nu_{vc}$) | **exploratory** |
| $\nu_{mv}$ | [−5.00, −0.90] × 5 | 0.05 | 2.271 | 0.62 ($\nu_{vm}$) | **exploratory** |

Deterministic recovery ($\sigma = 0$): $r = 1.00$ for all four → estimator is correct; non-identifiability is caused by stochastic noise (SNR 0.9–3.8). This is a property of flip-flop models, not only this implementation.

### B.2 DDM module

Procedure: $5^3 \times 20$ replicates, 3500 trials per replicate (~590 per coherence level, matching monkey N). 12 observables (6 accuracy + 6 mean RT).

| Parameter | Grid | $r$ | RMSE | Max cross-corr. | Class |
|-----------|------|----:|-----:|:----------------:|-------|
| $\kappa$ | [5.0, 30.0] × 5 | 0.997 | 0.740 | < 0.001 | **fitted** |
| $a$ | [0.5, 3.0] × 5 | 1.000 | 0.013 | < 0.001 | **fitted** |
| $T_{er}$ | [0.1, 0.5] × 5 | 1.000 | 0.002 | < 0.001 | **fitted** |

Resolution (minimum detectable difference at 80% discrimination, 3500 trials):

| Parameter | MDD | MDD % | Recommended knob step |
|-----------|----:|------:|----------------------:|
| $\kappa$ | 0.44 | 3.0% | 0.5 |
| $a$ (↑) | 0.045 | 3.0% | 0.05 |
| $a$ (↓) | 0.090 | 6.0% | 0.10 |
| $T_{er}$ | 0.010 | 3.0% | 0.01 |

Asymmetry of $a$: lowering $a$ by 3% produces noisier observables (more errors → higher binomial variance, shorter RT → higher relative variance) → MDD downward is 6%, twice the upward MDD.

### B.3 RL+Mood module

Procedure: $5^4 \times 20$ replicates, 500 trials, $p_{\text{reward}} = 0.75$, volatility = 0.3. 475 of 625 grid nodes stable ($\lambda < \lambda_{\text{crit}}$). 4 observables (win-stay, lose-switch, recovery speed, choice autocorrelation).

| Parameter | Grid | $r$ | Max cross-corr. | Class |
|-----------|------|----:|:----------------:|-------|
| $\alpha$ | [0.05, 0.35] × 5 | 0.67 | 0.08 ($\beta$) | **exploratory** (N = 500) |
| $\beta$ | [1.0, 6.0] × 5 | 0.90 | 0.10 ($\alpha$) | **fitted** |
| $\tau_m$ | [1.0, 10.0] × 5 | 0.01 | 0.09 ($\alpha$) | **exploratory** |
| $\lambda$ | [0.0, 0.4] × 5 | 0.005 | 0.12 ($\alpha$) | **exploratory** |

$\alpha$ recovery depends on sequence length:

| N trials | $r$ | 95% CI | Class |
|---------:|----:|--------|-------|
| 500 | 0.67 | [0.66, 0.69] | exploratory |
| 1000 | 0.77 | [0.76, 0.78] | **fitted** |
| 2000 | 0.86 | [0.86, 0.87] | **fitted** |

### B.4 MLE recovery (P6)

Navarro–Fuss density, three parameters ($\kappa$, $a$, $T_{er}$), $z = 0$, boundaries $\pm a/2$. Synthetic recovery: 50 parameter points, N = 500 trials each.

| Parameter | $r$ |
|-----------|----:|
| $\kappa$ | 0.945 |
| $a$ | 0.971 |
| $T_{er}$ | 0.992 |


## Appendix C: Parameter specifications

### C.1 Arousal parameters

Units: seconds (1 MTU = 1 s). Source: Phillips and Robinson (2007, 2008) except where noted.

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

| Parameter | Symbol | Calibrated | Low | High | Class | Determined by |
|-----------|--------|----------:|----:|-----:|-------|---------------|
| Sensitivity | $\kappa$ | 14.724 | 5 | 30 | **fitted** (standalone $r = 1.00$; MLE $r = 0.945$) | $\kappa a$ from accuracy, $a$ from RT |
| Threshold | $a$ | 1.492 | 0.5 | 3.0 | **fitted** (standalone $r = 1.00$; MLE $r = 0.971$) | $a^2$ from RT scale |
| Non-decision time | $T_{er}$ | 0.318 | 0.1 | 0.5 | **fitted** (standalone $r = 1.00$; MLE $r = 0.992$) | OLS intercept |
| Starting point | $z$ | 0 | — | — | fixed | Symmetric task |
| Diffusion noise | $\sigma$ | 1.0 | — | — | fixed | Identifiability convention |

### C.3 RL+Mood parameters

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
