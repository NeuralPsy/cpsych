# CPsych: Modular Computational Psychiatry Platform

[![CI](https://github.com/NeuralPsy/cpsych/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuralPsy/cpsych/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24015477.svg)](https://doi.org/10.5281/zenodo.24015477)

A platform for modelling mental functions as stochastic dynamical systems.
Four modules -- Arousal (sleep-wake), DDM (decision-making), RL+Mood (reinforcement
learning with momentary affect), and BAA (boundary attention) -- are coupled through
parameter modulation and analysed for stability, parameter recoverability, and
predictive limits.

## Installation

Requires Python >= 3.11.

```bash
git clone <repo-url>
cd cpsych
pip install -e ".[test]"
```

## Three Levels of Verification

### Level 1: Verify published numbers (~30 seconds)

Confirms that all 64 numbers cited in the platform manuscript match the
pre-computed result files shipped with the repository.

```bash
python article/verify_numbers.py
```

Expected output: `Verified: 64/64`.

For the transfer-test analysis (P10/P11), which belongs to a separate
manuscript, `article/report_p10_p11.py` prints every reported number straight
from `results/p10_model_comparison.json` and
`results/p11_free_fit_behaviour.json`:

```bash
python article/report_p10_p11.py results
```

`article/check_numbers.py` additionally compares each of those numbers against
the manuscript text and against `article/literature_values.json`, which holds
every figure taken from another paper together with the verbatim sentence it
came from. It takes the results directory and the manuscript file as
arguments; the manuscript is added here when that paper is published.

### Level 2: Run test suite (~13 minutes)

256 tests covering module contracts, parameter recovery, integrator correctness,
coupling stability, and the impossible map pipeline.

```bash
pytest -m "not slow"
```

To include long-running tests (simulated time T > 86400):

```bash
pytest -m ""
```

### Level 3: Reproduce from scratch (~3 hours + data download)

#### Step A. Download external datasets (~225 MB)

Datasets are not included in the repository. Download scripts fetch from
public archives (OSF, Zenodo) and verify SHA-256 checksums against manifests.

| Dataset | Script | Size | Source |
|---------|--------|------|--------|
| Confidence Database | `python data/fetch_confidence_db.py` | 123 MB | [Rahnev et al. 2020](https://doi.org/10.1038/s41562-019-0813-1) |
| Eckstein et al. 2022 | `python data/fetch_eckstein2022.py` | 4 MB | [Eckstein et al. 2022](https://doi.org/10.7554/eLife.75474) |
| OBF-Psychiatric | `python data/fetch_obf_psychiatric.py` | 98 MB | [Garcia-Ceja et al. 2025](https://doi.org/10.1038/s41597-025-04384-3) |

All datasets are CC-By 4.0. Download time depends on bandwidth (~10-30 min).

#### Step B. Run calibration experiments (~1 minute)

```bash
python experiments/arousal_calibration.py     # 8s
python experiments/ddm_calibration.py         # 10s
python experiments/rl_mood_calibration.py     # 14s
```

#### Step C. Run parameter recovery (~15 minutes)

```bash
python experiments/ddm_recovery.py            # 4 min
python experiments/rl_mood_recovery.py
python experiments/rl_mood_alpha_recovery_ci.py
python experiments/rl_mood_vol_invariance_ci.py
```

#### Step D. Run resolution and deferred observables (~10 minutes)

```bash
python experiments/ddm_resolution.py
python experiments/ddm_deferred.py
python experiments/rl_mood_resolution.py
python experiments/rl_mood_deferred.py
python experiments/rl_mood_lambda_crit.py
python experiments/arousal_validation.py
python experiments/arousal_vh_window_d18.py
python experiments/arousal_d18_final.py
python experiments/arousal_consistency_recompute.py
python experiments/arousal_sigma_h_recompute.py
```

#### Step E. Run coupling and stability analyses (~15 minutes)

```bash
python experiments/p3_value_coupling_sweep.py
python experiments/p3_value_coupling_fine.py
python experiments/p3_rho_s_degeneracy.py
python experiments/p4_scout.py
python experiments/p4_stability.py
python experiments/p4_hybrid_19a.py
python experiments/p4_benchmark.py
```

#### Step F. Run impossible maps (~95 minutes)

These are the most expensive computations: parameter-space sweeps of the
triple composite (arousal + DDM + RL, all couplings active).

```bash
python experiments/p4_impossible_map.py
```

This script produces multiple result files. Recorded wall times per map:
618 s, 667 s, 1032 s, 1645 s, 1690 s.

#### Step G. Run external data comparisons (~5 minutes)

Requires datasets from Step A.

```bash
python experiments/p6_ddm_comparison.py       # 69s
python experiments/p6_arousal_comparison.py
python experiments/p6_rl.py
python experiments/p6_ddm_mle_recovery.py     # 5s
```

#### Step H. Run the transfer test (~6 minutes)

Requires the Eckstein 2022 dataset from Step A.

```bash
python experiments/p10_model_comparison.py    # ~5 min, six models, 112 subjects
python experiments/p11_free_fit_behaviour.py  # ~2 s, needs P10's output
```

#### Step I. Rebuild and verify numbers

```bash
python article/build_numbers.py
python article/verify_numbers.py
python article/report_p10_p11.py results      # every transfer-test number
```

## Repository Structure

```
src/cpsych/           Core platform (SDE kernel, modules, couplings, observables)
  core/               Integrator, module contract, coupling, stability analysis
  modules/            Arousal, DDM, RL+Mood, BAA
  environments/       Task protocols (perceptual 2AFC, reversal bandit)
  fitting/            Parameter estimation (DDM MLE)
  observables/        Derived metrics (circular statistics)
tests/                256 pytest tests
experiments/          Analysis scripts, each writes to results/
results/              JSON outputs -- sole source of numbers in the manuscript
article/              Manuscript, numbers.json (64 entries), verification scripts
                      biblio.py (citation keys -> AMA numbering), check_numbers.py,
                      literature_values.json (sourced figures), report_p10_p11.py
specs/                Module specifications (written before implementation)
reports/              Progress log, 53 decisions, alternatives, Neuroblox evaluation
research/             Input literature review (read-only)
data/                 Download scripts + SHA-256 manifests (data not included)
sandbox/              Interactive parameter interface (ipywidgets)
```

**Note on `reports/`.** Development logs (`progress.md`, `decisions.md`,
`alternatives.md`) are working notes in Russian. They document every
architectural decision (D1--D53), tried-and-rejected variants, and the full
timeline of the six development gates. `neuroblox_evaluation.md` evaluates
Neuroblox.jl as an alternative kernel (also in Russian). These files are
included for transparency and auditability; the manuscript and code are in
English.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for the 22 rules governing development,
the six gates, the module contract, and the parameter classification scheme
(fitted / exploratory / untested).

## Key Results

- 4 modules, 6 couplings, 26 parameters (3 fitted, 18 exploratory, 5 untested)
- Zero instabilities across 675 parameter-space points (impossible map)
- Parameter recoverability does not predict real-data accuracy:
  Arousal (no recoverable parameters) gives the best prediction;
  DDM (all recoverable) gives no prediction
- 5 degeneracies identified; 53 decisions documented

## Citation

If you use this software, please cite:

> Vasilchenko, K. (2026). CPsych: Modular Computational Psychiatry Platform (v1.0).
> Zenodo. https://doi.org/10.5281/zenodo.24015477

## License

[MIT](LICENSE)
