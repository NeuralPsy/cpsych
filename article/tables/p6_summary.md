# P6 Summary Table

## Comparisons

| Module | Dataset | N subj | Estimated | Fixed | Criterion | Result | Verdict |
|--------|---------|-------:|-----------|-------|-----------|--------|---------|
| DDM (single) | Confidence DB | 20 datasets | kappa, a, Ter (MLE) | — | MAE < 0.038 | MAE=0.041 | **Indeterminate** |
| DDM (multi, coherence) | Confidence DB | 10 datasets | kappa, a, Ter (MLE) | — | MAE < 0.044 | MAE=0.086 | **Negative** |
| DDM (multi, all) | Confidence DB | 53 datasets | kappa, a, Ter (MLE) | — | MAE < 0.100 | MAE=0.102 | Indeterminate [35.8%,62.3%] |
| Arousal (onset) | OBF-Psychiatric | 32 controls | none (all exploratory) | all | Model 23:00, data 23:39-00:01 | diff 38-60 min | **Positive** (uncalibrated) |
| Arousal (duration) | OBF-Psychiatric | — | — | — | — | — | **Withdrawn** (D49) |
| Arousal variability | OBF-Psychiatric | 32 controls | none | sigma_h | — | within-SD 2.15-2.18h vs model 0.50h | **Structural gap** |
| RL | Eckstein 2022 | 114 adults | none (N<500) | alpha, beta, tau_m, lam | WS in IQR | 0.681 vs [0.94,1.00] | **Negative** |

## Revisions post-factum

| Decision | Original | Revised | Reason |
|----------|----------|---------|--------|
| D45 | Single accuracy threshold 0.047 | Three per-level: 0.038 / 0.100 / 0.257 | Mean across difficulty levels deflates SD |
| D49 | Duration comparison included | Duration withdrawn | Scoring algorithm not reproducible |
| D52 | 43 multi / 30 single | 53 multi / 20 single | Physical difficulty criterion formalized; is_multi bug fixed |

## What the model does not reproduce (complete list)

| Observable | Module | Reason | Type |
|------------|--------|--------|------|
| RT quantile profile | DDM | No inter-trial variability eta (D16) | Model limitation |
| Right-skew pattern | DDM | No eta | Model limitation |
| Slow errors | DDM | No eta | Model limitation |
| Confidence ratings | DDM | No metacognition module | Structural absence |
| Onset advance (deprivation) | Arousal | Insufficient resolution (D13 INDET) | Resolution |
| Free-living sleep variability | Arousal | **Structural:** no external sources (social schedule, weekends, caffeine) | Structural absence |
| Adaptive learning rate | RL | Rescorla-Wagner by construction | Model limitation |
| Clinical mood | RL | Momentary affect, not clinical (D19) | Scale |
| Behavior under unambiguous punishment | RL | alpha/beta calibrated on ambiguous feedback; optimal values context-dependent | Calibration-context |
| NREM/REM structure | Arousal | Two-process model | Structural absence |
| kappa transfer across stimulus scales | DDM | kappa*difficulty degeneracy: units arbitrary | Structural (5th degeneracy) |
