"""Closed-form calibration of DDM parameters from Roitman & Shadlen data.

Procedure (specs/ddm.md v1.2):
1. p = kappa*a from accuracy via logistic MLE (1 free parameter)
2. a^2 and Ter from OLS on mean RT = a^2 * h(p, coh) + Ter
3. kappa = p / a

Outputs: results/ddm_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# --- Load data ---
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "roitman_rts.csv"

df = pd.read_csv(DATA)
# Monkey 2 = monkey N (calibration), monkey 1 = monkey B (held-out)
monkey_n = df[df["monkey"] == 2].copy()

print(f"Monkey N: {len(monkey_n)} trials")
cohs = sorted(monkey_n["coh"].unique())
print(f"Coherences: {cohs}")

# --- Per-coherence summary ---
summary = []
for c in cohs:
    sub = monkey_n[monkey_n["coh"] == c]
    acc = sub["correct"].mean()
    mean_rt = sub["rt"].mean()  # all responses (unconditional)
    n = len(sub)
    summary.append({"coh": c, "acc": acc, "mean_rt": mean_rt, "n": n})
    print(f"  coh={c:.3f}: n={n}, acc={acc:.3f}, mean_RT={mean_rt:.3f}s")

summary_df = pd.DataFrame(summary)
coh_arr = summary_df["coh"].values
acc_arr = summary_df["acc"].values
rt_arr = summary_df["mean_rt"].values

# --- Step 1: p = kappa*a from accuracy ---
# P(correct | coh) = 1 / (1 + exp(-p * coh))
# At coh=0: P(correct) = 0.5 regardless of p (correct by construction)
# Fit p using logistic MLE on coh > 0


def neg_log_likelihood(p, cohs_data, correct_data):
    """Negative log-likelihood for logistic: P(correct) = sigmoid(p*coh)."""
    v = p * cohs_data
    # Clip to avoid overflow
    v = np.clip(v, -500, 500)
    log_p_correct = -np.log1p(np.exp(-v))
    log_p_error = -np.log1p(np.exp(v))
    nll = -np.sum(correct_data * log_p_correct
                  + (1 - correct_data) * log_p_error)
    return nll


# Use per-trial data for MLE (coh > 0 only, since coh=0 is uninformative)
trials_nonzero = monkey_n[monkey_n["coh"] > 0]
coh_trials = trials_nonzero["coh"].values
correct_trials = trials_nonzero["correct"].values

result = minimize_scalar(
    neg_log_likelihood,
    bounds=(0.1, 200.0),
    args=(coh_trials, correct_trials),
    method="bounded",
)
p = result.x
print(f"\nStep 1: p = kappa*a = {p:.4f}")

# Verify: predicted accuracy
for c in cohs:
    pred_acc = 1 / (1 + np.exp(-p * c))
    obs_acc = acc_arr[cohs.index(c)]
    print(f"  coh={c:.3f}: pred={pred_acc:.3f}, obs={obs_acc:.3f}")


# --- Step 2: a^2 and Ter from OLS ---
# mean_RT = a^2 * h(p, coh) + Ter
# h(p, coh) = tanh(p*coh/2) / (2*p*coh)  for coh > 0
# h(p, 0)   = 1/4  (limit)

def h_func(p, coh):
    """Predicted mean decision time coefficient: E[DT] = a^2 * h(p, coh)."""
    if coh == 0:
        return 0.25
    pc = p * coh
    return np.tanh(pc / 2) / (2 * pc)


h_arr = np.array([h_func(p, c) for c in cohs])

# OLS: mean_RT = a^2 * h + Ter
# Design matrix: [h, 1] * [a^2, Ter]^T = mean_RT
X = np.column_stack([h_arr, np.ones(len(cohs))])
beta, residuals, rank, sv = np.linalg.lstsq(X, rt_arr, rcond=None)
a_sq = beta[0]
Ter = beta[1]
a = np.sqrt(a_sq)
kappa = p / a

print(f"\nStep 2: a^2 = {a_sq:.6f}, a = {a:.4f}")
print(f"Step 3: Ter = {Ter:.4f} sec")
print(f"         kappa = p/a = {kappa:.4f}")

# --- Verify fit ---
print("\nFit verification:")
pred_rt = a_sq * h_arr + Ter
for i, c in enumerate(cohs):
    resid_pct = abs(pred_rt[i] - rt_arr[i]) / rt_arr[i] * 100
    print(f"  coh={c:.3f}: obs_RT={rt_arr[i]:.3f}s, "
          f"pred_RT={pred_rt[i]:.3f}s, error={resid_pct:.1f}%")

rmse = np.sqrt(np.mean((pred_rt - rt_arr) ** 2))
max_pct_err = np.max(np.abs(pred_rt - rt_arr) / rt_arr * 100)
print(f"  RMSE = {rmse:.4f}s, max_pct_error = {max_pct_err:.1f}%")

if max_pct_err > 15:
    print("WARNING: max percentage error > 15% — escalate")
    sys.exit(1)

# --- Convert to MTU ---
# In the DDM, time is in MTU. The data is in seconds.
# Ter in MTU: Ter_mtu = Ter / scale, where scale = seconds_per_MTU.
# The drift in the model is v = kappa * coh * direction, and sigma=1.
# The data RTs are in seconds. We need a time scaling factor.
# From the model: E[DT_model] = a^2 * h(p, coh) in MTU.
# From the data: E[DT_data] = mean_RT - Ter in seconds.
# So: seconds_per_MTU = E[DT_data] / E[DT_model].
# But E[DT_model] = a^2 * h(p, coh) and E[DT_data] = a^2 * h(p, coh) * scale.
# Wait: the OLS already absorbed the time scaling into a^2.
# a^2 here is in seconds (from data RTs in seconds).
# In the pure DDM with sigma=1: a^2 would be in MTU^2.
# So the scaling: 1 MTU = 1 second * (a_model / a_data).
#
# Simpler: we just use the a and kappa as-is. The model runs in its own
# time units; when we simulate, dt is in the same units as the data (seconds).
# So: kappa and a from the OLS directly give seconds-based parameters.
# Ter is also in seconds.
#
# For the DDM module, 1 MTU = 1 second.
print(f"\nFinal parameters (1 MTU = 1 second):")
print(f"  kappa = {kappa:.4f}")
print(f"  a     = {a:.4f}")
print(f"  Ter   = {Ter:.4f}")

# --- Save ---
out = {
    "kappa": round(float(kappa), 4),
    "a": round(float(a), 4),
    "Ter": round(float(Ter), 4),
    "p_kappa_a": round(float(p), 4),
    "a_squared": round(float(a_sq), 6),
    "rmse_sec": round(float(rmse), 6),
    "max_pct_error": round(float(max_pct_err), 2),
    "monkey": "N (monkey 2 in CSV)",
    "n_trials": int(len(monkey_n)),
    "procedure": "closed-form: p from logistic MLE, a^2 and Ter from OLS",
    "per_coherence": [
        {
            "coh": float(c),
            "n": int(summary[i]["n"]),
            "obs_acc": round(float(acc_arr[i]), 4),
            "pred_acc": round(float(1 / (1 + np.exp(-p * c))), 4),
            "obs_mean_rt": round(float(rt_arr[i]), 4),
            "pred_mean_rt": round(float(pred_rt[i]), 4),
        }
        for i, c in enumerate(cohs)
    ],
}

outpath = ROOT / "results" / "ddm_calibration.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}")
