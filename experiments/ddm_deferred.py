"""DDM deferred observables and held-out test.

Deferred (specs/ddm.md v1.2):
  1. RT quantiles (Q10, Q30, Q50, Q70, Q90) per coherence
  2. Right-skew Q90/Q50 per coherence
  3. RT distributions: correct vs error should be identical (positive prediction)
     Slow errors vs monkey N: reportable observation

Held-out (D8):
  Monkey B: accuracy ±0.10, mean RT ±25% per coherence

Outputs: results/ddm_deferred.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# --- Load data ---
df = pd.read_csv(ROOT / "results" / "roitman_rts.csv")
monkey_n = df[df["monkey"] == 2]
monkey_b = df[df["monkey"] == 1]
cal = json.load(open(ROOT / "results" / "ddm_calibration.json"))

COHERENCES = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
kappa = cal["kappa"]
a = cal["a"]
Ter = cal["Ter"]

# --- Simulate DDM at calibrated params ---
print(f"Calibrated: kappa={kappa}, a={a}, Ter={Ter}")

DT = 0.001
N_TRIALS = 10000  # large sample for clean quantiles
B = 6  # one replicate per coherence level (deterministic stimulus)
SEED = 777

rng = np.random.default_rng(SEED)

# Run separate simulations per coherence for clean per-coh stats
model_results = {}
for ci, coh in enumerate(COHERENCES):
    rts_correct = []
    rts_error = []
    direction = 1.0  # fix direction; by symmetry doesn't matter

    evidence = 0.0
    clock = 0.0
    v = kappa * coh * direction

    for trial in range(N_TRIALS):
        evidence = 0.0
        clock = 0.0
        while True:
            dW = rng.standard_normal()
            evidence += v * DT + dW * np.sqrt(DT)
            clock += DT
            if abs(evidence) >= a / 2:
                choice = 1.0 if evidence >= a / 2 else -1.0
                correct = choice == direction
                rt = clock + Ter
                if correct:
                    rts_correct.append(rt)
                else:
                    rts_error.append(rt)
                break
            if clock >= 11.25:  # T_MAX
                # truncated — skip
                break

    rts_correct = np.array(rts_correct)
    rts_error = np.array(rts_error)
    all_rts = np.concatenate([rts_correct, rts_error]) if len(rts_error) > 0 else rts_correct
    acc = len(rts_correct) / (len(rts_correct) + len(rts_error))

    # Quantiles on all RTs
    quantiles = {}
    for q in [10, 30, 50, 70, 90]:
        quantiles[f"Q{q}"] = float(np.percentile(all_rts, q))

    right_skew = quantiles["Q90"] / quantiles["Q50"] if quantiles["Q50"] > 0 else np.nan

    # RT distributions: correct vs error
    rt_equal = None
    mean_rt_correct = float(np.mean(rts_correct)) if len(rts_correct) > 0 else np.nan
    mean_rt_error = float(np.mean(rts_error)) if len(rts_error) > 0 else np.nan

    if len(rts_error) >= 10 and len(rts_correct) >= 10:
        # Deferred #3: test equal distributions
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(rts_correct, rts_error)
        rt_equal = {"ks_stat": round(float(stat), 4), "p": round(float(pval), 4)}

    model_results[coh] = {
        "accuracy": round(acc, 4),
        "mean_rt": round(float(np.mean(all_rts)), 4),
        "n_correct": len(rts_correct),
        "n_error": len(rts_error),
        "quantiles": {k: round(v, 4) for k, v in quantiles.items()},
        "right_skew": round(right_skew, 4) if not np.isnan(right_skew) else None,
        "mean_rt_correct": round(mean_rt_correct, 4),
        "mean_rt_error": round(mean_rt_error, 4) if not np.isnan(mean_rt_error) else None,
        "rt_equal_test": rt_equal,
    }
    print(f"coh={coh:.3f}: acc={acc:.3f}, mean_rt={np.mean(all_rts):.3f}, "
          f"Q50={quantiles['Q50']:.3f}, right_skew={right_skew:.3f}")


# --- Deferred #1: RT quantiles comparison with monkey N ---
print("\n=== Deferred #1: RT quantiles ===")
deferred1 = {}
for coh in COHERENCES:
    sub = monkey_n[monkey_n["coh"] == coh]
    data_q = {f"Q{q}": float(np.percentile(sub["rt"], q)) for q in [10, 30, 50, 70, 90]}
    model_q = model_results[coh]["quantiles"]
    deferred1[str(coh)] = {
        "data": data_q,
        "model": model_q,
        "diff_pct": {k: round((model_q[k] - data_q[k]) / data_q[k] * 100, 1)
                     for k in data_q},
    }
    print(f"coh={coh:.3f}:")
    for k in data_q:
        diff = (model_q[k] - data_q[k]) / data_q[k] * 100
        print(f"  {k}: data={data_q[k]:.3f}, model={model_q[k]:.3f}, diff={diff:+.1f}%")

# --- Deferred #2: Right-skew ---
print("\n=== Deferred #2: Right-skew Q90/Q50 ===")
deferred2 = {}
for coh in COHERENCES:
    sub = monkey_n[monkey_n["coh"] == coh]
    data_skew = float(np.percentile(sub["rt"], 90) / np.percentile(sub["rt"], 50))
    model_skew = model_results[coh]["right_skew"]
    deferred2[str(coh)] = {
        "data": round(data_skew, 4),
        "model": model_skew,
    }
    print(f"coh={coh:.3f}: data={data_skew:.3f}, model={model_skew:.3f}")

# --- Deferred #3: RT correct vs error ---
print("\n=== Deferred #3: RT correct vs error (positive prediction) ===")
deferred3_model = {}
deferred3_data = {}
for coh in COHERENCES:
    mr = model_results[coh]
    if mr["rt_equal_test"] is not None:
        status = "PASS" if mr["rt_equal_test"]["p"] > 0.05 else "FAIL"
        print(f"coh={coh:.3f}: model KS p={mr['rt_equal_test']['p']:.4f} ->{status}")
        deferred3_model[str(coh)] = {
            "status": status,
            "ks_stat": mr["rt_equal_test"]["ks_stat"],
            "p": mr["rt_equal_test"]["p"],
            "mean_rt_correct": mr["mean_rt_correct"],
            "mean_rt_error": mr["mean_rt_error"],
        }
    else:
        print(f"coh={coh:.3f}: too few errors to test ({mr['n_error']})")
        deferred3_model[str(coh)] = {"status": "insufficient_errors", "n_error": mr["n_error"]}

    # Reportable: slow errors in monkey N data
    sub = monkey_n[monkey_n["coh"] == coh]
    correct_rt = sub[sub["correct"] == 1]["rt"]
    error_rt = sub[sub["correct"] == 0]["rt"]
    if len(error_rt) >= 5 and len(correct_rt) >= 5:
        slow_errors = float(error_rt.mean() - correct_rt.mean())
        deferred3_data[str(coh)] = {
            "mean_rt_correct": round(float(correct_rt.mean()), 4),
            "mean_rt_error": round(float(error_rt.mean()), 4),
            "diff_error_minus_correct": round(slow_errors, 4),
            "slow_errors": slow_errors > 0,
        }
        print(f"  monkey N: correct={correct_rt.mean():.3f}, error={error_rt.mean():.3f}, "
              f"diff={slow_errors:+.3f}{'(slow errors)' if slow_errors > 0 else ''}")

# --- Held-out: Monkey B ---
print("\n=== Held-out: Monkey B ===")
held_out = {}
all_pass = True
for coh in COHERENCES:
    sub_b = monkey_b[monkey_b["coh"] == coh]
    data_acc = float(sub_b["correct"].mean())
    data_rt = float(sub_b["rt"].mean())
    model_acc = model_results[coh]["accuracy"]
    model_rt = model_results[coh]["mean_rt"]

    acc_diff = abs(model_acc - data_acc)
    rt_diff_pct = abs(model_rt - data_rt) / data_rt * 100

    acc_pass = acc_diff <= 0.10
    rt_pass = rt_diff_pct <= 25

    held_out[str(coh)] = {
        "data_acc": round(data_acc, 4),
        "model_acc": round(model_acc, 4),
        "acc_diff": round(float(acc_diff), 4),
        "acc_pass": acc_pass,
        "data_mean_rt": round(data_rt, 4),
        "model_mean_rt": round(model_rt, 4),
        "rt_diff_pct": round(float(rt_diff_pct), 1),
        "rt_pass": rt_pass,
    }

    status = "PASS" if acc_pass and rt_pass else "FAIL"
    if not (acc_pass and rt_pass):
        all_pass = False
    print(f"coh={coh:.3f}: acc diff={acc_diff:.3f} {'OK' if acc_pass else 'FAIL'}, "
          f"rt diff={rt_diff_pct:.1f}% {'OK' if rt_pass else 'FAIL'} ->{status}")

print(f"\nHeld-out overall: {'PASS' if all_pass else 'FAIL'}")

# --- Save ---
out = {
    "deferred_1_quantiles": deferred1,
    "deferred_2_right_skew": deferred2,
    "deferred_3_model_rt_equal": deferred3_model,
    "deferred_3_data_slow_errors": deferred3_data,
    "held_out_monkey_b": held_out,
    "held_out_pass": all_pass,
    "calibrated_params": {"kappa": kappa, "a": a, "Ter": Ter},
    "n_trials_simulated": N_TRIALS,
}

outpath = ROOT / "results" / "ddm_deferred.json"
with open(outpath, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}")
