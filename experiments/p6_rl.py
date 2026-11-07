"""P6 Step 5: RL comparison with Eckstein 2022 data.

Criterion mode, all params fixed, adults 18+.
Win-stay, lose-shift, accuracy, recovery speed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpsych.modules.rl_mood import RLMoodModule
from cpsych.environments.reversal_bandit import ReversalBandit

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eckstein2022")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Fixed params from P2.4 calibration
ALPHA, BETA = 0.15, 2.50
TAU_M, LAM = 2.163, 0.49
N_TRIALS = 120  # Eckstein protocol


def load_adults():
    demo = pd.read_csv(os.path.join(DATA_DIR,
                       "EcksteinEtAl_eLife2022_Demographics.csv"))
    adults = demo[demo["PreciseYrs"] >= 18]["ID"].values
    data = {}
    for sid in adults:
        path = os.path.join(DATA_DIR, f"PS_{sid}.csv")
        if os.path.exists(path):
            data[sid] = pd.read_csv(path)
    return data, demo


def empirical_observables(data):
    ws_list, ls_list, acc_list, rec_list = [], [], [], []
    for sid, df in data.items():
        n = len(df)
        if n < 50:
            continue
        choices = df["selected_box"].values
        rewards = df["reward"].values
        correct = df["ACC"].values.astype(float)

        nw = ns = nwst = nlsh = 0
        for t in range(1, n):
            if rewards[t-1] == 1:
                nw += 1
                if choices[t] == choices[t-1]:
                    nwst += 1
            else:
                ns += 1
                if choices[t] != choices[t-1]:
                    nlsh += 1

        ws_list.append(nwst / nw if nw else np.nan)
        ls_list.append(nlsh / ns if ns else np.nan)
        acc_list.append(correct.mean())

        switch_t = np.where(df["switch_trial"].values)[0]
        speeds = []
        for st in switch_t:
            found = False
            for k in range(1, 50):
                if st + k >= n:
                    break
                w = correct[st+max(1, k-3):st+k+1]
                if len(w) >= 4 and w.mean() >= 0.70:
                    speeds.append(k)
                    found = True
                    break
            if not found:
                speeds.append(50)
        if speeds:
            rec_list.append(np.mean(speeds))

    return (np.array(ws_list), np.array(ls_list),
            np.array(acc_list), np.array(rec_list))


def simulate_model(B, seed=42):
    rng = np.random.default_rng(seed)
    mod = RLMoodModule()
    env = ReversalBandit(p_reward=0.75, p_incorrect=0.0,
                         reversal_mode="criterion",
                         criterion_range=(7, 15))
    env.reset(B, rng)

    x = mod.default_state(B)
    theta = mod.default_params(B)
    theta[:, 0] = ALPHA
    theta[:, 1] = BETA
    theta[:, 2] = TAU_M
    theta[:, 3] = LAM
    mask = np.ones(B, dtype=bool)

    all_choices = np.zeros((N_TRIALS, B))
    all_rewards = np.zeros((N_TRIALS, B))
    all_correct = np.zeros((N_TRIALS, B))
    all_switch = np.zeros((N_TRIALS, B), dtype=bool)
    prev_arm = env._correct_arm.copy()

    for t in range(N_TRIALS):
        env.next_trial(rng, mask)
        all_switch[t] = env._correct_arm != prev_arm
        prev_arm = env._correct_arm.copy()

        out = mod.outputs(x, theta)
        choice = (rng.random(B) < out["choice_prob"]).astype(float)
        all_correct[t] = choice == env._correct_arm
        r = env.reward(choice, rng)["r"]
        all_choices[t] = choice
        all_rewards[t] = r
        x = mod.trial_update(x, theta, mask, {"r": r, "choice": choice})

    # Observables per simulated subject
    ws_list, ls_list, acc_list, rec_list = [], [], [], []
    for b in range(B):
        ch = all_choices[:, b]
        rw = all_rewards[:, b]
        cr = all_correct[:, b]

        nw = ns = nwst = nlsh = 0
        for t in range(1, N_TRIALS):
            if rw[t-1] == 1:
                nw += 1
                if ch[t] == ch[t-1]:
                    nwst += 1
            else:
                ns += 1
                if ch[t] != ch[t-1]:
                    nlsh += 1

        ws_list.append(nwst / nw if nw else np.nan)
        ls_list.append(nlsh / ns if ns else np.nan)
        acc_list.append(cr.mean())

        st_idx = np.where(all_switch[:, b])[0]
        speeds = []
        for st in st_idx:
            found = False
            for k in range(1, 50):
                if st + k >= N_TRIALS:
                    break
                w = cr[st+max(1, k-3):st+k+1]
                if len(w) >= 4 and w.mean() >= 0.70:
                    speeds.append(k)
                    found = True
                    break
            if not found:
                speeds.append(50)
        if speeds:
            rec_list.append(np.mean(speeds))

    return (np.array(ws_list), np.array(ls_list),
            np.array(acc_list), np.array(rec_list))


def simulate_diagnostic(B, alpha, beta, persistence=0.0, seed=42):
    """Simulate with custom params + optional choice persistence."""
    rng = np.random.default_rng(seed)
    mod = RLMoodModule()
    env = ReversalBandit(p_reward=0.75, p_incorrect=0.0,
                         reversal_mode="criterion",
                         criterion_range=(7, 15))
    env.reset(B, rng)

    x = mod.default_state(B)
    theta = mod.default_params(B)
    theta[:, 0] = alpha
    theta[:, 1] = beta
    theta[:, 2] = TAU_M
    theta[:, 3] = LAM
    mask = np.ones(B, dtype=bool)

    all_choices = np.zeros((N_TRIALS, B))
    all_rewards = np.zeros((N_TRIALS, B))
    all_correct = np.zeros((N_TRIALS, B))
    all_switch = np.zeros((N_TRIALS, B), dtype=bool)
    prev_arm = env._correct_arm.copy()
    prev_choice = np.full(B, 0.5)  # no previous choice at start

    for t in range(N_TRIALS):
        env.next_trial(rng, mask)
        all_switch[t] = env._correct_arm != prev_arm
        prev_arm = env._correct_arm.copy()

        if persistence == 0.0:
            out = mod.outputs(x, theta)
            choice = (rng.random(B) < out["choice_prob"]).astype(float)
        else:
            # Softmax with choice kernel: P(1) = sigmoid(beta*(V1-V0) + p*(2*prev-1))
            V0, V1 = x[:, 0], x[:, 1]
            logit = beta * (V1 - V0) + persistence * (2 * prev_choice - 1)
            prob1 = 1.0 / (1.0 + np.exp(-logit))
            choice = (rng.random(B) < prob1).astype(float)

        all_correct[t] = choice == env._correct_arm
        r = env.reward(choice, rng)["r"]
        all_choices[t] = choice
        all_rewards[t] = r
        prev_choice = choice
        x = mod.trial_update(x, theta, mask, {"r": r, "choice": choice})

    # Compute observables
    ws_list, acc_list, rec_list = [], [], []
    for b in range(B):
        ch = all_choices[:, b]
        rw = all_rewards[:, b]
        cr = all_correct[:, b]

        nw = nwst = 0
        for t in range(1, N_TRIALS):
            if rw[t-1] == 1:
                nw += 1
                if ch[t] == ch[t-1]:
                    nwst += 1
        ws_list.append(nwst / nw if nw else np.nan)
        acc_list.append(cr.mean())

        st_idx = np.where(all_switch[:, b])[0]
        speeds = []
        for st in st_idx:
            found = False
            for k in range(1, 50):
                if st + k >= N_TRIALS:
                    break
                w = cr[st+max(1, k-3):st+k+1]
                if len(w) >= 4 and w.mean() >= 0.70:
                    speeds.append(k)
                    found = True
                    break
            if not found:
                speeds.append(50)
        if speeds:
            rec_list.append(np.mean(speeds))

    return {"ws": float(np.nanmean(ws_list)),
            "acc": float(np.nanmean(acc_list)),
            "recovery": float(np.nanmean(rec_list)) if rec_list else float("nan")}


def main():
    print("Loading Eckstein data (adults 18+)...")
    data, demo = load_adults()
    print(f"  Adults with trial data: {len(data)}")

    print("Computing empirical observables...")
    e_ws, e_ls, e_acc, e_rec = empirical_observables(data)

    print("Simulating model (criterion mode)...")
    m_ws, m_ls, m_acc, m_rec = simulate_model(len(data))

    print()
    print("=== Comparison ===")
    for label, emp, mdl in [("Win-stay", e_ws, m_ws),
                             ("Lose-shift", e_ls, m_ls),
                             ("Accuracy", e_acc, m_acc)]:
        em = np.nanmean(emp)
        mm = np.nanmean(mdl)
        eiq = (np.nanpercentile(emp, 25), np.nanpercentile(emp, 75))
        in_iqr = eiq[0] <= mm <= eiq[1]
        print(f"  {label:12s}: data={em:.3f}, model={mm:.3f}, "
              f"diff={mm-em:+.3f}, IQR=[{eiq[0]:.3f},{eiq[1]:.3f}], "
              f"in_IQR={in_iqr}")

    er = np.nanmean(e_rec)
    mr = np.nanmean(m_rec)
    print(f"  {'Recovery':12s}: data={er:.1f}, model={mr:.1f}, "
          f"ratio={mr/er:.2f}")

    # Also full sample
    print()
    print("=== Full sample (all ages) ===")
    ps = pd.read_csv(os.path.join(DATA_DIR,
                     "EcksteinEtAl_eLife2022_ProbSwitch.csv"))
    print(f"  WS (data): {ps['WS'].mean():.3f}")
    print(f"  LS (data): {ps['LS'].mean():.3f}")
    print(f"  ACC (data): {ps['ACC'].mean():.3f}")

    # === Diagnostic: Eckstein parameters ===
    print()
    print("=== Diagnostic ===")
    B_diag = len(data)

    # Alpha only: Eckstein alpha, our beta
    diag_alpha = simulate_diagnostic(B_diag, alpha=0.76, beta=BETA, seed=100)
    print(f"  alpha_eckstein (a=0.76, b={BETA}): "
          f"WS={diag_alpha['ws']:.3f}, ACC={diag_alpha['acc']:.3f}")

    # Full Eckstein params with persistence
    diag_full = simulate_diagnostic(B_diag, alpha=0.76, beta=3.0,
                                     persistence=0.30, seed=101)
    print(f"  full Eckstein (a=0.76, b=3.0, p=0.3): "
          f"WS={diag_full['ws']:.3f}, ACC={diag_full['acc']:.3f}")

    # Concordance ratios
    ratio_alpha = round(0.76 / ALPHA, 1)
    ratio_beta = round(3.0 / BETA, 1)
    print(f"  ratio_alpha = {ratio_alpha}, ratio_beta = {ratio_beta}")

    result = {
        "n_adults": len(data),
        "empirical": {
            "ws": {"mean": float(np.nanmean(e_ws)),
                   "sd": float(np.nanstd(e_ws, ddof=1)),
                   "iqr": [float(np.nanpercentile(e_ws, 25)),
                           float(np.nanpercentile(e_ws, 75))]},
            "ls": {"mean": float(np.nanmean(e_ls)),
                   "sd": float(np.nanstd(e_ls, ddof=1)),
                   "iqr": [float(np.nanpercentile(e_ls, 25)),
                           float(np.nanpercentile(e_ls, 75))]},
            "acc": {"mean": float(np.nanmean(e_acc)),
                    "sd": float(np.nanstd(e_acc, ddof=1))},
            "recovery": {"mean": float(np.nanmean(e_rec)),
                         "median": float(np.nanmedian(e_rec))},
        },
        "model": {
            "ws": float(np.nanmean(m_ws)),
            "ls": float(np.nanmean(m_ls)),
            "acc": float(np.nanmean(m_acc)),
            "recovery": float(np.nanmean(m_rec)),
            "params": {"alpha": ALPHA, "beta": BETA,
                       "tau_m": TAU_M, "lam": LAM},
            "env": {"mode": "criterion", "p_reward": 0.75,
                    "p_incorrect": 0.0, "criterion_range": [7, 15]},
        },
        "diagnostic": {
            "alpha_eckstein": diag_alpha,
            "persistence": {
                "runs": {
                    "alpha076_beta300_p03": diag_full,
                },
            },
            "eckstein_concordance": {
                "ratio_alpha": ratio_alpha,
                "ratio_beta": ratio_beta,
            },
        },
    }

    out_path = os.path.join(RESULTS_DIR, "p6_rl_comparison.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
