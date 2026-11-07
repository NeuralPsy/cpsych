"""Recompute consistency checks #1-2 at sigma_h=0.008 (D18).

Paired baseline/deprivation, N=100, same method as D13.
Reports onset_advance, dur_excess, H_excess with bootstrap CIs.

Outputs: results/arousal_consistency_d18.json
"""
import json, time
import numpy as np
from pathlib import Path
from cpsych.observables.circular import circular_mean, circular_sd, circular_diff

ROOT = Path(__file__).resolve().parent.parent

V_VM=-1.9; V_MV=-1.9; V_VC=-3.2; A_TONIC=1.0; MU=3.6
TAU_V=10; TAU_M=10; CHI=39600; C0=1.0
OMEGA=2*np.pi/86400; ALPHA=57600
Q_MAX=100; THETA=10; SIGMA_Q=3; Q_THRESH=1.0
SIGMA_VV=0.003; SIGMA_VM=0.015; SIGMA_H=0.008
V_VH=0.325; DT=1.0; DEBOUNCE_S=60

def Q(V): return Q_MAX/(1+np.exp(-(V-THETA)/SIGMA_Q))

def run_single(seed, T_days, v_vh=V_VH, clamp_vm_min=None,
               clamp_start_h=None, clamp_end_h=None):
    """Run one replicate with optional V_m clamping for forced wakefulness.

    clamp_vm_min: if set, V_m >= this value during [clamp_start_h, clamp_end_h].
    Returns: onsets (absolute hours), sleep_durs (hours), H_at_each_onset.
    """
    rng=np.random.default_rng(seed)
    T=T_days*86400; n_steps=int(T/DT)
    vv=0.0; vm=0.0; h=15.0; clock=0.0
    was_wake=True; deb=0.0
    onsets_h=[]; sleep_durs=[]; last_ws=None; H_at_onset=[]

    for step in range(n_steps):
        t=step*DT; t_h=t/3600
        C=C0+np.cos(OMEGA*(clock-ALPHA)); D=V_VC*C+v_vh*h
        Qv=Q(vv); Qm_=Q(vm)
        dvv=(V_VM*Qm_+D-vv)/TAU_V; dvm=(A_TONIC+V_MV*Qv-vm)/TAU_M; dh=(MU*Qm_-h)/CHI
        dW=rng.standard_normal(3)
        vv+=dvv*DT+SIGMA_VV*np.sqrt(DT)*dW[0]
        vm+=dvm*DT+SIGMA_VM*np.sqrt(DT)*dW[1]
        h+=dh*DT+SIGMA_H*np.sqrt(DT)*dW[2]
        clock+=DT

        # Deprivation clamp: force V_m >= clamp_vm_min
        if clamp_vm_min is not None and clamp_start_h <= t_h <= clamp_end_h:
            vm = max(vm, clamp_vm_min)

        Qm_now=Q(vm); iw=Qm_now>Q_THRESH
        if deb>0: deb-=DT; continue
        if was_wake and not iw:
            onsets_h.append(t_h)
            H_at_onset.append(h)
            last_ws=t_h
            deb=DEBOUNCE_S; was_wake=False
        elif not was_wake and iw:
            if last_ws is not None:
                sleep_durs.append(t_h - last_ws)
            deb=DEBOUNCE_S; was_wake=True

    return onsets_h, sleep_durs, H_at_onset

N=100
print(f"Running N={N} paired baseline/deprivation at sigma_h={SIGMA_H}")
t0=time.time()

onset_advances=[]
dur_excesses=[]
h_excesses=[]

for i in range(N):
    # Baseline: 7 days, no clamping
    base_onsets_h, base_durs, base_H_at = run_single(seed=i, T_days=7)

    # Skip first day transient, need >= 3 cycles
    base_onsets_valid = [o for o in base_onsets_h if o > 24]
    base_durs_valid = base_durs[1:] if len(base_durs) > 1 else base_durs  # skip first
    base_H_valid = [h for o, h in zip(base_onsets_h, base_H_at) if o > 24]

    if len(base_onsets_valid) < 3:
        continue

    # Deprivation: clamp V_m >= 1.0 starting just before 2nd bedtime
    second_onset = base_onsets_valid[1]  # 2nd onset in hours
    clamp_start = second_onset - 0.5
    clamp_dur = 18.0  # 18h forced wake -> ~34h total continuous wake
    clamp_end = clamp_start + clamp_dur
    third_onset = base_onsets_valid[2]  # expected 3rd onset without dep

    dep_onsets_h, dep_durs, dep_H_at = run_single(
        seed=i, T_days=7, clamp_vm_min=1.0,
        clamp_start_h=clamp_start, clamp_end_h=clamp_end)

    # Recovery: first onset after clamp_end
    rec_onsets = [(o, h) for o, h in zip(dep_onsets_h, dep_H_at) if o > clamp_end]
    rec_durs = [d for o, d in zip(dep_onsets_h, dep_durs) if o > clamp_end]

    if not rec_onsets or not rec_durs:
        continue

    rec_onset_h = rec_onsets[0][0]
    rec_H = rec_onsets[0][1]
    rec_dur = rec_durs[0]

    base_mean_dur = np.mean(base_durs_valid) if len(base_durs_valid) > 0 else np.nan
    base_mean_H = np.mean(base_H_valid) if len(base_H_valid) > 0 else np.nan

    onset_adv = third_onset - rec_onset_h  # positive = earlier
    dur_exc = rec_dur - base_mean_dur
    h_exc = (rec_H - base_mean_H) / base_mean_H * 100 if base_mean_H > 0 else np.nan

    onset_advances.append(onset_adv)
    dur_excesses.append(dur_exc)
    if not np.isnan(h_exc):
        h_excesses.append(h_exc)

    if (i+1)%20==0:
        print(f"  {i+1}/{N} done, {time.time()-t0:.0f}s")

onset_advances = np.array(onset_advances)
dur_excesses = np.array(dur_excesses)
h_excesses = np.array(h_excesses)

# Circular stats for onset_advance
oa_cmean = circular_mean(onset_advances * 3600, 86400) / 3600 if len(onset_advances)>0 else np.nan
oa_csd = circular_sd(onset_advances * 3600, 86400) / 3600 if len(onset_advances)>0 else np.nan

# Bootstrap CIs
def boot_ci(arr, n_boot=2000, seed=999):
    rng=np.random.default_rng(seed)
    means=[np.mean(rng.choice(arr,len(arr),replace=True)) for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])

def boot_ci_circ(arr, period=24, n_boot=2000, seed=999):
    rng=np.random.default_rng(seed)
    means=[]
    for _ in range(n_boot):
        sample = rng.choice(arr, len(arr), replace=True)
        means.append(circular_mean(sample*3600, 86400)/3600)
    return np.percentile(means, [2.5, 97.5])

print(f"\nResults at sigma_h={SIGMA_H}:")
print(f"  N valid: onset={len(onset_advances)}, dur={len(dur_excesses)}, H={len(h_excesses)}")

if len(onset_advances)>5:
    ci_oa = boot_ci_circ(onset_advances)
    print(f"  onset_advance: circ mean={oa_cmean:.2f}h, CI [{ci_oa[0]:.2f}, {ci_oa[1]:.2f}]")
    oa_status = "PASS" if ci_oa[0]>=1.0 else ("INDETERMINATE" if ci_oa[1]>=1.0 else "FAIL")
    print(f"    Status: {oa_status} (threshold >= 1.0h)")

if len(dur_excesses)>5:
    ci_dur = boot_ci(dur_excesses)
    print(f"  dur_excess: mean={np.mean(dur_excesses):.2f}h, CI [{ci_dur[0]:.2f}, {ci_dur[1]:.2f}]")
    dur_status = "PASS" if ci_dur[0]>=1.0 else ("INDETERMINATE" if ci_dur[1]>=1.0 else "FAIL")
    print(f"    Status: {dur_status} (threshold >= 1.0h)")

if len(h_excesses)>5:
    ci_h = boot_ci(h_excesses)
    print(f"  H_excess: mean={np.mean(h_excesses):.1f}%, CI [{ci_h[0]:.1f}, {ci_h[1]:.1f}]")
    h_status = "PASS" if ci_h[0]>=20.0 else ("INDETERMINATE" if ci_h[1]>=20.0 else "FAIL")
    print(f"    Status: {h_status} (threshold >= 20%)")

elapsed = time.time()-t0
print(f"\nTotal: {elapsed:.0f}s")

out = {
    "sigma_h": SIGMA_H, "v_vh": V_VH, "N": N,
    "onset_advance": {"circ_mean": round(float(oa_cmean),2),
                      "ci": [round(float(ci_oa[0]),2), round(float(ci_oa[1]),2)] if len(onset_advances)>5 else None,
                      "n_valid": len(onset_advances),
                      "status": oa_status if len(onset_advances)>5 else "insufficient"},
    "dur_excess": {"mean": round(float(np.mean(dur_excesses)),2) if len(dur_excesses)>5 else None,
                   "ci": [round(float(ci_dur[0]),2), round(float(ci_dur[1]),2)] if len(dur_excesses)>5 else None,
                   "n_valid": len(dur_excesses),
                   "status": dur_status if len(dur_excesses)>5 else "insufficient"},
    "h_excess": {"mean": round(float(np.mean(h_excesses)),1) if len(h_excesses)>5 else None,
                 "ci": [round(float(ci_h[0]),1), round(float(ci_h[1]),1)] if len(h_excesses)>5 else None,
                 "n_valid": len(h_excesses),
                 "status": h_status if len(h_excesses)>5 else "insufficient"},
}
with open(ROOT/"results"/"arousal_consistency_d18.json","w") as f:
    json.dump(out, f, indent=2)
print(f"Saved results/arousal_consistency_d18.json")
