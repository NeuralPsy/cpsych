"""v_vh window boundaries at sigma_h=0.008 (D18).

Critical question: old window [0.30, 0.34] widened to [0.25, 0.35] by sleep
criterion alone. But at v_vh=0.25 deprivation metrics previously FAILED.
Check whether they pass at sigma_h=0.008.

Boundary points only: 0.25, 0.29, 0.30, 0.325, 0.34, 0.35.
N=30 paired baseline/deprivation per point.
"""
import numpy as np, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cpsych.observables.circular import circular_mean

V_VM=-1.9; V_MV=-1.9; V_VC=-3.2; A_TONIC=1.0; MU=3.6
TAU_V=10; TAU_M=10; CHI=39600; C0=1.0
OMEGA=2*np.pi/86400; ALPHA=57600
Q_MAX=100; THETA=10; SIGMA_Q=3; Q_THRESH=1.0
SIGMA_VV=0.003; SIGMA_VM=0.015; SIGMA_H=0.008
DT=1.0; DEBOUNCE_S=60

def Q(V): return Q_MAX/(1+np.exp(-(V-THETA)/SIGMA_Q))

def run(seed, v_vh, clamp_vm_min=None, clamp_start_h=None, clamp_end_h=None):
    rng=np.random.default_rng(seed)
    n_steps=7*86400
    vv=vm=0.0; h=15.0; clock=0.0; was_wake=True; deb=0.0
    onsets_h=[]; sleep_durs=[]; H_at=[]
    for step in range(n_steps):
        t_h=step/3600
        C=C0+np.cos(OMEGA*(step-ALPHA)); D=V_VC*C+v_vh*h
        Qv=Q(vv); Qm_=Q(vm)
        vv+=(V_VM*Qm_+D-vv)/TAU_V+SIGMA_VV*rng.standard_normal()
        vm+=(A_TONIC+V_MV*Qv-vm)/TAU_M+SIGMA_VM*rng.standard_normal()
        h+=(MU*Qm_-h)/CHI+SIGMA_H*rng.standard_normal()
        clock+=1
        if clamp_vm_min is not None and clamp_start_h<=t_h<=clamp_end_h:
            vm=max(vm,clamp_vm_min)
        iw=Q(vm)>Q_THRESH
        if deb>0: deb-=1; continue
        if was_wake and not iw:
            onsets_h.append(t_h); H_at.append(h); deb=DEBOUNCE_S; was_wake=False
        elif not was_wake and iw:
            if onsets_h: sleep_durs.append(t_h-onsets_h[-1])
            deb=DEBOUNCE_S; was_wake=True
    return onsets_h, sleep_durs, H_at

N=30
print(f"v_vh window at sigma_h={SIGMA_H}, N={N}")
t0=time.time()
results = {}

for v_vh in [0.25, 0.29, 0.30, 0.325, 0.34, 0.35]:
    sleeps=[]; oas=[]; durs=[]; hexs=[]
    for i in range(N):
        bo,bd,bh = run(i, v_vh)
        bv=[(o,h) for o,h in zip(bo,bh) if o>24]
        bdv=bd[1:] if len(bd)>1 else bd
        if len(bv)<3 or len(bdv)<1: continue
        sleeps.append(np.mean(bdv))
        so=bv[1][0]; cs=so-0.5; ce=cs+18; to=bv[2][0]
        bhm=np.mean([h for _,h in bv])
        do,dd,dh=run(i,v_vh,clamp_vm_min=1.0,clamp_start_h=cs,clamp_end_h=ce)
        rec=[(o,h) for o,h in zip(do,dh) if o>ce]
        recd=[d for o,d in zip(do,dd) if o>ce]
        if not rec or not recd: continue
        oas.append(to-rec[0][0])
        durs.append(recd[0]-np.mean(bdv))
        hexs.append((rec[0][1]-bhm)/bhm*100)

    ms=np.nanmean(sleeps) if sleeps else float('nan')
    def bci(a):
        if len(a)<5: return float('nan'),(float('nan'),float('nan'))
        r=np.random.default_rng(999)
        b=[np.mean(r.choice(a,len(a),replace=True)) for _ in range(1000)]
        return np.mean(a),(np.percentile(b,2.5),np.percentile(b,97.5))
    oam,oac=bci(oas); dm,dc=bci(durs); hm,hc=bci(hexs)
    s_ok=7<=ms<=9
    dur_ok=not np.isnan(dm) and dc[0]>=1.0
    h_ok=not np.isnan(hm) and hc[0]>=20
    all_ok=s_ok and dur_ok and h_ok
    results[v_vh]={'sleep':round(ms,2),'sleep_ok':s_ok,
        'oa':round(oam,2),'oa_ci':[round(oac[0],2),round(oac[1],2)],
        'dur':round(dm,2),'dur_ci':[round(dc[0],2),round(dc[1],2)],'dur_ok':dur_ok,
        'H':round(hm,1),'H_ci':[round(hc[0],1),round(hc[1],1)],'H_ok':h_ok,
        'all_ok':all_ok}
    print(f"v_vh={v_vh:.3f}: sleep={ms:.2f}h {'OK' if s_ok else 'NO'} | "
          f"oa={oam:.2f}[{oac[0]:.2f},{oac[1]:.2f}] | "
          f"dur={dm:.2f}[{dc[0]:.2f},{dc[1]:.2f}] {'OK' if dur_ok else 'NO'} | "
          f"H={hm:.1f}%[{hc[0]:.1f},{hc[1]:.1f}] {'OK' if h_ok else 'NO'} | "
          f"ALL={'OK' if all_ok else 'NO'}")

print(f"\nTotal: {time.time()-t0:.0f}s")
with open(Path(__file__).resolve().parent.parent/"results"/"arousal_vh_window_d18.json","w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved results/arousal_vh_window_d18.json")
