"""P4 Step 0: Batch benchmark for triple composite.

Assembles arousal + DDM + RL with all P3 couplings.
Measures wall time at B=1, 100, 1000, 5000 for T=86.4 (1 model day).
Reports us/replicate-step. Checks memory at B=5000.

Results: results/p4_benchmark.json
"""
import json
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpsych.core.coupling import Coupling
from cpsych.core.integrator import Integrator, Tier
from cpsych.modules.arousal import ArousalModule
from cpsych.modules.ddm import DDMModule
from cpsych.modules.rl_mood import RLMoodModule
from cpsych.environments.composite_task import CompositeTask


def make_all_couplings(s_kappa=2.0, s_alpha=2.0, s_value=3.0):
    """All P3 couplings for triple composite."""
    return [
        # gain -> kappa
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="ddm", target_param="kappa",
            h=lambda gain, kappa_base: kappa_base * (gain - 1),
            strength=s_kappa, max_strength=4.0, param_min=0.0,
            doc="gain->kappa",
        ),
        # gain -> alpha
        Coupling(
            source_module="arousal", source_output="gain",
            target_module="rl_mood", target_param="alpha",
            h=lambda gain, alpha_base: alpha_base * (gain - 1),
            strength=s_alpha, max_strength=4.0,
            param_min=0.0, param_max=1.0,
            doc="gain->alpha",
        ),
        # value_diff -> drift_bias
        Coupling(
            source_module="rl_mood", source_output="value_diff",
            target_module="ddm", target_param="drift_bias",
            strength=s_value, max_strength=20.0,
            doc="value_diff->drift_bias",
        ),
    ]


def make_composite(B, seed=42):
    """Build the triple composite integrator."""
    arousal = ArousalModule(q_m_wake_mean_baseline=4.2883)
    ddm = DDMModule()
    rl = RLMoodModule()

    env = CompositeTask(coherence=0.128, p_reward=0.75, volatility=0.3)
    couplings = make_all_couplings()

    integ = Integrator(
        modules={"arousal": arousal, "ddm": ddm, "rl_mood": rl},
        environment=env,
        couplings=couplings,
        tiers=[
            Tier("slow", 1.0, ["arousal"]),
            Tier("fast", 0.001, ["ddm", "rl_mood"]),
        ],
        trial_driver="ddm",
        gate_module="arousal",
        gate_output="is_wake",
        B=B,
        seed=seed,
        obs_dt=86.4,  # record only start and end (minimal memory)
    )
    return integ


def benchmark_run(B, T=86.4):
    """Run composite at given B, return timing info."""
    integ = make_composite(B)

    # Count total replicate-steps:
    # Slow tier: T/dt_slow steps of arousal (1 module, 4 states)
    # Fast tier: T/dt_fast steps of DDM+RL (ratio 1000:1)
    # Total fast steps per slow step = 1000
    # Total slow steps = T / 1.0 = 86 (for T=86.4)
    # Total EM steps = slow_steps * (1 + fast_ratio) per module
    # But the metric is replicate-steps = B * total_steps_all_modules
    dt_slow = 1.0
    dt_fast = 0.001
    n_slow = int(round(T / dt_slow))
    fast_ratio = int(round(dt_slow / dt_fast))

    # Each slow step: 1 arousal step + 1000 fast steps (DDM + RL each)
    # Total module-steps = n_slow * (1 + fast_ratio * 2)
    # = 86 * (1 + 2000) = 86 * 2001 = 172_086
    total_module_steps = n_slow * (1 + fast_ratio * 2)
    total_replicate_steps = B * total_module_steps

    t0 = time.perf_counter()
    state = integ.run(T=T)
    t1 = time.perf_counter()

    wall_s = t1 - t0
    us_per_rep_step = (wall_s * 1e6) / total_replicate_steps

    return {
        "B": B,
        "T": T,
        "wall_s": round(wall_s, 3),
        "total_replicate_steps": total_replicate_steps,
        "us_per_replicate_step": round(us_per_rep_step, 4),
        "trial_count_median": int(np.median(state.trial_count)),
        "n_recordings": len(next(iter(state.recordings.values()))),
    }


def benchmark_memory(B=5000, T=86.4):
    """Check peak memory at B=5000."""
    tracemalloc.start()
    integ = make_composite(B)
    state = integ.run(T=T)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "B": B,
        "peak_mb": round(peak / 1e6, 1),
        "current_mb": round(current / 1e6, 1),
        "n_recordings": len(next(iter(state.recordings.values()))),
    }


def main():
    B_values = [1, 100, 1000, 5000]
    T = 86.4  # 1 model day

    results = {"T": T, "benchmarks": [], "description":
        "Triple composite: arousal(slow,dt=1) + DDM+RL(fast,dt=0.001). "
        "All P3 couplings. CompositeTask. Gate by is_wake."}

    print(f"{'B':>6} {'wall_s':>8} {'us/rep-step':>12} {'trials':>7} {'recs':>5}")
    print("-" * 45)

    for B in B_values:
        print(f"{B:>6}  ...", end="", flush=True)
        r = benchmark_run(B, T)
        results["benchmarks"].append(r)
        print(f"\r{r['B']:>6} {r['wall_s']:>8.1f} {r['us_per_replicate_step']:>12.4f} "
              f"{r['trial_count_median']:>7} {r['n_recordings']:>5}")

    # Memory check at B=5000
    print("\nMemory check at B=5000...")
    mem = benchmark_memory(5000, T)
    results["memory"] = mem
    print(f"  Peak: {mem['peak_mb']:.1f} MB, Current: {mem['current_mb']:.1f} MB, "
          f"Recordings: {mem['n_recordings']}")

    # Speedup ratio
    if len(results["benchmarks"]) >= 2:
        us_b1 = results["benchmarks"][0]["us_per_replicate_step"]
        us_bmax = results["benchmarks"][-1]["us_per_replicate_step"]
        results["speedup_b1_to_bmax"] = round(us_b1 / us_bmax, 1)
        print(f"\nSpeedup B=1 -> B={B_values[-1]}: "
              f"{results['speedup_b1_to_bmax']}x")

    # Reference for regression test
    results["reference_us"] = results["benchmarks"][-1]["us_per_replicate_step"]
    results["reference_B"] = results["benchmarks"][-1]["B"]

    out_path = os.path.join(os.path.dirname(__file__), '..',
                             'results', 'p4_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    print("=== P4 Step 0: Batch benchmark ===\n")
    main()
