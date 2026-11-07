"""Compute Q_M_WAKE_MEAN_BASELINE and write to results/arousal_calibration.json.

Procedure: day-by-day integration until settling (see specs/arousal.md v2.2).
Run from project root:
    PYTHONPATH=src python experiments/arousal_calibration.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpsych.modules.arousal import (
    compute_calibration, _AROUSAL_SPEC, _I,
)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "arousal_calibration.json"


def main():
    q_ref, settled_day, daily_means = compute_calibration()

    # Baseline params as dict for JSON
    baseline_params = {p.name: p.default for p in _AROUSAL_SPEC.params}
    # Remove noise (calibration is deterministic)
    baseline_params["sigma_vv"] = 0.0
    baseline_params["sigma_vm"] = 0.0
    baseline_params["sigma_h"] = 0.0

    result = {
        "Q_M_WAKE_MEAN_BASELINE": round(q_ref, 4),
        "settled_day": settled_day,
        "settling_tol": 0.0001,
        "daily_means": [round(m, 4) for m in daily_means],
        "baseline_params": baseline_params,
        "procedure": (
            "day-by-day until |delta| < 0.0001, "
            "Q_thresh=1.0, T_trim=1800, day 0 transient skipped"
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Q_M_WAKE_MEAN_BASELINE = {q_ref:.4f}")
    print(f"Settled on day {settled_day}")
    print(f"Daily means: {[f'{m:.4f}' for m in daily_means]}")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
