"""Phillips-Robinson 2007/2008 arousal (sleep-wake) module.

3 physical state variables (V_v, V_m, H) plus a clock state variable
for the non-autonomous circadian drive C(t). The clock is a technical
device: drift = 1.0 s/s, no diffusion, not a physical variable.

Reference: specs/arousal.md v2.2, specs/arousal_sources.md.
Equations verified from deposited implementations (CellML, BioModels,
sleepR). Parameter set: PR08 with adjusted v_vh (0.19 -> 0.325, D13).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.specs import ModuleSpec, StateVar, Param, Temporality, Exposure
from ..core.module import SDEModule

Q_THRESH = 1.0  # s^{-1}, wake/sleep demarcation threshold
T_TRIM = 1800   # s, trimming from each end of wake periods
SETTLING_TOL = 0.0001  # s^{-1}, inter-day settling tolerance

_AROUSAL_SPEC = ModuleSpec(
    name="arousal",
    states=(
        StateVar("vv", Temporality.CONTINUOUS, 0.0,
                 doc="Mean potential VLPO (mV)"),
        StateVar("vm", Temporality.CONTINUOUS, 0.0,
                 doc="Mean potential MA (mV)"),
        StateVar("h", Temporality.CONTINUOUS, 15.0,
                 doc="Homeostatic sleep pressure (nM)"),
        StateVar("clock", Temporality.CONTINUOUS, 0.0,
                 doc="Model time (s), technical — for circadian C(t)"),
    ),
    params=(
        Param("tau_v", 10.0, 5.0, 50.0, Exposure.INTERNAL, "s",
              "VLPO time constant"),
        Param("tau_m", 10.0, 5.0, 50.0, Exposure.INTERNAL, "s",
              "MA time constant"),
        Param("v_vm", -1.9, -2.70, -0.50, Exposure.INTERNAL, "mV*s",
              "Inhibition MA->VLPO"),
        Param("v_mv", -1.9, -5.0, -0.90, Exposure.INTERNAL, "mV*s",
              "Inhibition VLPO->MA"),
        Param("A", 1.0, 0.5, 2.0, Exposure.INTERNAL, "mV",
              "Tonic input to MA"),
        Param("v_vc", -3.2, -10.0, -1.0, Exposure.INTERNAL, "mV",
              "Circadian coupling"),
        Param("v_vh", 0.325, 0.08, 0.50, Exposure.INTERNAL, "mV/nM",
              "Adenosine sensitivity (adjusted from PR08 0.19, D13)"),
        Param("Q_max", 100.0, 100.0, 100.0, Exposure.INTERNAL, "1/s",
              "Max firing rate (fixed)"),
        Param("theta_sig", 10.0, 10.0, 10.0, Exposure.INTERNAL, "mV",
              "Sigmoid threshold (fixed)"),
        Param("sigma_q", 3.0, 1.0, 6.0, Exposure.INTERNAL, "mV",
              "Sigmoid slope"),
        Param("chi", 39600.0, 28800.0, 50400.0, Exposure.INTERNAL, "s",
              "Somnogen time constant (11h)"),
        Param("mu", 3.6, 1.0, 10.0, Exposure.INTERNAL, "nM*s",
              "Somnogen production rate"),
        Param("c0", 1.0, 1.0, 1.0, Exposure.INTERNAL, "",
              "Circadian offset (fixed)"),
        Param("omega", 2 * np.pi / 86400,
              2 * np.pi / 86400, 2 * np.pi / 86400,
              Exposure.INTERNAL, "1/s",
              "Circadian frequency (fixed, 24h)"),
        Param("alpha", 57600.0, 0.0, 86400.0, Exposure.INTERNAL, "s",
              "Circadian phase (16h)"),
        Param("sigma_vv", 0.003, 0.0, 2.0, Exposure.INTERNAL, "mV/sqrt(s)",
              "Noise on V_v"),
        Param("sigma_vm", 0.015, 0.0, 5.0, Exposure.INTERNAL, "mV/sqrt(s)",
              "Noise on V_m"),
        Param("sigma_h", 0.008, 0.0, 0.5, Exposure.INTERNAL, "nM/sqrt(s)",
              "Noise on H (D18: argmax frac in [15,30] min)"),
    ),
    input_names=(),
    output_names=("gain", "Qm", "is_wake"),
    timescale=10.0,
    doc="Phillips-Robinson 2007/2008 flip-flop sleep-wake model.",
    circular_observables=(
        ("sleep_onset_tod", 24.0),
        ("sleep_offset_tod", 24.0),
        ("onset_advance", 24.0),
    ),
)

# Parameter indices for fast access in drift/diffusion/outputs
_I = {p.name: i for i, p in enumerate(_AROUSAL_SPEC.params)}


def _sigmoid(V, Q_max, theta_sig, sigma_q):
    """Firing rate Q(V) = Q_max / (1 + exp(-(V - theta) / sigma_q))."""
    return Q_max / (1.0 + np.exp(-(V - theta_sig) / sigma_q))


class ArousalModule(SDEModule):
    """Phillips-Robinson 2007/2008 sleep-wake flip-flop model.

    Physical state: V_v (VLPO), V_m (MA), H (sleep pressure).
    Technical state: clock (model time for circadian C(t)).

    Parameters
    ----------
    q_m_wake_mean_baseline : float
        Calibration constant from results/arousal_calibration.json.
        No default — must be provided explicitly (rule 1).
    """

    def __init__(self, q_m_wake_mean_baseline: float):
        self._q_ref = q_m_wake_mean_baseline

    @property
    def spec(self) -> ModuleSpec:
        return _AROUSAL_SPEC

    @property
    def q_m_wake_mean_baseline(self) -> float:
        return self._q_ref

    def drift(self, x, theta, u):
        vv = x[:, 0:1]
        vm = x[:, 1:2]
        h = x[:, 2:3]
        clock = x[:, 3:4]

        tau_v = theta[:, _I["tau_v"]:_I["tau_v"] + 1]
        tau_m = theta[:, _I["tau_m"]:_I["tau_m"] + 1]
        v_vm = theta[:, _I["v_vm"]:_I["v_vm"] + 1]
        v_mv = theta[:, _I["v_mv"]:_I["v_mv"] + 1]
        A = theta[:, _I["A"]:_I["A"] + 1]
        v_vc = theta[:, _I["v_vc"]:_I["v_vc"] + 1]
        v_vh = theta[:, _I["v_vh"]:_I["v_vh"] + 1]
        Q_max = theta[:, _I["Q_max"]:_I["Q_max"] + 1]
        theta_sig = theta[:, _I["theta_sig"]:_I["theta_sig"] + 1]
        sigma_q = theta[:, _I["sigma_q"]:_I["sigma_q"] + 1]
        chi = theta[:, _I["chi"]:_I["chi"] + 1]
        mu = theta[:, _I["mu"]:_I["mu"] + 1]
        c0 = theta[:, _I["c0"]:_I["c0"] + 1]
        omega = theta[:, _I["omega"]:_I["omega"] + 1]
        alpha = theta[:, _I["alpha"]:_I["alpha"] + 1]

        Qm = _sigmoid(vm, Q_max, theta_sig, sigma_q)
        Qv = _sigmoid(vv, Q_max, theta_sig, sigma_q)
        C = c0 + np.cos(omega * (clock - alpha))
        D = v_vc * C + v_vh * h

        dvv = (v_vm * Qm + D - vv) / tau_v
        dvm = (A + v_mv * Qv - vm) / tau_m
        dh = (mu * Qm - h) / chi
        dclock = np.ones_like(clock)

        return np.hstack([dvv, dvm, dh, dclock])

    def diffusion(self, x, theta):
        sigma_vv = theta[:, _I["sigma_vv"]:_I["sigma_vv"] + 1]
        sigma_vm = theta[:, _I["sigma_vm"]:_I["sigma_vm"] + 1]
        sigma_h = theta[:, _I["sigma_h"]:_I["sigma_h"] + 1]
        zero = np.zeros((x.shape[0], 1))
        return np.hstack([sigma_vv, sigma_vm, sigma_h, zero])

    def outputs(self, x, theta):
        vm = x[:, 1]
        Q_max = theta[:, _I["Q_max"]]
        theta_sig = theta[:, _I["theta_sig"]]
        sigma_q = theta[:, _I["sigma_q"]]
        Qm = _sigmoid(vm, Q_max, theta_sig, sigma_q)
        gain = Qm / self._q_ref
        is_wake = (Qm > Q_THRESH).astype(float)
        return {"gain": gain, "Qm": Qm, "is_wake": is_wake}


# -------------------------------------------------------------------
# Calibration helper
# -------------------------------------------------------------------

def compute_calibration(
    theta_override: dict[str, float] | None = None,
    dt: float = 1.0,
    q_thresh: float = Q_THRESH,
    t_trim: int = T_TRIM,
    settling_tol: float = SETTLING_TOL,
    max_days: int = 10,
) -> tuple[float, int, list[float]]:
    """Compute Q_M_WAKE_MEAN_BASELINE by day-by-day settling.

    Returns (q_m_wake_mean, settled_day, daily_means).
    settled_day is 0-indexed (day 0 = transient, day 1 = first usable).

    Raises RuntimeError if settling not achieved within max_days.
    """
    mod = ArousalModule(q_m_wake_mean_baseline=1.0)  # dummy, only drift used
    x = mod.default_state(1)
    theta = mod.default_params(1)
    if theta_override:
        for k, v in theta_override.items():
            theta[0, _I[k]] = v
    # Zero out noise for deterministic calibration
    theta[0, _I["sigma_vv"]] = 0.0
    theta[0, _I["sigma_vm"]] = 0.0
    theta[0, _I["sigma_h"]] = 0.0
    u = np.zeros((1, 0))

    steps_per_day = int(86400 / dt)
    trim_steps = int(t_trim / dt)
    min_wake_steps = int(3600 / dt)
    daily_means: list[float] = []
    prev_mean: float | None = None

    for day in range(max_days):
        # Integrate one day, track Q_m online
        qm_day = np.empty(steps_per_day)
        for step in range(steps_per_day):
            dx = mod.drift(x, theta, u)
            x = x + dx * dt
            qm_day[step] = _sigmoid(
                x[0, 1], theta[0, _I["Q_max"]],
                theta[0, _I["theta_sig"]], theta[0, _I["sigma_q"]])

        # Find wake periods and compute trimmed mean
        is_wake = qm_day > q_thresh
        changes = np.diff(is_wake.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if is_wake[0]:
            starts = np.concatenate([[0], starts])
        if is_wake[-1]:
            ends = np.concatenate([ends, [steps_per_day]])

        trimmed_samples: list[float] = []
        for s, e in zip(starts, ends):
            ts = s + trim_steps
            te = e - trim_steps
            if te - ts < min_wake_steps:
                continue
            trimmed_samples.extend(qm_day[ts:te].tolist())

        day_mean = float(np.mean(trimmed_samples)) if trimmed_samples else 0.0
        daily_means.append(day_mean)

        # Skip day 0 (transient). Check settling from day 2 onward.
        if day >= 2 and prev_mean is not None:
            if abs(day_mean - prev_mean) < settling_tol:
                return day_mean, day, daily_means
        prev_mean = day_mean

    raise RuntimeError(
        f"Calibration did not settle within {max_days} days. "
        f"Daily means: {daily_means}")


def load_arousal_calibration(path: str | Path) -> float:
    """Load Q_M_WAKE_MEAN_BASELINE from calibration JSON."""
    with open(path) as f:
        data = json.load(f)
    return float(data["Q_M_WAKE_MEAN_BASELINE"])


# -------------------------------------------------------------------
# Deterministic simulation helper (for tests and validation)
# -------------------------------------------------------------------

def run_deterministic(
    mod: ArousalModule,
    T_days: int = 7,
    dt: float = 1.0,
    theta_override: dict[str, float] | None = None,
) -> dict:
    """Run deterministic simulation and extract transitions online.

    Returns dict with transitions, sleep/wake durations, gain stats.
    Does NOT store full trajectory — online detection only.
    """
    x = mod.default_state(1)
    theta = mod.default_params(1)
    if theta_override:
        for k, v in theta_override.items():
            theta[0, _I[k]] = v
    theta[0, _I["sigma_vv"]] = 0.0
    theta[0, _I["sigma_vm"]] = 0.0
    theta[0, _I["sigma_h"]] = 0.0
    u = np.zeros((1, 0))

    T = int(T_days * 86400 / dt)
    transitions: list[dict] = []
    Q_max_val = theta[0, _I["Q_max"]]
    theta_sig_val = theta[0, _I["theta_sig"]]
    sigma_q_val = theta[0, _I["sigma_q"]]

    Qm_init = _sigmoid(x[0, 1], Q_max_val, theta_sig_val, sigma_q_val)
    was_wake = Qm_init > Q_THRESH
    period_start_h = 0.0

    # Per-wake-period gain tracking
    wake_qm_max = -np.inf
    wake_qm_min = np.inf
    h_at_transitions: list[tuple[float, float]] = []  # (time_h, H)

    # Per-day gain swing (days 2+, trim-aware)
    day_1_end = 86400
    trim_steps = int(T_TRIM / dt)

    # Track min gain seen overall
    gain_min_overall = np.inf
    gain_max_overall = -np.inf

    for step in range(T):
        dx = mod.drift(x, theta, u)
        x = x + dx * dt
        Qm = _sigmoid(x[0, 1], Q_max_val, theta_sig_val, sigma_q_val)
        t_h = (step + 1) * dt / 3600.0
        is_wake = Qm > Q_THRESH

        g = Qm / mod.q_m_wake_mean_baseline
        if g < gain_min_overall:
            gain_min_overall = g
        if g > gain_max_overall:
            gain_max_overall = g

        if is_wake != was_wake:
            trans_type = "SLEEP->WAKE" if is_wake else "WAKE->SLEEP"
            transitions.append({
                "time_h": round(t_h, 2),
                "type": trans_type,
                "H": round(float(x[0, 2]), 2),
            })
            was_wake = is_wake

    # Extract sleep/wake durations from transitions (days 2+)
    sleep_durations = []
    wake_durations = []
    for i in range(len(transitions) - 1):
        dur = transitions[i + 1]["time_h"] - transitions[i]["time_h"]
        if transitions[i]["type"] == "WAKE->SLEEP":
            sleep_durations.append(round(dur, 2))
        else:
            wake_durations.append(round(dur, 2))

    return {
        "transitions": transitions,
        "n_transitions": len(transitions),
        "sleep_durations": sleep_durations,
        "wake_durations": wake_durations,
        "mean_sleep_h": round(float(np.mean(sleep_durations[1:])), 2)
        if len(sleep_durations) > 1 else 0.0,
        "mean_wake_h": round(float(np.mean(wake_durations[1:])), 2)
        if len(wake_durations) > 1 else 0.0,
        "gain_range": [round(float(gain_min_overall), 4),
                       round(float(gain_max_overall), 4)],
    }
