# Methodology: Rules and Development Protocol

This document describes the methodological rules and development protocol governing
the CPsych platform. Rules 1--14 were established during a prior diagnostic project
(BAA) and carried forward. Rules 15--22 address risks specific to platform development.

## Core Philosophy

A sandbox that can produce anything explains nothing. If every parameter combination
yields a plausible result, what has been built is a plausibility generator, not a model.
Value comes from the opposite: what the model **refuses** to produce. From the very first
module, the platform must show not only "here is what happened" but also
"here is what cannot happen."

A second risk is specific to the development phase. The diagnostic phase guarded against
post-hoc fitting to already-obtained results. Development iterates over design variants,
and its failure mode is **rationalising what was found**. Rules 15--19 address this.

## Twenty-Two Rules

### Inherited from the diagnostic project (1--14)

1. **No number in reports or articles is typed by hand.** The sole source is
   `results/*.json`. If a number is not in results, it does not appear in text.

2. **No metric is used without a test.** Every metric is verified against an external
   reference or an analytical limit.

3. **Discrepancy between text and code is resolved by checking code, not editing text.**

4. **No placeholders in data.** No stubs that end up in a figure or table.
   Missing data causes an explicit error.

5. **No hidden branches.** Every behaviour that affects a result is documented in the
   module specification or removed.

6. **All randomness via `np.random.default_rng(seed)`.** Seeds are recorded in manifests.
   Calibration and evaluation sets do not overlap.

7. **Negative results are reported as-is.** If a module fails to assemble, if a coupling
   breaks stability, if a parameter is not recoverable -- this is written directly.

8. **Escalation to the author** on any decision that changes scientific content
   (as opposed to implementation).

9. **Input materials are read-only.** `research/` is not edited.

10. **References are not fabricated.** Every equation is checked against the primary
    source by DOI. Equations taken from secondary reviews are marked as unverified
    against the original.

11. **Atomic commits** with task identifiers in the message.

12. **No new functionality until the current phase gate is closed.**

13. **The seed-averaged vector of channel-indexed quantities is not reported.**
    The system is permutation-symmetric; such an average is flat regardless of
    effect strength. Acceptable: within-run sorting with rank-averaging;
    permutation-invariant summaries; full seed distributions.

14. **Any conclusion that depends on a free parameter is reported as a function of
    that parameter,** not as a verdict at one value. The parameter selection criterion
    is stated before computation.

### Development-specific (15--22)

15. **Success criteria are written before implementation.** In `specs/<module>.md`:
    which observables, target values, tolerances, and what constitutes failure.
    Implementation begins only after the specification is committed.

16. **Alternatives log.** Every tried variant of equations, couplings, or
    parameterisations is recorded in `reports/alternatives.md` -- including rejected
    ones and especially rejected ones. Entries are made at the time of trial,
    not retroactively.

17. **Deferred observables.** A subset of observables from the specification is marked
    held-out and is never used during module tuning. Evaluation against them is
    one-shot, at the end of the phase. If a held-out observable fails, tuning
    does not resume: that is the result.

18. **Three classes of parameters.** A parameter enters the sandbox interface after
    parameter recovery (generate data with known theta, estimate theta-hat,
    correlation and confusion matrix).
    - **Fitted:** r >= 0.7. Suitable for direct investigation and inverse inference.
    - **Exploratory:** recovery performed, r < 0.7. Displayed in the sandbox but
      explicitly marked: they can be manipulated to observe consequences, but cannot
      be inferred from observed behaviour. Not used in the inverse problem. Visually
      distinct from fitted. Recovery result is final.
    - **Untested:** recovery not performed. Status unknown. Further investigation
      is possible (unlike exploratory, where recovery was performed and yielded
      r < 0.7).

    **18a. Corollary for pharmacology.** A drug that modifies an exploratory or
    untested parameter produces effects that cannot be attributed back to that
    parameter. Targets in the exploratory/untested class are suitable for
    simulation but not for fitting a drug model to clinical data.

19. **No composition without stability verification.** Before coupling any two
    modules: linearise the composite, compute the spectrum, find the critical
    coupling strength. The coupling knob range is bounded below the threshold.

    **19a. Hybrid (switching) systems.** If the composite contains a discrete gate
    (e.g., is_wake), the system is piecewise-smooth. Linearisation is defined
    within a regime but does not characterise stability at the switching boundary.
    Stability analysis is performed: (a) separately for each regime;
    (b) numerically at the switching boundary -- run through the wake-sleep
    transition checking for diverging responses, latch-up, and parasitic
    oscillations near the threshold.

20. **Pure functions for portability.** `drift(x, theta, u)` and `diffusion(x, theta)`
    are written as pure functions without in-place mutation or side effects; all
    randomness arrives via an explicit generator argument; state is not stored in
    the object between calls.

21. **Tuning rules must determine parameter values uniquely.** If multiple targets
    admit a range rather than a point, this is declared in the specification
    before implementation as an extra degree of freedom.

22. **No caching of intermediate results without invalidation.** Silent loading from
    disk (`if path.exists(): np.load(...)`) is a built-in desynchronisation
    mechanism: the file outlives code changes, and the result becomes a mixture
    of old simulation and new analysis. Acceptable: (a) full recomputation on every
    run; (b) cache with hash verification of the generating code. This rule was
    introduced after D53: stale `.npy` checkpoints in arousal recovery produced
    non-reproducible r-values undetected by tests, `verify_numbers`, or red-team.

## Development Gates

| Gate | Condition |
|------|-----------|
| **P0** | Kernel decision (custom vs. Neuroblox) made, justified, recorded in decisions.md |
| **P1** | Module contract finalised; integrator core works; BAA regression suite ported and green |
| **P2** | Three starter modules implemented; each passed standalone validation, parameter recovery, and held-out |
| **P3** | All three pairwise couplings assembled; critical coupling strengths measured; knob ranges bounded |
| **P4** | Triple composite stable; impossible map obtained; BAA connected as fourth module |
| **P5** | Sandbox works: knobs, observables, stability-boundary warnings |

All six gates have been passed. See `reports/progress.md` for the full timeline
and `reports/decisions.md` for the 53 numbered decisions.

## Module Contract

Each module is an object with explicitly declared:
- `state` -- state variables, names, and dimensions
- `params` -- parameters with ranges, units, and exposed/internal designation
- `inputs` -- what it accepts from other modules
- `outputs` -- what it provides (state variables or derived quantities)
- `timescale` -- characteristic relaxation time in model units
- `drift(x, theta, u)` and `diffusion(x, theta)` -- right-hand sides
- `validation` -- reference to specification and standalone verification results

Coupling is a separate object: `theta_i <- h(x_j)`, i.e. one module's output
modulates another module's **parameter**. Direct mixing of state variables is
forbidden without separate justification: the literature couples through parameter
modulation, and such coupling can be bounded for stability.

## Technical Conventions

- Python >= 3.11. Integration: correct Euler-Maruyama (noise proportional to sqrt(dt))
  or higher-order scheme; integration step and observation frequency are **distinct**.
- Timescale separation: if module tau values differ by more than two orders of magnitude,
  split integration or slow-variable averaging is used. Checked at every composition.
- Time units are dimensionless unless explicitly stated otherwise. The word "seconds"
  is not used without grounding in real data.
- Code language: English. Reports in `reports/`: Russian.
