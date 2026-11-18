# (D1) Non-periodic input representation

Plan: `docs/proposal_D_nonperiodic_initial_conditions.qmd`, Section 4 and RQ1. Hand-off to (A): the choice of
time-axis computation (integrated schedule, month 6).

## First pass (2026-11-18): oracle comparison of input representations, true poles

`d1_oracle.py` → `results/d1_oracle.md`. Predictions are in the script docstring, committed before the run
(10d4cb9).

**Design.** No training. E1's systems p2, p4, p6 with their true poles and residues; T = 20, n = 1024. Inputs:
100 per family:

- step, ramp, PRBS and smooth random with mismatched endpoints (E1 ranges);
- a periodic control.

Each representation computes the response from the input samples, so the whole error is the representation's.

**Paths.**

- `fourier`: the LNO v1 layer in the textbook wiring, i.e. the response to the trigonometric interpolant. Checked
  against the float32 layer itself to 2.6e-7.
- `fourier_pad2`, `fourier_pad4`: the same, with zero padding.
- `zoh`, `foh`: exponential-integrator recursion with a zero-order or first-order hold.
- `cheb16`, `cheb64`, `cheb256`: least-squares Chebyshev projection of the samples, integrated exactly.

**Truths.**

- *Sample truth:* the exact response to the piecewise-linear interpolant of the same samples. This favours FOH,
  which is exact for that interpolant.
- *Fine truth:* the inputs are defined on a 32× finer grid and the paths see only the n samples. This is the fair
  comparison.

Fine truth, median relative L2 error, p2 / p4 / p6:

| family | fourier | fourier_pad2 | zoh | foh | cheb64 |
|---|---|---|---|---|---|
| step | 1.7–2.8e-3 | 1.6–2.7e-3 | 3.2–5.4e-3 | 1.6–2.7e-3 | 7.3e-3–1.1e-2 |
| ramp | 0.7–1.8e-3 | 1.8–2.9e-4 | 1.1–2.3e-3 | **1.2–2.8e-6** | 1.0–1.6e-4 |
| PRBS | 0.9–1.0e-2 | 0.9–1.0e-2 | 1.8–1.9e-2 | 0.9–1.0e-2 | 2.9–4.5e-2 |
| smooth, mismatched ends | 4.1–6.7e-4 | 3.0–4.9e-4 | 2.6–3.1e-3 | 5.9–7.5e-6 | **1.7–2.2e-8** |
| periodic (control) | 2.9–4.5e-6 | 2.9–4.4e-4 | 4.7–7.6e-3 | 1.8–4.0e-5 | **0.5–1.2e-7** |

**Predictions against results.**

1. **Partly holds.** The Fourier path's error on the four mismatched-endpoint families is 10–100× that on the
   periodic control. But it is not concentrated at the endpoints: the last-5 % endpoint error is about the same as,
   or lower than, the whole-window error. That part of the prediction **fails**. The error of the periodic
   interpolant is spread over the window.
2. **Holds.** The Fourier error falls as n^(−1.03 … −1.33) from n = 256 to 4096; the periodic control falls as
   n^(−2.07).
3. **Holds.** FOH reproduces the sample truth to 1e-16, and the ZOH error falls as n^(−1.0).
4. **Holds.** On smooth inputs `cheb64` is far better than `fourier` (4e-6 against 4e-4 on the sample truth). It
   is worse on step (7e-3–1e-2 against 4–7e-4) and PRBS (3–5e-2 against 1e-3), where the discontinuities lie
   inside the window.

**Further findings.**

- **A global Chebyshev fit of uniform samples is numerically unusable at high degree.** `cheb256` gives errors up
  to 1e7, which is the instability the plan's risk table anticipates.
- **Zero padding helps ramps (5×) and hurts the periodic control (100×).** On steps and PRBS it does nothing,
  because the sampling of jumps between grid points limits every path there.
- **On step and PRBS, every path is limited by the sampling.** A jump between samples cannot be located from the
  samples, so FOH, Fourier and padding all reach 2e-3 (step) and 1e-2 (PRBS).

**Reading for RQ1 and the hand-off to (A).**

- The Fourier path costs 1e-3–1e-4 relative error on non-periodic inputs, from periodicity alone. That is 100×
  more than FOH or Chebyshev on smooth inputs and ramps.
- The FOH exponential-integrator recursion is never worse than Fourier except on periodic inputs. It is O(n),
  causal by construction, and resolution-invariant in μ and β.
- A global Chebyshev path is best on smooth inputs but needs pieces at discontinuities and a stable evaluation.
- Proposed hand-off (a decision for the user, since the plan's default is Chebyshev): **FOH exponential-integrator
  recursion as the default time-axis computation, with piecewise Chebyshev as an option for smooth inputs.**
- This agrees with (C):
  - the FOH refit identifies p6's fast real poles (E4);
  - the textbook wiring makes the layer causal (E5, Proposition 7).

**Next.**

- Trained comparison: a single pole–residue layer per path on E1-style data, with pole recovery and test error.
- Piecewise Chebyshev with a stable recursion, and its cost.
- The S4D view: the FOH recursion is a diagonal SSM discretisation.
