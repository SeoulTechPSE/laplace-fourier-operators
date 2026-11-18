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

## Second pass (2026-11-18): trained layers, Fourier vs FOH time axis

`d1_train.py` → `results/d1_train.md`. Predictions are in the script, committed before the runs (059623d).

**Design.** A single tied pole–residue layer per path:

- `fourier`: the E1 tied layer;
- `foh`: the same layer with the exact sampled FOH response (`e2_train.FOHPoleResidue`).

The truth is computed on a 32× finer grid; E1 families and splits; `stable` init; `common` protocol + pole
lr × 10; slots = true poles; 10 seeds × p2 / p4 / p6, 60 runs.

Median over seeds of the per-family median test error:

| system | path | all | step | ramp | PRBS | smooth |
|---|---|---|---|---|---|---|
| p2 | fourier | 4.6e-3 | 3.2e-3 | 1.2e-3 | 1.1e-2 | 1.4e-3 |
| p2 | foh | 4.7e-3 | 3.3e-3 | **1.4e-4** | 1.2e-2 | 1.3e-3 |
| p4 | fourier | 4.2e-2 | 3.2e-2 | 1.1e-2 | 8.3e-2 | 4.5e-2 |
| p4 | foh | 4.2e-2 | 3.2e-2 | 1.0e-2 | 8.4e-2 | 4.5e-2 |
| p6 | fourier | 1.6e-2 | 1.1e-2 | 3.3e-3 | 3.4e-2 | 1.5e-2 |
| p6 | foh | 1.7e-2 | 1.2e-2 | 3.4e-3 | 3.7e-2 | 1.6e-2 |

**Predictions against results.**

1. FOH at most half of Fourier on ramp and smooth samples, in all three systems: **fails**. The gain appears only
   on p2 ramps (8×). The other half of the prediction (step and PRBS within 30 %) holds.
2. Complex-pole recovery within 1 seed in 10: **holds**. The two paths recover the same poles in the same number of
   seeds (p2 9/10; p4 3/10; p6 5/10 and 8/10).
3. FOH recovers each real pole at least as often: **holds, trivially**. Both paths recover p4's real poles in 0
   seeds, and p6's −1 in 1 seed and −2 in 0.

**Reading.**

- Under the published protocol the training error (3e-3 to 4e-2) is 10–1000× the representation error of either
  path (1e-6 to 2e-3, first pass). The time axis cannot matter until training reaches the representation floor.
  It does reach that floor on p2, the one system where E1's protocol fits well, and there FOH gains 8× on ramps.
- This matches (C): with a recipe that reaches the floor (E2: Adam → polish → restarts), the FOH layer lowered the
  test error 7–225× and recovered every p6 pole without noise (E4.7, multisine and E1 data).
- The D1 choice of time axis should therefore be judged with a training recipe that reaches the representation
  floor, not with the published protocol.

## Third pass (2026-11-18): Fourier vs FOH with the E2 recipe, fine-grid truth

Pre-registered in `results/d1_recipe_prereg.md` (d2e6d9d).

**Design.** `e2_train.py` conditions `p6_e1fine_T20` (band / Fourier layer) and `p6_e1fine_T20_foh` (FOH layer):
E1 families with inputs and truth on a 32× finer grid (`e2_richness.make`, family `e1fine`). The E2 recipe is
`damp8_ms` → `e2_polish.py` → `e2_rescue.py`. Seeds 10–17. Results are in `experiments/C/theory/results/e2_rescue.md`.

| condition | runs | train / oracle | test rel. L2 | recovered: −0.5±1.5i, −0.3±4i, −1, −2 |
|---|---|---|---|---|
| E4, sample truth, band (`p6_e1_T20`) | 20 | 1.01 | 6.7e-4 | 100 %, 100 %, 95 %, 70 % |
| E4, sample truth, FOH (`p6_e1_T20_foh`) | 20 | 21 | 4.7e-6 | 100 %, 100 %, 95 %, 80 % |
| **fine truth, band** | 8 | 0.93 | **3.7e-3** | 100 %, 100 %, 100 %, **75 %** |
| **fine truth, FOH** | 8 | 0.93 | **3.4e-3** | 100 %, 100 %, 100 %, **50 %** |

Oracle floors (true poles): band 3.7e-3, FOH 3.4e-3.

**Predictions against results.**

1. FOH test error within 20 % of band: **holds**. 3.4e-3 vs 3.7e-3: both runs reach their oracle floor, which is
   the sampling of steps and PRBS.
2. Recovery of −2 at most 1 run in 8 apart, and below the sample-truth rates for both: **fails**. The band layer
   recovers it in 6 of 8 runs, the FOH layer in 4 of 8. The band layer's 75 % is not below its 70 %.
3. Complex pairs recovered in every run by both: **holds**.

**Reading.**

- **The FOH layer's large advantage in (C) E4 on E1 data was a model-class match with the data's first-order
  hold.** With a truth that neither layer matches, the two layers have the same test error, and the FOH layer is
  not better at recovering the fast real pole (4 of 8 vs 6 of 8; too few runs to call the band layer better).
- **E1's finding 3 survives the fine-grid truth.** With the E2 recipe, −1 is recovered in every run and −2 in
  50–75 %, by either layer. So the fast real poles are identifiable from the sampled data, not only from data in
  the model class. (The paper's Section 3.3 states this for the sample truth; this pass supports it more
  generally.)
- **For the D1 hand-off,** pole recovery on sampled data does not decide the time axis. What does decide it:
  - the representation error on smooth inputs and ramps (first pass: FOH and Chebyshev 100× below Fourier);
  - causality;
  - O(n) cost.

  The FOH recursion still wins on the last two, and on ramps. Chebyshev wins on smooth inputs.

**Next.**

- Piecewise Chebyshev with a stable recursion, and its cost.
- The S4D view: the FOH recursion is a diagonal SSM discretisation.
- A qualifying note in (C) E4.7 / E4.8: the FOH layer's gains there assume data with a first-order hold.
