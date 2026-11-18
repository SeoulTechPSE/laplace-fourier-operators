# (A) Deep wavenumber pole operator

Plan: `docs/proposal_A_wavenumber_pole_operator.qmd`, Section 4.4 (deep model, post-hoc linear probe), RQ2. Shared
code: `lfo.models.WavenumberPoleOperator1d` (lift → [pole conv + pointwise] × L → projection; poles shared by all
channel pairs of a layer), `lfo.data.burgers` (forced viscous Burgers, pseudo-spectral IF-RK4). The linear recipe it
builds on is in `experiments/A/layer`.

## First deep validation (2026-11-18): forced Burgers, WPO against FNO

`a9_burgers.py` → `results/a9_burgers.md` (predictions eac2be5). ν = 0.1, n_x = 64, T = 5. The forcing scale is
log-uniform over [0.03, 1], so the data mix near-linear samples with strongly nonlinear ones (up to 91 % of u is not
the linear response). Both models have about 59 k parameters, use Adam lr 1e-3 with cosine decay, and train for
4000 steps; 3 seeds each.

| model | test (T = 5) | T = 10 grid: (0, 5] / (5, 10] | probe ν̂ (0.1) | probe dispersion error (kept k) |
|---|---|---|---|---|
| WPO | 0.091–0.137 (mean 0.117) | 0.066–0.106 / **3.2–8.5** | 0.032–0.081 | 0.96–1.01 (8–12 modes) |
| FNO | 0.103–0.107 (mean 0.105) | 0.40 / 0.30 | 0.104–0.109 | 0.38–0.42 (k = 3–6) |

Method check (the solver itself, measured before training): ν̂ = 0.10046, dispersion error 1.8 % (k = 2–9).

**Predictions against results.**

1. WPO mean test error ≤ 1.5× the FNO's: **holds** (1.11×), but only because both are poor (about 10 %).
2. WPO extrapolation error over (5, 10] ≤ 2× its test error, and the FNO's > 2×: **fails.** The WPO's error grows
   from 0.1 to 3–8. The FNO's grows 2.9×; it is also wrong on (0, 5] of the longer grid (0.40), because its
   t modes rescale with the window.
3. WPO probe ν̂ within 5 % and dispersion error ≤ 10 % in ≥ 2 of 3 seeds, and the FNO's probe worse: **fails.**
   The WPO's probe is the worse one.

**Diagnosis.**

- **Both models underfit.** Train and test errors are similar (WPO 0.10 / 0.14, FNO 0.08 / 0.11). Errors are
  largest in the lowest-amplitude quartile (WPO 0.15, FNO 0.18), where the samples are near-linear.
- **The WPO's poles barely moved from their initialisation.** In all four layers, Re μ_n(k) stays within ±0.5 and
  is almost flat in k, and some poles have Re μ > 0 (up to 0.21). The truth needs a k² coefficient of
  −ν k_ref² = −25.6 in k / k_ref units. With Adam at lr 1e-3, a parameter moves by at most about 4 in 4000 steps,
  so it cannot get there. The positive poles grow as e^{0.2 t}, which explains the extrapolation blow-up.
- This is the deep-model form of what (C) E2 and E5 found: the poles need their own, larger learning rate (the
  `plr10` variants there). The linear validations hid it, because L-BFGS rescales its steps.

**Next.** Pre-register a pole learning rate (×30) for the WPO, with and without the stable parameterisation, at the
same budget, then decide on longer training for both models.
