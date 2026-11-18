# (A) Pole layer and linear validation (RQ1, RQ4)

Plan: `docs/proposal_A_wavenumber_pole_operator.qmd`, Sections 4 and 7. Gate: dispersion relations recovered on
linear benchmarks (month 10).

**Layer.** `lfo.models.wavenumber.WavenumberPoleLayer1d` (837ca15, 09f19f0; tests in `tests/test_wavenumber.py`).
For each spatial Fourier mode k, the temporal kernel is a pole–residue sum with polynomial poles and residues:

  μ_n(k) = a0 + a1 k² + a2 k⁴ + i (c1 k + c3 k³),  β_n(k) = b0 + b1 k².

- **Time axis.** The exact first-order-hold recursion (the D1 decision): causal, with no periodicity in time. It is
  computed as a causal FFT convolution plus a first-sample correction.
- **Options.**
  - `modes`: keep the lowest wavenumbers only.
  - `re_max`: a smooth cap on Re μ.
  - `stable`: Re μ = −softplus.
- **Tests.**
  - The FOH step equals `lfo.data.lti`'s.
  - The convolution equals the recursion.
  - With the true diffusion poles, the layer gives the exact solution to 1e-10.
  - The layer is causal.
  - It transfers to a finer spatial grid.
  - Mode truncation and the growth cap behave as specified.

## First validation (2026-11-18): dispersion recovery, one layer

`a1_dispersion.py` → `results/a1_dispersion.md`. Predictions are in the script, committed before the runs
(cd2fa83). Prediction 1 was corrected before any training: the oracle floor rules out a 1e-3 test error.

**Data.**

- Periodic x in [0, 2π), n_x = 64, T = 5, n_t = 201, zero initial state.
- The forcing is 4 random spatial modes in k = 1..16, times E1-family time profiles.
- The truth is computed on a 32× finer time grid, so neither the data nor the layer shares a hold.

**Systems.**

- `diff`: λ = −0.1 k².
- `adr`: λ = −0.5 i k − 0.05 k² − 0.2.

**Training.** Adam, lr 1e-2 with cosine decay, 3000 full-batch steps; N = 1, 2; 5 seeds.

**First attempt: all 20 runs NaN.** The polynomial was evaluated up to k = 32, where a2 k⁴ made e^{μT} overflow.
Fixed by `modes = 17` (the forced band) and a smooth cap Re μ ≤ 0.5. The predictions were not changed.

**Second attempt.**

| system | N | oracle floor | test rel. L2 | dispersion error (dominant pole, k = 1..16) | recovered (N = 1) |
|---|---|---|---|---|---|
| diff | 1 | 8.3e-3 | 3 seeds **8.3e-3**; 2 seeds 0.37, 0.94 | 3 seeds **0.3–0.4 %**; 2 seeds > 40× | 3 seeds **D = 0.09998**, v ≈ 3e-6, k_r ≈ −1e-4 |
| diff | 2 | 8.3e-3 | 1.1e-2–1.3e-2 (4 seeds); 0.37 | 0.47 to 280 | – |
| adr | 1 | 8.7e-3 | 0.24–0.47 | 36 to 290 | D 0.013–0.25, v 0.26–0.57, k_r −0.005–0.27 |
| adr | 2 | 8.7e-3 | 1.2e-2–0.34 | 80 to 570 | – |

**Predictions against results.**

1. N = 1 recovers the dispersion to 1 % in every seed: **fails.**
   - On `diff`, 3 of 5 seeds recover D to 0.02 % at the oracle floor.
   - The other 2 stop with a positive k² coefficient (growth held by the cap) and a k⁴ term.
   - On `adr`, no seed fits (test 0.24–0.47): v is near 0.5 in some seeds, but spurious k³ and k⁴ terms remain.
2. N = 2 dominant pole within 5 % in ≥ 4 of 5 seeds: **fails.**
   - On `diff`, the fit nearly reaches the floor (1.1–1.3e-2) with a dispersion error ≥ 47 %. The two poles share
     the mode (Proposition 8's risk).

**Reading.**

- **The layer can represent the dispersion exactly.** The oracle and the 3 good diffusion seeds sit at the floor
  set by the sampling of steps and PRBS.
- **The polynomial fit is badly non-convex.**
  - Higher-order terms (k⁴, k³) let wrong basins fit the low k and grow elsewhere.
  - The oscillatory term v k (up to 8 rad/s here) adds phase-wrapping local minima, as in E1's frequency fits.
- **The failure is optimisation, not representation.** That is E1's lesson again, now in k.

**Next.**

- A two-stage identification:
  1. fit one pole per excited k (a small, well-conditioned problem per mode, with the E2 polish / restarts if
     needed);
  2. fit the polynomial μ(k) to those poles by least squares, which is linear for fixed degree;
  3. use the result to initialise or replace the joint fit.
- Alternatives: a curriculum in k (low k first) or multi-start; and degree selection (a k⁴ term without physics is
  the main source of wrong basins).
