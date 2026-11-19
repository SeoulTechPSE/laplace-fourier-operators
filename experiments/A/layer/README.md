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

## Second validation (2026-11-18): two-stage identification

`a2_twostage.py` → `results/a2_twostage.md` (predictions 13a8593).

**Stages.**

1. One pole per excited k (prior-free grid, then Nelder–Mead; the residue by least squares).
2. Polynomial least squares, with the degree chosen by BIC.
3. An Adam fine-tune of the layer from the stage-2 polynomial.

Data seeds 0–4; `diff` and `adr`.

**Predictions against results: all three fail.**

1. **Per-mode poles within 1 %.** Mid-k poles are within 0.5 %. Two groups miss:
   - high k (≥ 12, where |λ| Δt ≳ 0.4) is off by 1–4 %, because jumps between samples bias fast modes;
   - the slowest diffusion mode (k = 1) is off by 1–2 %.
2. **BIC picks the true degree.** It does so in only 2–3 of 5 seeds; it adds k⁴ / k³ terms to fit the biased
   high-k poles. Where the degree is right, D and v are within 1 %.
3. **The Adam fine-tune reaches the oracle.** It made several fits worse. With k up to 16, one Adam step on a k⁴
   coefficient moves Re μ by up to 65 lr: a scaling problem, which also explains part of the first validation's
   failure.

## Third validation (2026-11-18): resolution cut, scaled polynomials, L-BFGS fine-tune

`a3_twostage.py` → `results/a3_twostage.md` (predictions 885d64d).

**Changes,** diagnosed from the second validation:

- Stage 2 uses only modes with |μ̂_k| Δt ≤ 0.25 (from stage 1, no truth), weighted by 1 / residual².
- The layer evaluates its polynomials in k / k_ref (new option `k_ref = 16`).
- The fine-tune uses L-BFGS (200 iterations) instead of Adam.

| system | stage 1, kept modes: max error | stage 2: D / v / k_r | tuned test (oracle) | dispersion error | tuned D, v, k_r |
|---|---|---|---|---|---|
| diff (5 seeds) | 1.3–2.4 % (k = 1..9 or 1..10) | 0.1001–0.1015 / ≈ 0 / −0.007…−0.001 | 8.1e-3–9.1e-3 (8.3e-3) | 1.9e-6 – 5.0e-3 | 0.09999–0.10002, 0.00000, −0.00015…−0.00001 |
| adr (5 seeds) | 0.9–1.7 % (k = 1..12) | 0.0494–0.0503 / 0.4983–0.5012 / 0.195–0.205 | 8.7e-3–9.3e-3 (8.7e-3) | 1.3e-4 – 2.7e-3 | 0.04999–0.05000, 0.49996–0.50000, 0.19979–0.20002 |

**Predictions against results.**

1. Kept-mode poles within 2 %: **fails narrowly.** It holds in 9 of 10 runs; `diff` seed 3 is at 2.4 %, its k = 1
   mode.
2. BIC picks the true degree in ≥ 4 of 5 seeds: **fails.**
   - It picks the true degree in 1 of 5 seeds on `diff` and 3 of 5 on `adr`.
   - D and v from stage 2 are within 1 % except `diff` seed 4 (1.5 %). k_r is within 0.01 everywhere.
3. Fine-tuned test error ≤ 1.2× oracle in ≥ 4 of 5 seeds, and dispersion error ≤ 2 %: **holds in all 10 runs.**
   The tuned test error is 0.98–1.10× the oracle, and the dispersion error is 2e-6 to 5e-3.

**Reading.**

- **With this recipe the layer recovers the dispersion relation in every run.** After the L-BFGS fine-tune, D, v
  and k_r are within 0.08 % of the truth on both systems and all seeds, and the test error sits at the oracle
  floor. This is RQ1's first positive result on two 1D linear PDEs.
- **The spurious higher-order terms vanish in the fine-tune.** Stage 2's degree choice is unreliable, but the
  joint fit started near the truth drives the extra k⁴ / k³ coefficients to negligible values. So degree selection
  matters less than a good start and a well-scaled, second-order optimiser.
- **What made the difference.** Scaling (`k_ref`), a second-order optimiser, and an initialisation from per-mode
  identification restricted to well-resolved modes. Together they are the (A) analogue of the (C) E2 recipe.
- **Limits.**
  - Two 1D systems, with first-order dynamics per mode (one pole per k).
  - Stage 1 uses the explicit mode structure of a linear PDE.
  - The oracle floor (sampling of steps and PRBS) caps the test error.

**Next.**

- Second-order-in-time systems: the beam (±i c k², two poles per k), with N = 2.
- 2D diffusion.
- The joint fit with this recipe but without stage 1 (multi-start + L-BFGS + k_ref), to see whether stage 1 is
  needed.
- Then deeper models (lifting / nonlinearity) with the linear probe of (C) E5.

## Fourth validation (2026-11-18): the Euler–Bernoulli beam, two poles per wavenumber

`a4_beam.py` → `results/a4_beam.md` (predictions 531128c).

**Setup.**

- u_tt = −c² u_xxxx + f with c = 0.04. The output is the velocity: poles ±i c k² on the imaginary axis, residues
  1/2. The displacement would need residues 1/(2 i c k²), which polynomial residues cannot express.
- Layer change: Im μ may use any powers of k (`im_powers`, 2714262), because only k ≥ 0 is evaluated. The odd-only
  default could not express c k².
- The a3 recipe, extended to two poles per mode: a greedy two-pole grid, then Nelder–Mead on 4 parameters, labels
  by the sign of Im.

| seed | stage 1 error: k = 1 / 2 / 3 / k ≥ 4 | c: stage 2 → tuned | test: init → tuned (oracle) | dispersion error |
|---|---|---|---|---|
| 0 | 54 / 0.68 / 1.08 / ≤ 0.01 | −0.021 → −0.0099 | 28 → 0.35 (1.2e-2) | 3.2 |
| 1 | 60 / 0.26 / 0.03 / ≤ 0.005 | 0.0415 → **0.04000** | 0.60 → **1.18e-2** (1.18e-2) | 1.6e-4 |
| 2 | 70 / 0.17 / 0.005 / ≤ 0.005 | −0.166 → **0.04000** | 0.72 → **1.21e-2** (1.21e-2) | 3.9e-5 |
| 3 | 1.7 / 0.30 / 0.05 / ≤ 0.01 | 0.0403 → −0.038 | 5.1 → 0.37 (1.2e-2) | 2.1 |
| 4 | 61 / 0.81 / 0.03 / ≤ 0.01 | 0.0460 → **0.04002** | 0.89 → **1.14e-2** (1.14e-2) | 3.6e-4 |

**Predictions against results.**

1. Stage 1, both poles within 2 % for k ≥ 3: **fails.** k = 3 is off by 3–108 % in 4 of 5 seeds. k ≥ 4 is within
   1 % everywhere.
2. Fine-tuned c within 1 %, |Re μ| ≤ 0.01 |Im μ|, and test at the oracle in ≥ 4 of 5 seeds: **fails, 3 of 5.** In
   those three, c = 0.04000–0.04002, Re/Im ≤ 2.3e-4, and the test error equals the oracle floor. The other two end
   in wrong basins (Re/Im 4–44).

**Diagnosis.**

- **The low modes are not observable within the window.** Their phase advances over T = 5 by c k² T = 0.2, 0.8
  and 1.8 rad for k = 1, 2, 3: less than a period. Wrong poles fit them about as well: the stage-1 residuals there
  are as small as at well-resolved k (≈ 0.005).
- Because stage 2 weights modes by 1 / residual², these wrong poles enter with full weight and corrupt the
  polynomial. The bad starts then lead the fine-tune into wrong basins.
- The resolution cut of a3 (|μ̂| Δt ≤ 0.25) handles fast modes; this is the opposite end, slow modes.

**Next.**

- An observability cut from stage 1, without the truth: keep a mode only if its estimated pole is resolved within
  the window, max(|Re μ̂|, |Im μ̂|) T ≥ 2 (for the beam this drops k = 1–3). Alternatively, weight stage 2 by
  the per-mode curvature (Fisher information) instead of the residual.
- Then rerun the beam and the a3 systems with this cut, pre-registered.


## Fifth validation (2026-11-18): an observability cut in stage 2

`a5_observability.py` → `results/a5_observability.md` (predictions 377ad3b).

**Change.** Before the resolution cut, stage 2 keeps a mode only if every stage-1 pole of it satisfies
max(|Re μ̂|, |Im μ̂|) T ≥ 2, computed from the estimates alone. Below that, e^{μt} ≈ 1 + μt over the window, so the
pole and the residue trade off. Stage 1 is reused from a3 / a4 (deterministic), so the comparison isolates the cut.
`a4_beam.py` was refactored to expose `fit_beam` (stages 2–3); it reproduces a4 to the last digit.

**What the cut removed.** Beam: k = 1–3 in every seed. diff: k = 1, and k = 2 in seeds 0 and 3 (it sits on the
threshold, 1.99–2.01). adr: nothing.

| system | tuned test / oracle (5 seeds) | constants | change against a3 / a4 |
|---|---|---|---|
| beam | 1.00–1.02 | c = 0.03986–0.04002, Re/Im ≤ 2.3e-4, dispersion error ≤ 3.6e-4 | seeds 0 and 3 fixed (were 0.35–0.37 test, c wrong); init test 1.2–1.4e-2 (was 0.6–28) |
| diff | 0.98–1.10 | D = 0.09999–0.10002, k_r ≥ −1.5e-4 | none (same to 3 digits) |
| adr | 0.99–1.06 | D, v, k_r within 0.05 % | none (identical: no mode removed) |

**Predictions against results.**

1. Beam, c within 1 %, test ≤ 1.2× oracle and Re/Im ≤ 0.01 in ≥ 4 of 5 seeds: **holds, 5 of 5.** Worst c is seed 3,
   −0.35 %.
2. diff and adr, no loss against a3 (test ≤ 1.2× oracle, dispersion error ≤ 2 %, constants within 1 %, in all 10):
   **holds.**

**Reading.**

- The beam failures were caused by the stage-2 initial point alone. With the slow modes left out, the initial point
  is already at the oracle floor and L-BFGS only polishes it.
- The two cuts bracket the usable band from stage 1: |μ̂| Δt ≤ 0.25 (fast end, sampling) and |μ̂| T ≥ 2 (slow end,
  window). Both use only the estimates. The polynomial then extrapolates the dispersion relation outside the band;
  for the beam it does so to 1e-4 at k = 1–3, where the data alone could not fix the poles.
- The recipe now recovers all three 1D systems in every seed: per-mode identification, two cuts, weighted
  polynomials in k / k_ref, L-BFGS.

**Next.**

- 2D diffusion (the layer is 1D; a 2D version needs |k|² over a 2D wavenumber grid).
- The joint fit without stage 1 (multi-start + L-BFGS + k_ref), to see whether stage 1 is needed.
- Then deeper models (lifting / nonlinearity) with the linear probe of (C) E5.

## Sixth validation (2026-11-18): 2D diffusion and anisotropic 2D ADR

`a6_diffusion2d.py` → `results/a6_diffusion2d.md` (predictions 3f62df5). Layer: `WavenumberPoleLayer2d` (a4efb2b).

**Layer.** Re μ is a polynomial in |k|² (isotropic); Im μ = c · k (advection) plus optional isotropic powers |k|^p;
β is a polynomial in |k|². A real field needs μ(−k) = conj μ(k). The even imaginary terms (Im β, |k|^p in Im μ)
break this on the row k_y = 0, the only place where the rfft2 layout holds negative k_x. They are therefore
multiplied by a half-plane sign, which is the 1D convention (evaluate on k ≥ 0, extend by conjugation). The tests
caught this: the 2D layer did not reduce to the 1D layer for y-independent fields until the sign was added.

**Setup.** 32 × 32 grid, T = 5. Each sample is forced at 6 wavevectors from the half plane of |k_x|, k_y ≤ 8 (144
wavevectors), so each wavevector is seen by only 1–17 training samples (1D: about 50). diff2d: D = 0.1. adr2d:
D = 0.05, v = (0.4, −0.3), k_r = 0.2. The recipe is the 1D one (a3 + a5): per-wavevector poles, both cuts, weighted
polynomials, then L-BFGS with k_ref = 8.

| system | modes kept | stage 1 (kept): median / max error | tuned test / oracle | dispersion error | tuned D; v; k_r |
|---|---|---|---|---|---|
| diff2d | 129–133 / 144 | 0.7–1.0 % / 4.7–9.1 % | 1.000–1.002 | 5e-5 – 1.3e-3 | D = 0.09998–0.10003; v = 0; \|k_r\| ≤ 3.4e-4 |
| adr2d | 141–142 / 144 | 0.6–0.7 % / 3.1–5.3 % | 1.000–1.002 | 8e-5 – 3.2e-4 | D = 0.04999–0.05003; v within 0.03 %; k_r = 0.19974–0.20026 |

**Predictions against results.**

1. Every kept stage-1 pole within 2 %: **fails.** 4–28 kept modes per run are 2–9 % off. They are high-|k| modes
   (|μ| Δt ≈ 0.1–0.2) and modes seen by few samples (as few as 1–3). The 2 % bound came from 1D, where each mode
   had about 50 samples.
2. BIC drops |k|⁴ (5 of 5 in both systems), stage-2 D and v within 2 %, k_r within 0.01: **holds.** Stage 2
   averages over 130–140 modes, so the per-mode errors cancel.
3. Every seed at ≤ 1.2× the oracle, dispersion error ≤ 1 %, D and v within 1 %, k_r within 0.002: **holds**, in all
   10 runs (worst ratio 1.002, worst dispersion error 0.13 %).

**Reading.**

- The recipe carries over to 2D unchanged, including an anisotropic advection vector.
- Per-mode accuracy matters less in 2D, because there are many more modes per polynomial coefficient. The data per
  mode can be thin as long as the polynomial is shared.
- Limit: the layer's Re μ is isotropic by construction. Anisotropic diffusion would need a quadratic form
  k^T A k in place of D |k|².

**Next.**

- The joint fit without stage 1 (multi-start + L-BFGS + k_ref), in 1D and 2D, to see whether stage 1 is needed.
- Deeper models (lifting / nonlinearity) with the linear probe of (C) E5.

## Seventh validation (2026-11-18): is stage 1 needed? A joint fit from random starts

`a7_joint.py` → `results/a7_joint.md` (predictions 12e576a).

**Setup.** Same data, layers, k_ref, Re cap and L-BFGS as a3–a6, but no stages 1 and 2. All μ coefficients start at
random (N(0, 3²) in k / k_ref units). 16 starts per data seed in 1D and 8 in 2D. The start with the lowest
validation error is selected.

| system | starts at the oracle (per seed) | selected start: test / oracle | selected: constants | time per seed (all starts) |
|---|---|---|---|---|
| diff | 11 / 16 (every seed) | 1.000–1.001 (5 / 5) | D = 0.09999–0.10001 | 16–18 s |
| adr | 11 / 16 | 1.000–1.002 (5 / 5) | D, v, k_r within 0.05 % | 15–19 s |
| beam | 7–9 / 16 | 1.002–1.012 (5 / 5) | c = 0.03986–0.04002 | 39–43 s |
| diff2d | 5 / 8 | 1.000–1.002 (5 / 5) | D within 0.03 % | 26–28 s |
| adr2d | 5 / 8 | 1.000–1.001 (5 / 5) | D, v within 0.05 %, k_r within 0.0003 | 25–27 s |

**Predictions against results.**

1. diff, diff2d: at least half the starts succeed and the selected start succeeds in 5 of 5: **holds** (69 %, 63 %).
2. adr, adr2d: the selected start succeeds in ≥ 4 of 5: **holds** (5 of 5). The per-start success rate is lower
   than for diffusion: **fails**, it is the same. The phase v k did not add local minima here.
3. beam: at most a quarter of the starts succeed and the selected start in at most 2 of 5: **fails.** 44–56 % of
   the starts succeed and the selected start does so in 5 of 5. Stage 1 is not needed for the beam either.

**Why starts fail.**

- In diff, adr, diff2d and adr2d, the failing starts are the same indices in every data seed (1D: 2, 6, 7, 8, 11;
  2D: 2, 6, 7). The failure depends on the start, not on the data.
- A failed start ends with Re μ = 0.5 at every k, the value of the soft cap. Its residues shrink to about 0.05,
  and the training error stalls at 0.87. Above the cap, the soft minimum has almost zero gradient with respect to
  the polynomial, so the optimiser cannot bring Re μ back down. The cap was added in a1 to stop overflow, and it
  creates this trap.
- The beam's failing starts end at test errors 10–37× the oracle, like a4's wrong basins (validation 0.11–0.5).
  Their mechanism was not analysed.
- Validation error separates failed from successful starts by two orders of magnitude (0.9 against 0.01), so
  selection is never in doubt.

**Reading.**

- With k_ref scaling and L-BFGS, a joint fit from random starts plus selection by validation recovers every linear
  system, in every seed. Stage 1 is not needed. It did not fail in a1 because joint fitting is hard, but because of
  Adam and unscaled coefficients.
- Cost is comparable: 15–43 s per data seed for all starts, against 6–75 s for the two-stage recipe.
- Stage 1 remains useful as a diagnostic, because it reads the dispersion relation off the data mode by mode
  without a model class. It also gives a single deterministic start.
- What matters for deep models: the pole parameters can be trained end to end, provided they are scaled and the
  Re cap does not trap them.

**Next.**

- Remove the cap trap: start with Re μ ≤ 0, or use a cap with a nonzero gradient above it. Then rerun the trapped
  starts.
- Deeper models (lifting / nonlinearity) with the linear probe of (C) E5.

## Eighth validation (2026-11-18): removing the Re-cap trap

`a8_captrap.py` → `results/a8_captrap.md` (predictions 0cd6993). The a7 protocol with two remedies, same draws:

- `leak`: `re_leak = 0.05` (6dcc57d), which gives a slope of 0.05 above the cap;
- `stable`: the Re coefficients start at −|draw|.

The first launch stopped at beam seed 3: torch's strong-Wolfe line search raised an IndexError on a non-finite
loss. `a7_joint.fit` now treats that like a non-finite loss, and the run resumed from the finished entries.

| system | a7 | `leak` | `stable` |
|---|---|---|---|
| diff, adr (per seed, of 16) | 11 | **16** (every seed) | **16** (every seed) |
| diff2d, adr2d (of 8) | 5 | **8** | **8** |
| beam (of 16) | 7, 8, 9, 8, 8 | 13, 9, 15, 12, 12 | 8, 10, 9, 9, 8 |
| selected start, test / oracle (25 seeds) | ≤ 1.012 | ≤ 1.012 | ≤ 1.012 |

**Predictions against results.**

1. Diffusion and ADR, with each remedy, ≥ 15 of 16 (1D) and ≥ 7 of 8 (2D) in every seed: **holds**, every start
   succeeds.
2. Beam, each remedy adds 1–4 successful starts per seed: **fails** for both, in opposite directions. `leak` adds
   1–7 (two seeds exceed 4); `stable` adds 0–2 (two seeds add none).
3. The selected start is at the oracle in all 25 seeds with each remedy: **holds.**

**Reading.**

- For first-order systems, the cap trap was the only reason for failed starts. Either remedy removes it.
- On the beam the leak is the better remedy. With it, a start can pass through the cap on its way to the truth; for
  the beam the truth sits on the imaginary axis, close to the cap. A stable start only changes which draws fail:
  0, 3, 5, 7, 8 still fail, by the other mechanisms seen in a7 (dipole pairs, over-damping).
- Default for later work: `re_leak = 0.05` whenever `re_max` is used. Every deep model in `experiments/A/deep`
  uses it.

## RQ4 (2026-11-19): polynomial against MLP pole functions

`a19_mlp_band.py` → `results/a19_mlp_band.md` (predictions a06eab9). Layer option `mu_mlp` (29cf0ef): μ_n(k) from
an MLP of k / k_ref (two tanh layers of 32), with residues still polynomial.

**Setup.** Training and validation data are forced at k ≤ 8 only. One test set is forced in-band (k ≤ 8), the other
out of band (k = 9–16). Systems: diff, adr, beam; 5 data seeds. Both models use the a8 joint fit (8 starts, L-BFGS,
leaky cap).

| system | μ(k) | in-band test / oracle | out-of-band test / oracle | dispersion error, k ≤ 8 | k = 9–16 |
|---|---|---|---|---|---|
| diff | polynomial | 1.00 | **1.00** | 1e-4 – 5e-4 | **3e-4 – 2e-3** |
| diff | MLP | 1.01–1.05 | 28–44 | 1e-3 – 9e-3 | 0.47–0.55 |
| adr | polynomial | 1.00 | **1.00–1.01** | 3e-4 – 9e-4 | **1e-3 – 5e-3** |
| adr | MLP | 1.00–1.03 | 4.6–19 | 5e-4 – 4e-3 | 0.13–0.33 |
| beam | polynomial | 1.00–1.01 | **1.00–1.08** | 7e-5 – 2e-4 | **2e-4 – 1e-3** |
| beam | MLP | 1.03–1.05 | 29–41 | 6e-3 – 7e-3 | 0.27–0.38 |

**Predictions against results: all hold.**
1. Both fit in-band (≤ 1.2× the oracle).
2. Out-of-band dispersion error: polynomial ≤ 5 % (max 0.46 %); MLP > 10 % (13–55 %), in every seed.
3. Out-of-band test: polynomial ≤ 1.5× (max 1.08); MLP > 3× (4.6–44), in every seed.

**Reading (RQ4).** Inside the band the data excite, the MLP is almost as accurate as the polynomial: its test error
is within 5 %, though its dispersion error is 5–30× larger. Outside the band, only the polynomial carries the
physics, because its structure is the physics. The trade-off is not accuracy against interpretability inside the
band: the polynomial wins both whenever its form is right. What the MLP buys is freedom from a wrong form, which
none of these systems needed. A system whose λ(k) is not polynomial (nonlocal, fractional) is the test that would
favour the MLP.

## RQ4, second test (2026-11-19): a non-polynomial dispersion relation

`a22_fractional.py` → `results/a22_fractional.md` (predictions 3f100d0). Fractional diffusion,
λ(k) = −0.15 |k|^1.5. Before training, the best least-squares fit of {1, k², k⁴} is 2.3 % off over k = 1..16; fitted
on k ≤ 8 it is 71 % off over k = 9..16. a19 protocol, 5 data seeds. The oracle uses the exact λ(k).

| regime | μ(k) | in-band test / oracle | out-of-band test / oracle | dispersion error, in band | k = 9..16 |
|---|---|---|---|---|---|
| full (train k ≤ 16) | polynomial | 5.4–8.5 | – | 19–35 % | – |
| full | MLP | **1.02–1.04** | – | **0.55–1.4 %** | – |
| band (train k ≤ 8) | polynomial | 3.3–4.8 | 3500–5300 | 5.3–5.8 % | 96–98 % |
| band | MLP | **1.01** | **12–18** | **0.07–0.27 %** | 23–27 % |

**Predictions against results: both hold.**
1. `full`: the MLP reaches ≤ 1.2× the oracle with a dispersion error ≤ 1 % in 4 of 5 seeds (seed 3: 1.4 %). The
   polynomial is ≥ 1.5 % off and has a higher test error in every seed.
2. `band`: both are > 10 % off out of band in every seed.

**Reading (RQ4, with a19).**

- With the right form (a19), the polynomial wins in band and is the only one that extrapolates. With the wrong form
  (here), the MLP wins in band by 5–8×. Neither extrapolates.
- When the polynomial's form is wrong, its extrapolation is catastrophic: the k⁴ term drives the error to 3500×.
  The MLP degrades gracefully (12–18×).
- **A practical check.** Fit both. If the polynomial's in-band error matches the MLP's, its form is supported and
  its extrapolation can be trusted. If it is clearly worse (5–8× here, against 1.00 in a19), the form is wrong.
- The polynomial's learned dispersion error (19–35 %) is far above its least-squares floor (2.3 %). The joint fit
  does not reach the best polynomial when no polynomial is right. Not analysed further.
