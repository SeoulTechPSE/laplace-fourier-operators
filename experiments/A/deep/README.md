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
- **The WPO's poles barely moved from their initialization.** In all four layers, Re μ_n(k) stays within ±0.5 and
  is almost flat in k, and some poles have Re μ > 0 (up to 0.21). The truth needs a k² coefficient of
  −ν k_ref² = −25.6 in k / k_ref units. With Adam at lr 1e-3, a parameter moves by at most about 4 in 4000 steps,
  so it cannot get there. The positive poles grow as e^{0.2 t}, which explains the extrapolation blow-up.
- This is the deep-model form of what (C) E2 and E5 found: the poles need their own, larger learning rate (the
  `plr10` variants there). The linear validations hid it, because L-BFGS rescales its steps.

**Next.** Pre-register a pole learning rate (×30) for the WPO, with and without the stable parameterization, at the
same budget, then decide on longer training for both models.

## Second pass (2026-11-18): a separate pole learning rate

`a10_polelr.py` → `results/a10_polelr.md` (predictions 295b41d). As a9, but the WPO's pole coefficients train at
30× the learning rate (3e-2). Variants: `plr30` (the a9 layer: Re cap with leak) and `plr30_stable`
(Re μ = −softplus(·) ≤ 0). 3 seeds each; the a9 FNO is the reference.

| model | test (T = 5), mean | T = 10, (5, 10] | ratio (5, 10] / test | probe ν̂ (0.1) | probe dispersion error |
|---|---|---|---|---|---|
| a9 WPO | 0.117 | 3.2–8.5 | 35–62 | 0.03–0.08 | 0.96–1.01 |
| a9 FNO | 0.105 | 0.30 | 2.9 | 0.104–0.109 | 0.38–0.42 |
| plr30 | **0.048** (0.041–0.053) | 0.11–0.16 | 2.6–3.1 | 0.122–0.138 | 0.16–0.26 |
| plr30_stable | **0.035** (0.034–0.036) | 0.068–0.088 | 2.0–2.4 | 0.111–0.114 | 0.09–0.16 |

**Predictions against results.**

1. Probe ν̂ within 10 % and dispersion error ≤ 20 % in ≥ 2 of 3 seeds, per variant: **fails** for both.
   `plr30_stable` is 11–14 % off in ν̂ (dispersion error 9–16 %). `plr30` is 22–38 % off.
2. Mean test error ≤ 0.7× a9's WPO and below a9's FNO, per variant: **holds** (0.048 and 0.035, against 0.117 and
   0.105).
3. `plr30_stable` keeps the (5, 10] error ≤ 2× its test error in every seed: **fails** narrowly (2.04–2.43). `plr30`
   exceeds 2× in at least one seed: **holds** (2.6–3.1).

**Reading.**

- The pole learning rate was the bottleneck. With it, the WPO's error falls by 2.4–3.4× at the same budget, and the
  extrapolation blow-up is gone (a9: errors of 3–8 over (5, 10]). The stable parameterization helps further:
  lowest error, smallest growth, best probe.
- **The probe is biased toward stronger damping.** ν̂ is 0.111–0.114 across stable seeds, so the bias is
  systematic, not seed noise. The relative loss weights every sample equally, so the near-linear samples are not
  under-weighted. Three explanations remain, to be tested:
  - the probe amplitude (ε = 0.01) lies below the smallest training scale (0.03), so the probe extrapolates in
    amplitude;
  - the model's linear part absorbs an effective viscosity from the nonlinear transfer in the large samples;
  - limited capacity.
  This is (C)'s data gate in another form.
- **First-layer poles are not the physical poles.** The nearest first-layer pole is 30–70 % off −ν k² for
  k = 4–16. As in (C) E5, only the probe of the whole operator is meaningful in a deep model.
- **Not yet a fair accuracy comparison.** The WPO got a tuned pole learning rate and the FNO did not, and both are
  still short of convergence (a9: train ≈ test error).

**Next.**

- A fair comparison: learning-rate sweep on the validation split for both models (and the pole ratio for the WPO),
  longer training, 3 seeds.
- Then the probe bias: probe at several amplitudes, and train with data that reach smaller scales.

## Probe bias (2026-11-18): where it comes from

**Probe-amplitude scan.** `a12_probe_amp.py` → `results/a12_probe_amp.md` (predictions dab403a). The a10 stable
checkpoints were probed at ε = 0.003–0.3.

- ν̂ is 0.1110, 0.1122 and 0.1137 at every ε ≤ 0.03, unchanged to four digits, and barely moves at 0.3. The
  solver gives 0.1005 for ε ≤ 0.03.
- Both predictions hold: the solver is within 1 %, and the WPO's ν̂ is flat in ε. The bias is not an amplitude
  extrapolation.

**Best linear fits to the data.** `a12b_linear_fit.py` → `results/a12b_linear_fit.md` (predictions d719c42). A
single pole layer, fitted to all samples and to each quartile of forcing amplitude.

| subset | linear test error | ν_lin (k² coefficient) | offset (extra uniform decay rate) |
|---|---|---|---|
| all | 0.18 | 0.0953 | 0.041 |
| Q1 (smallest) | 0.029 | 0.0997 | 0.001 |
| Q2 | 0.072 | 0.0997 | 0.008 |
| Q3 | 0.16 | 0.1001 | 0.047 |
| Q4 (largest) | 0.38 | 0.0756 | 0.365 |

- Prediction 1 (all samples: ν_lin in [0.105, 0.13]) **fails**: 0.095.
- Prediction 2 **fails in part.** Q1 is within 3 % (0.0997), but ν_lin does not rise with amplitude. In the
  larger samples, the extra damping of the best linear model goes into the k-independent decay rate (offset 0.05
  in Q3, 0.37 in Q4), not into ν.

**Per-mode reading of the WPO probe** (from the a10 probe data, mode by mode against −ν k²):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| truth | −0.10 | −0.40 | −0.90 | −1.60 | −2.50 | −3.60 | −4.90 | −6.40 |
| stable WPO, seeds 0–2 | −0.095 to −0.096 | −0.396 to −0.397 | −0.891 to −0.895 | −1.59 to −1.61 | −2.50 to −2.52 | −3.62 to −3.65 | −5.13 to −5.21 | −7.10 to −7.19 |

**Reading.**

- The deep model's linearization is right to about 1 % for k ≤ 6. Over k = 7–8, the top of the forced band, it
  is 5–12 % too damped.
- The "ν̂ bias" is an artifact of summarizing that curve with {1, k²}. The fit absorbs the curvature into a
  positive offset (+0.19 to +0.26) and a larger k² coefficient. The per-mode probe is the better readout; the
  polynomial summary needs a check of its residual.
- The forcing amplitude falls as 1/k, so modes 7–8 are forced weakly. In the larger samples they are dominated
  by energy transferred from lower k (the nonlinear cascade). The linear response the model learns there is
  therefore less constrained, and the data's best linear description damps more there (the offsets above).
- For the trust rule: report per-mode probe poles with the band that the data excite strongly, rather than a
  single fitted constant.

## Third pass (2026-11-18): tuned learning rates, 3× training — and a flaw in the FNO configuration

`a11_fair.py` → `results/a11_fair.md` (predictions 232df05). Both models are tuned on the validation split (FNO:
lr ∈ {1e-3, 3e-3, 1e-2}; WPO stable: lr ∈ {1e-3, 3e-3} × pole ratio ∈ {10, 30}). The selected settings are then
trained for 12000 steps with 3 seeds.

- Selected: FNO lr 1e-3; WPO lr 3e-3 with pole ratio 30.

| model | train | test (T = 5) | (5, 10] / test | probe ν̂ (0.1) | probe dispersion error |
|---|---|---|---|---|---|
| FNO (n_modes 12 × 12) | 0.062–0.064 | 0.106–0.109 | 2.78–2.87 | 0.105–0.109 | 0.34–0.38 |
| WPO (stable) | 0.011 | **0.023–0.024** | **1.75–1.92** | **0.101–0.103** | **0.010–0.034** |

Per-mode WPO probe (all seeds): within 1 % of −ν k² for k = 1–7, and within 2.4 % at k = 8 (a10: 12 % too damped
at k = 8).

**Predictions against results.**

1. WPO mean test error below the FNO's: **holds** (0.024 against 0.107). See the flaw below.
2. Extrapolation ratio ≤ 2.5 for the WPO and > 2.5 for the FNO in every seed: **holds.**
3. The probe bias persists (ν̂ in [0.105, 0.13]): **fails.** It is gone (0.101–0.103). The a10 over-damping at
   k = 7–8 came from undertraining.
4. Train error ≥ 0.8× test error for both: **fails.** Both models now show a generalization gap (WPO 0.47×, FNO
   0.59×) with 200 training samples.

**Flaw in the FNO configuration (found after the run).** In neuralop 2.0, `n_modes = (12, 12)` keeps 12 spatial
modes in total (|k| ≤ 6) and 12 // 2 + 1 = 7 temporal frequencies (weight shape (w, w, 12, 7)). The forcing band is
k ≤ 8, and steps and PRBS need many more temporal frequencies. The FNO's error floor of about 0.1 and its broken
probe at k ≥ 7 (Re μ̂ from −23 to −1500) follow from this truncation. The setting was chosen in a9 to match the
parameter budget, without checking the band. The accuracy comparison in a9–a11 is therefore not fair to the FNO.
The WPO-only findings stand: the pole learning rate, the probe and extrapolation without blow-up.

**Next.** A band-covering FNO. Full modes do not fit a 59 k budget (width 4 with 32 × 32 modes already has 135 k
parameters). So the comparison uses the Tucker-factorized TFNO from neuraloperator, with n_modes covering the band,
at the matched budget, plus a full-rank FNO at a larger budget as a reference. Both get the same learning-rate
sweep.

## Fourth pass (2026-11-18): band-covering FNO baselines

`a13_baselines.py` → `results/a13_baselines.md` (predictions 9055eb7). n_modes (32, 64): |k| ≤ 16 and 33 temporal
frequencies. Same sweep and protocol as a11. The WPO is a11's.

- Selected learning rates: TFNO 3e-3, FNO 1e-3.

| model | params | train | test (T = 5) | (5, 10] | ratio | probe residual (median) | probe dispersion error: kept modes / k ≤ 8 |
|---|---|---|---|---|---|---|---|
| WPO (stable) | 59.5 k | 0.011 | **0.023–0.024** | **0.042–0.046** | **1.8–1.9** | **0.002–0.003** | **0.010–0.034** / 0.010–0.019 |
| TFNO (Tucker, matched) | 59.1 k | 0.009–0.010 | 0.027–0.033 | 0.29 | 8.6–10.6 | 0.008–0.25 | 0.90–0.95 / 0.016–0.020 |
| FNO (full rank, 18×) | 1.08 M | 0.006–0.007 | 0.041–0.043 | 0.29 | 6.7–7.1 | 0.36–0.49 | 0.90–0.93 / 0.007–0.015 |

**Predictions against results.**

1. The matched TFNO's mean test error is above the WPO's: **holds** (0.031 against 0.024).
2. The 18× FNO's mean test error is below the WPO's: **fails.** It is 0.042. It fits the training set best (0.006)
   but overfits with 200 samples.
3. Extrapolation ratio > 2.5 for both FNO variants in every seed: **holds** (6.7–10.6). Their (5, 10] error is 0.29
   for every seed; the WPO's is 0.04.
4. Probe dispersion error above the WPO's in every seed: **holds** (0.90–0.95), but see below.

**Reading (post hoc, from the stored per-mode probes).**

- **Inside the forced band (k ≤ 8), every model's linearization is right to 1–2 %.** The FNO-type models learn the
  physical linearization there just as the WPO does. The registered dispersion error separates them only because
  of the modes above the band.
- **Above the band (k = 9–16), the models differ.** The WPO continues the dispersion relation: k = 9 gives −8.17
  to −8.45 against −8.1. The probe keeps only k ≤ 9 for the WPO, because higher modes decay faster than the
  resolution cut allows, as in the truth. The FNO-type models return arbitrary poles there (Re −0.03 to −7,
  |Im| up to 8), and those spurious poles pass both cuts.
- **The WPO's linear response is LTI-consistent.** Its median probe residual is 0.002–0.003, against 0.008–0.49 for
  the FNO variants: the FNO's odd small-amplitude response is not well described by one causal pole per mode.
- **Accuracy at the matched budget:** WPO 0.024 against TFNO 0.031, so the WPO is 23 % lower. That is a modest
  margin. The 18× FNO is worse than both.
- **Extrapolation in time is the clear difference:** 0.04 against 0.29 over (5, 10]. The FNO's t modes rescale with
  the window. The WPO's recursion does not depend on the window.

**Status of the a9–a13 sequence.** On forced Burgers at a 59 k budget, the WPO (stable, separate pole learning rate)
leads the band-covering TFNO by 23 % in-window and by 7× in time extrapolation. In-band, the probes of both are
right. Out of band, only the WPO's is, together with an LTI-consistent linearization.

## Resolution transfer (2026-11-18), zero-shot

`a15_resolution.py` → `results/a15_resolution.md` (predictions in its docstring, committed before the run). The
models trained at n_x = 64 and dt = 0.025 are evaluated unchanged on 100 new samples. The truth comes from a
128-point solve.

| model | n_x = 32 | n_x = 128 | dt / 2 (n_t = 401) |
|---|---|---|---|
| WPO | 1.01–1.02× | 1.00× | **0.79–0.80×** |
| TFNO | 1.01× | 1.00× | 0.92–0.94× |
| FNO 18× | 1.01–1.09× | 1.00–1.03× | 0.94–0.96× |

(Ratios are to each model's own error at n_x = 64 and dt on the same samples: WPO 0.018–0.019, TFNO 0.022–0.027,
FNO 0.033–0.035.)

**Predictions against results.**

1. WPO within 1.2× at n_x = 128 and dt / 2: **holds.**
2. TFNO within 1.2× at n_x = 128 and dt / 2: **holds.**
3. Both exceed 1.2× at n_x = 32: **fails.** Aliasing is negligible at 32 points, because the solution's content
   above k = 16 is tiny for ν = 0.1.

**Reading.**

- In space, all three models transfer to within a few percent. Resolution invariance does not separate them on
  this problem.
- On a finer time grid, every model improves. Part of the error at dt = 0.025 is the sampling floor seen since a1:
  steps and PRBS jumps fall between samples, and a finer grid resolves them. The WPO gains the most (−20 %
  against −4 to −8 %). Its FOH recursion uses the finer dt exactly, while the FNO's temporal modes are fixed in
  number.

## Cahn–Hilliard (2026-11-19): an unstable band (RQ3)

`a16_cahn_hilliard.py` → `results/a16_cahn_hilliard.md` (predictions 1f52ba7). Forced 1D Cahn–Hilliard
(`lfo.data.cahn_hilliard`), M = 0.1 and κ = 0.04, so λ(k) = M k²(1 − κ k²): k_c = 5, k_max = 3.54. The forcing
scale is log-uniform over [0.009, 0.3]. Configurations come from a11 / a13 (no new sweep), 12000 steps, 3 seeds.
Method check before training: the solver's own probe gives k_c = 5.107.

| model | test | (5, 10] / test | probe k_c | signs (kept modes) | per-mode Re μ̂, k = 1..4 (truth 0.096, 0.336, 0.576, 0.576) |
|---|---|---|---|---|---|
| WPO (Re μ > 0 allowed) | 0.025–0.027 | 13.6–18.1 | **4.99–5.03** | right (3/3) | 0.095–0.098, 0.334–0.335, 0.578–0.579, 0.575–0.577 |
| WPO stable (control) | 0.025–0.027 | 9.9–11.9 | nan, 1.9, 4.4 | right in 2/3 | 0.096–0.097, 0.334–0.335, 0.579, 0.575–0.576 |
| TFNO (matched) | 0.025–0.032 | 20.4–25.7 | nan (3/3) | wrong (3/3) | seed 2 as WPO; seeds 0–1 noisy (residual 0.1–0.3) |

**Predictions against results.**

1. WPO: k_c within 5 % and every sign right in ≥ 2 of 3 seeds: **holds** (3 of 3; k_c within 0.7 %, better than
   the solver's own probe).
2. The stable control has a higher test error than the WPO in every seed: **fails.** It is equal (0.025–0.027).
3. WPO mean test error ≤ TFNO's: **holds** (0.026 against 0.029).
4. TFNO's k_c within 10 % in ≥ 2 of 3 seeds: **fails** (no k_c in any seed). In-band, its per-mode poles are right
   in seed 2. But spurious poles above the band (k = 8–16) pass the cuts and spoil the {1, k², k⁴} fit, as on Burgers.
5. WPO extrapolation ratio below the TFNO's in every seed: **holds** (paired by seed). All models extrapolate badly
   here: phase separation continues past T = 5.

**The stable control.** A network whose layer poles all have Re μ ≤ 0 reproduces the growing modes in the probe,
to three digits, with residuals of 0.002. Its layer poles show how: in the unstable band they sit just below zero
(Re μ ≈ −0.01 to −0.02 in layers 0–1), so they act as near-integrators. A cascade of near-integrators produces
t^n growth, which matches e^{0.58 t} on [0, 5] to within the probe's resolution. The WPO with Re μ > 0 allowed
instead puts positive poles in the band (0.1–0.33 per layer). No single layer pole equals λ(k): the growth is shared
across layers.

**Longer probe window.** `a16b_long_probe.py` (predictions 6a335a0) repeats the probe over T = 10 to separate true
growth from emulated growth. It does not.
- Prediction 1 **fails**: the solver itself turns nonlinear at k = 3, 4 over T = 10, because e^{0.576 · 10} ≈ 300
  amplifies even ε = 0.003 into the cubic regime.
- Both WPOs (trained on T = 5) break down similarly beyond the window. Prediction 2 holds for the stable control,
  but the unstable one fails it too, so the test does not discriminate.

**Reading.**

- RQ3 holds for the WPO's probe: the unstable band (k_c, k_max, the sign of every mode) is recovered from a deep
  nonlinear model.
- **Identifiability limit.** Within a finite window, a deep stack can emulate an unstable pole with stable
  near-integrators. The probe of the whole operator cannot tell the two apart, and an unstable linearization
  cannot be probed far beyond the window either, because it leaves the linear regime. So the claim "the model has
  an unstable pole" needs either the layer structure (Re μ > 0 allowed and used) or data over longer windows. The
  probe alone does not support it. This fits (C)'s finding that probe and layer readings must agree before a pole
  is trusted.

## Data efficiency (2026-11-19)

`a14_data_eff.py` → `results/a14_data_eff.md` (predictions b17f853). Nested training sets on forced Burgers. The
WPO uses the a11 configuration and the TFNO the a13 one; 12000 steps, 3 seeds, a9 test set.

| N | WPO test | TFNO test | TFNO / WPO |
|---|---|---|---|
| 50 | 0.036 | 0.070 | **1.96** |
| 100 | 0.029 | 0.042 | 1.42 |
| 200 | 0.024 | 0.031 | 1.28 |
| 400 | 0.021 | 0.023 | 1.09 |

**Predictions against results.**

1. WPO below TFNO at every N, with the largest ratio at N = 50: **holds.**
2. WPO at N = 100 is at or below TFNO at N = 200: **holds**, narrowly (0.029 against 0.031; the seed ranges
   overlap).
3. Both improve monotonically with N: **holds.**

**Reading.** The WPO's advantage is a data-efficiency advantage. It halves the TFNO's error at 50 samples, and the
margin shrinks to 9 % at 400. That is what an inductive bias gives: the wavenumber-pole structure substitutes for
data. Both models still have a train–test gap at N = 400 (WPO 0.012 against 0.021).

## 2D deep operator (2026-11-19): forced 2D Cahn–Hilliard

`a17_ch2d.py` → `results/a17_ch2d.md` (predictions c380ece). Shared code: `WavenumberPoleConv2d`,
`WavenumberPoleOperator2d`, `cahn_hilliard.solve2d` (7a1fb79, tests).

**Setup.** 32² grid, T = 5, M = 0.1, κ = 0.04, so k_c = 5. Each sample is forced at 4 wavevectors with |k_x|,
k_y ≤ 6; s is log-uniform over [0.009, 0.3]. Models: WPO-2D (width 16, 4 layers, 4 poles, isotropic Re μ in |k|²,
18.7 k parameters) and a 3D Tucker FNO (n_modes 16 × 16 × 64, 15.0 k, the closest rank; 20 % fewer). 6000 steps,
batch 10, a11 / a13 learning rates, 3 seeds. The probe uses wavevectors (k, 0) and (k, k), k = 1..6. Solver
probe: k_c = 5.003.

| model | train | test | probe k_c | per-mode error, kept modes (median / max) |
|---|---|---|---|---|
| WPO-2D | 0.023–0.029 | **0.030–0.036** | **4.98–5.09** | ~0.5 % / 14 % (the (5, 5) mode in seed 1) |
| TFNO | 0.055–0.076 | 0.100–0.125 | 4.49–4.96 | 2–19 % (in-band (3, 0): 0.45–0.49 against 0.576) |

**Predictions against results.**

1. WPO-2D: k_c within 5 % and every kept sign right in ≥ 2 of 3 seeds: **holds** (3 of 3, within 1.9 %).
2. WPO-2D mean test error ≤ TFNO's: **holds** (0.032 against 0.116).
3. TFNO's k_c within 10 % in at most 1 of 3 seeds: **fails.** Two seeds are within 10 % (4.64, 4.96). The probe here
   uses only k ≤ 6 along two directions, so no out-of-band modes enter the fit, and the TFNO's in-band errors
   (up to 19 %) partly cancel in k_c.

**Reading.**

- The 1D findings carry over to 2D. The WPO's probe is right to a fraction of a percent mode by mode, including
  the isotropy: (k, 0) and (k, k) fall on the same λ(|k|). Its k_c is within 2 %.
- The TFNO underfits at this budget (train 0.055–0.076, against 0.023–0.029). It had 20 % fewer parameters and no
  learning-rate sweep, so the 3.6× accuracy gap overstates what a tuned TFNO would show. The probe comparison (WPO
  per-mode exact, TFNO 2–19 % off in-band) is less sensitive to that.

## 2D Cahn–Hilliard with a tuned, budget-matched TFNO (2026-11-19)

`a18_ch2d_tuned.py` → `results/a18_ch2d_tuned.md` (predictions 9d0cfa7).

**Setup.** Two Tucker configurations within 10 % of the WPO-2D's budget: width 12 with rank 0.003 (20.4 k), and
width 20 with rank 0.001 (20.5 k). Each is swept over lr ∈ {1e-3, 3e-3, 1e-2} for 2000 steps. The selected
setting, width 12 with lr 3e-3, is trained for 6000 steps over 3 seeds. Width 20 with rank 0.001 did not train at
any learning rate (validation error 1.0).

| model | params | test | probe k_c | median in-band per-mode error |
|---|---|---|---|---|
| WPO-2D (a17) | 18.7 k | **0.030–0.036** | **4.98–5.09** | **0.2–0.6 %** |
| TFNO tuned | 20.4 k | 0.071–0.089 | 3.76, 4.98, 5.06 | 3.9–8.3 % |
| TFNO a17 (untuned, 15.0 k) | 15.0 k | 0.100–0.125 | 4.49–4.96 | 3–16 % |

**Predictions against results: both hold.**
1. The tuned TFNO's test error is below a17's TFNO (0.082 against 0.116) and above the WPO-2D's (0.032).
2. Its k_c is within 10 % in 2 of 3 seeds (seed 1: 3.76), and its median per-mode error stays above the WPO-2D's.

**Reading.** Tuning closes about a third of the gap. At a matched budget the WPO-2D keeps a 2.6× lower test error,
and its per-mode linearization is an order of magnitude more accurate.

## Application 2, stage 2 (2026-11-21): rollout into 2D coarsening (a31)

`a31_coarsening.py` → `results/a31_coarsening.md` (predictions b1b5cdc; solver extensions 71961b0, 1a7deee).
Unforced 2D Cahn–Hilliard, M = 1, κ = 0.02, critical quench from white noise; truth on 128² (checked against
256²: 4e-5), models on 64². A window maps the state at t0 to [t0, t0 + 1]; training windows start at t0 = 1..20
(coarsening regime); rollout of 20 windows from the true state at t = 1. Truth: L from 1.51 to 3.52, exponent
n = 0.271 on t = 2..21 (t^{1/3} with early-time and box corrections). 3 seeds, 6000 steps.
- `wpo`: WPO-2D (a17's settings, modes 12), state broadcast over the window; 18.7 k.
- `tfno`: Tucker FNO on (x, y, t); 15.5 k.
- `fno2d`: FNO on (x, y), state to state at t0 + 1 (the usual autoregressive surrogate); 21.3 k.

| model | window-end error | rollout error, window 1 / 5 / 20 | L(21) / truth | n (truth 0.271) | S(k, 21) error | max abs u(21) |
|---|---|---|---|---|---|---|
| wpo | 0.033–0.035 | 0.22 / 0.45 / diverges (seed 0) | 0.51; 0.975–0.991 | −0.10; 0.261–0.268 | 8e8; 0.16–0.18 | 4e5; 1.05–1.65 |
| tfno | **0.015–0.022** | **0.13 / 0.22 / 0.37** | 0.967–0.974 | 0.255–0.257 | 0.15–0.18 | 1.04–1.07 |
| fno2d | 0.035–0.043 | 0.23 / 0.52 / 0.88 | **0.968–1.015** | **0.247–0.267** | **0.03–0.16** | 1.03–1.04 |

For `wpo`, seed 0 first, then seeds 1–2. References: persistence 0.192 per window; a rollout frozen at t = 1 has
L(21) / truth 0.43.

**Predictions against results.**
1. Every model beats persistence on the window-end error: **holds** (≤ 0.043 against 0.192).
2. `tfno` ≤ `wpo` in mean window-end error: **holds** (0.018 against 0.034).
3. `wpo`: L(21) within 10 % and n within ±0.05, in ≥ 2 of 3 seeds: **holds** (seeds 1, 2); seed 0 diverges.
4. `fno2d`: the same, ≥ 2 of 3 seeds: **holds** (all three).
5. S(k, 21) error ≤ 0.25 for every model and seed: **fails** (`wpo` seed 0). Every other run: 0.03–0.18.

**Diagnosis of the divergence** (post hoc; seed 0 retrained and reproduced exactly). In seed 0, layers 2–4 have
poles at the growth cap (max Re μ 1.10, 1.01, 1.01; seed 1: 0.85, 0.50, 0.80). In rollout the amplitude grows
(max |u| 2.0, 15, 2e3, 4e5 at windows 5, 10, 15, 20) without a pile-up at high k (energy beyond |k| = 12 stays
~2e-3). Within one window the pointwise nonlinearities hold the growth back on training states; off them,
compounding over windows, nothing does. Seed 2 (max |u| 1.65) shows the start of the same drift.

**Reading.**
- Coarsening statistics are easy to get roughly right. All stable runs give L(21) within 3.5 % and n within 0.025
  of the truth, against 0.43 for a frozen field. They are a weak test of the operator; the per-window errors
  separate the models more.
- The TFNO is the best pointwise model (2× lower one-window and rollout error), as on 1D Cahn–Hilliard with initial
  fields (D2, d2g/d2h). The plain autoregressive FNO-2D has the worst pointwise rollout but the best statistics.
- The pole structure gives no advantage here and costs robustness. Re μ > 0 is needed for the spinodal band
  (a16, a17, a29), but about the separated phases ±1 the linearization is stable
  (λ = −Mk²(2 + κk²)). A WPO allowed to grow can drift out of the data in rollout. A state-dependent or
  stage-dependent cap would be needed; not tried. (a32 below: the cap is not the cause.)

## Application 2, stage 2b (2026-11-21): the WPO-2D growth cap on coarsening (a32, a32b)

`a32_growth_cap.py` → `results/a32_growth_cap.md` (predictions 53b2cac). a31's setup with the WPO-2D pole constraint
changed (same 18,737 parameters, 3 seeds): `wpo_stable` (Re μ = −softplus < 0 in every layer), `wpo_cap03`
(re_max 0.3, no leak), `wpo_cap1` (re_max 1.0, no leak). a31's `wpo` (cap 1.0, leak 0.05) and `tfno` are the
references.

| model | window-end error | diverged seeds (max \|u(21)\| > 2) | stable seeds: window-20 error, L(21) / truth, n | max Re μ |
|---|---|---|---|---|
| a31 wpo | 0.033–0.035 | 1 of 3 (4e5) | 0.67–0.74, 0.975–0.991, 0.261–0.268 | at the cap in seed 0 |
| a31 tfno | **0.015–0.022** | 0 | **0.34–0.39**, 0.967–0.974, 0.255–0.257 | — |
| wpo_stable | 0.037–0.040 | **2 of 3** (268, 171) | 0.76, 1.002, 0.271 (seed 1) | < 0 (−0.02 to −1.6) |
| wpo_cap03 | 0.037–0.048 | **3 of 3** (9e6–1e14) | — | 0.30 in every layer and seed |
| wpo_cap1 | 0.035–0.038 | **2 of 3** (22, 2e4) | 0.77, 0.987, 0.266 (seed 1) | 1.00 in 2–3 layers per seed |

**Predictions against results.**
1. `wpo_stable` never diverges and gets L, n right in every seed: **fails** (2 of 3 diverge).
2. `wpo_stable` window-end error ≤ 0.043: **holds** (0.039).
3. `wpo_cap03` never diverges: **fails** (all 3 diverge, the most violently).
4. `wpo_cap1` reaches max Re μ ≥ 0.95 in some layer in ≥ 1 seed: **holds** (all 3).
5. Every variant's window-20 error above the TFNO's 0.37: **holds** (the stable seeds give 0.76–0.77).

**Post-hoc diagnosis** (`a32b_amplitude.py` → `results/a32b_amplitude.md`, no predictions; runs retrained with the
same seeds). True test states at t0 = 5, 10, 15 are scaled by a, and the gain is max |M(a s)| / max |a s| at the
window end. The truth's gain is from the solver on 128² (64² agrees to 0.3 %).

| run | gain at a = 1 / 1.1 / 1.25 / 1.5 / 2 | max \|u\| after window 5 / 10 / 15 / 20 |
|---|---|---|
| truth | 1.00 / 0.91 / 0.80 / 0.67 / 0.50 | — |
| tfno s0 | 1.00 / 0.94 / 0.89 / 0.82 / 0.76 | 1.05 / 1.03 / 1.04 / 1.06 |
| wpo s0 (diverges) | 1.02 / 1.02 / 1.21 / 1.65 / 2.35 | 2.0 / 15 / 2e3 / 4e5 |
| wpo s1 | 1.02 / 0.94 / 1.13 / 1.60 / 1.63 | 1.9 / 1.3 / 1.1 / 1.05 |
| wpo_stable s0 (diverges) | 1.02 / 0.98 / 1.15 / 1.35 / 1.51 | 1.3 / 1.9 / 6.4 / 268 |
| wpo_stable s1 | 1.02 / 1.05 / 1.56 / 2.95 / 4.84 | 1.1 / 1.2 / 1.3 / 1.2 |

**Reading.**
- The growth cap is not the cause of a31's divergence. Removing growth entirely (stable poles) does not help (2 of
  3 seeds diverge), and a tight cap of 0.3 is worse (3 of 3). The poles are not where the instability lives.
- What is missing is saturation. The true window map pulls an overshoot back toward ±1 (gain 1/a), and the TFNO
  does so too, more weakly (0.76 at a = 2). Every WPO-2D run, with stable poles or not, amplifies a scaled state
  (gain 1.5–4.8 at a = 2), so it has no restoring force outside the amplitudes it was trained on (|u| ≲ 1.05). An
  overshoot past ±1 is then kept or amplified. Whether a seed diverges within 20 windows depends on whether its
  rollout reaches such states (wpo s1 reaches 1.9 by window 5 and recovers; wpo_stable s1 has the largest gain
  but stays below 1.3), so the gain explains the failure mode but does not predict which seed fails.
- Why the cap of 0.3 is worst is open; with every layer pinned at the cap the one-window fit is also the worst
  (up to 0.048).
- Candidates (not tried): training on amplitude-perturbed starts (a·s mapped to the truth from a·s); a saturating
  output map; rollout (pushforward) training on this system. These address the missing restoring force, not
  the poles.


## Application 2, stage 2c (2026-11-21): training on amplitude-perturbed starts (a33)

`a33_amplitude_aug.py` → `results/a33_amplitude_aug.md` (predictions 410fd88). a32b's diagnosis: the WPO-2D map
has no restoring force on overshoots. Remedy: each of a31's 960 training windows gets one scaled copy, start a·s
with a ~ U[0.7, 1.6], target the truth from a·s (128² solver, dt / 2; checked against dt / 4 to 5e-6). Training
set 1920 windows, otherwise a31 (6000 steps, same models, seeds, parameters).

| model | window-end error | diverged seeds | window 1 / 5 / 20 error | L(21) / truth | n | gain at a = 1.25 / 1.5 / 2 |
|---|---|---|---|---|---|---|
| a31 wpo | 0.033–0.035 | 1 of 3 | 0.22 / 0.45 / 0.67–0.74 (stable seeds) | 0.975–0.991 (stable) | 0.261–0.268 (stable) | 1.13–1.21 / 1.60–1.65 / 1.63–2.35 (a32b) |
| wpo_aug | 0.035–0.039 | **0 of 3** | 0.22 / 0.47 / 0.81 | 0.961–0.990 | 0.244–0.262 | **0.82–0.83 / 0.70–0.72 / 0.58–0.66** |
| a31 tfno | 0.015–0.022 | 0 | 0.13 / 0.22 / 0.37 | 0.967–0.974 | 0.255–0.257 | 0.89 / 0.82 / 0.76 (a32b, s0) |
| tfno_aug | **0.018–0.020** | 0 | **0.14 / 0.24 / 0.41** | 0.974–0.976 | 0.255–0.257 | 0.80 / 0.67 / 0.51 |

Truth's gain: 0.80 / 0.67 / 0.50.

**Predictions against results.**
1. `wpo_aug` never diverges: **holds** (max |u(21)| 1.04–1.08 in all 3 seeds).
2. `wpo_aug` gain < 1 at a = 1.25 and 1.5 in every seed: **holds** (0.82–0.83, 0.70–0.72).
3. `wpo_aug` window-end error ≤ 0.041: **holds** (mean 0.037).
4. `wpo_aug` window-20 rollout error ≤ 0.74: **fails** (0.81; seeds 0.70, 0.85, 0.89).
5. `tfno_aug` best pointwise: **holds** (0.41 against 0.81).

**Reading.**
- One cheap data change fixes the failure mode. With amplitude-perturbed starts, every WPO-2D seed restores
  overshoots (gain close to the truth's, also at a = 2, outside the training range), and no seed diverges, where
  a31 / a32 lost 1–3 of 3 seeds under every pole constraint. The divergence was a data-coverage problem (the
  training starts never left |u| ≲ 1.05), not a pole problem.
- The coarsening statistics are now right in every WPO-2D seed (L(21) within 4 %, n within 0.03).
- It does not buy pointwise accuracy. The stable WPO-2D rollouts are slightly worse than a31's two surviving
  seeds (0.81 against 0.71), at a slightly higher one-window error (0.037 against 0.034: half the training windows
  are now off the trajectory distribution). The TFNO learns the restoring map almost exactly with the same data
  (gain 0.51 at a = 2) and stays 2× better pointwise.
- Net for application 2, stage 2: the pole operator reaches the coarsening statistics robustly once trained on
  perturbed starts, but the TFNO is the better surrogate in this nonlinear regime. The pole structure's value in
  Cahn–Hilliard stays in the linear regime (a29, a30: spinodal band, k_c, parametric extrapolation).

## Application 2, stage 2d (2026-11-21): transfer to a 4π box without retraining (a34)

`a34_domain.py` → `results/a34_domain.md` (predictions a93e03f). a33's `wpo_aug` and `tfno_aug`, retrained with
the same data and seeds and saved, evaluated on the 2π box and on a 4π box at the same grid spacing (128²).
- Truth: 256² with M / 4 and κ / 4 on a 2π grid (an exact rescaling), the same physical initial noise, 16
  trajectories.
- WPO-2D: each layer gets length 4π and 24 modes, the same physical band.
- TFNO: unchanged. Its weights are indexed by mode number, so on 4π they act on half the physical wavenumbers.

Checks before any fit: on a 2π state tiled 2 × 2, the transferred WPO-2D reproduces the tiled 2π output to 3.9e-16.
Truth on 4π: L(21) = 3.45, n = 0.261, against 3.52 and 0.271 on 2π. A fifth prediction (n on 4π above 0.271) was
withdrawn before the commit because the truth had already been computed.

| model | window-end error 2π | window-end error 4π | ratio | 4π rollout, window 1 / 5 / 20 | L(21) / truth | n (truth 0.261) | max \|u(21)\| |
|---|---|---|---|---|---|---|---|
| wpo_aug | 0.035–0.039 | **0.035–0.041** | **1.02–1.03** | **0.24 / 0.58 / 0.85** | **1.04–1.08** | **0.272–0.286** | 1.07–1.09 |
| tfno_aug | 0.018–0.020 | 0.66–0.71 | 35–38 | 1.13 / 1.34 / 1.41 | 2.15–2.19 | 0.31–0.35 | 1.03 |

**Predictions against results.**
1. `wpo_aug` 4π error ≤ 1.3× its 2π error, every seed: **holds** (1.02–1.03).
2. `tfno_aug` 4π error ≥ 2× its 2π error, every seed: **holds** (35–38×).
3. On 4π, `wpo_aug` below `tfno_aug`: **holds** (0.038 against 0.69).
4. `wpo_aug` on 4π: no divergence, L(21) within 10 %, n within ±0.05, every seed: **holds** (1.04–1.08, 0.272–0.286).

**Reading.**
- The pole operator transfers to a larger domain without retraining, because its poles are functions of the
  physical wavenumber. The one-window error is unchanged (1.02–1.03×), and 20-window rollouts on 4π reproduce the
  coarsening law (L within 8 %, n within 0.026).
- The TFNO does not transfer. Its spectral weights belong to mode indices, which on the larger box are other
  physical wavenumbers. The one-window error rises 35× and the domains grow 2.2× too fast. This tests transfer, not
  accuracy: a TFNO trained on the 4π box (or on several box sizes) would be the fair accuracy comparison, and on the
  training box it stays 2× better.
- This is the first coarsening result where the pole structure carries a property the baseline lacks. It is the
  nonlinear counterpart of the linear-regime results: the parametrization in physical k is what transfers.
- Truth: the 4π box gives the same law as 2π (n 0.261 against 0.271). The 2π box was not finite-size limited up
  to t = 21; n below 1/3 is the early-time correction.

## Application 2, stage 2e (2026-11-21): a TFNO trained on the 4π box (a35)

`a35_tfno_4pi.py` → `results/a35_tfno_4pi.md` (predictions ca6f6e3). The accuracy reference for a34. TFNOs trained
on the 4π box itself.
- Training data: 12 trajectories (the same total area as a31's 48 on 2π), with a33's amplitude-scaled copies.
- `tfno4`: a31's TFNO unchanged (n_modes 16, physical |k| < 4 on 4π).
- `tfno4_m32`: n_modes 32, the same physical band as on 2π, 19,109 parameters.

| model | trained on | window-end error 4π | 4π rollout, window 1 / 5 / 20 | L(21) / truth | n (truth 0.261) |
|---|---|---|---|---|---|
| tfno4 | 4π | 0.031–0.038 | 0.24 / 0.56 / 0.83 | 1.031–1.042 | 0.270–0.274 |
| tfno4_m32 | 4π | **0.019–0.023** | **0.15 / 0.38 / 0.63** | **1.020–1.034** | **0.265–0.273** |
| wpo_aug (a34) | 2π only | 0.035–0.041 | 0.24 / 0.58 / 0.85 | 1.038–1.078 | 0.272–0.286 |

**Predictions against results.**
1. `tfno4_m32` ≤ 0.025: **holds** (0.021).
2. `tfno4_m32` below the transferred WPO-2D (0.038): **holds**.
3. `tfno4` ≥ 1.2× `tfno4_m32`: **holds** (0.034, 1.65×).
4. `tfno4_m32` L within 10 % and n within ±0.05 in every seed: **holds**.

**Reading.** Trained on the larger box, the TFNO is again the most accurate model: 1.8× better than the
transferred WPO-2D per window, and 1.35× better at window 20. It needs its spectral band rescaled to the box: with
a31's mode count it gives up most of that edge (0.034). The transferred WPO-2D, which never saw the 4π box, is
within 10 % of a TFNO trained there with the wrong band (0.038 against 0.034). It is 1.8× behind the one with the
right band. Application 2, stage 2 in one line: the TFNO is the better surrogate on any box it is trained on, and
the pole operator is the one that moves between boxes without data or retraining.

## Next-level step 1 (2026-11-22): poles per mode index vs per physical wavenumber; LNO; real budgets (a36)

`a36_index_poles.py` → `results/a36_index_poles.md` (predictions ebcce69). Literature re-check first
(`experiments/A/literature.md`): per-Fourier-mode generators with exponential updates exist (SGNO, S4D-type SSMs;
LGN-KM ties damping to |k|^p, unforced). So the (A) claim to test is the physical-wavenumber form. a11's forced
Burgers, a11's protocol, 3 seeds:
- `wpo_index`: the WPO with each layer's poles as a free table over the 22 mode indices (residues polynomial),
  initialized at the polynomial's values;
- `lno`: the original LNO v2, steady modes (9, 9) to cover the band, at the WPO's real budget (width 6);
- `tfno` (a13 retrained, 110,569 real parameters) and `tfno_r` (rank 0.01, 54,649 real).
- New metrics: probe pole error outside the forced band (k = 9..16) and a set forced only at k = 9..12.

Found while setting up: a13's "matched" TFNO counted complex weights once (`numel`); in real degrees of freedom it
had 1.86× the WPO's. Budgets are now matched with `lfo.metrics.n_real_params` (STATUS decision 2026-11-22).

| model | real params | test | (5, 10] / test | probe ν̂ | probe outside the band | unseen modes | unseen / test |
|---|---|---|---|---|---|---|---|
| wpo (a11 checkpoints) | 59,529 | **0.0237** | 1.83 | 0.1016 | **0.116** | **0.032** | **1.34** |
| wpo_index | 60,465 | **0.0234** | 1.71 | 0.1016 | 0.985 | 4.38 | 188 |
| tfno_r | 54,649 | 0.0279 | 10.3 | 0.0987 | 0.887 | 0.91 | 32 |
| tfno | 110,569 | 0.0304 | 9.53 | 0.0988 | 0.865 | 0.67 | 22 |
| lno | 55,834 | 0.376 | 1.20 | 0.060 | 1.063 | 1.09 | 2.9 |

(The probe's "in-band dispersion error" in the result file is computed over each model's kept modes; the
observability cut keeps out-of-band modes for some models, so that column mixes in- and out-of-band. At k ≤ 8
every model except the LNO is within a few % of −νk², as ν̂ shows.)

**Predictions against results.**
1. `wpo_index` test within ±15 % of the WPO's: **holds** (0.0234 against 0.0237).
2. `wpo_index` out-of-band probe error ≥ 3× the WPO's and above 0.142 in every seed: **holds** (0.985, 8.5×).
3. Unseen-mode ratio ≥ 2× the WPO's for `wpo_index` and `tfno`: **holds** (188 and 22 against 1.34).
4. `wpo_index` extrapolation ratio ≤ 2 in every seed: **holds** (1.70–1.73).
5. `lno`: test ≥ 1.5× the WPO's **holds** (16×); out-of-band ≥ 3× **holds**; extrapolation ratio > 2.5 in every
   seed **fails** (1.06–1.40: its in-window error is already 0.38).
6. `tfno_r` within 1.2× of `tfno`: **holds**; it is 8 % better with half the parameters.

**Reading.**
- **The physical-wavenumber form is what pays, not having poles.** Free per-index poles give the same in-band
  accuracy and the same time extrapolation (both come from the FOH recursion), but outside the excited band they
  stay near their initialization: the probe there is wrong (0.99), and forcing at unseen modes gives a response
  4× too large (error 188× the test error). With poles in k the unseen modes cost 1.3×.
- The original LNO does not fit forced Burgers at this budget (0.38), as its spatial "poles" have no Fourier-mode
  structure; its low extrapolation ratio reflects a poor fit, not good extrapolation.
- The TFNO's earlier 2× real budget did not help it here: at the WPO's real budget it is slightly better. The
  WPO's lead on forced Burgers (15 %) holds at equal real budgets.

## Real-budget rerun (2026-11-22): 2D coarsening (a38)

`a38_real_budget_2d.py` → `results/a38_real_budget_2d.md` (predictions b77bffe). a31 / a33's TFNO had 27,409 real
parameters (1.46× the WPO-2D's 18,737). `tfno_r`: width 12, rank 0.01, 19,037 real; otherwise a33.

| model | real params | window-end error | rollout, window 1 / 5 / 20 | L(21) / truth | n |
|---|---|---|---|---|---|
| wpo_aug (a33) | 18,737 | 0.0366 | 0.223 / 0.472 / 0.812 | 0.961–0.990 | 0.244–0.262 |
| tfno_aug (a33) | 27,409 | 0.0189 | 0.140 / 0.241 / 0.407 | 0.974–0.976 | 0.255–0.257 |
| tfno_r | 19,037 | 0.0197 (0.0185–0.0218) | 0.140 / 0.239 / 0.462 | 0.981–0.987 | 0.259–0.263 |

**Predictions against results.** 1. window-end below 0.028: **holds** (0.0197). 2. window-20 below `wpo_aug`'s:
**holds** (0.46 against 0.81).

**Reading.** At an equal real budget the TFNO keeps its 1.9× per-window lead; the budget miscount did not create
it.

## Next-level step 3 (2026-11-22): PDEBench 1D Burgers, a public benchmark (a37)

`a37_pdebench_burgers.py` → `results/a37_pdebench_burgers.md` (predictions c12faba; post hoc rerun 1cdc1b5 and
the a37 rerun commit). The first (A) result on data not generated here: PDEBench 1D Burgers, nu = 0.01, initial
field → trajectory on [0, 2], 256 × 41, 9,000 / 1,000 split, PDEBench's nRMSE.
- Checks before any fit: the repository solver through the exact map to [0, 2π) (w = 2π u, nu' = 0.04π)
  reproduces PDEBench to nRMSE 2.9e-4, and the probe on the solver returns nu' exactly.
- Budgets in real degrees of freedom (`lfo.metrics.n_real_params`): WPO 61,130, TFNO 64,745.

**The pre-registered WPO runs ended in nan, all three.** Diagnosis (same configuration, logged): the loss fell
normally while the fastest poles (m up to 63) grew to Re mu ≈ −14,000; at step 6,270 the gradient overflowed. The
cause was a bug in the shared `foh_kernel`: a masked branch evaluated exp(−mu dt), which is inf for Re(mu) dt <
−709, and the masked inf turned the gradient into nan. Fixed with a test (1cdc1b5); no earlier experiment reached
that range. The three WPO runs were repeated unchanged with the fixed kernel (post hoc, marked as such).

| model | nRMSE 256 | nRMSE 1024 (zero-shot) | probe nu' (truth 0.1257) | dispersion error, m = 1..10 |
|---|---|---|---|---|
| WPO (pre-registered) | nan, every seed | | | |
| WPO, rerun after the fix | 0.045 (0.044–0.046) | 0.045 | 0.106 (0.081–0.123) | 0.24 (0.12–0.41) |
| TFNO, real budget | **0.026** (0.025–0.026) | **0.026** | 0.041 (0.035–0.054) | 0.67 (0.56–0.74) |

**Predictions against results** (the WPO numbers are from the post hoc rerun).
1. WPO nu' within 5 % in every seed, and the TFNO's dispersion error ≥ 3× the WPO's: **fails**. nu' is 2.4 %,
   9 % and 35 % off; the error ratio is 2.86.
2. The TFNO more accurate, the WPO within 2×: **holds** (1.76×).
3. Both within 1.3× at 1024 points: **holds** (1.00 for both).

**Reading.**
- On a public benchmark the picture is the one from (A)'s own data: the TFNO is the more accurate surrogate
  (1.8×), and both are resolution-invariant.
- The linear probe separates the models less than on forced Burgers. Per mode, both get the slowest modes
  (m ≤ 4) right; at m = 5–7 both are off; at m = 8–10 two WPO seeds are within 10 % and no TFNO seed is. The
  initial fields of PDEBench excite a broad band but no forcing isolates modes, so the small-amplitude response is
  only partly constrained by the data. The WPO's nu' is right in one seed of three (2.4 %), near in one (9 %).
- The bug fix matters beyond a37: any WPO with very fast poles (fine grids, wide bands) could hit it.

## Next-level step 4 (2026-11-22): odd residues and local mixing against the nonlinear gap (a39)

`a39_odd_local.py` → `results/a39_odd_local.md` (predictions 513e658; `beta_powers` in 4cee292). The hypothesis:
the WPO cannot apply the derivative symbol i k. Its residues are polynomials in k², so every pole layer is even in
k, and the pointwise channel mixes apply no symbol at all; Burgers' convection needs i k at every layer. Two minimal
changes keep the poles as functions of k:
- `wpo_odd`: residues with powers (0, 1, 2, 3) of k / k_ref.
- `wpo_local`: the pointwise mix replaced by a 5-tap periodic convolution over x.

Widths were reduced to the WPO's real budget (P: 17 / 22; F: 17 / 22). Training was the WPO's selected
configuration with 3 seeds.

| part | model | real params | error | other |
|---|---|---|---|---|
| P (PDEBench) | WPO (a37 rerun) | 61,130 | 0.0452 (0.0440–0.0460) | 1024 / 256 1.00; dispersion error 0.24 |
| P | TFNO, real budget (a37) | 64,745 | **0.0257** | 1.00; 0.67 |
| P | `wpo_odd` | 61,568 | 0.0467 (0.0459–0.0475) | 1.00; 0.18 (0.10–0.22) |
| P | `wpo_local` | 63,763 | 0.0539 (0.0526–0.0561) | 1.29 (1.19–1.36); 0.20 |
| F (forced Burgers) | WPO (a11) | 59,529 | **0.0237** | probe outside the band 0.12; unseen / test 1.34 |
| F | TFNO, real budget (a36) | 54,649 | 0.0279 | 0.89; 32 |
| F | `wpo_odd` | 58,083 | 0.0250 (0.0230–0.0262) | 0.115 (0.108–0.124); 1.98 (1.55–2.26) |
| F | `wpo_local` | 57,933 | 0.0246 (0.0239–0.0259) | 0.315 (0.263–0.383); 7.5 (6.8–8.0) |

**Predictions against results.**
1. P: `wpo_odd` ≤ 0.036: **fails** (0.0467, no gain over the WPO).
2. P: `wpo_local` ≤ 0.036: **fails** (0.0539, worse than the WPO).
3. P: neither reaches the TFNO: **holds**.
4. P: `wpo_local`'s 1024 / 256 ratio > 1.3 and `wpo_odd`'s ≤ 1.1: **fails** on the first part (1.29, mean just
   below the threshold; seeds 1.37 / 1.19 / 1.31). The second part holds (1.00).
5. F: `wpo_odd` keeps the form's benefits (unseen ratio ≤ 3, probe outside the band ≤ 0.3, every seed): **holds**.
6. F: `wpo_odd`'s test ≤ the WPO's 0.0237: **fails** (0.0250; within the seed spread).

**Reading.**
- The missing odd symbol is not what separates the WPO from the TFNO. Giving the residues odd powers changes
  neither accuracy (P 0.047 vs 0.045) nor time extrapolation. The derivative is evidently already available to
  the deep model; the pointwise nonlinearity between layers can produce the convective term from even-symbol
  features.
- Odd residues are free in the other respects: the form's benefits stay (unseen modes 2.0× against 1.3×; probe
  outside the band unchanged), and the probe's dispersion error on PDEBench is a little lower and less spread
  (0.18 against 0.24).
- Local mixing costs. Its stencil is a symbol in k dx learned in the band, so it does not continue outside it
  (probe 0.32, unseen modes 7.5×; the TFNO is at 0.89 and 32×), and it loses resolution invariance (1.29×).
- The gap therefore sits elsewhere than in the linear symbol. The remaining candidates are the nonlinear block
  (pointwise mixing at width ~20–30 against the TFNO's full spectral mixing of channels) and the time axis (the
  TFNO's Fourier modes in t mix the whole window; the WPO's FOH recursion is causal).

## Next-level step 4b (2026-11-22): the nonlinear block and the time axis (a40)

`a40_gap_source.py` → `results/a40_gap_source.md` (predictions 547db6c). The neuraloperator FNO differs from the
WPO in two ways beyond its spectral weights:
- **Block.** Each FNO block adds a channel MLP (width → width / 2 → width) and a gated skip of the block input,
  so it has two nonlinearities per layer; the WPO has one.
- **Time axis.** The FNO mixes the whole padded window through 21 temporal modes and gets (x, t) as grid channels;
  the WPO's recursion is causal.

Three variants, each at the WPO's real budget, with an lr sweep at 4000 steps and then 3 seeds × 12,000 steps:
- `wpo_mlp`: the WPO with the FNO block order;
- `tfno_nomlp`: the a37 TFNO without channel MLPs;
- `fno_x`: an FNO over x applied to each time sample, u(·, t) = G(u0, t), with no mixing across time.

| model | real params | nRMSE 256 | 1024 / 256 | probe nu' (truth 0.1257) | dispersion error |
|---|---|---|---|---|---|
| WPO (a37 rerun) | 61,130 | 0.0452 (0.0440–0.0460) | 1.00 | 0.106 | 0.235 |
| `wpo_mlp` | 63,382 | 0.0357 (0.0347–0.0368) | 1.00 | 0.109 (0.095–0.122) | **0.165** (0.12–0.24) |
| TFNO (a37) | 64,745 | **0.0257** (0.0252–0.0264) | 1.00 | 0.041 | 0.672 |
| `tfno_nomlp` | 62,201 | 0.0299 (0.0289–0.0311) | 1.00 | 0.047 | 0.632 |
| `fno_x` | 62,257 | 0.0373 (0.0324–0.0419) | 1.00 | 0.047 | 0.643 |

**Predictions against results.**
1. `wpo_mlp` ≤ 0.036: **holds**, narrowly (0.0357).
2. `tfno_nomlp` ≥ 0.032: **fails** (0.0299). Without the MLP the TFNO loses about a fifth of its edge (0.0042
   of 0.0195), not a third.
3. `fno_x` ≤ 0.030: **fails** (0.0373). Without temporal mixing the FNO loses more than half of its edge.
4. `wpo_mlp` keeps the WPO's properties (1024 / 256 ≤ 1.1, dispersion error ≤ 0.3): **holds** (1.00; 0.165).

**Reading.**
- The gap has two parts, and neither part alone accounts for it.
- **The block.** The FNO block on the WPO closes half of the gap (0.045 → 0.036) and costs nothing. The probe
  improves as well (dispersion error 0.165, the lowest of any model on this benchmark), and resolution invariance
  is kept. On the TFNO side the MLP is worth a smaller share (0.026 → 0.030).
- **The time axis.** Taking temporal mixing out of the FNO costs it more (0.026 → 0.037). The rest of the TFNO's
  edge is therefore consistent with its non-causal mixing over the window, together with t as an input; the WPO's
  recursion is causal.
- This bears on the claim. For initial-value data a causal time axis gives up accuracy that a whole-window method
  can use; what the causal axis buys is time extrapolation and the FOH semantics for forcing (a36: (5, 10]
  extrapolation 1.8× against the TFNO's 10×).
- Not tested here: whether `wpo_mlp` keeps the form's benefits on forced Burgers (out of band, unseen modes), and
  whether the two parts add up.

## Next-level step 4c (2026-11-22): the FNO block on forced Burgers (a41)

`a41_block_forced.py` → `results/a41_block_forced.md` (predictions 0362d7a). This tests whether a40's `wpo_mlp`
keeps the form's benefits. Setup as a36: width 23, 57,011 real parameters; the WPO's configuration; 3 seeds.

| model | real params | test | (5, 10] / test | probe outside the band | unseen / test |
|---|---|---|---|---|---|
| WPO (a11) | 59,529 | **0.0237** (0.0230–0.0244) | 1.83 | **0.116** (0.091–0.142) | 1.34 (1.11–1.71) |
| TFNO, real budget (a36) | 54,649 | 0.0279 | 10.3 | 0.887 | 32 |
| `wpo_mlp` | 57,011 | 0.0242 (0.0223–0.0265) | 2.02 (1.93–2.09) | 0.386 (0.139–0.837) | 2.16 (1.81–2.36) |

**Predictions against results.**
1. Mean test ≤ 0.0215: **fails**. At 0.0242 the error is the WPO's; the block's PDEBench gain does not carry over.
2. In every seed, unseen-mode ratio ≤ 3 and probe outside the band ≤ 0.3: **fails**. The unseen-mode part holds
   in every seed (≤ 2.36). The probe part fails in seed 0 (0.84).
3. Mean (5, 10] / test ≤ 2.5: **holds** (2.02).

**Reading.**
- **Seed 0's probe.** It is right to k = 11 (≤ 18 %) and then diverges: 31 % at k = 12, 88 % at k = 14 and 3× at
  k = 16. The other two seeds stay within 35 %. The extra nonlinear capacity lets the polynomial pole laws drift
  where the data put no constraint. That was a stated risk; per-index poles (a36) fail the same way, though more.
- **No accuracy gain here.** On forced Burgers the WPO was already ahead of the TFNO, and the block adds nothing.
  Its gain on PDEBench is specific to the initial-value task.
- **Decision.** The default deep block stays the WPO's. `wpo_mlp` remains an experiment-level variant, not
  promoted to `lfo.models`. The conditional reruns (2D coarsening, D2) are not run.

## Five seeds with intervals for the headline claims (2026-11-22, a42)

`a42_five_seeds.py` → `results/a42_five_seeds.md` (predictions cf1822b). This adds seeds 3 and 4, otherwise
unchanged, to a36's form comparison and a37's PDEBench comparison. Reported: mean ± sd [95 % t-interval] over 5
seeds.

| setup | model | test | probe outside the band | unseen / test |
|---|---|---|---|---|
| F | WPO, poles in k | 0.0239 ± 0.0006 [0.0232, 0.0245] | **0.098** [0.058, 0.139] | **1.26** [0.92, 1.61] |
| F | WPO, poles per index | 0.0230 ± 0.0008 [0.0219, 0.0240] | 0.980 [0.948, 1.012] | 181 [106, 255] |
| F | TFNO, real budget | 0.0287 ± 0.0020 [0.0262, 0.0311] | 0.874 [0.779, 0.969] | 30.5 [18.7, 42.3] |

| setup | model | nRMSE | probe nu' | nu' within 5 % | dispersion error |
|---|---|---|---|---|---|
| P | WPO | 0.0450 ± 0.0008 [0.0440, 0.0459] | 0.101 [0.079, 0.124] | 1 of 5 | 0.25 [0.10, 0.39] |
| P | TFNO, real budget | **0.0254** ± 0.0006 [0.0247, 0.0261] | 0.040 [0.030, 0.051] | 0 of 5 | 0.69 [0.59, 0.79] |

Time extrapolation, (5, 10] / test: WPO 1.85 [1.76, 1.95], per index 1.75 [1.67, 1.83], TFNO 10.0 [9.2, 10.9].

**Predictions against results: all five hold.**
1. The unseen-mode intervals separate (WPO's upper bound 1.61; the others' lower bounds 106 and 18.7).
2. The out-of-band probe intervals separate.
3. The in-band test intervals of the two WPO forms overlap. The per-index form is marginally better, by 4 %.
4. The TFNO's nRMSE interval lies below the WPO's; ratio of means 1.77.
5. The WPO's nu' is within 5 % in 1 of 5 seeds.

**Reading.** The three-seed headline numbers stand, with intervals:
- the form separates out-of-band behavior by an order of magnitude or more;
- the PDEBench gap is 1.77×, with tight intervals;
- the WPO's probe ν' on PDEBench is not reliable seed by seed (0.079–0.124 interval).

## Cost per model (2026-11-22, a43)

`a43_cost.py` → `results/a43_cost.md` (predictions 34352ab). Each model was measured alone on the RTX 4090:
- training step: forward, loss, backward and Adam at batch 25, median of 5 × 20 steps after warm-up;
- inference: per sample at batch 50;
- peak training memory.

| setup | model | dtype | train step (ms) | × TFNO | inference (ms / sample) | peak memory (MB) |
|---|---|---|---|---|---|---|
| F | WPO | float64 | 131 | 4.66 | 1.67 | 2197 |
| F | WPO per index | float64 | 128 | 4.55 | 1.66 | 2197 |
| F | TFNO | float32 | 28 | 1.00 | 0.46 | 1277 |
| F | LNO v2 | float32 | 27 | 0.97 | 0.35 | 2238 |
| P | WPO | float64 | 79 | 4.36 | 1.07 | 1570 |
| P | `wpo_mlp` | float64 | 98 | 5.44 | 1.32 | 1931 |
| P | TFNO | float32 | 18 | 1.00 | 0.29 | 975 |
| P | `fno_x` | float32 | 19 | 1.06 | 0.14 | 822 |

**Predictions against results.**
1. P: WPO step 1.3–3× the TFNO's: **fails** (4.36×).
2. F: ≥ 3×: **holds** (4.66×).
3. WPO peak memory ≥ the TFNO's in both setups: **holds** (1.7× and 1.6×).

**Reading.**
- The WPO costs 4–5× the TFNO per training step and 3.6× per inference sample.
- The comparison mixes precisions. The WPO runs in float64 (the FOH recursion and the pole fits), and the 4090's
  float64 throughput is a small fraction of its float32 throughput. Part of the factor is therefore precision,
  not the method; a float32 WPO was not measured.
- The pole form itself costs nothing: per-index poles cost the same as poles in k.
- Correction to the a43 docstring: the LNO port runs in float32, not float64.

## The WPO in float32: cost and accuracy (2026-11-23, a44)

`a44_float32.py` → `results/a44_float32.md` (predictions a0edd34). `lfo.models` gained a float32 path for the 1D
deep WPO (379f1b4): activations, pointwise layers and the pole layers' channel-sized work in float32 / complex64;
pole and residue functions, FOH coefficients and pole parameters stay float64 (same real budget). The default
float64 path is bit-identical.

**Cost** (alone on the RTX 4090, a43's procedure; with the fixed layer, below):

| setup | model | train step (ms) | × TFNO | × WPO float64 | inference (ms / sample) | peak memory (MB) |
|---|---|---|---|---|---|---|
| F | WPO float64 | 152 | 4.96 | 1.00 | 2.04 | 2197 |
| F | WPO float32 | 32 | 1.05 | 0.21 | 0.49 | 1112 |
| F | TFNO | 31 | 1.00 | 0.20 | 0.53 | 1277 |
| P | WPO float64 | 88 | 4.46 | 1.00 | 1.23 | 1570 |
| P | WPO float32 | 26 | 1.31 | 0.29 | 0.30 | 815 |
| P | TFNO | 20 | 1.00 | 0.22 | 0.33 | 975 |

The pre-registered measurement (before the fix) gave the same ratios: F 1.00×, P 1.31× the TFNO; 0.22× / 0.32×
the float64 WPO. Absolute times of the second session are about 10 % higher for every model.

**Accuracy** (5 seeds, 95 % t-intervals; float64 rows from a42):

| setup | model | test / nRMSE | unseen / test | probe outside the band / ν' |
|---|---|---|---|---|
| F | WPO float64 | 0.0239 [0.0232, 0.0245] | 1.26 [0.92, 1.61] | 0.098 [0.058, 0.139] |
| F | WPO float32 | 0.0238 [0.0233, 0.0243] | 1.33 [1.00, 1.67] | 0.121 [0.028, 0.213] |
| F | per index / TFNO (a42) | 0.0230 / 0.0287 | 181 [106, 255] / 30.5 [18.7, 42.3] | 0.98 / 0.87 |
| P | WPO float64 | 0.0450 [0.0440, 0.0459] | – | ν' 0.101 |
| P | WPO float32, pre-registered | 0.774 (3 of 5 seeds at 1.0) | – | nan |
| P | WPO float32, rerun with the fixed layer (post hoc) | 0.0459 [0.0439, 0.0479] | – | ν' 0.091 |

**A defect found on the way (post hoc).** The pre-registered P runs failed. At the same trained weights the
float32 forward differed from float64 by 144 %, although every FFT, kernel and contraction agreed to 1e-7. Cause:
complex residues give the k = 0 bin an imaginary part, which irfft must ignore. CPU and float64 cuFFT do, but
complex64 cuFFT C2R does not for large batches (torch 2.6, CUDA 12.4): 6 % error on random spectra at P's shapes
(n_x = 256 from 64 bins), none at F's (n_x = 64 from 22 bins), so the F runs are valid. The layer now drops that
imaginary part explicitly (9d48999), with a regression test that fails without the fix. The float64 path is
bit-identical, so no earlier result changes.

**Predictions against results.**
1. float32 step ≤ 0.5× the float64 WPO's in both setups: **holds** (0.22, 0.32; rerun 0.21, 0.29).
2. float32 step still ≥ 1.3× the TFNO's in both: **fails** on F (1.00; rerun 1.05), holds on P (1.31).
3. float32 peak memory ≤ 0.7× float64's: **holds** (0.51, 0.52).
4. F: form claim kept, test within 5 %: **holds** (unseen 1.33 [1.00, 1.67], disjoint from 106–255 and 18.7–42.3;
   test 0.0238 against 0.0239).
5. P: nRMSE within 5 % of float64's: **fails** as registered (0.774, the cuFFT defect). Post hoc, with the fixed
   layer: 0.0459, 2 % above.

**Reading.**
- a43's 4.4–5× cost factor is precision, not method. In float32 the WPO matches the TFNO's training step on
  forced Burgers (201 time samples) and takes 1.3× on PDEBench, with less memory than the TFNO in both.
- float32 keeps the results: the form claim on F and the PDEBench accuracy (post hoc) are unchanged within the
  intervals.
- Mixed-precision code on the GPU needs a Hermitian check before irfft; a unit test at initialization (real
  residues) did not catch it.

## Missing baselines: SSM, LNO, ALNO, LFNO at the WPO's budget (2026-11-23, a45)

`a45_baselines.py` → `results/a45_baselines.md` (predictions 30d44a7; S4D layer vendored in 40fd989). Four
operators of the related work, each at the WPO's real budget (54.8–64.5 k real against 59.5 k / 61.1 k), on a36's
forced Burgers (F) and a37's PDEBench 1D Burgers (P), 5 seeds, 95 % t-intervals:
- `s4d`: per layer GELU(S4D along t (the authors' layer, `lfo.models.s4d`) + SpectralConv along x + 1×1);
- `lno`: the original LNO v2 (F seeds 0–2 from a36, 3–4 added; P new);
- `alno`, `lfno`: `lfo.models.ALNO2d` / `LFNO2d` (separable poles in x and t, textbook contraction).

Sweep (seed 0, 4000 steps, validation): lr 1e-3 / 3e-3 / 1e-2, pole lr ×1 / ×30. Selected: s4d 3e-3 (F) and 1e-2
(P); alno 1e-2 ×1 (F, P); lfno 1e-2 ×1 (F) and ×30 (P); lno 1e-3 (P). Final 12000 steps. 3.3 GPU-hours.

| F | test | (5, 10] / test | probe outside the band | unseen / test |
|---|---|---|---|---|
| WPO (a42) | **0.0239** [0.0232, 0.0245] | **1.85** [1.76, 1.95] | **0.098** [0.058, 0.139] | **1.26** [0.92, 1.61] |
| S4D-type SSM | 0.0390 [0.0359, 0.0422] | 2.61 [2.11, 3.11] | 1.00 [0.90, 1.10] | 29.4 [23.9, 34.8] |
| ALNO | 0.0773 [0.0704, 0.0842] | 4.14 [3.58, 4.70] | > 10 in every seed | 114 [−29, 258] |
| LFNO | 0.0945 [0.0824, 0.1067] | 166 [−187, 519] | > 10 in every seed | 9.2 [7.9, 10.6] |
| LNO v2 | 0.325 [0.214, 0.435] | 1.35 [0.92, 1.77] | 1.08 [0.96, 1.19] | 3.6 [2.0, 5.3] |

| P | nRMSE 256 (= 1024) | probe ν' (truth 0.01 → 0.1257 m²) | ν' within 5 % | dispersion error |
|---|---|---|---|---|
| TFNO, real budget (a42) | **0.0254** [0.0247, 0.0261] | 0.040 | 0 of 5 | 0.69 |
| WPO (a42) | 0.0450 [0.0440, 0.0459] | 0.101 | 1 of 5 | **0.25** |
| S4D-type SSM | 0.0513 [0.0485, 0.0541] | 0.039 | 0 of 5 | 0.71 |
| ALNO | 0.0879 [0.0754, 0.1005] | 0.030 | 0 of 5 | 0.80 |
| LFNO | 0.275 [−0.161, 0.710] | 0.014 | 0 of 5 | 0.90 |
| LNO v2 | 0.878 [0.876, 0.880] | 0.012 | 0 of 5 | 0.92 |

**Predictions against results: three of six hold.**
1. Failed. Unseen / test intervals above the WPO's: S4D, LFNO and LNO yes; ALNO's mean is 114 but its interval
   reaches below the WPO's upper bound (seeds 12.7–252).
2. Held. Every baseline's mean out-of-band probe error ≥ 0.35 (S4D 1.00, LNO 1.08, ALNO and LFNO > 10).
3. Failed. S4D test within 0.8–1.6× the WPO's: it is 1.63×. LNO (13.6×), ALNO (3.2×), LFNO (4.0×) ≥ 1.5× held.
4. Held. S4D (5, 10] / test ≤ 3 (2.61); ALNO (4.14) and LFNO (166) > 3.
5. Failed. S4D nRMSE below the WPO's 0.0450: it is 0.0513, above. LNO, ALNO and LFNO above the WPO held.
6. Held. No baseline has ν' within 5 % in more than 2 of 5 seeds (all 0 of 5).

**Post hoc context** (not pre-registered). The ALNO and LFNO probe means (≈ 3e13) come from fits that find no
meaningful out-of-band pole in any seed; the medians are 6e12 and 3e13. Medians of unseen / test: S4D 27.5, ALNO
66.5, LFNO 9.6. LFNO's (5, 10] mean is driven by two seeds (670 and 81; median 69). LFNO P seed 4 (0.90) did not
train; without it the LFNO nRMSE is 0.118. Several rows selected an edge of the lr grid; a46 extends it.

**Reading.**
- The failed predictions were optimistic about the SSM: at equal budget the WPO is 1.6× more accurate than the
  S4D-type operator on forced Burgers and 1.14× on PDEBench, and extrapolates better in time.
- Only poles in the physical wavenumber continue outside the forced band: every per-mode or per-index generator
  (S4D, the WPO's own per-index form, ALNO, LFNO, LNO) has an out-of-band probe error of order 1 or worse, and an
  unseen-mode error 3.6–181× its test error (means), against 1.3× for the WPO.
- The pole operators of the related work (LNO, ALNO, LFNO) do not reach the FNO / SSM accuracy level at this
  budget; the original LNO does not train on PDEBench (nRMSE 0.88).
- The TFNO remains the most accurate model on PDEBench; the WPO is second.

## The lr sweep beyond a45's edges (2026-11-23, a46)

`a46_lr_edges.py` → `results/a46_lr_edges.md` (predictions 8d0a23c). a45 selected the upper edge of its lr grid
(1e-2) for F:alno, F:lfno, P:s4d, P:alno and P:lfno, and the lower edge (1e-3) for P:lno. Added at a45's sweep
protocol (4000 steps, seed 0): lr 3e-2 and 1e-1 (pole lr ×1 / ×30 for ALNO and LFNO), and 3e-4 and 1e-4 for P:lno.

| job | a45 best (vali) | best new point (vali) | new points diverged or at ≈ 1 |
|---|---|---|---|
| F:alno | 1e-2, ×1 (0.0843) | 3e-2, ×1 (0.0901) | 3 of 4 |
| F:lfno | 1e-2, ×1 (0.0952) | 3e-2, ×1 (0.1236) | 3 of 4 |
| P:s4d | 1e-2 (0.0672) | 3e-2 (0.1036) | 1 of 2 |
| P:alno | 1e-2, ×1 (0.1227) | 3e-2, ×30 (0.1785) | 2 of 4 |
| P:lfno | 1e-2, ×30 (0.1400) | 3e-2, ×1 (0.1543) | 3 of 4 |
| P:lno | 1e-3 (0.8877) | 3e-4 (0.8908) | – (all ≈ 0.89–0.91) |

No selection changed, so no final run was repeated; a46's tables equal a45's. 0.8 GPU-hours.

**Predictions against results: five of six hold.**
1. Failed. P:s4d below a45's 0.0513: the selection did not change, so the nRMSE is a45's. Its validation fell
   up to 1e-2 and rises beyond it.
2. Held. P:s4d at or above the WPO's 0.0450 (0.0513).
3. Held. F:alno and F:lfno test ≥ 1.5× the WPO's (0.077, 0.094).
4. Held. F:alno and F:lfno unseen / test > 3 (114, 9.2).
5. Held. P:lno > 0.5 at every lr (validation 0.89–0.91).
6. Held. P:alno and P:lfno above the WPO's 0.0450 (0.088, 0.275).

**Reading.** The a45 baselines sit at a validation minimum of the extended grid, not at a truncated edge; the
comparison in a45 stands as reported.

## A dispersive system: forced KdV–Burgers (2026-11-24, a47)

`a47_kdv_burgers.py` → `results/a47_kdv_burgers.md` (checks and predictions c8bd6de; solver term 35493e4).
u_t + u u_x = ν u_xx − δ u_xxx + f, ν = 0.1, δ = 0.01: λ(k) = −νk² + iδk³, complex poles. a36's forced protocol
unchanged otherwise; WPO, per-index WPO, real-budget TFNO and a45's S4D-type operator, 5 seeds each.
Run time 13 h summed over runs (WPO-type runs in float64; two processes shared the GPU, about 8 h wall time).

Checks before training:
- nonlinear part 0.8 / 8.5 / 49 % (p10 / p50 / p90), up to 174 %;
- solver probe ν̂ 0.5 %, δ̂ 0.85 %, out-of-band error 0.024.

| model | selected | test | (5, 10] / test | unseen / test | probe outside the band | ν̂ | δ̂ | ν̂ 5 % / δ̂ 10 % |
|---|---|---|---|---|---|---|---|---|
| WPO | 3e-3, ×30 | 0.0245 [0.0241, 0.0250] | **1.83** [1.73, 1.93] | **2.10** [1.08, 3.12] | **0.208** [0.041, 0.376] | 0.1004 | 0.01030 | 5 / 5 |
| per index | 3e-3, ×30 | **0.0227** [0.0222, 0.0232] | **1.80** [1.68, 1.93] | 364 [−48, 776] | 0.988 [0.970, 1.005] | 0.1011 | 0.01012 | 5 / 5 |
| TFNO, real | 1e-2 | 0.0338 [0.0283, 0.0393] | 8.81 [7.45, 10.18] | 19.6 [14.2, 25.1] | 0.870 [0.808, 0.931] | 0.0985 | 0.00839 | 4 / 3 |
| S4D-type | 1e-3 | 0.0404 [0.0384, 0.0424] | 2.88 [2.54, 3.22] | 37.6 [21.6, 53.6] | ≥ 0.99 (2 seeds fail) | 0.0976 | 0.00795 | 4 / 2 |

Per seed, unseen / test: WPO 1.4–3.3; per index 168–946; TFNO 15–27; S4D 27–59. Out-of-band error per seed for the
WPO: 0.09, 0.12, 0.14, 0.26, 0.42 (median 0.14).

**Predictions against results: four of six hold.**
1. Held. The WPO's test (0.0245) is below the TFNO's (0.0338) and the S4D operator's (0.0404).
2. Held. The per-index test is −7 % from the WPO's (within ±15 %).
3. Failed. Unseen / test: the WPO's interval [1.08, 3.12] lies below the TFNO's and the S4D operator's, but the
   per-index interval reaches below zero (one seed at 946). Per seed every other model is above every WPO seed.
4. Failed. The WPO's mean out-of-band error is 0.208 (predicted ≤ 0.2); the others' ≥ 0.35 held. Seed 0 (0.42)
   overshoots Im λ at k ≥ 14 (fitted 36i and 72i against 27i and 34i at k = 14, 15).
5. Held. The WPO's ν̂ within 5 % and δ̂ within 10 % in 5 of 5 seeds (δ̂ within 1.9–3.8 %).
6. Held. (5, 10] / test: WPO 1.83 ≤ 2.5; TFNO 8.8 > 5.

**Post hoc.** The pre-set in-band dispersion error (`disp_err`) is ≈ 0.9–1.0 for most runs because the kept-mode cut
(|μ| dt ≤ 0.25, |μ| T ≥ 2) admits slow spurious fits at unexcited modes k ≥ 9; on Burgers those fits were fast and
cut. Restricted to k = 2..8 it is 0.015 / 0.018 / 0.016 / 0.018 (WPO / per index / TFNO / S4D). ν̂ and δ̂ are
residual-weighted and not affected. The TFNO selected the upper edge of its lr grid (1e-2).

**Reading.**
- The form claim carries over to complex poles: in band every model's linearization is right; only poles in k
  keep unseen modes near the test error, and the WPO recovers the dispersion coefficient in every seed.
- The continuation outside the band is shorter than on Burgers (0.21 against 0.10): a cubic Im μ fitted on
  k ≤ 8 drifts by k = 14–16.
- t-intervals are a poor separation test for ratios with heavy upper tails (this and a45's ALNO); per-seed ranges
  are reported alongside.

## The TFNO's lr beyond a47's edge (2026-11-24, a48)

`a48_kdv_tfno_edge.py` → `results/a48_kdv_tfno_edge.md` (predictions f7185f8). a47's TFNO selected the upper
edge of its grid (1e-2, validation 0.0292). Added at the sweep protocol: lr 3e-2 → 0.0333, 1e-1 → 1.000 (no
training). The selection stays 1e-2, so a47's TFNO runs stand; a48's tables equal a47's. Four minutes.

**Predictions against results: three of three hold.**
1. Held. The selection does not change.
2. Held. The TFNO's test (0.0338) stays above the WPO's (0.0245).
3. Held. Unseen / test 19.6 > 5; probe outside the band 0.87 ≥ 0.35.
