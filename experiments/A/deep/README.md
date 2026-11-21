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
  extrapolation blow-up is gone (a9: errors of 3–8 over (5, 10]). The stable parameterisation helps further:
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

- The deep model's linearisation is right to about 1 % for k ≤ 6. Over k = 7–8, the top of the forced band, it
  is 5–12 % too damped.
- The "ν̂ bias" is an artefact of summarising that curve with {1, k²}. The fit absorbs the curvature into a
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
4. Train error ≥ 0.8× test error for both: **fails.** Both models now show a generalisation gap (WPO 0.47×, FNO
   0.59×) with 200 training samples.

**Flaw in the FNO configuration (found after the run).** In neuralop 2.0, `n_modes = (12, 12)` keeps 12 spatial
modes in total (|k| ≤ 6) and 12 // 2 + 1 = 7 temporal frequencies (weight shape (w, w, 12, 7)). The forcing band is
k ≤ 8, and steps and PRBS need many more temporal frequencies. The FNO's error floor of about 0.1 and its broken
probe at k ≥ 7 (Re μ̂ from −23 to −1500) follow from this truncation. The setting was chosen in a9 to match the
parameter budget, without checking the band. The accuracy comparison in a9–a11 is therefore not fair to the FNO.
The WPO-only findings stand: the pole learning rate, the probe and extrapolation without blow-up.

**Next.** A band-covering FNO. Full modes do not fit a 59 k budget (width 4 with 32 × 32 modes already has 135 k
parameters). So the comparison uses the Tucker-factorised TFNO from neuraloperator, with n_modes covering the band,
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

- **Inside the forced band (k ≤ 8), every model's linearisation is right to 1–2 %.** The FNO-type models learn the
  physical linearisation there just as the WPO does. The registered dispersion error separates them only because
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
right. Out of band, only the WPO's is, together with an LTI-consistent linearisation.

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
  near-integrators. The probe of the whole operator cannot tell the two apart, and an unstable linearisation
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
and its per-mode linearisation is an order of magnitude more accurate.

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
  (a16, a17, a29), but about the separated phases ±1 the linearisation is stable
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
- What is missing is saturation. The true window map pulls an overshoot back towards ±1 (gain 1/a), and the TFNO
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

