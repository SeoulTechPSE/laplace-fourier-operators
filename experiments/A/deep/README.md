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
