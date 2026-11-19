# (A) Application 1: process dynamics with time delay

Plan: `docs/proposal_A_wavenumber_pole_operator.qmd`, Section 5. Shared code: `lfo.models.PoleResidueDelay`
(`lfo/models/delay.py`). It implements K(s) = Σ β_n e^{−θ_n s} / (s − μ_n) with an exact delayed first-order hold:
the undelayed FOH state plus a partial FOH step to t_j − θ. This is exact for piecewise-linear inputs and
differentiable in θ, including delays that are not multiples of the sample time. Tests: zero and integer delays,
the FOPDT step response at a fractional delay, and the gradient against finite differences.

## First step (2026-11-19): FOPDT and SOPDT

`a20_fopdt.py` → `results/a20_fopdt.md` (predictions 1989e24).

**Setup.**
- Systems: FOPDT (K = 2, τ = 1.5, θ = 1.37) and SOPDT (K = 1, τ = 2.0 / 0.5, θ = 0.83).
- Data: dt = 0.05 and T = 20. Inputs are E1-family signals. The truth is the exact undelayed response on a 32×
  finer grid, shifted by θ. No noise; data seeds 0–4.
- Oracle floor: about 1 %, from steps and PRBS jumps between samples.

**Models.**
- `pdelay`: the delay layer at the true order (10 starts, L-BFGS, selection by validation).
- `pole_only`: 4 complex poles without delay (the LNO-type representation).
- `arx`: ARX with na = order and nb = order + 1, and the integer delay nk chosen by validation.

| system | model | test / oracle | θ̂ (truth) | K̂ | τ̂ |
|---|---|---|---|---|---|
| FOPDT | pdelay | **1.00** | **1.3701** (1.37) | **2.0000** | **1.4998–1.4999** |
| FOPDT | pole only | 1.57–1.95 | – | – | – |
| FOPDT | ARX | 1.12–1.18 | 1.35 (one sample off) | 2.002–2.003 | 1.514–1.519 |
| SOPDT | pdelay | **1.00** | 0.823–0.839 (0.83) | **1.0000** | **2.0000; 0.4996–0.4999** |
| SOPDT | pole only | 1.06–1.09 | – | – | – |
| SOPDT | ARX | 1.08–1.12 | 0.85 | 0.9999–1.0003 | 1.998–2.004; 0.499–0.501 |

**Predictions against results.**

1. `pdelay`: θ, K and every τ within 1 % in every seed: **fails narrowly.**
   - FOPDT is exact to 1e-4.
   - SOPDT's θ is off by 0.04–1.1 % (seed 0 at 1.1 %), while K and τ are within 0.08 %.
   - The test error is at the oracle in every seed.
2. `pole_only` ≥ 5× the oracle: **fails.** It reaches 1.06–1.95×. Four complex poles approximate the delay (as a
   Padé approximant would) to within about the 1 % sampling floor.
3. ARX: θ̂ within one sample, with a test error above `pdelay`'s: **holds.**

**Reading.**

- **Only the delay layer gives θ as a continuous, accurate parameter.** ARX resolves θ to the sample (0.05), and
  its time constants are 1 % off on FOPDT. The pole-only model fits nearly as well but has no delay to read.
  Test error again does not certify structure, as in (C).
- **SOPDT's θ is the least identifiable parameter.** A fast lag (τ₂ = 0.5) and a delay are partly
  interchangeable at the output, so θ scatters by ~1 % while τ₂ is right to 0.08 %.
- The sampling floor (~1 %) hides the pole-only model's structural error here. With smooth inputs only, or noise
  added, the comparison would change. Next steps: noise, then the CSTR (linear-probe poles) and the
  axial-dispersion reactor (delay as −i v k).

## Output noise (2026-11-19)

`a21_noise.py` → `results/a21_noise.md` (predictions 9ab74fe). White noise (1 % or 5 % of each signal's rms) is
added to the training and validation outputs; the test set is clean. Setup and fits are a20's.

| system | noise | model | test / oracle | θ error (max) | K error | τ error (max) |
|---|---|---|---|---|---|---|
| FOPDT | 1 % | delay layer | 1.01 | 0.06 % | 0.03 % | 0.13 % |
| FOPDT | 1 % | ARX | 1.46–1.90 | 2.2 % | 0.9 % | 5.5 % |
| FOPDT | 5 % | delay layer | 1.03–1.05 | 0.08 % | 0.10 % | 0.19 % |
| FOPDT | 5 % | ARX | 15–22 | 17 % | 9.5 % | 64 % |
| SOPDT | 1 % | delay layer | 1.01–1.03 | 3.1 % | 0.07 % | 0.64 % |
| SOPDT | 1 % | ARX | 7.5–11 | 33 % | 10 % | 87 % |
| SOPDT | 5 % | delay layer | 1.03–1.08 | 3.3 % | 0.19 % | 1.3 % |
| SOPDT | 5 % | ARX | 6.8–11 | 57 % | 5.5 % | 86 % |

**Predictions against results.**

1. Delay layer within 2 % at 1 % noise and within 5 % at 5 % noise: **fails narrowly.** SOPDT's θ is off by 3.1 %
   at 1 % noise; everything else is within bounds, and FOPDT is within 0.2 % even at 5 % noise.
2. ARX: some τ more than 5 % off at 5 % noise in ≥ 4 of 5 seeds: **holds** (52–86 % in every seed).

**Reading.**

- The delay layer is fitted by output error, simulating the model against the data, so noise only adds variance.
  At 5 % noise it stays at the oracle (≤ 1.08×).
- ARX is fitted by equation error, so noisy past outputs bias it. On SOPDT it breaks down already at 1 % noise.
- SOPDT's θ remains the least identifiable parameter (θ–τ₂ trade-off, as in a20). It scatters by about 3 % under
  noise while τ₁ and τ₂ hold to 1.3 %.

## Lumped delay against distributed dispersion (2026-11-19)

`a24_dispersion_reactor.py` → `results/a24_dispersion_reactor.md` (predictions 5c4d793).

**Setup.**
- A periodic analogue of a dispersion reactor: v = 0.5, D = 0.02, k_r = 0.1.
- A Gaussian source, with sensors at distances L = 1, 2, 3 (transport times 2, 4, 6).
- Lumped FOPDT and SOPDT delay layers are fitted to each sensor signal. The wavenumber layer is fitted to the
  whole field.

| L | L / v | t̄ (truth) | FOPDT θ / θ + τ | SOPDT θ / θ + τ₁ + τ₂ | lumped test error (FOPDT / SOPDT) |
|---|---|---|---|---|---|
| 1 | 2 | 2.230 | 1.28 / 2.30 | 1.01 / 2.20 | 0.10 / 0.056 |
| 2 | 4 | 4.077 | 2.99 / 4.42 | 2.65 / 4.24 | 0.14 / 0.081 |
| 3 | 6 | 6.044 | 4.68 / 6.73 | 4.30 / 6.33 | 0.17 / 0.10 |

The wavenumber layer on the field gives v = 0.50001, D = 0.01999, k_r = 0.10000, at a test error of 1.0 % (the
sampling floor).

**Predictions against results.**

1. SOPDT mean delay within 5 % of t̄ at every L, FOPDT within 10 %: **fails** for FOPDT at L = 3 (11 %). SOPDT holds
   (1.3, 3.9, 4.7 %).
2. Slope of the SOPDT mean delay against L within 5 % of t̄'s slope (1.907): **fails**. It is 2.063, 8 % above t̄'s
   slope and 3 % above 1 / v. The lumped fit extrapolates the tail that the window truncates, so it lands
   closer to the true transport. The wavenumber layer's v is within 0.002 %: **holds**.
3. Lumped θ below L / v at every L, for both models: **holds**.

**Reading.**
- A lumped θ is not a physical constant. It is the dead time of an approximation: smaller than L / v, because
  dispersion brings the first arrival forward, and growing faster than linearly in L.
- The lumped models also leave 5–17 % of the sensor signal unexplained, and more the further the sensor is. One
  dispersion relation, λ(k) = −ivk − Dk² − k_r, explains every sensor at once, with the transport speed as an
  exact parameter.
- This is the plan's Section 5.2 link: in a distributed system the delay *is* the imaginary part of λ(k), and the
  wavenumber pole layer reads it directly.

## Delay and dispersion together: the wavenumber pole layer with an input delay (2026-11-19)

`a25_wpo_delay.py` → `results/a25_wpo_delay.md` (predictions 45eddc5). Layer option `delay=` (756e41b): one shared
input delay, applied exactly per mode with `delay_shift`. System: a1's ADR with a delayed input f(x, t − 0.73).
5 data seeds; 10 starts each.

| model | test / oracle | θ̂ (0.73) | D̂ (0.05) | v̂ (0.5) | k̂_r (0.2) |
|---|---|---|---|---|---|
| WPO + delay (N = 1) | **1.00** | **0.7298–0.7301** | **0.0500** | **0.4999–0.5001** | **0.1998–0.2001** |
| WPO, N = 1, no delay | 38–44 | – | – | – | – |
| WPO, N = 3, no delay | 14–21 | – | – | – | – |

**Predictions against results: both hold.**
1. The delayed layer is at the oracle with θ, D, v within 0.04 % (k_r within 0.0002), in every seed.
2. Without a delay, the error is 14–44× the oracle in every seed. Two extra poles per mode do not replace the
   delay at this precision.

**Reading.** Delay and dispersion are different structures. Transport inside the domain is −ivk in λ(k) (a24). An
input or measurement dead time is a factor e^{−θs} common to all modes. One layer with both reads both exactly.

## Nonlinear CSTR: probe poles at a steady state (2026-11-19)

`a23_cstr.py` → `results/a23_cstr.md` (predictions f224ba0).

**Setup.**
- Exothermic CSTR (Da 0.072, B 8, β 0.3, γ 20) at its upper steady state. The Jacobian pair is −0.7694 ± 0.9597 i.
- Input: coolant-temperature deviations; output: reactor temperature.
- 7 % of the median sample is not linear, and up to 86 % of the largest ones, where extinction events occur.
- Models:
  - WPO on time only (n_x = 1, 18.7 k parameters);
  - FNO-1D (17.7 k);
  - the best linear model (one complex pair).
- The probe is a small-amplitude odd response; the solver's own probe is within 0.3 %.

| model | test | probe pole | probe error | probe residual |
|---|---|---|---|---|
| WPO | **0.016–0.029** | −0.763 to −0.775 ± 0.956–0.961 i | **0.1–0.6 %** | **1–2e-3** |
| FNO | 0.041–0.056 | −0.770 to −0.781 ± 0.964–0.966 i | 0.4–1.0 % | 6–8e-3 |
| best linear model | 0.15 | −0.764 ± 0.964 i | 0.5 % | – |

**Predictions against results.**
1. WPO probe within 5 % in ≥ 2 of 3 seeds: **holds** (3 of 3, within 0.6 %).
2. The best linear model's pole more than 10 % off: **fails**. It is within 0.5 %. Most samples are near-linear and
   the nonlinearity is not symmetric, so it raises the linear model's error (0.15) without biasing its pole. On
   Burgers, by contrast, the nonlinearity added uniform damping (a12b).
3. WPO mean test error ≤ FNO's: **holds** (0.021 against 0.046).

**Reading.** At a steady state with a stable focus, every reading of the linearisation agrees within 1 %: WPO probe,
FNO probe and the best linear fit. That matches Burgers in band. What separates the models is accuracy on the
nonlinear samples (WPO 2.2× lower error) and how LTI the small-signal response is (residual 4× lower).

## A reactor with real inlet / outlet boundaries (Danckwerts), inlet → outlet (2026-11-19)

`a27_danckwerts.py` → `results/a27_danckwerts.md` (predictions 0bcea43).

**Setup.**
- v = 1, D = 0.05 (Péclet 20), k_r = 0.5, with Danckwerts boundaries. The output is the outlet concentration.
- Truth: finite differences on 400 cells with Crank–Nicolson. Reference eigenvalues: −5.85, −6.91, −8.75, −11.45, …
  (all real).
- Models, all fitted by output error with 10 starts:
  - FOPDT and SOPDT;
  - 6 free real poles;
  - a modal model, λ_n = −(a₀ + a₁ n²) over n = 1..8 with free residues.

| model | test | slowest poles | other |
|---|---|---|---|
| FOPDT | 0.042 | −2.82 | θ 0.64 |
| **SOPDT** | **0.020** | −4.40, −4.68 | θ 0.53 |
| 6 free poles | 0.042 | −1.23, −1.65, −1.86 | – |
| modal, 8 modes | 0.032 | −4.71, −4.92, −5.27 | a₀ 4.64, a₁ 0.070 |

**Predictions against results: all three fail.**
1. Modal model best, lumped ≥ 3× worse: **fails.** SOPDT with a delay is best (0.020); the modal model is at 0.032.
2. Modal λ₁..λ₃ within 8 / 5 / 5 %: **fails.** The fit compresses the spectrum (a₁ = 0.07 against ~0.4) to
   −4.7 … −9.1, against −5.8 … −31.5.
3. The slowest free pole within 5 % of λ₁: **fails** (−1.23 against −5.85).

**Reading.**
- At Péclet 20 the inlet → outlet response is a dispersed front arriving after about one residence time.
  - A truncated sum of the true modes represents a front poorly: the residues alternate in sign and converge
    slowly.
  - Output-error fits therefore use the poles to approximate a delay (as the pole-only model did in a20), not to
    report the eigenvalues.
- The convection–dispersion operator with these boundaries is strongly non-normal. Its eigenvalues describe the
  long-time decay, but the input → output dynamics within the window are dominated by transport. Eigenvalues are
  not what the outlet signal identifies.
- What it does identify efficiently is a dead time plus a few lags (SOPDT θ = 0.53, mean delay 0.53 + 0.23 + 0.21
  = 0.97, against the residence time 1.0).
- For (A) this marks a boundary. The pole readout is the physical one when the modes are Fourier modes of a
  translation-invariant problem (a24, periodic): there λ(k), including −ivk, is identified exactly. In open-boundary
  convective systems, the useful structures are a delay plus dispersion, or the boundary handling of project (D).

## CSTR from closed-loop data with measurement noise (2026-11-19)

`a26_cstr_closed_loop.py` → `results/a26_cstr_closed_loop.md` (predictions df503db). A PI loop (Kc 1, Ki 0.5),
with noise σ equal to 2 % of the median rms of y. Every model is tested on clean open-loop data.

| regime | WPO probe error (3 seeds) | WPO clean test | linear (output error) pole | ARX(2, 2) pole |
|---|---|---|---|---|
| open | 0.3–0.7 % | 0.018–0.020 | −0.765 ± 0.964 i | wrong (−0.02 ± 0.18 i) |
| open, noisy | 0.06–0.5 % | 0.018–0.019 | −0.758 ± 0.969 i | wrong |
| closed | 0.3–0.6 % | 0.029–0.030 | −0.771 ± 0.966 i | wrong |
| closed, noisy | 0.4–1.1 % | 0.037–0.047 | −0.772 ± 0.973 i | wrong |

**Predictions against results.**
1. WPO probe within 2 % in every regime: **holds** (≤ 1.1 %).
2. Output-error linear pole within 2 % in every regime: **holds** (0.5–1.1 %). The first table reported 1.6 for the
   closed-loop rows because `err` compared the conjugate root with the +i Jacobian root. The script now compares
   against both roots, and the table was regenerated from the stored poles (`--summarize`); no runs changed.
3. ARX within 2 % without noise: **fails.** ARX(2, 2) is wrong in every regime, noise-free included. Equation-error
   ARX on this nonlinear plant with these orders does not give the linear poles; its noisy-closed-loop bias cannot
   be separated from that.

**Reading.** Output-error fits, both the deep WPO's probe and the linear pair, are unaffected by closed-loop
operation with white measurement noise, as theory says. Closed-loop data cost the WPO accuracy on the open-loop
test set (0.018 → 0.03–0.047): the loop narrows the input excitation.

## Why ARX failed on the CSTR: orders and an amplitude gate (2026-11-19)

`a28_arx_orders.py` → `results/a28_arx_orders.md` (predictions committed before the run). The a26 data. ARX orders
2, 4 and 6 are fitted on all samples, or on the quarter with the smallest input rms. Entries are the relative error
of the slowest oscillatory pole.

| regime | samples | ARX(2,2) | ARX(4,4) | ARX(6,6) |
|---|---|---|---|---|
| open | all | 0.88 | 0.90 | 34 |
| open | smallest quarter | 0.31 | 0.092 | **0.071** |
| open, noisy | smallest quarter | 52 | 28 | 23 |
| closed | smallest quarter | 0.38 | 0.12 | 0.093 |
| closed, noisy | smallest quarter | 52 | 28 | 22 |

**Predictions against results.**
1. Noise-free open loop: the gate gives the pair within 5 % at some order, and no order does without it: **fails
   narrowly.** The best is 7.1 %, while every order without the gate is ≥ 88 % off.
2. Noisy closed loop, gated: every order more than 10 % off: **holds.**

**Reading.**
- Most of ARX's failure in a26 came from the nonlinear samples dominating the equation-error fit. Restricting it to
  the smallest quarter cuts the error twelvefold.
- The remaining ~7 % reflects order and sampling: ARX maps poles through z = e^{μ dt} from a fit of one-step
  predictions.
- Even 2 % measurement noise destroys the equation-error estimate. The small samples have the lowest
  signal-to-noise ratio, so the gate that removes the nonlinearity amplifies the noise.
- The output-error fits (a26) need neither a gate nor an order choice. That is the practical case for the pole
  layers as identification tools.
