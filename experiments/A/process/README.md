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
