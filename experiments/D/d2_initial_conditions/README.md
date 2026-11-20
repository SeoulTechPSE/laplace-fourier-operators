# (D2) Initial conditions as an explicit input; rollout by chaining windows

Plan: `docs/proposal_D_nonperiodic_initial_conditions.qmd`, Section 5. Shared code: the `ic=True` option of
`lfo.models.WavenumberPoleLayer1d` (7b70758). With it, `forward(f, u0)` adds the free response
û₀(k) Σ γ_n(k) e^{μ_n(k) t}, which shares the poles with the forced response. A test checks that two chained windows
reproduce the exact solution to 1e-9.

## First pass (2026-11-19): periodic ADR

`d2_ic_rollout.py` → `results/d2_ic_rollout.md` (predictions 1191c1c).

**Setup.**
- a1's ADR with a random initial field per window, forcing at half amplitude, T = 5, and 5 data seeds.
- The `ic` layer is compared with a layer that sees only the forcing (`noic`).
- Rollout: 4 chained windows over [0, 20], each starting from the previous window's last field.

| model | test / oracle | D̂, v̂, k̂_r | max abs(γ(k) − 1) | rollout error | rollout / oracle rollout |
|---|---|---|---|---|---|
| ic | **1.00** | 0.0500, 0.5000, 0.2000 | **≤ 1.1e-5** | 1.3–2.2e-3 | **1.000–1.001** |
| noic | 570–750 | – | – | – | – |

**Predictions against results.**
1. `ic` at the oracle, with D, v, k_r and γ within 1 %: **holds** (γ within 1e-5).
2. `noic` ≥ 5× the oracle: **holds** (570–750×: the free response dominates these windows).
3. Rollout error ≤ 1.5× the single-window test error in every seed: **fails** in seed 3 (1.73; others
   0.95–1.42).
   - The model's rollout error equals the oracle's rollout error in every seed (ratio 1.000–1.001), so chaining
     adds nothing.
   - The ratio to the single-window error reflects the different sample sets, through the sampling floor of steps
     and PRBS. The prediction compared against the wrong reference; the oracle rollout is the right one.

**Reading.** Sharing the poles between the free and forced responses makes chaining exact for a linear system. The
learned layer is at the oracle over 4 windows. For deep nonlinear models the IC path and chaining need a state
readout and the semigroup regulariser of the plan; that is the next step.

## Second pass (2026-11-19): deep rollout on forced Burgers

`d2b_burgers_rollout.py` → `results/d2b_burgers_rollout.md` (predictions 42bc898). Shared code (4b67655): an
initial field in `lfo.data.burgers.solve`, and the initial-state path of the deep `WavenumberPoleOperator1d`. With
`ic=True`, the lifted initial field enters the first pole layer as a free response with its poles. Its γ matrices
are constant in k, and the width is 23, to keep the budget.

**Setup.** Forced Burgers (ν = 0.1) with a random initial field per window (modes 1–8, scale log-uniform over
[0.03, 1]), T = 5, 3 seeds, 6000 steps. Rollout: 4 chained windows over [0, 20] on 50 trajectories.

| model | params | window test (mean) | rollout over [0, 20] (mean) | rollout / test |
|---|---|---|---|---|
| WPO, IC path sharing the poles | 61.1 k | 0.0254 | **0.0541** | 2.05–2.18 |
| WPO, u₀ as an input channel | 59.6 k | 0.0273 | 0.0603 | 2.02–2.35 |
| TFNO, u₀ as an input channel | 59.1 k | 0.0250 | 0.0611 | 2.34–2.51 |

**Predictions against results.**
1. The IC-path WPO's window error is at or below both others: **fails narrowly.** It beats the channel WPO
   (0.0254 against 0.0273) but not the TFNO (0.0250).
2. Rollout / window ≤ 2 for the IC-path WPO in every seed, and > 2 for the TFNO: **fails.** The IC-path WPO is at
   2.05–2.18; the TFNO is above 2, as predicted.
3. The IC-path WPO has the lowest rollout error: **holds** (0.054 against 0.060 and 0.061, about 10 % lower).

**Reading.**
- In a nonlinear system, every model roughly doubles its error over 4 chained windows. Each window starts from
  a predicted rather than a true field.
- Sharing the poles between the initial state and the forced response gives the smallest growth and the lowest
  rollout error, but only by about 10 %. That is a modest structural gain, not the exactness of the linear case.
- Next steps from the plan: the semigroup regulariser ‖G(t+s)[u₀] − G(s)[G(t)[u₀]]‖ during training, and training
  on predicted initial fields (pushforward).

## Third pass (2026-11-19): training for rollout (pushforward, semigroup regulariser)

`d2c_rollout_training.py` → `results/d2c_rollout_training.md` (predictions 2776b70). The d2b IC-path WPO (width 23,
a11 settings, 3 seeds, 6000 steps), trained three ways on the same 200 two-window trajectories ([0, 10]):

- `base`: both windows as independent samples from their true initial fields (400 windows);
- `push`: window 2 starts from the model's own (detached) prediction at t = 5;
- `semi`: `base` plus the semigroup penalty ‖G_full[T/2:] − G(f[T/2:], G(f[:T/2], u₀)[end])‖ / ‖G_full[T/2:]‖, weight 1.

| training | window test (mean, per seed) | rollout over [0, 20] (mean, per seed) | rollout / test |
|---|---|---|---|
| base | 0.0221 (0.0227, 0.0213, 0.0222) | **0.0424** (0.0411, 0.0451, 0.0411) | 1.81–2.12 |
| push | 0.0232 (0.0236, 0.0235, 0.0225) | 0.0482 (0.0535, 0.0478, 0.0432) | 1.92–2.27 |
| semi | 0.0251 (0.0220, 0.0282, 0.0252) | 0.0450 (0.0396, 0.0483, 0.0470) | 1.71–1.86 |

**Predictions against results.**
1. `push` and `semi` each lower the mean rollout error below `base`: **fails** for both (0.048 and 0.045 against
   0.042).
2. `push` has rollout / test ≤ 1.7 in every seed: **fails** (1.92–2.27).
3. Neither remedy raises the mean window test error by more than 10 %: **holds for `push`** (+5 %), **fails for
   `semi`** (+14 %; its ratio is lowest because its window error is higher, not its rollout error lower).

**Reading.**
- Neither remedy helps. The cheapest change helped most: `base` here reaches 0.042 against 0.054 in d2b (−22 %),
  with the same architecture. It differs from d2b in its training windows: half of them start from developed
  Burgers fields (the state at t = 5 of a trajectory), which is what windows 2–4 of a rollout start from, and there
  are twice as many (400 against 200). The two effects are confounded here (post hoc).
- Once the training set covers the states a rollout visits, starting window 2 from the model's own prediction
  (`push`) adds only the model's own error as noise, and the semigroup penalty trades window accuracy for
  self-consistency. The rollout error still roughly doubles over 4 windows; what remains is the accumulation of
  per-window errors, not a distribution shift.
- For the paper: train on windows that start from developed states; the plan's two remedies are recorded as
  negative results at this budget.

## Fourth pass (2026-11-20): developed start states against window count

`d2d_start_states.py` → `results/d2d_start_states.md` (predictions 450deaa). A 2 × 2 design separating the two
changes that were confounded in d2c's `base` (same model, 6000 steps, batch 25, 3 seeds; `dev400` is d2c `base`):

| condition | windows | start states | window test (mean) | rollout over [0, 20] (mean; per seed) | rollout / test |
|---|---|---|---|---|---|
| fresh200 | 200 | fresh (as d2b) | 0.0236 | 0.0599 (0.0552, 0.0614, 0.0630) | 2.37–2.68 |
| fresh400 | 400 | fresh | **0.0194** | 0.0498 (0.0497, 0.0509, 0.0487) | 2.51–2.67 |
| dev200 | 200 | half developed | 0.0269 | 0.0503 (0.0478, 0.0490, 0.0541) | 1.83–1.93 |
| dev400 | 400 | half developed | 0.0221 | **0.0424** (0.0411, 0.0451, 0.0411) | 1.81–2.12 |

**Predictions against results.**
1. `dev200` rolls out better than `fresh400`: **fails**, narrowly (0.0503 against 0.0498; within seed spread).
2. At equal count, developed start states lower the rollout error by at least 10 %: **holds** (dev200 / fresh200 =
   0.84, dev400 / fresh400 = 0.85).
3. `fresh400` has a window test error at or below `dev400`'s: **holds** (0.0194 against 0.0221).

**Reading.**
- Both effects are real and of similar size: doubling the windows lowers the rollout error by 15–17 %, and making
  half of them developed lowers it by 15–16 %. They add up (fresh200 → dev400: −29 %). The d2c gain over d2b was
  about half each, not mostly start states as prediction 1 assumed.
- The two act differently. More fresh windows lower the window error (−18 %) and the rollout error follows with the
  same ratio (~2.5). Developed start states *raise* the error on fresh test windows (+14 % at both counts) but cut
  the rollout / test ratio from ~2.5 to ~1.9: they buy coverage of the states a rollout visits at the cost of the
  initial-state distribution the window test measures.
- For the paper: report rollout error, not window error, when choosing training data for chained windows; include
  developed start states.

## Fifth pass (2026-11-20): initial fields outside the training range; rollout over 10 windows

`d2e_ic_generalisation.py` → `results/d2e_ic_generalisation.md` (predictions 554d0f4). The three d2b models,
all trained on d2c's developed-start windows (400 windows, 3 seeds; `wpo_ic` reproduces d2c `base` exactly:
rollout 0.0411, 0.0451, 0.0411). Evaluated on:
- `id`: fresh windows;
- `amp`: initial-field scale 1–3× (training ≤ 1);
- `high`: initial field on modes 9–16 only (training 1–8);
- `roll4`: d2c's 4-window set;
- `roll10`: 10 windows over [0, 50], with the forcing drawn per window from the training family.

| model | params | id | amp (ratio to id) | high (ratio to id) | roll4 | roll10 | roll10 / its first 4 windows |
|---|---|---|---|---|---|---|---|
| wpo_ic | 61.1 k | 0.0221 | 0.252 (10.8–11.9) | **0.0159 (0.70–0.76)** | **0.0424** | 0.0550 | 1.48–1.59 |
| wpo_chan | 59.6 k | 0.0251 | 0.297 (11.0–12.6) | 0.0271 (0.95–1.28) | 0.0478 | 0.0703 | 1.51–1.56 |
| tfno_chan | 59.1 k | 0.0215 | **0.189 (8.5–9.3)** | 0.0714 (3.1–3.6) | 0.0441 | **0.0490** | **1.33–1.39** |

Per-window error on `roll10` (mean over seeds), windows 1–10:
- wpo_ic: 0.022, 0.034, 0.039, 0.033, 0.033, 0.037, 0.036, 0.037, 0.038, 0.047;
- wpo_chan: 0.025, 0.044, 0.085, 0.071, 0.062, 0.066, 0.055, 0.064, 0.061, 0.066;
- tfno_chan: 0.021, 0.032, 0.047, 0.043, 0.041, 0.043, 0.037, 0.039, 0.042, 0.048.

**Predictions against results.**
1. `wpo_ic` has the lowest `roll4` error, at most 0.9× `tfno_chan`'s: **fails.** It is lowest, but only by 4 %
   (0.96×). Developed-start training helped the TFNO more (0.061 in d2b → 0.044) than the IC-path WPO (0.054 →
   0.042).
2. `high`: `wpo_ic`'s ratio at most half of both others': **fails.** It holds against the TFNO (0.72 against 3.3),
   not against `wpo_chan` (1.08).
3. `amp`: every ratio ≥ 2, and `wpo_ic` not more than 10 % below `wpo_chan`: **holds** (8.5–12.6; 11.4 against
   11.9).
4. `roll10` ≤ 1.5× its first 4 windows for `wpo_ic` in every seed: **fails** (1.48, 1.54, 1.59).

**Reading.**
- **Unseen modes in the initial field.** The TFNO's error triples. The WPOs, whose time axis is a pole response in
  every mode, do not degrade (the ratio is below 1, because high modes decay within the window). The IC path is
  better than the channel input (0.72 against 1.08), but by less than predicted. This is the one clear structural
  gain over the TFNO.
- **Larger initial amplitudes** cost every model about 10×. The TFNO degrades least (8.8×). A stronger nonlinear
  regime is not something the pole structure extrapolates.
- **Long rollout.** No model blows up. Per-window errors saturate after window 3 (`wpo_ic` at 0.033–0.038 in windows
  3–9). All three models rise in window 10, so the rise most likely comes from that window's forcing
  draws (not checked). Prediction 4's ratio is set mostly by that window. The TFNO has the lowest error over 10 windows (0.049
  against 0.055). With developed-start training, the IC-path WPO's rollout advantage from d2b is gone over long
  horizons.

## Sixth pass (2026-11-20): a damped beam with initial displacement and velocity

`d2f_beam_ic.py` → `results/d2f_beam_ic.md` (predictions 19ce7dc). u_tt = −c²u_xxxx + η u_xxt + f (c = 0.04,
η = 0.002), observed through its displacement u, with random initial displacement u₀ and velocity v₀. Per mode, the
forcing residues and the coefficient of v₀ are ±1/(μ₁ − μ₂) ∝ 1/k²: not polynomial in k, but fixed by the poles
(a₁ = (v₀ − μ₂u₀)/(μ₁ − μ₂), a₂ = (μ₁u₀ − v₀)/(μ₁ − μ₂)). Poles from sum and product laws in k². 5 data seeds,
L-BFGS from 8 starts. Rollout: 4 windows, each model passing (u, u_t), with u_t = μ₁x₁ + μ₂x₂ from its own modal
states, since only u is observed.

| model | reals | test / oracle | ĉ (0.04) | η̂ (0.002) | rollout / oracle rollout |
|---|---|---|---|---|---|
| struct: residues and IC coefficients from the poles | 8 | **1.000–1.001** | **0.040000** | **0.00200** (≤ 0.007 %) | **0.997–1.001** |
| poly: free polynomial residue and IC laws | 30 | 51–88 | 0.0414–0.0415 | 0.022–0.023 | 120–187 |
| struct_u0: v₀ withheld | 8 | 94–168 | 0.0399–0.0400 | 0.0017–0.0022 | 64–120 |

Oracle test error 2.1–3.6e-3; oracle rollout error 0.6–1.1e-2.

**Predictions against results.**
1. `struct` at ≤ 1.2× the oracle with c and η within 1 % and rollout ≤ 1.05× the oracle's, in every seed:
   **holds** (c and η within 0.007 %).
2. `poly` ≥ 3× the oracle: **holds** (51–88×).
3. `struct_u0` ≥ 5× the oracle: **holds** (94–168×).

**Reading.**
- For a second-order system, the initial-state coefficients are a known function of the poles: the inverse of the
  2 × 2 Vandermonde matrix of each mode. Writing them that way adds no parameters and is exact. The layer then
  identifies c and η to 1e-4 relative, and chained windows reproduce the oracle's rollout.
- The constraint matters. With free polynomial laws in their place (30 reals), the fit is 50–90× worse and η is
  biased tenfold: the 1/k² coefficients cannot be expressed, and the poles absorb the misfit.
- Both initial fields are needed (v₀ withheld: ~100×). Only u is observed, but the model's own modal states give
  the velocity at the window end, so chaining needs no velocity measurement: the latent-state option of the plan
  (Section 5, "Defining the output state") is exact here.

## Seventh pass (2026-11-20): Cahn–Hilliard as a second rollout system

`d2g_ch_rollout.py` → `results/d2g_ch_rollout.md` (predictions 81b756f). The d2e protocol on forced 1D
Cahn–Hilliard (M = 0.1, κ = 0.04, the a16 forcing), which has an unstable band k < 5 and saturates at ±1.
- WPOs allow Re μ > 0.
- Developed-start training: 400 windows, 6000 steps, 3 seeds.
- `amp` uses s0 in [1, 2]: the solver diverged on [1, 3], found before any fit.

| model | id | amp (ratio) | high (ratio) | roll4 | roll10 | window 10 / window 3 |
|---|---|---|---|---|---|---|
| wpo_ic | 0.0255 | 0.110 (4.0–4.6) | **0.060** (2.2–2.5) | 0.0498 | 0.0604 | 1.60 |
| wpo_chan | 0.0259 | 0.137 (5.0–5.7) | 0.089 (3.3–3.6) | 0.0575 | 0.0622 | 1.65 |
| tfno_chan | **0.0126** | **0.067** (4.6–5.8) | 0.131 (9.4–11.8) | **0.0240** | **0.0373** | 2.11 |

**Predictions against results.**
1. Unseen initial modes: the TFNO's ratio high / id ≥ 2× `wpo_ic`'s: **holds** (10.5 against 2.35, 4.5×). Here
   the IC-path WPO is also lower in absolute error (0.060 against 0.131).
2. Larger amplitudes: every ratio ≥ 2 in every seed: **holds** (4.0–5.8).
3. Saturation, window 10 ≤ 2× window 3 for every model: **fails** for the TFNO (2.11). The WPOs reach 1.60 and 1.65.
   Errors keep growing slowly in all three models, unlike Burgers.
4. `wpo_ic`'s `roll4` within ±10 % of the TFNO's: **fails.** The TFNO is 2× better (0.024 against 0.050), and also
   2× better in-distribution (0.013 against 0.026).

**Reading.**
- **What carries over from Burgers.** The robust structural result: pole-based models tolerate initial fields on
  unseen modes far better (ratio 2.4 against 10.5), and the IC path beats the channel input (2.4 against 3.5).
  Larger amplitudes cost every model.
- **What does not.** On this system the TFNO is the better model in-distribution and in rollout, by 2×. In a16
  (zero initial state, 12000 steps) the WPO and the TFNO were at parity (0.026 against 0.029). This pass used the
  D2 protocol (6000 steps) and no Cahn–Hilliard tuning, so the gap may partly be training budget (not checked).
  Whatever its cause, the pole structure gives no rollout advantage here, and the IC path's edge over the channel
  WPO (0.050 against 0.058 over 4 windows) is the only structural gain in rollout.
- **Error growth.** Errors keep growing over 10 windows in every model (1.6–2.1× from window 3 to 10). In Burgers
  they saturated. In the unstable band, small state errors grow within each window.
