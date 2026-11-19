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
