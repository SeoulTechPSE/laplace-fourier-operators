# (D) Non-periodic space: time-varying boundaries as a channel that shares the poles

Plan: `docs/proposal_D_nonperiodic_initial_conditions.qmd`, Table 6.1, "Diffusion + time-varying boundary/source".
The periodic layer cannot represent boundaries. On [0, L] with Dirichlet values a(t), b(t), the sine transform
diagonalises the operator, and the boundary values enter mode m as a forcing c(k_m)(a − (−1)^m b) with
c(k) = 2Dk/L, an odd function of k. The model is the wavenumber pole layer on sine modes plus a boundary channel
c(k) = c₁k + c₃k³ that shares the poles μ(k).

## First pass (2026-11-19): diffusion with Dirichlet boundaries

`d3_dirichlet.py` → `results/d3_dirichlet.md` (predictions 97c9d30; device fix ff33238).

**Setup.**
- D = 0.02 on 63 interior points, T = 5.
- Forcing on sine modes 1–16; boundary values from E1 profiles, starting at rest.
- Truth: the 63-mode sine-Galerkin solution with an exact FOH on a 32× finer grid. A finite-difference truth was
  dropped before any fit: its discrete eigenvalues are not polynomial over all modes, so it would have tested the
  discretisation rather than the model.

| model | test / oracle | D̂ (0.02) | ĉ₁ (0.04) |
|---|---|---|---|
| sine modes + boundary channel | **1.00–1.04** | **0.01992–0.02007** | **0.03995–0.04001** |
| sine modes, no boundary channel | 82–104 | – | – |
| periodic layer | 84–105 | – | – |

**Predictions against results: both hold.**
1. With the boundary channel: at the oracle, D within 0.4 % and c₁ within 0.13 %, in every seed.
2. Without it, sine or periodic: 82–105× the oracle in every seed.

**Reading.**
- Boundary forcing is one more input that shares the dispersion relation. Its coefficient is itself a law in k
  (c ∝ k), and the layer identifies it together with D.
- This handles the non-periodic boundaries that a27 could not: there the boundary was an inlet in a convective
  problem observed only at the outlet. Here the boundary data and the whole field are observed, and the operator
  is self-adjoint.
- Next: Neumann or Robin boundaries (cosine modes, or modes with a boundary-dependent basis), and an advective
  case.

## Second pass (2026-11-19): Dirichlet, Neumann and mixed boundaries

`d3b_bc_types.py` → `results/d3b_bc_types.md` (predictions 07a570a; resume fbab0d6).

**Setup.**
- Each boundary channel gets a free cubic law c_j(k) = c₀ + c₁k + c₂k² + c₃k³. The sign pattern φ_m(boundary)
  comes from the basis.
- By Green's identity, a boundary *value* (Dirichlet) enters as 2Dk/L, proportional to k. A boundary *flux*
  (Neumann) enters as 2/L, constant.
- Cases: DD (sine basis), NN (cosine basis; the k = 0 mode is an integrator), DN (quarter-wave sine basis).
- 32 modes, D = 0.02, 5 seeds.

| case | with channels: test / oracle | D̂ (0.02) | law error, all 32 modes | law error, modes m ≤ 8 (post hoc) | without channels |
|---|---|---|---|---|---|
| DD | 1.00–1.02 | 0.01976–0.02006 | 1.5–27 % | **0.1–1.3 %** | 79–102× |
| NN | 1.00 | 0.02000–0.02005 | 0.8–7.4 % | **0.06–0.45 %** | 240–270×, one seed non-finite |
| DN | 1.00–1.02 | 0.01981–0.02007 | 5–60 % | **0.1–4.0 %** | 187–283× |

**Predictions against results.**

1. With channels, every case and seed at ≤ 1.2× the oracle, D within 1 % and each law within 2 % over all modes:
   **fails.**
   - The test error and D hold, except DD seed 2, where D is 1.2 % off.
   - The laws over all 32 modes are 1.5–60 % off.
2. Without channels, ≥ 3× the oracle: **holds** in every case and seed. NN seed 1 had no finite start at all.

**Reading.**
- Boundary signals excite mode m with a response that decays as e^{−D k_m² t}, so the high modes leave almost no
  trace in the data. The cubic law is therefore pinned only where the data are informative.
- On modes 1–8, post hoc, each channel's law is right within 0.06–1.3 % (DN seed 0: 4 %), and its form matches the
  physics: Dirichlet channels ∝ k, Neumann channels constant, and both in the mixed case.
- The all-modes criterion asked the model to extrapolate the law into a band the data do not constrain. This is
  the same lesson as project A's out-of-band dispersion (a19): a law is identified where it is excited.

## Third pass (2026-11-19): advection–diffusion with Dirichlet values (one learned weight)

`d3c_advective.py` → `results/d3c_advective.md` (predictions 2c19dcc). u_t = −v u_x + D u_xx + f on [0, 1], v = 0.2,
D = 0.05, Dirichlet values a(t), b(t). The weight w = e^{−αx}u with α = v/(2D) = 2 removes the advection, so the sine
modes of w diagonalise the problem with λ(k) = −Dk² − v²/(4D). Model `adv`: the d3b DD layer on the weighted fields
with one learned α. Control `noadv`: α = 0. Truth: 32-mode Galerkin solution of the weighted problem on a 32× finer
time grid. 5 data seeds.

| model | test / oracle | α̂ (2) | D̂ (0.05) | v̂ = 2D̂α̂ (0.2) | offset −a₀ (0.2) | v̂²/(4D̂) |
|---|---|---|---|---|---|---|
| adv | 1.06–1.39 | **1.992–2.002** | 0.0460–0.0517 | 0.184–0.207 | 0.173–0.239 | 0.184–0.207 |
| noadv | 14.8–17.5 | – | – | – | – | – |

**Predictions against results.**
1. `adv` at ≤ 1.2× the oracle with α and D within 2 %, v̂ within 3 % and the offset within 5 % of v̂²/(4D̂), in every
   seed: **fails.** α holds in every seed (≤ 0.4 %). The error ratio holds in 4 of 5 seeds (seed 1: 1.39). D̂ is
   off by up to 8 % (only seed 1 within 2 %), v̂ by up to 8 %, and the offset by up to 15 % from v̂²/(4D̂).
2. `noadv` ≥ 3× the oracle: **holds** (15–17×).

**Diagnosis (post hoc, seeds 0 and 4 refitted; scratch script, not committed).**
- The fitted in-band poles μ(k_m) are off by 7–10 % on every mode, m = 1 included, with a spurious imaginary part.
  β(k) and the boundary laws compensate: the laws (β included) stay within 0.2–12 % of their true values on m ≤ 12.
- The oracle itself has 0.9–1.1 % error, from the first-order hold on the coarse grid against the fine-grid truth.
  Refitting seed 0 on a truth computed on the model's own grid (oracle 2e-15) brings the in-band pole errors down to
  0.4–2 % (fit 3.6e-3). So most of the bias comes from the parameters absorbing the representation floor. The
  rest is optimisation: the fit stops at 3.6e-3 rather than at the oracle.

**Reading.**
- The one new parameter, the weight e^{αx} that removes the advection, is identified to 0.4 % in every seed. The
  layer then handles an advective, non-self-adjoint problem with Dirichlet boundaries at 1.1× the oracle (4 of 5
  seeds), against 15× without the weight.
- D and v²/(4D) carry a bias of up to 10 % because, at a 1 % representation floor, the pole law trades accuracy
  against the residues and laws. This is the same mechanism as the (C) E2 finding that a model-class mismatch biases
  the tied estimate. Claims about recovered physical constants need the representation floor well below the fit
  error, as in d3/d3b (oracle ≈ 1e-3 relative to the fit).

## Fourth pass (2026-11-20): Robin boundaries, the basis from a learned coefficient

`d3d_robin.py` → `results/d3d_robin.md` (predictions ff9596d; fixes before any result e966be8, 8a9ac59). A Robin
boundary D u_x(1) + H u(1) = F1(t) has the basis sin(k_m x) with k_m the roots of k cos k + β sin k = 0, β = H / D,
so the wavenumbers themselves are unknown. The model builds its basis from a learned β (bisection plus the implicit
function theorem for the gradient). The laws are claimed as before: the value channel ∝ D k_m / N_m and the flux
channel ∝ sin k_m / N_m. D = 0.02, β = 2, 32 modes, 5 data seeds, the a7 joint fit with 8 starts.

| model | test / oracle | β̂ | D̂ | value-law error m ≤ 8 | flux-law error m ≤ 8 |
|---|---|---|---|---|---|
| robin (β learned, β0 ~ U[0.25, 1]) | 6.37–8.35 | 0.0000–0.0006 | 0.0142–0.0251 | 0.15–0.34 | 0.13–0.34 |
| fixed (β = 2 given) | **1.00–1.01** | 2 | **0.0199–0.0202** | 3e-4–1.2e-2 | 6e-4–7.8e-3 |
| dn (β = 0, the Dirichlet–Neumann basis) | 5.86–7.64 | 0 | 0.0133–0.0140 | 0.06–0.15 | 0.25–0.30 |

**Predictions against results.**
1. `robin` at ≤ 1.2× the oracle with β within 2 %, D within 1 % and the laws within 2 % on m ≤ 8, in every seed:
   **fails** in every seed. β collapses to the Neumann end (β̂ ≤ 6e-4), and the fit lands near `dn`.
2. `fixed`, the same criteria: **holds** (laws ≤ 1.2 %, D ≤ 0.8 %).
3. `dn` ≥ 3× the oracle: **holds** (5.9–7.6×).

**Reading.** With the basis given, the layer is exact for Robin boundaries. What fails is finding β jointly with
everything else: from random poles and laws the early gradient pushes β to 0, where softplus flattens it, and the fit
settles in the Neumann basin.

## Fifth pass (2026-11-20): Robin coefficient by a profile, then released

`d3e_robin_profile.py` → `results/d3e_robin_profile.md` (predictions 8a9ac59). The rest is fitted with β fixed at
each point of the grid {0.3, 0.6, 1.2, 2.4, 4.8, 9.6} (which excludes the truth; 2 starts each). β is then
released from the best grid fit and L-BFGS runs on (300 iterations). Same data, oracle and seeds as d3d.

| seed | test / oracle | grid pick | β̂ | D̂ | law errors m ≤ 8 (value, flux) |
|---|---|---|---|---|---|
| 0 | 1.00 | 2.4 | 1.9978 | 0.02004 | 1.7e-3, 1.2e-3 |
| 1 | 1.07 | 2.4 | 1.9951 | 0.02052 | 3.7e-2, 2.2e-2 |
| 2 | 1.01 | 2.4 | 1.9908 | 0.01989 | 3.8e-3, 1.5e-3 |
| 3 | 1.11 | 2.4 | 1.9996 | 0.01863 | 5.2e-2, 3.3e-2 |
| 4 | 1.01 | 2.4 | 1.9992 | 0.02013 | 6.4e-3, 4.6e-3 |

Profile validation errors (seed 0, by grid β): 3.0e-2, 2.5e-2, 1.4e-2, **7.8e-3**, 3.5e-2, 7.0e-2. The profile is
unimodal with its minimum at 2.4 in every seed.

**Predictions against results.**
1. In every seed: β within 2 %, D within 1 %, ≤ 1.2× the oracle and the laws within 2 % on m ≤ 8: **fails in 2 of 5
   seeds.** β (≤ 0.5 %) and the error ratio (≤ 1.11) hold in every seed. D and the laws hold in seeds 0, 2 and 4.
   Seed 1 is off by 2.6 % in D and 3.7 % in the laws, seed 3 by 6.9 % and 5.2 %.
2. The profile minimum is at the grid point 2.4 in every seed: **holds.**

**Diagnosis (post hoc; seeds 1 and 3 refitted, which reproduces the table exactly; scratch script, not committed).**
Continuing L-BFGS with β free for 300 more iterations brings both seeds onto every criterion. Seed 1: 1.003×, β̂
1.9965, D̂ 0.02000, laws 2.5e-3 and 1.6e-3. Seed 3: 1.004×, β̂ 1.9954, D̂ 0.02014, laws 5.6e-3 and 4.6e-3. A further
900 iterations change nothing. The misses were an iteration budget too short after the release, not a bias.

**Reading.**
- An unknown boundary coefficient that sets the basis is identifiable, but not by the joint fit. Fitting the rest at
  fixed β gives a profile that is unimodal and sharply peaked (its minimum is 2× lower than the next grid point), and
  releasing β from its minimum converges to the truth (β within 0.5 % in every seed).
- The profile is cheap here (one scalar, 6 points). The general lesson for the paper: parameters that change the
  basis, rather than the coefficients in it, should be found by an outer search, not by gradient descent jointly
  with the poles. α in d3c is the exception so far: it was found jointly (to 0.4 %), because the weight
  e^{αx} changes the basis smoothly and does not flatten at a boundary of its range as softplus(ρ) does at β → 0.
- The pre-registered budget (300 iterations after release) was too short in 2 of 5 seeds. The recorded result
  stays a partial failure. With 600 iterations, all criteria hold in the two seeds checked.

## Sixth pass (2026-11-20): a variable coefficient, with the basis learned

`d3f_variable.py` → `results/d3f_variable.md` (predictions 90228c3). u_t = (D(x) u_x)_x + f on [0, 1] with
u = 0 at both ends and D(x) = 0.02 (1 + 0.5 cos πx). The eigenfunctions are not sines: the overlap of eigenvector m
with sine m is 0.98 for m = 1 and falls to 0.13 for m = 8. Truth and models share the 32-mode sine space, so the
only floor is the time axis.
- `mix`: a learned orthogonal basis change Q = exp(S − Sᵀ), starting from the sine basis, plus one free pole per
  mode (528 reals).
- `diag`: the sine basis with free poles (32 reals).
- L-BFGS, 600 iterations, 4 starts, 5 data seeds. Claims are on the excited eigenmodes m ≤ 8.

| model | test / oracle | eigenvalue error m ≤ 6 | eigenvalue error m = 7, 8 | min abs(⟨v_m, v̂_m⟩), m ≤ 7 / m = 8 |
|---|---|---|---|---|
| mix | 1.09–1.21 | ≤ 0.6 % | ≤ 1.1 % / ≤ 2.5 % | 0.9993 / 0.985–0.999 |
| diag | 15–27 | up to 10 % | – | 0.28 / 0.13 (the sine basis) |

Oracle test error 5.4–9.5e-3.

**Predictions against results.**
1. `mix` at ≤ 1.2× the oracle, λ₁..₈ within 1 % and |⟨v_m, v̂_m⟩| ≥ 0.999 for m ≤ 8, in every seed: **fails.** The
   error ratio holds in 4 of 5 seeds (seed 4: 1.21). The eigenvalues hold in 3 of 5 (seed 2 λ₇ 1.1 %, seed 4 λ₈
   2.5 %). The eigenvectors hold for m ≤ 7 in every seed, but m = 8 is below 0.999 in 4 of 5 (0.985–0.998).
2. `diag` ≥ 3× the oracle: **holds** (15–27×).

**Diagnosis (post hoc; seed 4 refitted, which reproduces the table exactly; scratch script, not committed).**
- Eigenmode 8 carries 0.9 % of the training forcing energy (mode 1: 64 %, mode 4: 3.6 %).
- Continuing L-BFGS converges to the oracle:
  - +600 iterations: 1.04×; λ₈ 0.55 %; v₈ 0.9976.
  - +1200: 1.006×; λ₈ 0.13 %; v₈ 0.99976.
  - +1800: 1.001×; λ₁..₈ within 0.06 %; every eigenvector ≥ 0.99987.
- So the misses were an iteration budget too short for a weakly excited mode with 528 parameters, not a limit of
  identifiability. As in d3e, the pre-registered result stays a partial failure.

**Reading.**
- A variable coefficient breaks the sine basis (the best diagonal model is 15–27× the oracle). An orthogonal basis
  change learned together with the poles recovers the Sturm–Liouville eigenvalues and eigenvectors of the excited
  modes, given enough iterations (1800 more than pre-registered for the weakest claimed mode).
- What was learned is the operator's eigen-decomposition, not D(x). Recovering D(x) from it is an inverse
  Sturm–Liouville problem, not attempted here. Unlike d3–d3e, the poles here are free per mode, with no law in k.

## Seventh pass (2026-11-20): d3e and d3f again, with the fit stopped by convergence

`d3g_converged.py` → `results/d3g_converged.md` (predictions 3ea4944). Each script's own fit is unchanged. The
selected model then continues with L-BFGS in chunks of 300 iterations until a chunk lowers the training loss by
less than a relative 1e-4, with a cap of 6000. All seeds are rerun.

| case | test / oracle | estimates (all seeds) | iterations after the original fit |
|---|---|---|---|
| Robin | 1.0004–1.011 | β̂ 1.9945–1.9992 (≤ 0.27 %), D̂ within 0.7 %, laws ≤ 0.64 % | 300–600 |
| variable coefficient | 1.0005–1.0018 | λ₁..₈ within 0.16 %, overlaps ≥ 0.99987 | 2700–3900; 6000 (cap) in seed 3 |

**Predictions against results.**
1. Robin, every seed, on the d3e criteria: **holds** (it failed in 2 of 5 seeds with the fixed budget).
2. Variable coefficient, every seed, on the d3f criteria: **holds** (it failed in 4 of 5).
3. Every run stops by the convergence test: **fails** in one run. Variable seed 3 reached the cap, although it was
   already within all criteria (λ within 0.16 %).

**Reading.** Both earlier failures were fit budgets, now shown on every seed rather than on post hoc refits. The
Robin release needs 300–600 iterations beyond d3e's 300. The 528-parameter learned basis needs 2700–6000, and
the relative-drop rule does not always fire there (slow, steady progress along the weakly excited modes). Fits
whose claims include weakly excited modes should be run to convergence, with the stopping rule reported.

## Eighth pass (2026-11-20): the variable coefficient learned as a function D(x)

`d3h_coefficient.py` → `results/d3h_coefficient.md` (predictions 3ea4944; fix after the first launch 54fc498). The
parameter is D(x) = Σ_{j<6} d_j cos(jπx) itself (6 reals). The layer's basis and poles are the eigen-decomposition
of the Galerkin operator A(d) = Σ d_j A_j, which is linear in d (`torch.linalg.eigh`). Same system, data and
seeds as d3f (`cos`, in the span), plus `exp`: D(x) = 0.01·3^x, not in the span (best 6-cosine approximation
1.06 % off). 4 starts, each run to convergence with the d3g rule.

| case | test / oracle | D(x) error (relative L2) | eigenvalue error m ≤ 8 | iterations |
|---|---|---|---|---|
| cos | **1.0000–1.0006** | **0.96–2.5e-4** | ≤ 1.8e-4 | 600 |
| exp | 1.08–1.13 | 1.07–1.08 % (floor 1.06 %) | ≤ 1.9e-3 | 600 |

**Fix after the first launch** (recorded in the docstring). In `cos` seeds 0 and 2, an L-BFGS line search stepped
to a D(x) with negative values. The poles turned positive, the response overflowed and eigh failed. The failure was
caught as a chunk without progress, which stopped the fit at 9–10× the oracle; seed 3 then crashed. Seed 1 stayed
in the valid region and was at the oracle. Fix, with model and predictions unchanged: d = 0.02 θ, and a penalty in
place of the eigen-solve where min D(x) < 1e-4. All seeds were rerun.

**Predictions against results.**
1. `cos`: ≤ 1.2× the oracle, D(x) within 1 %, λ₁..₈ within 1 %, every seed: **holds** (D(x) within 2.5e-4).
2. `exp`: ≤ 1.2× and D(x) within 2 %, every seed: **holds.** The error is the truncation floor: the fit lands on the
   best 6-cosine approximation (1.07–1.08 % against 1.06 %). The 1.08–1.13× is that approximation's own error
   (1.12× at the best approximation, measured before the fit).
3. Median iterations below d3f `mix`'s 2400 + 1800: **holds** (600, the minimum of two chunks).

**Reading.**
- Parameterising the coefficient instead of the basis turns a 528-parameter fit that needed thousands of
  iterations into a 6-parameter fit that converges in 600, and recovers the physical coefficient to 1e-4.
- Outside the span, the fit finds the best approximation in the span. The error then reflects the expansion, not
  the identification.
- With d3e, this is the second case where a parameter that sets the basis is best learned as the physical parameter
  (β, D(x)), with the basis computed from it. Free basis entries (d3f `mix`) also converge, but slowly and without
  giving the coefficient.
- The positivity guard was needed because a line search can leave the physical region. With D(x) > 0, the operator
  stays negative definite and the eigen-solve well conditioned.
