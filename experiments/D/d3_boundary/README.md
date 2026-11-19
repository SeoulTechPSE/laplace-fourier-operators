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
