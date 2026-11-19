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
