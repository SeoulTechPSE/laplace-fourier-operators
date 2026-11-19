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
