# E4: minimal-form refit, order test and profile intervals, first-order-hold refits, errors-in-variables ML with the finite-record variance

As `--hold foh_ml`, but the cost is `EIVMLT`: the variance of each frequency line is the exact marginal variance of the propagated input noise over the finite record (zero initial state), σ_y² + σ_u² (‖x_i‖ / ‖y_i‖)² mean_m |G_m(iω)|², G_m the transfer of the impulse response truncated to m samples; every fit is bounded to stable poles (Re μ ≤ −1e-3). A reported pole that is a duplicate, aliased or undamped (|Re μ| T_rec < 1) is unresolved.

Stage `rescue_na` (noise-aware restarts), groups of 5 seeds (one noise draw each), truth-free up to the scores. Certified: S − S_min ≤ χ²_p(0.99) σ̂². Report: the consensus of the certified runs (≥ 2), else the best run's effective poles. The report is refitted as a minimal tied layer (one slot per reported pole); order test: S_minimal − S_best ≤ χ²_Δp(0.99) σ̂²; if the consensus fails it, the best run's poles are tried. Each fitted pole: 95 % profile interval of Re μ (others re-fitted, c ∈ [1/3, 3]); identified = within 0.1 of a true pole with the interval closed. Cell per true pole: identified groups / intervals covering the true Re μ; RMS relative error of the minimal fit; mean relative half-width of the interval; relative CRB (e2_noise.py).

## p4, input noise (errors-in-variables), `p4_ms16_T20_n1e-2` + η_u

Noise on the stored train / validation inputs as well (outputs from the clean inputs), output noise 1e-2. The pipeline above unchanged (σ̂² from the outputs). Covered: by the profile interval / by the truth-free EIV bootstrap (32 draws; σ_u² from the inputs by a robust estimate, bias-corrected Re μ ± 1.96 std). Half-width: profile / bootstrap.

| η_u | η̂_u | groups | certified runs (mean) | report from consensus | order test passed | -0.4+3i: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -1: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -0.2: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | false identified |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.29 | 1 | 1.0 | 0/1 | 0/1 | 0 / 0 / 0, –, – / – | 0 / 0 / 0, –, – / – | 0 / 0 / 0, –, – / – | 0 |
