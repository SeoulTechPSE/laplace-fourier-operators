# E4: minimal-form refit, order test and profile intervals, first-order-hold refits, errors-in-variables ML

As `--hold foh`, but every refit minimizes the frequency-domain errors-in-variables ML cost Σ |R|² / (σ_y² + σ_u² (‖x_i‖ / ‖y_i‖)² |G_θ(iω)|²) (`EIVML`; poles and residues jointly, the weights move with θ). S is in noise units and every test uses σ² = 1 (no calibration). A reported pole within 10 % of another reported pole (duplicate) or with |Im μ| > π/dt (aliased) is unresolved: neither identified nor false.

Stage `rescue_na` (noise-aware restarts), groups of 5 seeds (one noise draw each), truth-free up to the scores. Certified: S − S_min ≤ χ²_p(0.99) σ̂². Report: the consensus of the certified runs (≥ 2), else the best run's effective poles. The report is refitted as a minimal tied layer (one slot per reported pole); order test: S_minimal − S_best ≤ χ²_Δp(0.99) σ̂²; if the consensus fails it, the best run's poles are tried. Each fitted pole: 95 % profile interval of Re μ (others re-fitted, c ∈ [1/3, 3]); identified = within 0.1 of a true pole with the interval closed. Cell per true pole: identified groups / intervals covering the true Re μ; RMS relative error of the minimal fit; mean relative half-width of the interval; relative CRB (e2_noise.py).

## p4

| η | groups | certified runs (mean) | report from consensus | order test passed | -0.4+3i: identified / covered, error, half-width, CRB | -1: identified / covered, error, half-width, CRB | -0.2: identified / covered, error, half-width, CRB | false identified |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 4 | 5.0 | 4/4 | 4/4 | 4/3, 2.7e-05, 3.7e-05, 2.5e-05 | 4/4, 2.1e-04, 4.8e-04, 2.5e-04 | 4/4, 9.9e-05, 2.8e-04, 1.4e-04 | 0 |

## p6, E1 data

| η | groups | certified runs (mean) | report from consensus | order test passed | -0.5+1.5i: identified / covered, error, half-width, CRB | -0.3+4i: identified / covered, error, half-width, CRB | -1: identified / covered, error, half-width, CRB | -2: identified / covered, error, half-width, CRB | false identified |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 4 | 5.0 | 4/4 | 4/4 | 4/4, 1.9e-04, 4.2e-04, 2.8e-04 | 4/4, 4.8e-05, 5.9e-05, 4.1e-05 | 4/4, 3.9e-03, 9.3e-03, 4.7e-03 | 4/4, 1.1e-02, 2.9e-02, 1.5e-02 | 0 |

## p4, input noise (errors-in-variables), `p4_ms16_T20_n1e-2` + η_u

Noise on the stored train / validation inputs as well (outputs from the clean inputs), output noise 1e-2. The pipeline above unchanged (σ̂² from the outputs). Covered: by the profile interval / by the truth-free EIV bootstrap (32 draws; σ_u² from the inputs by a robust estimate, bias-corrected Re μ ± 1.96 std). Half-width: profile / bootstrap.

| η_u | η̂_u | groups | certified runs (mean) | report from consensus | order test passed | -0.4+3i: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -1: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -0.2: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | false identified |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6.7e-07 | 4 | 5.0 | 4/4 | 4/4 | 4 / 3 / 3, 2.7e-05, 3.7e-05 / 2.9e-05 | 4 / 4 / 4, 2.1e-04, 4.8e-04 / 4.7e-04 | 4 / 4 / 4, 9.9e-05, 2.8e-04 / 2.6e-04 | 0 |
| 0.01 | 0.01 | 4 | 4.0 | 3/4 | 4/4 | 4 / 4 / 3, 3.4e-05, 5.6e-05 / 3.8e-05 | 4 / 4 / 3, 4.3e-04, 7.9e-04 / 5.4e-04 | 4 / 3 / 3, 2.4e-03, 1.6e-03 / 4.6e-04 | 0 |
| 0.03 | 0.03 | 4 | 4.0 | 3/4 | 4/4 | 4 / 4 / 3, 8.1e-05, 1.3e-04 / 1.0e-04 | 4 / 4 / 4, 4.2e-04, 1.4e-03 / 1.5e-03 | 4 / 4 / 4, 4.6e-04, 9.7e-04 / 1.0e-03 | 0 |
| 0.1 | 0.1 | 8 | 1.0 | 0/8 | 7/8 | 8 / 6 / 7, 5.0e-04, 3.8e-04 / 5.2e-04 | 8 / 3 / 6, 2.0e-02, 5.7e-03 / 7.2e-03 | 7 / 4 / 6, 6.0e-03, 2.2e-03 / 5.9e-03 | 2 |
| 0.3 | 0.29 | 12 | 1.2 | 0/12 | 10/12 | 10 / 4 / 6, 1.7e-02, 7.2e+61 / 9.7e-02 | 7 / 1 / 3, 3.8e-02, 6.1e-01 / 2.3e-01 | 10 / 2 / 7, 3.9e-02, 5.5e-02 / 4.9e-02 | 14 |

## p6, input noise (errors-in-variables), `p6_e1_T20_n1e-2` + η_u

Noise on the stored train / validation inputs as well (outputs from the clean inputs), output noise 1e-2. The pipeline above unchanged (σ̂² from the outputs). Covered: by the profile interval / by the truth-free EIV bootstrap (32 draws; σ_u² from the inputs by a robust estimate, bias-corrected Re μ ± 1.96 std). Half-width: profile / bootstrap.

| η_u | η̂_u | groups | certified runs (mean) | report from consensus | order test passed | -0.5+1.5i: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -0.3+4i: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -1: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -2: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | false identified |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.1e-08 | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 1.9e-04, 4.2e-04 / 3.8e-04 | 4 / 4 / 4, 4.8e-05, 5.9e-05 / 6.3e-05 | 4 / 4 / 4, 3.9e-03, 9.3e-03 / 9.0e-03 | 4 / 4 / 4, 1.1e-02, 2.9e-02 / 2.8e-02 | 0 |
| 0.01 | 0.01 | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 3.4e-04, 6.0e-04 / 6.0e-04 | 4 / 4 / 4, 4.9e-05, 8.6e-05 / 7.5e-05 | 4 / 4 / 4, 6.9e-03, 1.3e-02 / 1.3e-02 | 4 / 4 / 4, 2.1e-02, 3.9e-02 / 4.0e-02 | 0 |
| 0.03 | 0.031 | 4 | 5.0 | 4/4 | 4/4 | 4 / 3 / 3, 8.1e-04, 1.2e-03 / 1.3e-03 | 4 / 4 / 4, 1.0e-04, 1.7e-04 / 1.6e-04 | 4 / 4 / 4, 1.4e-02, 2.8e-02 / 3.2e-02 | 4 / 4 / 4, 4.0e-02, 7.4e-02 / 8.4e-02 | 0 |
| 0.05 | 0.051 | 8 | 4.5 | 7/8 | 8/8 | 8 / 7 / 6, 1.3e-03, 1.7e-03 / 1.8e-03 | 8 / 7 / 6, 2.1e-04, 2.4e-04 / 2.3e-04 | 8 / 7 / 7, 2.8e-02, 3.9e-02 / 4.7e-02 | 6 / 6 / 6, 3.6e-02, 1.1e-01 / 1.3e-01 | 2 |
| 0.1 | 0.1 | 12 | 1.3 | 1/12 | 12/12 | 12 / 5 / 7, 4.5e-03, 2.3e-03 / 2.8e-03 | 11 / 6 / 8, 6.6e-04, 3.8e-04 / 4.8e-04 | 8 / 6 / 6, 5.6e-02, 6.2e-02 / 8.0e-02 | 5 / 4 / 4, 6.4e-02, 1.8e-01 / 1.8e-01 | 12 |
