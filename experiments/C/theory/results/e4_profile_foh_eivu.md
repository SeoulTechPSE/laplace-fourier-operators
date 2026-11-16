# E4: minimal-form refit, order test and profile intervals, calibrated pipeline with aliasing fold and unresolved flags

As `--hold foh_eivo`, with two truth-free rules. Aliasing fold: before its calibrated reduction every run's poles with |Im μ| > π/dt are folded into (−π/dt, π/dt] (the sampled data cannot tell them apart). Unresolved: a calibrated step that drops a pole next to a kept one (|z_removed − z_kept| ≤ |z_kept|) with χ²_Δp(0.99) < ΔS ≤ κ χ²_Δp(0.99), i.e. taken only because of the calibration, flags the kept pole; a reported pole is unresolved if the report's steps or any certified run flag it. An unresolved pole counts as neither identified nor false (its interval is treated as open).

Stage `rescue_na` (noise-aware restarts), groups of 5 seeds (one noise draw each), truth-free up to the scores. Certified: S − S_min ≤ χ²_p(0.99) σ̂². Report: the consensus of the certified runs (≥ 2), else the best run's effective poles. The report is refitted as a minimal tied layer (one slot per reported pole); order test: S_minimal − S_best ≤ χ²_Δp(0.99) σ̂²; if the consensus fails it, the best run's poles are tried. Each fitted pole: 95 % profile interval of Re μ (others re-fitted, c ∈ [1/3, 3]); identified = within 0.1 of a true pole with the interval closed. Cell per true pole: identified groups / intervals covering the true Re μ; RMS relative error of the minimal fit; mean relative half-width of the interval; relative CRB (e2_noise.py).

## p4, input noise (errors-in-variables), `p4_ms16_T20_n1e-2` + η_u

Noise on the stored train / validation inputs as well (outputs from the clean inputs), output noise 1e-2. The pipeline above unchanged (σ̂² from the outputs). Covered: by the profile interval / by the truth-free EIV bootstrap (32 draws; σ_u² from the inputs by a robust estimate, bias-corrected Re μ ± 1.96 std). Half-width: profile / bootstrap.

| η_u | η̂_u | groups | certified runs (mean) | report from consensus | order test passed | -0.4+3i: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -1: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -0.2: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | false identified |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.1 | 8 | 5.0 | 8/8 | 8/8 | 8 / 3 / 7, 4.3e-04, 3.3e-04 / 4.6e-04 | 7 / 3 / 7, 7.0e-03, 3.9e-03 / 6.6e-03 | 4 / 5 / 7, 3.8e-03, 2.6e-03 / 4.7e-03 | 0 |
| 0.3 | 0.29 | 4 | 5.0 | 4/4 | 4/4 | 4 / 0 / 3, 7.1e-03, 8.1e-04 / 2.5e-03 | 3 / 0 / 3, 9.3e-02, 8.9e-03 / 3.2e-02 | 3 / 1 / 4, 2.5e-02, 6.3e-03 / 2.7e-02 | 1 |

## p6, input noise (errors-in-variables), `p6_e1_T20_n1e-2` + η_u

Noise on the stored train / validation inputs as well (outputs from the clean inputs), output noise 1e-2. The pipeline above unchanged (σ̂² from the outputs). Covered: by the profile interval / by the truth-free EIV bootstrap (32 draws; σ_u² from the inputs by a robust estimate, bias-corrected Re μ ± 1.96 std). Half-width: profile / bootstrap.

| η_u | η̂_u | groups | certified runs (mean) | report from consensus | order test passed | -0.5+1.5i: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -0.3+4i: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -1: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | -2: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap) | false identified |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.031 | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 3, 7.4e-04, 1.2e-03 / 1.3e-03 | 4 / 4 / 4, 1.1e-04, 1.7e-04 / 1.6e-04 | 4 / 4 / 4, 1.3e-02, 2.6e-02 / 2.8e-02 | 4 / 4 / 4, 3.9e-02, 7.4e-02 / 8.4e-02 | 0 |
| 0.1 | 0.1 | 4 | 3.5 | 2/4 | 4/4 | 4 / 2 / 2, 2.0e-02, 2.6e-03 / 2.5e-03 | 4 / 2 / 3, 2.0e-03, 4.2e-04 / 5.2e-04 | 2 / 2 / 2, 4.3e-02, 5.6e-02 / 5.2e-02 | 0 / 0 / 0, –, – / – | 8 |
