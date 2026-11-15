# E4: minimal-form refit, order test and profile intervals, first-order-hold refits

Refits, certificate and order-test reference use the exact first-order-hold response (the data generator's input model) instead of the layer's band-limited one. Each run is refitted and reduced (backward elimination) before the certificate.

Stage `rescue_na` (noise-aware restarts), groups of 5 seeds (one noise draw each), truth-free up to the scores. Certified: S − S_min ≤ χ²_p(0.99) σ̂². Report: the consensus of the certified runs (≥ 2), else the best run's effective poles. The report is refitted as a minimal tied layer (one slot per reported pole); order test: S_minimal − S_best ≤ χ²_Δp(0.99) σ̂²; if the consensus fails it, the best run's poles are tried. Each fitted pole: 95 % profile interval of Re μ (others re-fitted, c ∈ [1/3, 3]); identified = within 0.1 of a true pole with the interval closed. Cell per true pole: identified groups / intervals covering the true Re μ; RMS relative error of the minimal fit; mean relative half-width of the interval; relative CRB (e2_noise.py).

## p4

| η | groups | certified runs (mean) | report from consensus | order test passed | -0.4+3i: identified / covered, error, half-width, CRB | -1: identified / covered, error, half-width, CRB | -0.2: identified / covered, error, half-width, CRB | false identified |
|---|---|---|---|---|---|---|---|---|
| 0.001 | 4 | 5.0 | 4/4 | 4/4 | 4/3, 2.7e-06, 3.7e-06, 2.5e-06 | 4/4, 2.1e-05, 4.8e-05, 2.5e-05 | 4/4, 9.9e-06, 2.8e-05, 1.4e-05 | 0 |
| 0.01 | 4 | 5.0 | 4/4 | 4/4 | 4/3, 2.7e-05, 3.7e-05, 2.5e-05 | 4/4, 2.1e-04, 4.8e-04, 2.5e-04 | 4/4, 9.9e-05, 2.8e-04, 1.4e-04 | 0 |
| 0.1 | 4 | 5.0 | 4/4 | 4/4 | 4/3, 2.7e-04, 3.7e-04, 2.5e-04 | 4/4, 2.1e-03, 4.8e-03, 2.5e-03 | 4/4, 1.0e-03, 2.8e-03, 1.4e-03 | 0 |
| 0.3 | 4 | 5.0 | 4/4 | 4/4 | 4/3, 8.1e-04, 1.1e-03, 7.5e-04 | 4/4, 6.4e-03, 1.4e-02, 7.4e-03 | 4/4, 3.0e-03, 8.5e-03, 4.3e-03 | 0 |

## p6

| η | groups | certified runs (mean) | report from consensus | order test passed | -0.5+1.5i: identified / covered, error, half-width, CRB | -0.3+4i: identified / covered, error, half-width, CRB | -1: identified / covered, error, half-width, CRB | -2: identified / covered, error, half-width, CRB | false identified |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 4 | 5.0 | 4/4 | 4/4 | 4/4, 2.0e-06, 6.2e-06, 4.1e-06 | 4/2, 7.5e-07, 8.4e-07, 5.9e-07 | 4/4, 3.3e-05, 1.4e-04, 7.2e-05 | 4/4, 1.2e-04, 4.2e-04, 2.1e-04 | 0 |
| 0.01 | 4 | 5.0 | 4/4 | 4/4 | 4/4, 2.0e-05, 6.2e-05, 4.1e-05 | 4/2, 7.5e-06, 8.4e-06, 5.9e-06 | 4/4, 3.3e-04, 1.4e-03, 7.2e-04 | 4/4, 1.2e-03, 4.2e-03, 2.1e-03 | 0 |
| 0.1 | 4 | 5.0 | 4/4 | 4/4 | 4/4, 2.0e-04, 6.2e-04, 4.1e-04 | 4/2, 7.5e-05, 8.4e-05, 5.9e-05 | 4/4, 3.3e-03, 1.4e-02, 7.2e-03 | 4/4, 1.2e-02, 4.2e-02, 2.1e-02 | 0 |
| 0.3 | 4 | 4.8 | 4/4 | 4/4 | 4/4, 6.1e-04, 1.9e-03, 1.2e-03 | 4/2, 2.3e-04, 2.5e-04, 1.8e-04 | 4/4, 1.1e-02, 4.8e-02, 2.2e-02 | 4/4, 3.8e-02, 1.3e-01, 6.4e-02 | 0 |

## p6, runs of the FOH-trained layer (`*_foh`)

| η | groups | certified runs (mean) | report from consensus | order test passed | -0.5+1.5i: identified / covered, error, half-width, CRB | -0.3+4i: identified / covered, error, half-width, CRB | -1: identified / covered, error, half-width, CRB | -2: identified / covered, error, half-width, CRB | false identified |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 4 | 5.0 | 4/4 | 4/4 | 4/4, 2.0e-05, 6.2e-05, 4.1e-05 | 4/2, 7.5e-06, 8.4e-06, 5.9e-06 | 4/4, 3.3e-04, 1.4e-03, 7.2e-04 | 4/4, 1.2e-03, 4.2e-03, 2.1e-03 | 0 |
| 0.1 | 4 | 5.0 | 4/4 | 4/4 | 4/4, 2.0e-04, 6.2e-04, 4.1e-04 | 4/2, 7.5e-05, 8.4e-05, 5.9e-05 | 4/4, 3.3e-03, 1.4e-02, 7.2e-03 | 4/4, 1.2e-02, 4.2e-02, 2.1e-02 | 0 |

## p4, noise-free data (σ² = float32 rounding of the outputs)

Fits minimise S / (n σ²) and end with damped Newton steps. Covered: by the 95 % profile interval (σ² from the outputs only) / by the bootstrap interval Re μ ± 1.96 std (32 dithers of the stored inputs and fitted outputs by ± ulp/2). All groups of a condition share one data set, so coverage is one realisation per condition.

| condition | groups | certified runs (mean) | report from consensus | order test passed | -0.4+3i: identified / covered (profile / bootstrap), error, half-width | -1: identified / covered (profile / bootstrap), error, half-width | -0.2: identified / covered (profile / bootstrap), error, half-width | false identified |
|---|---|---|---|---|---|---|---|---|
| p4_ms16_T20 | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 5.6e-11, 9.3e-11 | 4 / 4 / 4, 2.5e-10, 1.2e-09 | 4 / 4 / 4, 3.0e-10, 7.1e-10 | 0 |
| p4_ms16_T20_foh | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 5.6e-11, 9.3e-11 | 4 / 4 / 4, 2.5e-10, 1.2e-09 | 4 / 4 / 4, 3.0e-10, 7.1e-10 | 0 |
| p4_e1_T20 | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 8.6e-12, 1.5e-10 | 4 / 4 / 4, 1.5e-09, 1.8e-09 | 4 / 0 / 4, 1.4e-09, 8.5e-10 | 0 |
| p4_e1_T20_foh | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 8.6e-12, 1.5e-10 | 4 / 4 / 4, 1.5e-09, 1.8e-09 | 4 / 0 / 4, 1.4e-09, 8.5e-10 | 0 |

## p6, noise-free data (σ² = float32 rounding of the outputs)

Fits minimise S / (n σ²) and end with damped Newton steps. Covered: by the 95 % profile interval (σ² from the outputs only) / by the bootstrap interval Re μ ± 1.96 std (32 dithers of the stored inputs and fitted outputs by ± ulp/2). All groups of a condition share one data set, so coverage is one realisation per condition.

| condition | groups | certified runs (mean) | report from consensus | order test passed | -0.5+1.5i: identified / covered (profile / bootstrap), error, half-width | -0.3+4i: identified / covered (profile / bootstrap), error, half-width | -1: identified / covered (profile / bootstrap), error, half-width | -2: identified / covered (profile / bootstrap), error, half-width | false identified |
|---|---|---|---|---|---|---|---|---|---|
| p6_ms16_T80 | 4 | 5.0 | 4/4 | 4/4 | 4 / 0 / 4, 4.4e-10, 3.1e-10 | 4 / 4 / 4, 4.2e-11, 4.3e-11 | 4 / 0 / 4, 7.7e-09, 7.2e-09 | 4 / 0 / 4, 2.5e-08, 2.1e-08 | 0 |
| p6_ms16_T80_foh | 4 | 5.0 | 4/4 | 4/4 | 4 / 0 / 4, 4.4e-10, 3.1e-10 | 4 / 4 / 4, 4.2e-11, 4.3e-11 | 4 / 0 / 4, 7.7e-09, 7.2e-09 | 4 / 0 / 4, 2.5e-08, 2.1e-08 | 0 |
| p6_ms16_T80_N4096 | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 1.9e-10, 1.6e-10 | 4 / 4 / 4, 2.2e-11, 2.1e-11 | 4 / 0 / 4, 3.7e-09, 3.6e-09 | 4 / 4 / 4, 1.0e-08, 1.1e-08 | 0 |
| p6_ms16_T80_N4096_foh | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 1.9e-10, 1.6e-10 | 4 / 4 / 4, 2.2e-11, 2.1e-11 | 4 / 0 / 4, 3.7e-09, 3.6e-09 | 4 / 4 / 4, 1.0e-08, 1.1e-08 | 0 |
| p6_e1_T20 | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 4.7e-10, 1.1e-09 | 4 / 4 / 4, 5.1e-11, 1.5e-10 | 4 / 4 / 4, 4.8e-10, 2.3e-08 | 4 / 4 / 4, 1.4e-08, 7.2e-08 | 0 |
| p6_e1_T20_foh | 4 | 5.0 | 4/4 | 4/4 | 4 / 4 / 4, 4.7e-10, 1.1e-09 | 4 / 4 / 4, 5.1e-11, 1.5e-10 | 4 / 4 / 4, 4.8e-10, 2.3e-08 | 4 / 4 / 4, 1.4e-08, 7.2e-08 | 0 |
