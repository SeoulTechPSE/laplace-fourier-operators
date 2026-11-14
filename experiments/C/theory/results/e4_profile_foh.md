# E4: minimal-form refit, order test and profile intervals, first-order-hold refits

Refits, certificate and order-test reference use the exact first-order-hold response (the data generator's input model) instead of the layer's band-limited one.

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
