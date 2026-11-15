# E4: the noise-free pole bias (float32 rounding of inputs vs outputs)

Minimal first-order-hold refit started at the truth, training split (200 samples). Per pole: relative error of Re μ for the stored data (x32 y32), exact inputs (x64 y32), exact outputs (x32 y64) and both exact (x64 y64); Monte-Carlo std of the error over 32 uniform ±ulp/2 dithers of both / of the outputs only; profile half-width / 1.96 of `e4_profile_foh` (σ² = output rounding), the std the intervals assume; truth-free parametric bootstrap std (32 dithers of the stored inputs and fitted outputs / of the fitted outputs only); z = |error of the stored data| / bootstrap std (both).

| data | pole | x32 y32 | x64 y32 | x32 y64 | x64 y64 | MC std, both | MC std, outputs | half-width / 1.96 | boot std, both | boot std, outputs | z (profile) | z (boot) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p4_ms16_T20 | -0.4+3i | +3.0e-10 | +2.8e-10 | +2.2e-11 | +4.2e-16 | 6.0e-10 | 2.2e-10 | 3.6e-10 | 4.5e-10 | 3.7e-10 | 0.8 | 0.7 |
| p4_ms16_T20 | -1 | +2.4e-10 | +6.6e-11 | +1.8e-10 | -2.4e-15 | 7.9e-10 | 6.2e-10 | 6.2e-10 | 8.6e-10 | 6.0e-10 | 0.4 | 0.3 |
| p4_ms16_T20 | -0.2 | +3.0e-10 | -3.3e-10 | +6.3e-10 | +8.3e-16 | 5.4e-10 | 3.6e-10 | 3.6e-10 | 5.3e-10 | 3.0e-10 | 0.8 | 0.6 |
| p6_ms16_T80 | -0.5+1.5i | +1.0e-09 | +2.8e-10 | +7.4e-10 | -8.9e-16 | 6.9e-10 | 4.5e-10 | 5.0e-10 | 7.8e-10 | 5.8e-10 | 2.0 | 1.3 |
| p6_ms16_T80 | -0.3+4i | -5.1e-10 | -2.5e-10 | -2.6e-10 | +2.2e-15 | 3.6e-10 | 2.7e-10 | 2.9e-10 | 4.4e-10 | 3.5e-10 | 1.8 | 1.2 |
| p6_ms16_T80 | -1 | +7.7e-09 | +3.0e-09 | +4.6e-09 | -1.3e-15 | 4.4e-09 | 2.8e-09 | 3.7e-09 | 5.5e-09 | 4.1e-09 | 2.1 | 1.4 |
| p6_ms16_T80 | -2 | -2.5e-08 | -7.5e-09 | -1.7e-08 | +5.3e-15 | 1.3e-08 | 8.2e-09 | 1.1e-08 | 1.7e-08 | 1.2e-08 | 2.3 | 1.5 |
| p6_ms16_T80_N4096 | -0.5+1.5i | +4.7e-10 | -2.8e-11 | +5.0e-10 | +1.8e-14 | 3.4e-10 | 2.3e-10 | 2.5e-10 | 4.0e-10 | 2.1e-10 | 1.9 | 1.2 |
| p6_ms16_T80_N4096 | -0.3+4i | -8.3e-11 | +3.8e-11 | -1.2e-10 | +3.7e-15 | 1.7e-10 | 1.2e-10 | 1.4e-10 | 1.6e-10 | 1.5e-10 | 0.6 | 0.5 |
| p6_ms16_T80_N4096 | -1 | +3.7e-09 | +9.9e-10 | +2.7e-09 | -3.1e-15 | 2.6e-09 | 1.7e-09 | 1.8e-09 | 2.5e-09 | 1.8e-09 | 2.0 | 1.5 |
| p6_ms16_T80_N4096 | -2 | -1.0e-08 | -1.5e-09 | -8.6e-09 | -8.7e-15 | 7.2e-09 | 5.4e-09 | 5.4e-09 | 7.8e-09 | 5.0e-09 | 1.9 | 1.3 |
| p4_e1_T20 | -0.4+3i | +2.6e-11 | -5.0e-10 | +5.2e-10 | +2.5e-15 | 1.2e-09 | 5.2e-10 | 5.9e-10 | 1.5e-09 | 5.9e-10 | 0.0 | 0.0 |
| p4_e1_T20 | -1 | -1.5e-09 | +9.6e-10 | -2.4e-09 | -4.4e-15 | 1.7e-09 | 8.7e-10 | 8.9e-10 | 1.2e-09 | 9.4e-10 | 1.7 | 1.2 |
| p4_e1_T20 | -0.2 | +1.4e-09 | -3.2e-10 | +1.7e-09 | +3.2e-15 | 8.4e-10 | 4.9e-10 | 4.4e-10 | 8.4e-10 | 5.2e-10 | 3.2 | 1.7 |
| p6_e1_T20 | -0.5+1.5i | +1.5e-09 | +8.6e-10 | +6.2e-10 | +0.0e+00 | 2.8e-09 | 2.0e-09 | 1.7e-09 | 2.8e-09 | 1.8e-09 | 0.9 | 0.5 |
| p6_e1_T20 | -0.3+4i | +6.6e-10 | +8.6e-10 | -2.0e-10 | +0.0e+00 | 1.9e-09 | 7.3e-10 | 1.0e-09 | 2.0e-09 | 9.3e-10 | 0.6 | 0.3 |
| p6_e1_T20 | -1 | -4.2e-10 | -3.6e-09 | +3.1e-09 | +0.0e+00 | 2.0e-08 | 1.2e-08 | 1.2e-08 | 2.0e-08 | 1.4e-08 | 0.0 | 0.0 |
| p6_e1_T20 | -2 | -1.4e-08 | +1.2e-10 | -1.4e-08 | -6.7e-16 | 5.4e-08 | 3.8e-08 | 3.7e-08 | 6.0e-08 | 4.4e-08 | 0.4 | 0.2 |

Within ± 1.96 std (95 % interval): profile (σ² = output rounding) 13/18, bootstrap (inputs + outputs) 18/18. Every group of a condition shares these data, so this is one realisation per data set.


S at the truth-started fit / n σ²_out: p4_ms16_T20 1.04 (exact inputs 1.00), p6_ms16_T80 1.04 (exact inputs 1.00), p6_ms16_T80_N4096 1.01 (exact inputs 1.00), p4_e1_T20 1.31 (exact inputs 1.00), p6_e1_T20 1.21 (exact inputs 1.00)
