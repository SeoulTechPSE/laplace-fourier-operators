# (A) Deep wavenumber pole operator on forced Burgers (ν = 0.1), against FNO

Solver probe (method check): ν̂ = 0.10046, dispersion error 1.8e-02, kept k = 2–9.

Probe: kept k listed per run in the JSON; no kept mode → ν̂ = nan, dispersion error inf.

| model | seed | params | test (T = 5) | T = 10: (0, 5] / (5, 10] | probe ν̂ | probe dispersion error | probe residual (median) |
|---|---|---|---|---|---|---|---|
| wpo | 0 | 59529 | 1.369e-01 | 1.056e-01 / 8.478e+00 | 0.03187 | 1.0e+00 | 3.4e-01 |
| wpo | 1 | 59529 | 1.231e-01 | 9.474e-02 / 7.485e+00 | 0.07350 | 1.0e+00 | 8.1e-02 |
| wpo | 2 | 59529 | 9.075e-02 | 6.623e-02 / 3.226e+00 | 0.08100 | 9.6e-01 | 1.1e-01 |
| fno | 0 | 59110 | 1.059e-01 | 3.995e-01 / 3.026e-01 | 0.10566 | 3.8e-01 | 7.1e-02 |
| fno | 1 | 59110 | 1.030e-01 | 4.003e-01 / 3.024e-01 | 0.10426 | 3.8e-01 | 7.1e-02 |
| fno | 2 | 59110 | 1.067e-01 | 3.993e-01 / 3.035e-01 | 0.10881 | 4.2e-01 | 7.2e-02 |

WPO first layer: relative error of the pole closest to −ν k² at k = 1, 4, 8, 16:

- seed 0: 0.33, 0.67, 0.88, 0.93
- seed 1: 0.13, 0.83, 0.93, 0.93
- seed 2: 0.40, 0.84, 0.93, 0.93
