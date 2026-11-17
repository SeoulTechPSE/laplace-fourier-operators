# E5: data-side gates of the trust rule

v2 gate: all-sample linear residual < 0.1. v3 gate: lowest 20% of samples by peak |f| linear (residual < 0.1) and equivalent-pole shift all vs low < 0.1. A trusted run is correct if its Jacobian pole is within 10 % of the small-signal pole: −0.25 ± 0.9682i (damped Duffing, pendulum), ± i (`c0_*`, undamped Duffing) or −0.9287 ± 4.1476i (`lz_*`, Lorenz ρ = 5 at C+; centred data).

| set | all: pole, resid | low band (peak |f| ≤): pole, resid | shift | v2 | v3 | v1 trusted: correct / all (errors) | v2 false / missed | v3 false / missed |
|---|---|---|---|---|---|---|---|---|
| `small` | -0.251+0.971i, 0.005 | (0.04) -0.250+0.969i, 0.001 | 0.002 | pass | pass | 6 / 6 (0.00–0.01) | 0 / 0 | 0 / 0 |
| `mixed` | -0.893+1.753i, 0.439 | (1.01) -0.424+1.161i, 0.263 | 0.611 | fail | fail | 0 / 2 (0.43–0.86) | 0 / 0 | 0 / 0 |
| `medium` | -0.280+1.044i, 0.095 | (0.18) -0.253+0.981i, 0.016 | 0.067 | pass | pass | 6 / 6 (0.05–0.06) | 0 / 0 | 0 / 0 |
| `moderate` | -0.359+1.176i, 0.200 | (0.35) -0.263+1.016i, 0.053 | 0.179 | fail | fail | 0 / 4 (0.15–0.17) | 0 / 0 | 0 / 0 |
| `pend_orig` | -0.255+0.924i, 0.121 | (2.00) -0.250+0.967i, 0.004 | 0.043 | fail | pass | 3 / 3 (0.02–0.02) | 0 / 3 | 0 / 0 |
| `pend_medium` | -0.245+0.947i, 0.036 | (0.19) -0.249+0.964i, 0.004 | 0.018 | pass | pass | 6 / 6 (0.01–0.01) | 0 / 0 | 0 / 0 |
| `pend_large` | -0.255+0.889i, 0.236 | (0.45) -0.244+0.941i, 0.027 | 0.055 | fail | pass | 5 / 5 (0.04–0.05) | 0 / 5 | 0 / 0 |
| `wide` | -0.294+1.063i, 0.132 | (0.13) -0.251+0.974i, 0.007 | 0.098 | fail | pass | 7 / 7 (0.05–0.07) | 0 / 7 | 0 / 0 |
| `pend_wide` | -0.260+0.909i, 0.244 | (0.20) -0.250+0.966i, 0.003 | 0.059 | fail | pass | 5 / 5 (0.02–0.03) | 0 / 5 | 0 / 0 |
| `c0_orig` | -0.229+1.260i, 0.884 | (2.00) -0.004+1.022i, 0.208 | 0.320 | fail | fail | 0 / 0 (–) | 0 / 0 | 0 / 0 |
| `c0_small` | -0.002+1.010i, 0.078 | (0.04) -0.000+1.002i, 0.025 | 0.008 | pass | pass | 6 / 6 (0.01–0.01) | 0 / 0 | 0 / 0 |
| `c0_medium` | -0.032+1.049i, 0.356 | (0.11) -0.006+1.013i, 0.154 | 0.043 | fail | fail | 5 / 5 (0.04–0.04) | 0 / 5 | 0 / 5 |
| `lz_orig` | -0.661+3.883i, 0.527 | (2.00) -1.484+4.722i, 0.179 | 0.238 | fail | fail | 0 / 0 (–) | 0 / 0 | 0 / 0 |
| `lz_small` | -0.999+4.055i, 0.027 | (0.14) -1.008+4.066i, 0.016 | 0.003 | pass | pass | 0 / 0 (–) | 0 / 0 | 0 / 0 |
| `lz_wide` | -1.115+4.023i, 0.183 | (1.39) -1.027+4.083i, 0.041 | 0.025 | fail | pass | 0 / 1 (0.96–0.96) | 0 / 0 | 1 / 0 |

Totals: v1 alone 7 false trusts; v2 0 false, 25 correct runs not trusted; v3 1 false, 5 correct runs not trusted.
