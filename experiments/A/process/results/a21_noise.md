# (A) Delay identification under output noise

Noise σ = 1 % or 5 % of each signal's rms on training / validation outputs; clean test. Max relative error over seeds 0–4.

| system | noise | model | test / oracle | θ error | K error | τ error (max) |
|---|---|---|---|---|---|---|
| fopdt | 1% | pdelay | 1.01–1.01 | 0.06% | 0.03% | 0.13% |
| fopdt | 1% | arx | 1.46–1.90 | 2.19% | 0.89% | 5.51% |
| fopdt | 5% | pdelay | 1.03–1.05 | 0.08% | 0.10% | 0.19% |
| fopdt | 5% | arx | 15.05–22.40 | 16.79% | 9.53% | 63.77% |
| sopdt | 1% | pdelay | 1.01–1.03 | 3.08% | 0.07% | 0.64% |
| sopdt | 1% | arx | 7.52–11.31 | 32.53% | 10.04% | 87.46% |
| sopdt | 5% | pdelay | 1.03–1.08 | 3.32% | 0.19% | 1.32% |
| sopdt | 5% | arx | 6.83–11.21 | 56.63% | 5.54% | 86.26% |
