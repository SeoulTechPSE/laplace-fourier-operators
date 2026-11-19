# (A) Lumped delay against distributed dispersion (periodic dispersion reactor)

v = 0.5, D = 0.02, k_r = 0.1; source at x0, sensors at distance L.

| L | L / v | t̄ (truth) | FOPDT θ | FOPDT θ + τ | SOPDT θ | SOPDT θ + τ₁ + τ₂ | FOPDT / SOPDT test |
|---|---|---|---|---|---|---|---|
| 1 | 2.00 | 2.230 | 1.281 | 2.301 | 1.009 | 2.201 | 1.04e-01 / 5.62e-02 |
| 2 | 4.00 | 4.077 | 2.991 | 4.415 | 2.652 | 4.237 | 1.43e-01 / 8.13e-02 |
| 3 | 6.00 | 6.044 | 4.682 | 6.727 | 4.298 | 6.327 | 1.72e-01 / 1.01e-01 |

Slope of the mean delay against L: SOPDT 2.0630, FOPDT 2.2128 (1 / v = 2.0000).
Wavenumber layer on the field: v = 0.50001, D = 0.01999, k_r = 0.10000 (test 1.01e-02).
