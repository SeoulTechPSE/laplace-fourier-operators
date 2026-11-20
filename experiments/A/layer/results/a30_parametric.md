# (A) Cahn–Hilliard linear regime with M, kappa, phi0 as inputs: extrapolation in (M, kappa)

Test error / oracle, mean over φ0 ∈ {0.15, 0.45, 0.7}; range over data seeds 0–4. Points (M, κ): interp (0.075, 0.045), hiM (0.2, 0.08), lowk (0.025, 0.015), hiMk (0.15, 0.045).

| model | interp | hiM | lowk | hiMk | worst (point, φ0) |
|---|---|---|---|---|---|
| phys | 1.00–1.00 | 1.00–1.00 | 1.00–1.00 | 1.00–1.00 | 1.01 |
| bilin | 1.00–1.49 | 1.16–5.42 | 1.25–8.98 | 1.01–2.10 | 17.39 |
| mlp | 26608.18–459580.37 | 895.13–1123.84 | 479.25–174546574601296.47 | 579.69–720.04 | 419845550229062.62 |
| fixed | 149.97–184.83 | 576.97–826.47 | 341.43–437.01 | 257.45–324.35 | 1875.47 |

k_c(0) relative error: phys: interp 9.6e-06–9.6e-06, hiM 1.5e-04–1.5e-04, lowk 1.2e-04–1.2e-04, hiMk 9.6e-06–9.6e-06; bilin: interp 9.6e-06–2.2e-04, hiM 1.5e-04–1.5e-04, lowk 4.2e-06–9.8e-04, hiMk 9.6e-06–2.2e-04.
