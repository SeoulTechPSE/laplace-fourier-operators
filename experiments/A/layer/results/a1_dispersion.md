# (A) First linear validation: dispersion recovery by the wavenumber pole layer

One layer, FOH time axis, polynomial μ(k), truth on a 32× finer time grid. Dispersion error over the forced band k = 1..16 (dominant pole). 5 seeds.

| system | N | oracle floor | test rel. L2 (range) | dispersion error (range) | D | v | k_r |
|---|---|---|---|---|---|---|---|
| diff | 1 | 8.3e-03 | 8.3e-03–9.4e-01 | 3.4e-03–1.6e+02 | -0.1334–0.1000 (0.1) | 0.0000–0.3485 (0) | -0.0001–0.0716 (0) |
| diff | 2 | 8.3e-03 | 1.1e-02–3.7e-01 | 4.7e-01–2.8e+02 | – | – | – |
| adr | 1 | 8.7e-03 | 2.4e-01–4.7e-01 | 3.6e+01–2.9e+02 | 0.0130–0.2529 (0.05) | 0.2642–0.5685 (0.5) | -0.0050–0.2714 (0.2) |
| adr | 2 | 8.7e-03 | 1.2e-02–3.4e-01 | 8.0e+01–5.7e+02 | – | – | – |
