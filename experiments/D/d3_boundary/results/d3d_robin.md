# (D) Robin boundary with a learned coefficient: basis from learned eigenvalues

D = 0.02, H = 0.04: beta = H / D = 2.0. Range over data seeds 0–4.

| model | test / oracle | β̂ | D̂ | value-law error m ≤ 8 | flux-law error m ≤ 8 | law errors, all modes |
|---|---|---|---|---|---|---|
| robin | 6.37–8.35 | 0.0000–0.0006 | 0.01419–0.02508 | 1.5e-01–3.4e-01 | 1.3e-01–3.4e-01 | 6.8e-01–3.7e+00 |
| fixed | 1.00–1.01 | 2.0000–2.0000 | 0.01990–0.02016 | 2.7e-04–1.2e-02 | 6.0e-04–7.8e-03 | 1.2e-02–2.5e-01 |
| dn | 5.86–7.64 | 0.0000–0.0000 | 0.01327–0.01398 | 5.8e-02–1.5e-01 | 2.5e-01–3.0e-01 | 6.0e-01–7.4e-01 |
