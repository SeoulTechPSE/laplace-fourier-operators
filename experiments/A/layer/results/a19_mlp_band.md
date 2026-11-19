# (A) RQ4: polynomial against MLP pole functions; training forced at k ≤ 8

Test / oracle on data forced in-band (k ≤ 8) and out-of-band (k = 9..16); dispersion error over k = 1..8 and 9..16. Range over data seeds 0–4.

| system | μ(k) | in-band test / oracle | out-of-band test / oracle | dispersion error k ≤ 8 | k = 9..16 |
|---|---|---|---|---|---|
| diff | poly | 1.00–1.00 | 1.00–1.00 | 9.3e-05–4.8e-04 | 3.2e-04–2.0e-03 |
| diff | mlp | 1.01–1.05 | 27.62–43.74 | 1.2e-03–9.4e-03 | 4.7e-01–5.5e-01 |
| adr | poly | 1.00–1.00 | 1.00–1.01 | 2.9e-04–9.1e-04 | 1.3e-03–4.6e-03 |
| adr | mlp | 1.00–1.03 | 4.57–18.67 | 5.3e-04–3.7e-03 | 1.3e-01–3.3e-01 |
| beam | poly | 1.00–1.01 | 1.00–1.08 | 7.3e-05–2.4e-04 | 2.2e-04–1.0e-03 |
| beam | mlp | 1.03–1.05 | 29.28–41.21 | 5.8e-03–7.3e-03 | 2.7e-01–3.8e-01 |
