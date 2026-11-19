# (A) RQ4: fractional diffusion λ = −0.15 |k|^1.5, polynomial against MLP μ(k)

Range over data seeds 0–4. `full`: trained and tested on k = 1..16. `band`: trained on k ≤ 8, tested in and out of band.

| regime | μ(k) | test / oracle (in) | test / oracle (out) | dispersion error (in) | dispersion error k = 9..16 |
|---|---|---|---|---|---|
| full | poly | 5.43–8.54 | – | 1.9e-01–3.5e-01 | – |
| full | mlp | 1.02–1.04 | – | 5.5e-03–1.4e-02 | – |
| band | poly | 3.28–4.77 | 3511.62–5331.87 | 5.3e-02–5.8e-02 | 9.6e-01–9.8e-01 |
| band | mlp | 1.01–1.01 | 12.22–17.52 | 6.7e-04–2.7e-03 | 2.3e-01–2.7e-01 |
