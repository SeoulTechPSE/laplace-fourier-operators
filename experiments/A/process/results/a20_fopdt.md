# (A) Delay identification on FOPDT / SOPDT

FOPDT: K 2, τ 1.5, θ 1.37. SOPDT: K 1, τ 2.0 / 0.5, θ 0.83. dt 0.05, T 20, noise-free, data seeds 0–4. Ranges over seeds.

| system | model | test / oracle | θ̂ | K̂ | τ̂ |
|---|---|---|---|---|---|
| fopdt | pdelay | 1.00–1.00 | 1.3701–1.3701 | 2.0000–2.0000 | 1.4998–1.4999 |
| fopdt | pole_only | 1.57–1.95 | – | – | – |
| fopdt | arx | 1.12–1.18 | 1.3500–1.3500 | 2.0019–2.0032 | 1.5137–1.5194 |
| sopdt | pdelay | 1.00–1.00 | 0.8231–0.8391 | 1.0000–1.0000 | 2.0000–2.0001; 0.4996–0.4999 |
| sopdt | pole_only | 1.06–1.09 | – | – | – |
| sopdt | arx | 1.08–1.12 | 0.8500–0.8500 | 0.9999–1.0003 | 1.9982–2.0042; 0.4986–0.5010 |
