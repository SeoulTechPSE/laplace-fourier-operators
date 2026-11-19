# (D2) Initial-condition path and window chaining (ADR, periodic)

Range over data seeds 0–4. Rollout: 4 chained windows on [0, 20]; ratio to the single-window test error.

| model | test / oracle | D̂ | v̂ | k̂_r | max abs(γ(k) − 1) | rollout error | rollout / test |
|---|---|---|---|---|---|---|---|
| ic | 1.00–1.00 | 0.0500–0.0500 | 0.5000–0.5000 | 0.2000–0.2000 | 3.3e-06–1.1e-05 | 1.33e-03–2.25e-03 | 0.95–1.73 |
| noic | 569.62–750.35 | – | – | – | – | – | – |
