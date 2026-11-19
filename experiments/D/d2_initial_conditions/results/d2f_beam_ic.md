# (D2) Damped beam with initial displacement and velocity, displacement output

c = 0.04, eta = 0.002; IC scale 1.24. Range over data seeds 0–4.

| model | test / oracle | ĉ | η̂ | rollout / oracle rollout (4 windows) |
|---|---|---|---|---|
| struct | 1.00–1.00 | 0.04000–0.04000 | 0.002000–0.002000 | 0.997–1.001 |
| poly | 50.53–87.54 | 0.04136–0.04152 | 0.021968–0.023486 | 120.084–187.139 |
| struct_u0 | 94.01–168.14 | 0.03993–0.04004 | 0.001735–0.002175 | 64.313–120.103 |

Oracle test error 2.08e-03–3.57e-03; oracle rollout error 6.16e-03–1.08e-02.
