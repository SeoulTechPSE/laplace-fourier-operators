# E3 deep, first pass: bounded slots in full ALNO / LFNO stacks

p4, multisine16, T = 20, dt = 20/1024; E1 deep budget; `common` + pole lr × 10. Per true pole: fraction of seeds with relative error < 0.1, pooled over all slots (E1 metric) / over slots with ≥ 2% of their layer's impulse-response weight. Tied-layer oracle 5.9e-04.

| model | variant | params | seeds | NaN / dead | test rel. L2 (Adam → final) | unstable poles | slots (weighted / all) | -0.4+3i: pooled / weighted | -1: pooled / weighted | -0.2: pooled / weighted |
|---|---|---|---|---|---|---|---|---|---|---|
| ALNO | plr10 | 5977 | 5 | 0 / 0 | 9.9e-03 ± 4.8e-04 → 9.9e-03 ± 4.8e-04 | 3% | 22 / 768 | 100% / 40% | 0% / 0% | 0% / 0% |
| ALNO | bounded | 5977 | 5 | 0 / 0 | 1.8e-02 ± 1.8e-03 → 9.5e-03 ± 1.1e-03 | 0% | 9 / 768 | 100% / 80% | 0% / 0% | 100% / 0% |
| LFNO | plr10 | 5977 | 5 | 2 / 0 | 9.6e-03 ± 7.9e-04 → 9.6e-03 ± 7.9e-04 | 7% | 4 / 1024 | 60% / 0% | 0% / 0% | 0% / 0% |
| LFNO | bounded | 5977 | 5 | 0 / 3 | 2.9e-02 ± 9.3e-03 → 1.7e-02 ± 7.1e-03 | 0% | 29 / 1024 | 40% / 40% | 40% / 0% | 40% / 0% |

NaN: training diverged to NaN (a slot pole reached mu = 0, where the steady response beta / (i omega - mu) at omega = 0 is singular). Dead: test rel. L2 >= 0.5 (output stays near 0). Test error, unstable poles and weighted slots: mean over the other runs.
