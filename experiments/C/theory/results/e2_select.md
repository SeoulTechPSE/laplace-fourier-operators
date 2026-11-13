# E2: best-of-5 selection by validation loss, independent runs

`mixed_off_ms`, seeds 10-59 (not used for the bootstrap estimate), 10 disjoint groups of 5. 'all recovered' = every true pole within relative error 0.1.

| condition | runs | single run: all recovered | best of 5 by validation: all recovered | groups with ≥ 1 good run | per pole, single → selected |
|---|---|---|---|---|---|
| p4_ms16_T20 | 50 | 38% | 80% (8/10) | 80% | -0.4+3i 100% → 100%, -1 44% → 80%, -0.2 58% → 100% |
| p6_ms16_T80 | 50 | 32% | 70% (7/10) | 100% | -0.5+1.5i 100% → 100%, -0.3+4i 100% → 100%, -1 70% → 100%, -2 32% → 70% |

Per group (seeds, good runs, selected seed, selected ok, selected test rel. L2):

- p4_ms16_T20: 10-14: 3/5, s14 ok 5.6e-04; 15-19: 0/5, s17 miss 5.7e-04; 20-24: 0/5, s24 miss 5.6e-04; 25-29: 2/5, s25 ok 6.1e-04; 30-34: 2/5, s30 ok 5.7e-04; 35-39: 2/5, s39 ok 5.6e-04; 40-44: 2/5, s44 ok 5.9e-04; 45-49: 3/5, s49 ok 5.8e-04; 50-54: 3/5, s51 ok 7.3e-04; 55-59: 2/5, s55 ok 5.5e-04
- p6_ms16_T80: 10-14: 1/5, s14 ok 4.2e-04; 15-19: 2/5, s18 ok 3.5e-04; 20-24: 1/5, s23 miss 4.5e-04; 25-29: 2/5, s29 ok 3.4e-04; 30-34: 1/5, s33 ok 3.2e-04; 35-39: 2/5, s38 ok 3.5e-04; 40-44: 3/5, s41 ok 3.1e-04; 45-49: 1/5, s46 miss 4.4e-04; 50-54: 2/5, s52 ok 4.0e-04; 55-59: 1/5, s56 miss 3.3e-04
