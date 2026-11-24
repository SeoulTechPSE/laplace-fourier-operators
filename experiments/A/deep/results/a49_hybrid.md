# (A) a49: WPO + even FNO hybrid against the PDEBench gap

Mean [95 % t-interval] over 5 seeds. References in the script's docstring (a42, a44).

Budgets: {'P': {'target': 61130, 'rank': 0.006, 'tfno': 30769, 'width': 16, 'wpo': 30057}, 'F': {'target': 59529, 'rank': 0.004, 'tfno': 29897, 'width': 17, 'wpo': 30339}}
Diverged: none

## F: forced Burgers

| model | cfg | test | (5, 10] / test | probe outside the band | unseen / test |
|---|---|---|---|---|---|
| hyb_even | {'lr': 0.01, 'plr': 30.0} | 0.0358 [0.0320, 0.0397] | 1.75 [1.69, 1.81] | 0.404 [0.164, 0.644] | 3.48 [1.92, 5.05] |
| hyb_sum | {'lr': 0.003, 'plr': 30.0} | 0.0287 [0.0262, 0.0312] | 2.41 [2.11, 2.71] | 1561637344610.371 [-2773467924026.709, 5896742613247.450] | 8.08 [1.83, 14.33] |

## P: PDEBench 1D Burgers

| model | cfg | nRMSE 256 | nRMSE 1024 | probe ν' (truth 0.1257) | dispersion error |
|---|---|---|---|---|---|
| hyb_even | {'lr': 0.01, 'plr': 30.0} | 0.0473 [0.0441, 0.0506] | 0.0473 [0.0441, 0.0505] | 0.1112 [0.0942, 0.1282] | 0.162 [0.033, 0.291] |
| hyb_sum | {'lr': 0.01, 'plr': 30.0} | 0.0279 [0.0235, 0.0322] | 0.0279 [0.0235, 0.0322] | 0.0429 [0.0345, 0.0513] | 0.683 [0.616, 0.750] |

Sweep (seed 0, 4000 steps, validation):

- P hyb_sum {'lr': 0.001, 'plr': 30.0}: 0.0703
- P hyb_even {'lr': 0.001, 'plr': 30.0}: 0.1512
- P hyb_sum {'lr': 0.003, 'plr': 30.0}: 0.0466
- P hyb_sum {'lr': 0.01, 'plr': 30.0}: 0.0392
- P hyb_even {'lr': 0.003, 'plr': 30.0}: 0.0980
- P hyb_even {'lr': 0.01, 'plr': 30.0}: 0.0821
- F hyb_sum {'lr': 0.001, 'plr': 30.0}: 0.0318
- F hyb_sum {'lr': 0.003, 'plr': 30.0}: 0.0289
- F hyb_sum {'lr': 0.01, 'plr': 30.0}: 0.0376
- F hyb_even {'lr': 0.001, 'plr': 30.0}: 0.0597
- F hyb_even {'lr': 0.003, 'plr': 30.0}: 0.0487
- F hyb_even {'lr': 0.01, 'plr': 30.0}: 0.0387
