# (A) Fair WPO–FNO comparison on forced Burgers

Sweep: model seed 0, 4000 steps, selection by validation. Final: 12000 steps, seeds 0–2. WPO stable.

| sweep | config | validation |
|---|---|---|
| fno | lr 0.001 | 1.150e-01 |
| fno | lr 0.003 | 1.213e-01 |
| fno | lr 0.01 | 1.346e-01 |
| wpo | lr 0.001, plr 10 | 3.536e-02 |
| wpo | lr 0.001, plr 30 | 3.074e-02 |
| wpo | lr 0.003, plr 10 | 2.832e-02 |
| wpo | lr 0.003, plr 30 | 2.596e-02 |

| model | seed | params | train | test (T = 5) | T = 10: (0, 5] / (5, 10] | (5, 10] / test | probe ν̂ | probe dispersion error |
|---|---|---|---|---|---|---|---|---|
| fno | 0 | 59110 | 6.369e-02 | 1.069e-01 | 3.967e-01 / 3.019e-01 | 2.82 | 0.10944 | 3.8e-01 |
| fno | 1 | 59110 | 6.184e-02 | 1.056e-01 | 3.990e-01 / 3.032e-01 | 2.87 | 0.10547 | 3.4e-01 |
| fno | 2 | 59110 | 6.253e-02 | 1.088e-01 | 3.966e-01 / 3.030e-01 | 2.78 | 0.10576 | 3.7e-01 |
| wpo | 0 | 59529 | 1.123e-02 | 2.442e-02 | 1.338e-02 / 4.284e-02 | 1.75 | 0.10104 | 1.0e-02 |
| wpo | 1 | 59529 | 1.108e-02 | 2.298e-02 | 1.353e-02 / 4.163e-02 | 1.81 | 0.10285 | 3.4e-02 |
| wpo | 2 | 59529 | 1.117e-02 | 2.375e-02 | 1.378e-02 / 4.566e-02 | 1.92 | 0.10104 | 1.7e-02 |
