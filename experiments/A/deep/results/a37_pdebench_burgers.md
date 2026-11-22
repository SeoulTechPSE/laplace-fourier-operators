# (A) PDEBench 1D Burgers (nu = 0.01): initial field -> trajectory, 256 x 41

Checks: repository solver vs PDEBench, nRMSE 2.91e-04; solver probe nu' 0.1257 (truth 0.1257), dispersion error 0.000. Budget (real): {'wpo': 61130, 'tfno_r': 64745, 'tfno_rank': 0.015}. Means over 3 seeds (range).

| model | real params | nRMSE 256 | nRMSE 1024 | 1024 / 256 | probe ν' | dispersion error |
|---|---|---|---|---|---|---|
| wpo | – | diverged (nan) in every seed | | | | |
| wpo, rerun after the kernel fix (post hoc) | 61130 | 0.0452 (0.0440–0.0460) | 0.0452 (0.0440–0.0459) | 1.00 (1.00–1.00) | 0.1060 (0.0811–0.1226) | 0.235 (0.118–0.409) |
| tfno_r | 64745 | 0.0257 (0.0252–0.0264) | 0.0257 (0.0252–0.0264) | 1.00 (1.00–1.00) | 0.0413 (0.0346–0.0543) | 0.672 (0.556–0.736) |

| sweep | cfg | validation |
|---|---|---|
| wpo | {'lr': 0.001, 'plr': 30.0} | 0.1370 |
| wpo | {'lr': 0.003, 'plr': 30.0} | 0.0894 |
| tfno_r | {'lr': 0.001} | 0.0751 |
| tfno_r | {'lr': 0.003} | 0.0464 |
| tfno_r | {'lr': 0.01} | 0.0388 |
