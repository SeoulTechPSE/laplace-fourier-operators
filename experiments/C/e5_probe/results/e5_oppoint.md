# E5: equivalent-linear (secant) poles along the amplitude

Equivalent pole = best linear pair (first-order hold) for one input–response pair at amplitude A. Data: the stored response; model: its own response to the same input. Cells: Im of the equivalent pole (median over 5 seeds for the models), and in brackets the model's relative output error at that sample (median). Re of the poles in the JSON.

## train waveform

| source | A = 0.1 | A = 0.5 | A = 1 | A = 2 | A = 3 | A = 5 | A = 7 | A = 10 |
|---|---|---|---|---|---|---|---|---|
| data | 0.968 (Re -0.250; resid 0.00) | 0.970 (Re -0.250; resid 0.00) | 0.975 (Re -0.250; resid 0.01) | 0.995 (Re -0.251; resid 0.02) | 1.024 (Re -0.252; resid 0.04) | 1.104 (Re -0.259; resid 0.10) | 1.204 (Re -0.275; resid 0.15) | 1.425 (Re -0.298; resid 0.24) |
| base | 0.975 (0.30) | 0.957 (0.10) | 0.968 (0.11) | 0.993 (0.08) | 1.028 (0.05) | 1.110 (0.06) | 1.203 (0.09) | 1.358 (0.32) |
| lno_plr10 | 0.993 (0.46) | 0.962 (0.10) | 0.970 (0.13) | 0.998 (0.12) | 1.028 (0.10) | 1.111 (0.10) | 1.208 (0.15) | 1.327 (0.35) |
| lno_stable_plr10 | 0.927 (0.15) | 0.979 (0.04) | 0.989 (0.05) | 1.005 (0.04) | 1.027 (0.02) | 1.103 (0.04) | 1.213 (0.04) | 1.351 (0.17) |
| tied4_stable_plr10 | 1.116 (0.36) | 1.116 (0.35) | 1.116 (0.34) | 1.116 (0.30) | 1.116 (0.23) | 1.116 (0.09) | 1.116 (0.25) | 1.116 (0.67) |

## test waveform

| source | A = 0.14 | A = 0.48 | A = 1 | A = 2 | A = 3 | A = 5 | A = 7 | A = 8.9 |
|---|---|---|---|---|---|---|---|---|
| data | 0.968 (Re -0.250; resid 0.00) | 0.970 (Re -0.250; resid 0.00) | 0.975 (Re -0.250; resid 0.01) | 0.994 (Re -0.250; resid 0.02) | 1.022 (Re -0.251; resid 0.05) | 1.100 (Re -0.258; resid 0.12) | 1.198 (Re -0.275; resid 0.19) | 1.325 (Re -0.297; resid 0.27) |
| base | 0.973 (0.19) | 0.958 (0.18) | 0.968 (0.18) | 0.991 (0.16) | 1.025 (0.14) | 1.101 (0.11) | 1.189 (0.13) | 1.273 (0.23) |
| lno_plr10 | 0.988 (0.45) | 0.966 (0.23) | 0.976 (0.23) | 1.004 (0.24) | 1.034 (0.24) | 1.106 (0.24) | 1.196 (0.23) | 1.276 (0.28) |
| lno_stable_plr10 | 0.928 (0.44) | 0.996 (0.40) | 1.006 (0.39) | 1.017 (0.38) | 1.027 (0.38) | 1.066 (0.39) | 1.136 (0.42) | 1.244 (0.47) |
| tied4_stable_plr10 | 1.114 (0.39) | 1.114 (0.39) | 1.114 (0.37) | 1.114 (0.33) | 1.114 (0.26) | 1.114 (0.11) | 1.114 (0.29) | 1.114 (0.58) |

