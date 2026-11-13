# E2 phase 1: local identifiability vs data (tied layer, at the truth)

ratio = profiled error of a 10 % change in Re μ / representation floor; **bold** = practically identifiable (ratio ≥ 10). Im μ of the complex poles is listed in the JSON (always ≥ the Re ratio here).

## p4

| sweep | family | K | T | floor | cond(F) | Re -0.4+3i | Re -1 | Re -0.2 |
|---|---|---|---|---|---|---|---|---|
| richness | multisine | 1 | 20 | 4.8e-04 | 4.1e+03 | 8.9 | 8.3 | **12.2** |
| richness | multisine | 2 | 20 | 1.8e-03 | 4.9e+04 | 3.0 | 2.6 | 6.7 |
| richness | multisine | 3 | 20 | 1.2e-03 | 1.9e+04 | 4.1 | 5.3 | **11.2** |
| richness | multisine | 4 | 20 | 7.7e-04 | 1.0e+04 | 8.8 | 8.9 | **18.9** |
| richness | multisine | 6 | 20 | 7.2e-04 | 5.5e+03 | **10.7** | **11.5** | **21.3** |
| richness | multisine | 8 | 20 | 8.5e-04 | 4.5e+03 | **13.7** | **10.2** | **17.9** |
| richness | multisine | 12 | 20 | 6.1e-04 | 4.2e+03 | **22.9** | **14.5** | **26.0** |
| richness | multisine | 16 | 20 | 6.5e-04 | 3.9e+03 | **24.1** | **13.9** | **23.8** |
| richness | multisine | 32 | 20 | 7.0e-04 | 3.5e+03 | **23.9** | **12.9** | **21.4** |
| window | e1 | – | 5 | 2.1e-03 | 1.4e+06 | 2.6 | 0.3 | 0.2 |
| window | e1 | – | 10 | 2.6e-03 | 8.1e+04 | 3.4 | 1.7 | 2.0 |
| window | e1 | – | 40 | 5.6e-04 | 3.2e+04 | **18.2** | **11.3** | **30.0** |
| window | e1 | – | 80 | 4.2e-04 | 3.3e+04 | **24.6** | **14.9** | **39.9** |
| window | multisine | 16 | 5 | 1.7e-03 | 3.0e+05 | 6.1 | 0.6 | 0.3 |
| window | multisine | 16 | 10 | 1.2e-03 | 7.4e+03 | **12.5** | 5.1 | 5.1 |
| window | multisine | 16 | 40 | 4.5e-04 | 3.5e+03 | **34.7** | **21.9** | **44.7** |
| window | multisine | 16 | 80 | 3.4e-04 | 3.7e+03 | **46.3** | **29.9** | **64.8** |
| family | e1 | – | 20 | 1.3e-03 | 4.1e+04 | 7.0 | 4.6 | 9.5 |
| family | step | – | 20 | 5.6e-04 | 1.4e+05 | 7.6 | 6.3 | **16.8** |
| family | ramp | – | 20 | 2.4e-03 | 1.2e+07 | 0.3 | 0.4 | 1.5 |
| family | prbs | – | 20 | 9.5e-04 | 1.8e+03 | **19.5** | 9.7 | **13.0** |
| family | smooth | – | 20 | 7.5e-04 | 9.3e+04 | 5.0 | 6.0 | **19.7** |
| family | pulse | – | 20 | 2.3e-03 | 1.2e+03 | **10.9** | 3.7 | 4.3 |

## p6

| sweep | family | K | T | floor | cond(F) | Re -0.5+1.5i | Re -0.3+4i | Re -1 | Re -2 |
|---|---|---|---|---|---|---|---|---|---|
| richness | multisine | 1 | 20 | 4.0e-04 | 4.5e+07 | 4.9 | 4.1 | 0.5 | 0.2 |
| richness | multisine | 2 | 20 | 1.6e-03 | 1.3e+07 | 1.0 | 1.8 | 0.2 | 0.1 |
| richness | multisine | 3 | 20 | 1.1e-03 | 9.2e+06 | 1.9 | 2.2 | 0.4 | 0.1 |
| richness | multisine | 4 | 20 | 8.6e-04 | 6.2e+06 | 3.3 | 3.6 | 0.5 | 0.2 |
| richness | multisine | 6 | 20 | 6.4e-04 | 1.1e+07 | 4.1 | **12.3** | 0.7 | 0.2 |
| richness | multisine | 8 | 20 | 8.9e-04 | 4.4e+06 | 4.1 | 5.2 | 0.6 | 0.2 |
| richness | multisine | 12 | 20 | 6.4e-04 | 2.8e+06 | 7.7 | **11.9** | 1.0 | 0.4 |
| richness | multisine | 16 | 20 | 6.9e-04 | 2.7e+06 | 7.4 | **13.9** | 1.0 | 0.4 |
| richness | multisine | 32 | 20 | 7.0e-04 | 2.4e+06 | 7.8 | **16.1** | 1.0 | 0.4 |
| window | e1 | – | 5 | 1.5e-03 | 8.8e+10 | 0.0 | 0.4 | 0.0 | 0.0 |
| window | e1 | – | 10 | 1.0e-03 | 1.3e+07 | 2.5 | 4.7 | 0.3 | 0.1 |
| window | e1 | – | 40 | 6.3e-04 | 7.6e+06 | 5.6 | 9.8 | 0.7 | 0.2 |
| window | e1 | – | 80 | 5.8e-04 | 7.6e+06 | 6.3 | **11.1** | 0.8 | 0.3 |
| window | multisine | 16 | 5 | 1.4e-03 | 1.9e+10 | 0.0 | 0.9 | 0.0 | 0.0 |
| window | multisine | 16 | 10 | 1.1e-03 | 5.2e+06 | 3.7 | 7.6 | 0.4 | 0.2 |
| window | multisine | 16 | 40 | 5.3e-04 | 2.4e+06 | **10.3** | **18.6** | 1.4 | 0.5 |
| window | multisine | 16 | 80 | 4.2e-04 | 2.3e+06 | **13.1** | **22.9** | 1.8 | 0.6 |
| family | e1 | – | 20 | 7.8e-04 | 7.5e+06 | 4.2 | 6.9 | 0.6 | 0.2 |
| family | step | – | 20 | 4.4e-04 | 4.1e+07 | 3.1 | 4.1 | 0.6 | 0.1 |
| family | ramp | – | 20 | 7.4e-04 | 4.6e+09 | 0.2 | 0.1 | 0.0 | 0.0 |
| family | prbs | – | 20 | 1.2e-03 | 2.5e+06 | 5.1 | 9.1 | 0.5 | 0.2 |
| family | smooth | – | 20 | 6.0e-04 | 2.5e+07 | 2.7 | 3.1 | 0.6 | 0.1 |
| family | pulse | – | 20 | 3.6e-03 | 2.5e+06 | 2.1 | 5.7 | 0.2 | 0.1 |
