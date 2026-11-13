# E4: pole error under output noise vs the Cramér–Rao bound

Tied layer, prior-free `damp8_ms` + polish + restarts, seeds 10-29 (4 noise draws, one per group of 5). Per pole: RMS relative error of single runs / of the group consensus (truth-free, e2_floor.py; ∞ = pole missing from the consensus) / relative CRB (trace bound, unbiased estimators). Recovered = single runs with relative error < 0.1. η = 0: no noise (bias / optimisation only).

## p4 (n_t = 1024)

| η | -0.4+3i: single / consensus / CRB, recovered | -1: single / consensus / CRB, recovered | -0.2: single / consensus / CRB, recovered |
|---|---|---|---|
| 0 | 4.9e-03 / 5.5e-03 / 0.0e+00, 100% | 3.2e-02 / 9.1e-03 / 0.0e+00, 100% | 3.9e-02 / 2.0e+00 / 0.0e+00, 90% |
| 0.001 | 6.1e-03 / 4.5e-03 / 2.5e-06, 100% | 3.2e-02 / 2.6e-02 / 2.5e-05, 100% | 8.0e-03 / 6.2e-03 / 1.4e-05, 100% |
| 0.01 | 1.1e-02 / 7.0e-03 / 2.5e-05, 100% | 6.5e-02 / 4.7e-02 / 2.5e-04, 90% | 1.2e-01 / 1.8e-02 / 1.4e-04, 80% |
| 0.1 | 1.8e-02 / 1.3e-02 / 2.5e-04, 100% | 9.5e-02 / 1.5e+00 / 2.5e-03, 60% | 1.6e-01 / 7.5e+00 / 1.4e-03, 80% |
| 0.3 | 2.5e-02 / 1.4e-02 / 7.5e-04, 100% | 1.7e-01 / 4.1e-01 / 7.4e-03, 70% | 1.7e-01 / 7.4e-02 / 4.3e-03, 60% |

## p6 (n_t = 16384)

| η | -0.5+1.5i: single / consensus / CRB, recovered | -0.3+4i: single / consensus / CRB, recovered | -1: single / consensus / CRB, recovered | -2: single / consensus / CRB, recovered |
|---|---|---|---|---|
| 0 | 6.6e-03 / 2.0e-03 / 0.0e+00, 100% | 1.9e-04 / 8.7e-05 / 0.0e+00, 100% | 4.4e-02 / 2.9e-02 / 0.0e+00, 90% | 3.4e-02 / 1.4e-02 / 0.0e+00, 100% |
| 0.001 | 1.1e-02 / 3.4e-03 / 4.1e-06, 100% | 2.5e-03 / 8.7e-04 / 5.9e-07, 100% | 3.7e-02 / 2.5e-02 / 7.2e-05, 100% | 8.7e-02 / 3.6e-01 / 2.1e-04, 85% |
| 0.01 | 3.1e-02 / 1.2e-02 / 4.1e-05, 100% | 4.9e-03 / 1.5e-03 / 5.9e-06, 100% | 9.3e-02 / 3.9e-02 / 7.2e-04, 85% | 1.8e-01 / 2.5e-01 / 2.1e-03, 70% |
| 0.1 | 4.3e-02 / 3.3e-02 / 4.1e-04, 100% | 8.2e-03 / 1.6e-03 / 5.9e-05, 100% | 2.8e-01 / 8.2e-01 / 7.2e-03, 20% | 6.3e-01 / 7.8e-01 / 2.1e-02, 5% |
| 0.3 | 4.8e-02 / 4.5e-02 / 1.2e-03, 100% | 8.8e-03 / 2.1e-03 / 1.8e-04, 100% | 2.9e-01 / 8.8e-01 / 2.2e-02, 10% | 7.2e-01 / 7.9e-01 / 6.4e-02, 0% |
