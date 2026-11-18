# D1 second pass: trained tied layer, Fourier vs FOH time axis

Truth on a 32× finer grid; E1 families; `stable` init, `common` + pole lr × 10; slots = true poles. Test error: median over seeds of the per-family median relative L2. Recovery: seeds with pole error < 0.1.

| system | path | seeds | test (all) | step | ramp | PRBS | smooth | recovered per true pole |
|---|---|---|---|---|---|---|---|---|
| p2 | fourier | 10 | 4.6e-03 | 3.2e-03 | 1.2e-03 | 1.1e-02 | 1.4e-03 | -0.5+2i: 9 |
| p2 | foh | 10 | 4.7e-03 | 3.3e-03 | 1.4e-04 | 1.2e-02 | 1.3e-03 | -0.5+2i: 9 |
| p4 | fourier | 10 | 4.2e-02 | 3.2e-02 | 1.1e-02 | 8.3e-02 | 4.5e-02 | -0.4+3i: 3, -1: 0, -0.2: 0 |
| p4 | foh | 10 | 4.2e-02 | 3.2e-02 | 1.0e-02 | 8.4e-02 | 4.5e-02 | -0.4+3i: 3, -1: 0, -0.2: 0 |
| p6 | fourier | 10 | 1.6e-02 | 1.1e-02 | 3.3e-03 | 3.4e-02 | 1.5e-02 | -0.5+1.5i: 5, -0.3+4i: 8, -1: 1, -2: 0 |
| p6 | foh | 10 | 1.7e-02 | 1.2e-02 | 3.4e-03 | 3.7e-02 | 1.6e-02 | -0.5+1.5i: 5, -0.3+4i: 8, -1: 1, -2: 0 |
