# (A) Euler–Bernoulli beam (velocity output): two poles per wavenumber

c = 0.04; poles ±i c k²; truth on a 32× finer time grid; recipe of `a3_twostage.py` with two poles per mode. Data seeds 0–4.

| seed | stage 1 max error, k ≥ 3 | k = 1, 2 | c: stage 2 → tuned | max Re/Im (k ≥ 3) | test: init → tuned (oracle) | dispersion error |
|---|---|---|---|---|---|---|
| 0 | 1.1e+00 | 5.4e+01, 6.8e-01 | -0.02087 → -0.00985 | 4.1e+00 | 2.81e+01 → 3.54e-01 (1.17e-02) | 3.2e+00 |
| 1 | 2.9e-02 | 6.0e+01, 2.6e-01 | 0.04153 → 0.04000 | 1.8e-04 | 6.04e-01 → 1.18e-02 (1.18e-02) | 1.6e-04 |
| 2 | 4.8e-03 | 7.0e+01, 1.7e-01 | -0.16591 → 0.04000 | 5.9e-05 | 7.18e-01 → 1.21e-02 (1.21e-02) | 3.9e-05 |
| 3 | 5.3e-02 | 1.7e+00, 3.0e-01 | 0.04028 → -0.03829 | 4.4e+01 | 5.12e+00 → 3.74e-01 (1.24e-02) | 2.1e+00 |
| 4 | 3.4e-02 | 6.1e+01, 8.1e-01 | 0.04602 → 0.04002 | 2.3e-04 | 8.92e-01 → 1.14e-02 (1.14e-02) | 3.6e-04 |
