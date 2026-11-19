# (A) CSTR from closed-loop data with measurement noise

Jacobian pair -0.7694 ± 0.9597 i. PI Kc 1.0, Ki 0.5. Noise σ 0.0017 (2 % of the median rms of y). Test: clean open-loop data.

| regime | model | pole (seeds) | pole error | clean test error |
|---|---|---|---|---|
| open | wpo | -0.764 ± 0.958 i; -0.772 ± 0.962 i; -0.778 ± 0.958 i | 5.0e-03, 2.9e-03, 7.3e-03 | 0.018, 0.020, 0.020 |
| open | linear | -0.765 ± 0.964 i | 5.3e-03 | 0.126 |
| open | arx | -0.022 ± 0.180 i | 8.8e-01 | – |
| open_noisy | wpo | -0.766 ± 0.956 i; -0.769 ± 0.960 i; -0.775 ± 0.957 i | 4.0e-03, 5.8e-04, 5.2e-03 | 0.018, 0.018, 0.019 |
| open_noisy | linear | -0.758 ± 0.969 i | 1.1e-02 | 0.127 |
| open_noisy | arx | 0.057 ± 0.000 i | 1.0e+00 | – |
| closed | wpo | -0.766 ± 0.958 i; -0.777 ± 0.961 i; -0.776 ± 0.959 i | 3.3e-03, 6.3e-03, 5.6e-03 | 0.029, 0.029, 0.030 |
| closed | linear | -0.771 ± 0.966 i | 1.6e+00 | 0.126 |
| closed | arx | -0.061 ± 0.175 i | 8.6e-01 | – |
| closed_noisy | wpo | -0.765 ± 0.956 i; -0.773 ± 0.957 i; -0.757 ± 0.953 i | 4.4e-03, 3.5e-03, 1.1e-02 | 0.047, 0.044, 0.037 |
| closed_noisy | linear | -0.772 ± 0.973 i | 1.6e+00 | 0.127 |
| closed_noisy | arx | 0.050 ± 0.000 i | 1.0e+00 | – |
