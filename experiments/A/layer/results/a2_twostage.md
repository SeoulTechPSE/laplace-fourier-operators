# (A) Two-stage dispersion identification, then fine-tuning

Stage 1: one pole per excited k (grid + Nelder–Mead, residue by least squares). Stage 2: polynomial LS, degree by BIC. Stage 3: layer initialized from stage 2, Adam 1000 steps. Data seeds 0–4.

| system | data seed | stage 1: max pole error (k = 1..16) | degree (Re; Im) | D | v | k_r | test: init → tuned | oracle | dispersion error |
|---|---|---|---|---|---|---|---|---|---|
| diff | 0 | 3.9e-02 | a0,a1; c1,c3 | 0.10071 | -0.00276 | -0.02249 | 1.92e-02 → 1.41e-02 | 8.27e-03 | 1.0e-01 |
| diff | 1 | 4.2e-02 | a0,a1,a2; c1 | 0.10765 | -0.00008 | -0.14389 | 9.46e-02 → 2.78e-02 | 8.27e-03 | 4.8e-01 |
| diff | 2 | 2.3e-02 | a0,a1,a2; c1,c3 | 0.10310 | 0.00042 | -0.05792 | 3.58e-02 → 2.67e-02 | 8.27e-03 | 1.4e-01 |
| diff | 3 | 1.2e-01 | a0,a1,a2; c1 | 0.08803 | -0.00021 | 0.21638 | 1.01e-01 → 2.06e-02 | 8.27e-03 | 4.5e-01 |
| diff | 4 | 3.2e-02 | a0,a1; c1 | 0.09832 | 0.00027 | 0.06032 | 3.66e-02 → 1.35e-02 | 8.27e-03 | 8.9e-02 |
| adr | 0 | 2.0e-02 | a0,a1; c1 | 0.05005 | 0.50139 | 0.20049 | 9.95e-03 → 2.56e-02 | 8.71e-03 | 1.9e-02 |
| adr | 1 | 1.7e-02 | a0,a1,a2; c1,c3 | 0.05183 | 0.50630 | 0.16538 | 2.72e-02 → 4.31e-02 | 8.71e-03 | 7.3e-01 |
| adr | 2 | 9.9e-03 | a0,a1; c1 | 0.05007 | 0.50066 | 0.20346 | 9.52e-03 → 1.02e-02 | 8.71e-03 | 2.1e-02 |
| adr | 3 | 5.4e-02 | a0,a1,a2; c1,c3 | 0.04752 | 0.48799 | 0.24467 | 3.44e-02 → 2.09e-02 | 8.71e-03 | 3.1e-01 |
| adr | 4 | 1.7e-02 | a0,a1; c1 | 0.04959 | 0.49761 | 0.21382 | 1.45e-02 → 2.54e-02 | 8.71e-03 | 6.3e-02 |
