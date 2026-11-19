# E5: linear-probe poles of Duffing models, `c0_small` data set

Undamped Duffing x'' + x + x³ = f, `e5_data.py` set `c0_small` (diverse E1 waveforms), 5 seeds per variant. Reference: the exact small-signal pole **± i** (on the imaginary axis).
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.002+1.010i, residual 0.08.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.619 | 0/256, 0.037+0.086i (0.91) | -36.055-0.000i (36.07), 0.90–0.94 | 0.45, 0.34 | 0.068+0.026i (0.98) | no, 0.04 | no |
| base | 1 | 0.648 | 0/256, 0.086+0.132i (0.87) | -39.356-0.000i (39.37), 0.87–0.96 | 0.40, 0.47 | -0.016+0.116i (0.88) | no, 0.38 | no |
| base | 2 | 0.683 | 0/256, 0.070+0.096i (0.91) | -64.444+0.000i (64.45), 0.68–0.77 | 0.29, 0.33 | 0.018+0.000i (1.00) | no, 0.10 | no |
| base | 3 | 0.622 | 0/256, 0.016+0.101i (0.90) | -106.791-0.000i (106.80), 0.57–0.88 | 0.10, 0.41 | -0.023+0.000i (1.00) | no, 0.38 | no |
| base | 4 | 0.691 | 0/256, 0.051+0.086i (0.92) | -41.111-0.000i (41.12), 0.76–0.84 | 0.30, 0.39 | 0.026+0.000i (1.00) | no, 0.25 | no |
| lno_plr10 | 0 | 0.604 | 0/256, 0.062+0.504i (0.50) | -0.051+0.517i (0.49), 0.94–0.99 | 0.41, 0.79 | -0.226+0.311i (0.73) | no, 0.36 | no |
| lno_plr10 | 1 | 0.567 | 0/256, 0.070+0.369i (0.63) | 0.006-0.347i (1.35), 0.89–0.96 | 0.50, 0.53 | -0.294+0.310i (0.75) | no, 0.49 | no |
| lno_plr10 | 2 | 0.581 | 0/256, -0.063+0.362i (0.64) | -0.042+0.405i (0.60), 0.95–0.99 | 0.52, 0.71 | -0.350+0.281i (0.80) | no, 0.37 | no |
| lno_plr10 | 3 | 0.588 | 0/256, -0.043+0.661i (0.34) | 0.030-0.000i (1.00), 0.91–0.99 | 0.44, 0.69 | -0.294+0.363i (0.70) | no, 0.43 | no |
| lno_plr10 | 4 | 0.587 | 0/256, 0.045+0.391i (0.61) | -0.071+0.373i (0.63), 0.78–0.98 | 0.50, 0.48 | -0.191+0.299i (0.73) | no, 0.15 | no |
| lno_stable_plr10 | 0 | 0.034 | 12/256, 0.011+0.990i (0.01) | 0.000+1.006i (0.01), 0.06–0.90 | 0.02, 0.02 | -0.001+1.009i (0.01) | yes, 0.02 | **yes** |
| lno_stable_plr10 | 1 | 0.045 | 3/256, 0.040+1.016i (0.04) | -0.003+1.009i (0.01), 0.21–0.78 | 0.12, 0.04 | -0.004+1.015i (0.02) | no, 0.04 | no |
| lno_stable_plr10 | 2 | 0.057 | 15/256, -0.007+0.994i (0.01) | -0.002+1.005i (0.01), 0.26–0.72 | 0.17, 0.04 | -0.005+1.018i (0.02) | no, 0.01 | no |
| lno_stable_plr10 | 3 | 0.053 | 8/256, -0.013+0.995i (0.01) | -0.003+1.006i (0.01), 0.21–0.82 | 0.13, 0.04 | -0.003+1.013i (0.01) | no, 0.02 | no |
| lno_stable_plr10 | 4 | 0.082 | 4/256, 0.057+1.019i (0.06) | -0.001+1.007i (0.01), 0.32–0.50 | 0.18, 0.05 | 0.001+1.010i (0.01) | no, 0.06 | no |
| tied4_stable_plr10 | 0 | 0.049 | 1/4, -0.000+1.006i (0.01) | 0.000+1.005i (0.00), 0.05–0.05 | 0.00, 0.02 | -0.000+1.006i (0.01) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 1 | 0.049 | 1/4, -0.001+1.006i (0.01) | 0.001+1.005i (0.01), 0.05–0.05 | 0.00, 0.02 | 0.000+1.006i (0.01) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 2 | 0.049 | 1/4, -0.000+1.006i (0.01) | -0.000+1.008i (0.01), 0.05–0.05 | 0.00, 0.02 | -0.000+1.006i (0.01) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 3 | 0.049 | 1/4, -0.001+1.006i (0.01) | 0.000+1.008i (0.01), 0.08–0.08 | 0.00, 0.02 | -0.000+1.006i (0.01) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 4 | 0.049 | 1/4, -0.001+1.006i (0.01) | 0.000+1.005i (0.00), 0.03–0.03 | 0.00, 0.02 | 0.000+1.006i (0.01) | yes, 0.00 | **yes** |
| lnotb_stable_plr10 | 0 | 0.029 | 5/256, 0.029+1.004i (0.03) | -0.001+1.005i (0.00), 0.05–0.85 | 0.00, 0.02 | -0.000+1.006i (0.01) | yes, 0.03 | **yes** |
| lnotb_stable_plr10 | 1 | 0.050 | 4/256, -0.018+1.020i (0.03) | -0.002+1.007i (0.01), 0.08–0.63 | 0.01, 0.03 | -0.003+1.016i (0.02) | yes, 0.01 | **yes** |
| lnotb_stable_plr10 | 2 | 0.048 | 6/256, -0.005+1.012i (0.01) | -0.001+1.007i (0.01), 0.04–0.83 | 0.01, 0.02 | -0.001+1.010i (0.01) | yes, 0.00 | **yes** |
| lnotb_stable_plr10 | 3 | 0.022 | 10/256, -0.016+0.991i (0.02) | -0.000+1.004i (0.00), 0.07–0.88 | 0.01, 0.02 | -0.001+1.007i (0.01) | yes, 0.02 | **yes** |
| lnotb_stable_plr10 | 4 | 0.056 | 6/256, 0.004+0.963i (0.04) | -0.002+1.006i (0.01), 0.10–0.43 | 0.01, 0.02 | -0.003+1.014i (0.01) | yes, 0.04 | **yes** |
| lnots_stable_plr10 | 0 | 0.048 | 2/16, 0.001+0.995i (0.00) | -0.001+1.007i (0.01), 0.04–0.73 | 0.00, 0.02 | -0.000+1.009i (0.01) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 1 | 0.049 | 1/16, -0.000+1.006i (0.01) | -0.001+1.007i (0.01), 0.08–0.46 | 0.01, 0.02 | -0.002+1.010i (0.01) | yes, 0.00 | **yes** |
| lnots_stable_plr10 | 2 | 0.051 | 1/16, -0.000+1.006i (0.01) | -0.002+1.009i (0.01), 0.11–0.76 | 0.01, 0.01 | -0.006+1.019i (0.02) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 3 | 0.049 | 1/16, -0.000+1.006i (0.01) | -0.001+1.006i (0.01), 0.06–0.68 | 0.01, 0.02 | -0.003+1.014i (0.01) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 4 | 0.052 | 1/16, 0.000+1.006i (0.01) | 0.001+0.997i (0.00), 0.17–0.43 | 0.02, 0.02 | -0.003+1.010i (0.01) | yes, 0.01 | **yes** |
| lnots4_stable_plr10 | 0 | 0.048 | 1/4, -0.000+1.006i (0.01) | -0.001+1.007i (0.01), 0.04–0.61 | 0.00, 0.02 | -0.001+1.008i (0.01) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 1 | 0.050 | 1/4, 0.001+1.006i (0.01) | -0.002+1.007i (0.01), 0.11–0.26 | 0.01, 0.02 | -0.003+1.010i (0.01) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 2 | 0.049 | 1/4, -0.000+1.006i (0.01) | -0.001+1.008i (0.01), 0.05–0.64 | 0.00, 0.02 | -0.001+1.009i (0.01) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 3 | 0.049 | 1/4, 0.000+1.006i (0.01) | -0.001+1.004i (0.00), 0.09–0.68 | 0.00, 0.02 | -0.001+1.007i (0.01) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 4 | 0.051 | 1/4, 0.001+1.007i (0.01) | -0.003+1.009i (0.01), 0.14–0.31 | 0.01, 0.02 | -0.004+1.012i (0.01) | yes, 0.01 | **yes** |
| lnots2_stable_plr10 | 0 | 0.047 | 1/2, -0.000+1.006i (0.01) | -0.001+1.007i (0.01), 0.03–0.63 | 0.00, 0.02 | -0.001+1.007i (0.01) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 1 | 0.049 | 1/2, -0.000+1.006i (0.01) | -0.000+1.006i (0.01), 0.01–0.22 | 0.00, 0.02 | -0.000+1.006i (0.01) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 2 | 0.049 | 1/2, -0.000+1.006i (0.01) | -0.001+1.006i (0.01), 0.03–0.65 | 0.00, 0.02 | -0.001+1.007i (0.01) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 3 | 0.601 | 0/2, -0.356+13.215i (12.22) | -1004.324+0.000i (1004.32), 0.02–0.26 | 0.00, 0.00 | -1.553+36.506i (35.54) | yes, 0.15 | no |
| lnots2_stable_plr10 | 4 | 0.114 | 1/2, 0.003+1.033i (0.03) | 0.013+1.032i (0.03), 0.33–0.35 | 0.00, 0.01 | 0.003+1.031i (0.03) | yes, 0.00 | **yes** |

Trusted runs (rule v1): 25 of 40. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): lno_stable_plr10 s0 0.01 (0.00), tied4_stable_plr10 s0 0.01 (0.00), tied4_stable_plr10 s1 0.01 (0.00), tied4_stable_plr10 s2 0.01 (0.00), tied4_stable_plr10 s3 0.01 (0.00), tied4_stable_plr10 s4 0.01 (0.00), lnotb_stable_plr10 s0 0.01 (0.00), lnotb_stable_plr10 s1 0.02 (0.01), lnotb_stable_plr10 s2 0.01 (0.00), lnotb_stable_plr10 s3 0.01 (0.00), lnotb_stable_plr10 s4 0.01 (0.00), lnots_stable_plr10 s0 0.01 (0.00), lnots_stable_plr10 s1 0.01 (0.00), lnots_stable_plr10 s2 0.02 (0.01), lnots_stable_plr10 s3 0.01 (0.01), lnots_stable_plr10 s4 0.01 (0.00), lnots4_stable_plr10 s0 0.01 (0.00), lnots4_stable_plr10 s1 0.01 (0.00), lnots4_stable_plr10 s2 0.01 (0.00), lnots4_stable_plr10 s3 0.01 (0.00), lnots4_stable_plr10 s4 0.01 (0.00), lnots2_stable_plr10 s0 0.01 (0.00), lnots2_stable_plr10 s1 0.01 (0.00), lnots2_stable_plr10 s2 0.01 (0.00), lnots2_stable_plr10 s4 0.03 (0.02)

Rule v2 linearity gate: the data's best linear pair leaves 0.078 of the output (gate 0.1): **passed**, v2 trusts the same runs as v1.
