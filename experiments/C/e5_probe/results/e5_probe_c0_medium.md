# E5: linear-probe poles of Duffing models, `c0_medium` data set

Undamped Duffing x'' + x + x³ = f, `e5_data.py` set `c0_medium` (diverse E1 waveforms), 5 seeds per variant. Reference: the exact small-signal pole **± i** (on the imaginary axis).
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.032+1.049i, residual 0.36.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.595 | 0/256, -0.048+0.188i (0.81) | -105.497-0.000i (105.50), 0.37–0.83 | 0.04, 0.26 | -0.045+0.118i (0.88) | no, 0.13 | no |
| base | 1 | 0.647 | 0/256, 0.045+0.097i (0.90) | -19.401+0.000i (19.43), 0.95–0.95 | 0.52, 0.43 | -0.031+0.130i (0.87) | no, 0.40 | no |
| base | 2 | 0.660 | 0/256, 0.049+0.093i (0.91) | -769.736+358.592i (848.74), 0.73–0.83 | 0.27, 0.53 | 0.025+0.000i (1.00) | no, 0.12 | no |
| base | 3 | 0.657 | 0/256, 0.071+0.115i (0.89) | -53.877+0.000i (53.89), 0.67–0.75 | 0.12, 0.65 | -0.041+0.195i (0.81) | no, 0.56 | no |
| base | 4 | 0.664 | 0/256, 0.022+0.086i (0.91) | -49.237+0.000i (49.25), 0.83–0.85 | 0.27, 0.58 | 0.045+0.000i (1.00) | no, 0.07 | no |
| lno_plr10 | 0 | 0.507 | 1/256, -0.006+0.917i (0.08) | -0.092-0.376i (1.38), 0.77–0.96 | 0.33, 0.18 | -0.202+0.765i (0.31) | no, 0.26 | no |
| lno_plr10 | 1 | 0.597 | 0/256, 0.013+0.620i (0.38) | -0.019+0.418i (0.58), 0.79–0.89 | 0.38, 0.63 | -0.043+0.322i (0.68) | no, 0.09 | no |
| lno_plr10 | 2 | 0.543 | 0/256, -0.139+0.456i (0.56) | -0.210+0.148i (0.88), 0.67–0.85 | 0.13, 0.57 | -0.236+0.404i (0.64) | no, 0.02 | no |
| lno_plr10 | 3 | 0.478 | 0/256, -0.020+0.898i (0.10) | -0.255+0.713i (0.38), 0.48–0.95 | 0.24, 0.42 | -0.273+0.822i (0.33) | no, 0.31 | no |
| lno_plr10 | 4 | 0.580 | 0/256, 0.164+0.406i (0.62) | -0.036+0.331i (0.67), 0.79–0.88 | 0.39, 0.33 | -0.116+0.305i (0.70) | no, 0.07 | no |
| lno_stable_plr10 | 0 | 0.155 | 9/256, 0.012+0.966i (0.04) | 0.008+1.022i (0.02), 0.17–0.95 | 0.11, 0.18 | 0.002+1.027i (0.03) | no, 0.04 | no |
| lno_stable_plr10 | 1 | 0.195 | 13/256, -0.019+0.980i (0.03) | -0.002+1.030i (0.03), 0.32–0.79 | 0.15, 0.07 | 0.000+1.037i (0.04) | no, 0.03 | no |
| lno_stable_plr10 | 2 | 0.173 | 13/256, 0.031+1.006i (0.03) | 0.001+1.022i (0.02), 0.29–0.89 | 0.20, 0.14 | -0.005+1.036i (0.04) | no, 0.02 | no |
| lno_stable_plr10 | 3 | 0.172 | 9/256, 0.002+0.985i (0.02) | 0.009+1.026i (0.03), 0.30–0.96 | 0.16, 0.12 | 0.001+1.037i (0.04) | no, 0.02 | no |
| lno_stable_plr10 | 4 | 0.249 | 3/256, 0.040+0.960i (0.06) | -0.011+1.046i (0.05), 0.41–0.59 | 0.16, 0.05 | -0.009+1.064i (0.06) | no, 0.01 | no |
| tied4_stable_plr10 | 0 | 0.247 | 1/4, -0.006+1.041i (0.04) | -0.000+1.035i (0.03), 0.20–0.20 | 0.00, 0.02 | -0.005+1.037i (0.04) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 1 | 0.247 | 1/4, -0.009+1.040i (0.04) | -0.001+1.038i (0.04), 0.24–0.24 | 0.01, 0.02 | -0.006+1.040i (0.04) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 2 | 0.247 | 1/4, -0.009+1.040i (0.04) | -0.004+1.047i (0.05), 0.25–0.25 | 0.01, 0.02 | -0.006+1.039i (0.04) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 3 | 0.247 | 1/4, -0.008+1.041i (0.04) | -0.004+1.046i (0.05), 0.28–0.28 | 0.00, 0.02 | -0.006+1.039i (0.04) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 4 | 0.247 | 1/4, -0.011+1.041i (0.04) | -0.006+1.036i (0.04), 0.08–0.08 | 0.01, 0.02 | -0.007+1.040i (0.04) | yes, 0.00 | **yes** |
| lnotb_stable_plr10 | 0 | 0.115 | 5/256, 0.046+0.993i (0.05) | 0.001+1.015i (0.01), 0.21–0.90 | 0.00, 0.02 | -0.004+1.030i (0.03) | yes, 0.06 | **yes** |
| lnotb_stable_plr10 | 1 | 0.099 | 2/256, -0.037+0.943i (0.07) | 0.007+1.010i (0.01), 0.09–0.92 | 0.00, 0.02 | 0.005+1.005i (0.01) | yes, 0.07 | **yes** |
| lnotb_stable_plr10 | 2 | 0.117 | 9/256, 0.015+0.976i (0.03) | 0.002+1.016i (0.02), 0.09–0.88 | 0.00, 0.02 | -0.002+1.024i (0.02) | yes, 0.04 | **yes** |
| lnotb_stable_plr10 | 3 | 0.116 | 6/256, -0.031+0.996i (0.03) | 0.000+1.018i (0.02), 0.16–0.88 | 0.01, 0.02 | -0.005+1.028i (0.03) | yes, 0.04 | **yes** |
| lnotb_stable_plr10 | 4 | 0.228 | 6/256, 0.008+1.005i (0.01) | -0.007+1.045i (0.05), 0.13–0.72 | 0.01, 0.02 | -0.006+1.051i (0.05) | yes, 0.02 | **yes** |
| lnots_stable_plr10 | 0 | 0.146 | 2/16, 0.054+1.006i (0.05) | 0.004+1.026i (0.03), 0.05–0.90 | 0.00, 0.02 | 0.004+1.027i (0.03) | yes, 0.05 | **yes** |
| lnots_stable_plr10 | 1 | 0.244 | 1/16, -0.009+1.039i (0.04) | -0.007+1.041i (0.04), 0.11–0.31 | 0.00, 0.02 | -0.004+1.042i (0.04) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 2 | 0.238 | 1/16, -0.007+1.039i (0.04) | -0.006+1.045i (0.05), 0.13–0.62 | 0.01, 0.01 | -0.008+1.053i (0.05) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 3 | 0.206 | 1/16, 0.005+1.029i (0.03) | -0.004+1.043i (0.04), 0.09–0.95 | 0.01, 0.02 | -0.002+1.050i (0.05) | yes, 0.02 | **yes** |
| lnots_stable_plr10 | 4 | 0.243 | 1/16, -0.008+1.040i (0.04) | -0.007+1.043i (0.04), 0.20–0.29 | 0.02, 0.01 | -0.013+1.053i (0.05) | yes, 0.01 | **yes** |
| lnots4_stable_plr10 | 0 | 0.177 | 1/4, -0.002+1.037i (0.04) | -0.001+1.040i (0.04), 0.06–0.77 | 0.00, 0.02 | -0.002+1.040i (0.04) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 1 | 0.245 | 1/4, -0.009+1.040i (0.04) | -0.009+1.046i (0.05), 0.14–0.19 | 0.00, 0.02 | -0.010+1.046i (0.05) | yes, 0.01 | **yes** |
| lnots4_stable_plr10 | 2 | 0.240 | 1/4, -0.009+1.040i (0.04) | -0.004+1.041i (0.04), 0.10–0.45 | 0.00, 0.02 | -0.005+1.043i (0.04) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 3 | 0.193 | 1/4, -0.001+1.037i (0.04) | -0.002+1.039i (0.04), 0.07–0.93 | 0.00, 0.02 | -0.002+1.039i (0.04) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 4 | 0.243 | 1/4, -0.008+1.040i (0.04) | -0.010+1.047i (0.05), 0.16–0.25 | 0.01, 0.01 | -0.012+1.049i (0.05) | yes, 0.01 | **yes** |
| lnots2_stable_plr10 | 0 | 0.177 | 1/2, -0.001+1.037i (0.04) | -0.002+1.037i (0.04), 0.03–0.80 | 0.00, 0.02 | -0.001+1.037i (0.04) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 1 | 0.245 | 1/2, -0.008+1.040i (0.04) | -0.007+1.040i (0.04), 0.05–0.12 | 0.00, 0.02 | -0.008+1.040i (0.04) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 2 | 0.241 | 1/2, -0.007+1.040i (0.04) | -0.006+1.040i (0.04), 0.06–0.37 | 0.00, 0.02 | -0.008+1.041i (0.04) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 3 | 0.605 | 0/2, -0.379+13.195i (12.20) | -483.832+0.000i (483.83), 0.02–0.18 | 0.00, 0.00 | -2.682+0.000i (2.86) | yes, 4.99 | no |
| lnots2_stable_plr10 | 4 | 0.245 | 1/2, -0.008+1.041i (0.04) | -0.006+1.040i (0.04), 0.05–0.11 | 0.00, 0.02 | -0.008+1.039i (0.04) | yes, 0.00 | **yes** |

Trusted runs (rule v1): 24 of 40. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): tied4_stable_plr10 s0 0.04 (0.03), tied4_stable_plr10 s1 0.04 (0.03), tied4_stable_plr10 s2 0.04 (0.03), tied4_stable_plr10 s3 0.04 (0.03), tied4_stable_plr10 s4 0.04 (0.03), lnotb_stable_plr10 s0 0.03 (0.03), lnotb_stable_plr10 s1 0.01 (0.05), lnotb_stable_plr10 s2 0.02 (0.04), lnotb_stable_plr10 s3 0.03 (0.03), lnotb_stable_plr10 s4 0.05 (0.02), lnots_stable_plr10 s0 0.03 (0.04), lnots_stable_plr10 s1 0.04 (0.03), lnots_stable_plr10 s2 0.05 (0.02), lnots_stable_plr10 s3 0.05 (0.03), lnots_stable_plr10 s4 0.05 (0.02), lnots4_stable_plr10 s0 0.04 (0.03), lnots4_stable_plr10 s1 0.05 (0.02), lnots4_stable_plr10 s2 0.04 (0.03), lnots4_stable_plr10 s3 0.04 (0.03), lnots4_stable_plr10 s4 0.05 (0.02), lnots2_stable_plr10 s0 0.04 (0.03), lnots2_stable_plr10 s1 0.04 (0.02), lnots2_stable_plr10 s2 0.04 (0.02), lnots2_stable_plr10 s4 0.04 (0.02)

Rule v2 linearity gate: the data's best linear pair leaves 0.356 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
