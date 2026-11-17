# E5: linear-probe poles of Duffing models, `pend_large` data set

Driven pendulum x'' + 0.5 x' + sin x = f, `e5_data.py` set `pend_large` (diverse E1 waveforms), pendulum LNO architecture (grid input, 8 modes), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.255+0.889i, residual 0.24.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.210 | 0/128, -0.031+0.475i (0.54) | -1.938+0.000i (1.95), 0.42–0.42 | 0.02, 0.18 | -0.317+0.472i (0.50) | no, 0.50 | no |
| base | 1 | 0.285 | 0/128, -0.111+0.348i (0.64) | -0.340+0.419i (0.56), 0.28–0.30 | 0.14, 0.67 | -0.253+0.331i (0.64) | no, 0.34 | no |
| base | 2 | 0.320 | 0/128, -0.049+0.312i (0.69) | -0.687-0.000i (1.06), 0.59–0.59 | 0.06, 0.19 | -0.206+0.298i (0.67) | no, 0.40 | no |
| base | 3 | 0.295 | 0/128, -0.057+0.322i (0.67) | -1.160-0.000i (1.33), 0.41–0.46 | 0.09, 0.63 | -0.197+0.314i (0.66) | no, 0.38 | no |
| base | 4 | 0.335 | 0/128, -0.016+0.211i (0.79) | -0.270-0.000i (0.97), 0.64–0.68 | 0.03, 0.65 | -0.246+0.297i (0.67) | no, 0.60 | no |
| lno_plr10 | 0 | 0.105 | 0/128, -0.132+0.921i (0.13) | -0.262+0.881i (0.09), 0.35–0.36 | 0.04, 0.03 | -0.231+0.833i (0.14) | yes, 0.15 | no |
| lno_plr10 | 1 | 0.125 | 0/128, -0.140+0.962i (0.11) | -0.288+0.873i (0.10), 0.46–0.47 | 0.03, 0.09 | -0.248+0.777i (0.19) | yes, 0.18 | no |
| lno_plr10 | 2 | 0.116 | 0/128, -0.005+1.015i (0.25) | -0.268+0.861i (0.11), 0.40–0.42 | 0.01, 0.06 | -0.236+0.789i (0.18) | yes, 0.33 | no |
| lno_plr10 | 3 | 0.127 | 0/128, -0.043+0.841i (0.24) | -0.290+0.854i (0.12), 0.45–0.47 | 0.02, 0.13 | -0.258+0.741i (0.23) | no, 0.30 | no |
| lno_plr10 | 4 | 0.120 | 0/128, -0.129+0.951i (0.12) | -0.278+0.870i (0.10), 0.45–0.46 | 0.04, 0.11 | -0.245+0.793i (0.18) | no, 0.19 | no |
| lno_stable_plr10 | 0 | 0.106 | 3/128, -0.275+0.974i (0.03) | -0.270+0.926i (0.05), 0.17–0.17 | 0.19, 0.04 | -0.258+0.923i (0.05) | no, 0.03 | no |
| lno_stable_plr10 | 1 | 0.108 | 7/128, -0.253+0.964i (0.00) | -0.270+0.924i (0.05), 0.14–0.14 | 0.11, 0.02 | -0.260+0.927i (0.04) | no, 0.04 | no |
| lno_stable_plr10 | 2 | 0.107 | 4/128, -0.237+0.991i (0.03) | -0.264+0.918i (0.05), 0.19–0.19 | 0.17, 0.05 | -0.253+0.915i (0.05) | no, 0.05 | no |
| lno_stable_plr10 | 3 | 0.105 | 6/128, -0.251+0.978i (0.01) | -0.267+0.924i (0.05), 0.15–0.15 | 0.12, 0.04 | -0.254+0.920i (0.05) | no, 0.02 | no |
| lno_stable_plr10 | 4 | 0.108 | 4/128, -0.217+0.975i (0.03) | -0.267+0.921i (0.05), 0.17–0.18 | 0.16, 0.09 | -0.264+0.928i (0.04) | no, 0.03 | no |
| tied4_stable_plr10 | 0 | 0.105 | 1/4, -0.246+0.919i (0.05) | -0.250+0.916i (0.05), 0.13–0.13 | 0.00, 0.01 | -0.250+0.922i (0.05) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 1 | 0.106 | 1/4, -0.255+0.896i (0.07) | -0.255+0.926i (0.04), 0.18–0.18 | 0.01, 0.01 | -0.258+0.924i (0.04) | yes, 0.03 | **yes** |
| tied4_stable_plr10 | 2 | 0.106 | 1/4, -0.251+0.903i (0.07) | -0.280+0.939i (0.04), 0.16–0.16 | 0.01, 0.01 | -0.252+0.921i (0.05) | yes, 0.02 | **yes** |
| tied4_stable_plr10 | 3 | 0.105 | 1/4, -0.249+0.902i (0.07) | -0.288+0.939i (0.05), 0.20–0.20 | 0.01, 0.01 | -0.253+0.920i (0.05) | yes, 0.02 | **yes** |
| tied4_stable_plr10 | 4 | 0.106 | 1/4, -0.288+0.887i (0.09) | -0.263+0.915i (0.05), 0.12–0.12 | 0.01, 0.01 | -0.256+0.927i (0.04) | yes, 0.05 | **yes** |

Trusted runs (rule v1): 5 of 20. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): tied4_stable_plr10 s0 0.05 (0.04), tied4_stable_plr10 s1 0.04 (0.04), tied4_stable_plr10 s2 0.05 (0.03), tied4_stable_plr10 s3 0.05 (0.03), tied4_stable_plr10 s4 0.04 (0.04)

Rule v2 linearity gate: the data's best linear pair leaves 0.236 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
