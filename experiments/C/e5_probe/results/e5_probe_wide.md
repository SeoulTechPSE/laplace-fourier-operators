# E5: linear-probe poles of Duffing models, `wide` data set

Simulated damped Duffing x'' + 0.5 x' + x + x³ = f, `e5_data.py` set `wide` (diverse E1 waveforms), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.294+1.063i, residual 0.13.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.267 | 0/256, -0.120+0.316i (0.67) | -0.309+0.344i (0.63), 0.22–0.46 | 0.17, 0.27 | -0.263+0.344i (0.62) | no, 0.30 | no |
| base | 1 | 0.394 | 0/256, -0.039+0.134i (0.86) | -45.156+0.000i (44.92), 0.48–0.52 | 0.04, 0.26 | -0.053+0.132i (0.86) | no, 0.09 | no |
| base | 2 | 0.387 | 0/256, -0.062+0.082i (0.91) | -30.924+0.000i (30.69), 0.53–0.54 | 0.05, 0.23 | -0.045+0.125i (0.87) | no, 0.34 | no |
| base | 3 | 0.374 | 0/256, -0.029+0.215i (0.78) | -0.249+0.000i (0.97), 0.62–0.72 | 0.12, 0.79 | -0.204+0.096i (0.87) | no, 0.73 | no |
| base | 4 | 0.433 | 0/256, -0.003+0.125i (0.88) | -33.162-0.000i (32.93), 0.67–0.67 | 0.18, 0.33 | 0.026+0.055i (0.95) | no, 0.00 | no |
| lno_plr10 | 0 | 0.145 | 0/256, -0.074+0.920i (0.18) | -1.931+0.001i (1.94), 0.43–0.62 | 0.04, 0.14 | -0.348+0.789i (0.20) | no, 0.31 | no |
| lno_plr10 | 1 | 0.167 | 0/256, -0.006+0.914i (0.25) | -0.349+0.755i (0.23), 0.25–0.41 | 0.08, 0.13 | -0.345+0.719i (0.27) | no, 0.29 | no |
| lno_plr10 | 2 | 0.119 | 0/256, 0.002+0.847i (0.28) | -0.315+0.928i (0.08), 0.40–0.55 | 0.02, 0.09 | -0.290+0.842i (0.13) | yes, 0.33 | no |
| lno_plr10 | 3 | 0.138 | 0/256, -0.077+0.911i (0.18) | -1.835+0.001i (1.86), 0.43–0.61 | 0.03, 0.17 | -0.331+0.793i (0.19) | no, 0.29 | no |
| lno_plr10 | 4 | 0.224 | 0/256, -0.087+0.622i (0.38) | -0.330+0.513i (0.46), 0.27–0.35 | 0.07, 0.52 | -0.318+0.539i (0.43) | no, 0.31 | no |
| lno_stable_plr10 | 0 | 0.048 | 3/256, -0.261+0.894i (0.07) | -0.249+1.004i (0.04), 0.10–0.36 | 0.07, 0.02 | -0.266+1.015i (0.05) | yes, 0.10 | **yes** |
| lno_stable_plr10 | 1 | 0.056 | 3/256, -0.261+0.970i (0.01) | -0.253+1.008i (0.04), 0.10–0.38 | 0.11, 0.02 | -0.261+1.016i (0.05) | no, 0.01 | no |
| lno_stable_plr10 | 2 | 0.049 | 2/256, -0.231+1.016i (0.05) | -0.246+1.002i (0.03), 0.13–0.37 | 0.08, 0.02 | -0.272+1.021i (0.06) | yes, 0.04 | **yes** |
| lno_stable_plr10 | 3 | 0.051 | 5/256, -0.216+0.909i (0.07) | -0.236+0.996i (0.03), 0.11–0.43 | 0.13, 0.03 | -0.259+1.008i (0.04) | no, 0.08 | no |
| lno_stable_plr10 | 4 | 0.065 | 0/256, -0.149+0.951i (0.10) | -0.232+1.005i (0.04), 0.23–0.45 | 0.16, 0.04 | -0.296+1.052i (0.10) | no, 0.02 | no |
| tied4_stable_plr10 | 0 | 0.104 | 1/4, -0.273+1.053i (0.09) | -0.226+1.019i (0.06), 0.20–0.20 | 0.00, 0.01 | -0.262+1.032i (0.07) | yes, 0.02 | **yes** |
| tied4_stable_plr10 | 1 | 0.108 | 1/4, -0.285+1.018i (0.06) | -0.237+1.024i (0.06), 0.28–0.28 | 0.02, 0.01 | -0.274+1.029i (0.07) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 2 | 0.104 | 1/4, -0.274+1.043i (0.08) | -0.248+1.037i (0.07), 0.16–0.16 | 0.01, 0.01 | -0.264+1.030i (0.06) | yes, 0.02 | **yes** |
| tied4_stable_plr10 | 3 | 0.105 | 1/4, -0.274+1.039i (0.07) | -0.250+1.038i (0.07), 0.22–0.22 | 0.00, 0.01 | -0.265+1.030i (0.06) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 4 | 0.104 | 0/4, -0.338+1.044i (0.12) | -0.264+1.028i (0.06), 0.04–0.04 | 0.01, 0.01 | -0.268+1.033i (0.07) | yes, 0.07 | **yes** |
| lnotb_stable_plr10 | 0 | 0.040 | 2/256, -0.245+0.942i (0.03) | -0.249+0.992i (0.02), 0.05–0.31 | 0.01, 0.01 | -0.258+1.001i (0.03) | yes, 0.06 | **yes** |
| lnotb_stable_plr10 | 1 | 0.044 | 1/256, -0.246+1.007i (0.04) | -0.242+0.992i (0.02), 0.06–0.37 | 0.00, 0.01 | -0.253+0.999i (0.03) | yes, 0.01 | **yes** |
| lnotb_stable_plr10 | 2 | 0.047 | 4/256, -0.261+1.020i (0.05) | -0.249+1.003i (0.03), 0.09–0.37 | 0.01, 0.03 | -0.266+1.018i (0.05) | yes, 0.01 | **yes** |
| lnotb_stable_plr10 | 3 | 0.039 | 0/256, -0.239+1.068i (0.10) | -0.249+0.990i (0.02), 0.06–0.37 | 0.01, 0.01 | -0.262+1.005i (0.04) | yes, 0.06 | **yes** |
| lnotb_stable_plr10 | 4 | 0.052 | 1/256, -0.193+0.992i (0.06) | -0.222+0.994i (0.04), 0.15–0.46 | 0.02, 0.05 | -0.276+1.031i (0.07) | yes, 0.07 | **yes** |
| lnots_stable_plr10 | 0 | 0.045 | 0/16, -0.128+0.983i (0.12) | -0.247+0.999i (0.03), 0.07–0.36 | 0.00, 0.02 | -0.258+1.008i (0.04) | yes, 0.13 | no |
| lnots_stable_plr10 | 1 | 0.062 | 1/16, -0.243+0.917i (0.05) | -0.246+1.012i (0.04), 0.08–0.29 | 0.00, 0.02 | -0.261+1.020i (0.05) | yes, 0.10 | **yes** |
| lnots_stable_plr10 | 2 | 0.072 | 1/16, -0.236+0.954i (0.02) | -0.248+1.024i (0.06), 0.08–0.46 | 0.01, 0.01 | -0.271+1.045i (0.08) | yes, 0.09 | **yes** |
| lnots_stable_plr10 | 3 | 0.059 | 1/16, -0.205+0.905i (0.08) | -0.247+1.018i (0.05), 0.07–0.48 | 0.01, 0.01 | -0.263+1.031i (0.06) | yes, 0.13 | no |
| lnots_stable_plr10 | 4 | 0.070 | 1/16, -0.180+0.976i (0.07) | -0.241+1.024i (0.06), 0.11–0.41 | 0.02, 0.01 | -0.276+1.054i (0.09) | yes, 0.11 | no |

Trusted runs (rule v1): 14 of 30. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): lno_stable_plr10 s0 0.05 (0.05), lno_stable_plr10 s2 0.06 (0.04), tied4_stable_plr10 s0 0.07 (0.04), tied4_stable_plr10 s1 0.07 (0.04), tied4_stable_plr10 s2 0.06 (0.04), tied4_stable_plr10 s3 0.06 (0.04), tied4_stable_plr10 s4 0.07 (0.04), lnotb_stable_plr10 s0 0.03 (0.07), lnotb_stable_plr10 s1 0.03 (0.07), lnotb_stable_plr10 s2 0.05 (0.05), lnotb_stable_plr10 s3 0.04 (0.06), lnotb_stable_plr10 s4 0.07 (0.03), lnots_stable_plr10 s1 0.05 (0.05), lnots_stable_plr10 s2 0.08 (0.03)

Rule v2 linearity gate: the data's best linear pair leaves 0.132 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
