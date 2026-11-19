# E5: linear-probe poles of Duffing models, `pend_orig` data set

Driven pendulum x'' + 0.5 x' + sin x = f, the original LNO data (`1D_Pendulum_c05`), pendulum LNO architecture (grid input, 8 modes), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.255+0.924i, residual 0.12; A <= 1 (20): -0.250+0.968i, residual 0.00; 1 < A <= 3 (40): -0.250+0.963i, residual 0.01; A > 3 (140): -0.255+0.906i, residual 0.12.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.114 | 0/128, 0.055+0.173i (0.85) | -30.961-0.023i (30.73), 0.86–0.86 | 0.80, 0.59 | -0.676+0.000i (1.06) | no, 0.92 | no |
| base | 1 | 0.191 | 0/128, -0.036+0.140i (0.86) | -103.662-0.295i (103.42), 0.81–0.85 | 0.86, 0.48 | -0.205+0.858i (0.12) | no, 0.84 | no |
| base | 2 | 0.179 | 0/128, 0.005+0.151i (0.86) | -55.167+34.569i (64.38), 0.79–0.80 | 0.81, 0.51 | -0.237+0.864i (0.10) | no, 0.84 | no |
| base | 3 | 0.165 | 0/128, 0.015+0.215i (0.80) | -10.480-0.004i (10.28), 0.84–0.86 | 0.85, 0.51 | 0.160+0.000i (1.05) | no, 0.59 | no |
| base | 4 | 0.174 | 0/128, -0.009+0.235i (0.77) | -81.612-36.974i (89.77), 0.81–0.83 | 0.83, 0.53 | -0.313+1.030i (0.09) | no, 0.79 | no |
| lno_plr10 | 0 | 0.162 | 0/128, -0.054+0.919i (0.20) | -52.479-0.059i (52.24), 0.85–0.85 | 0.76, 0.51 | 0.238+0.000i (1.08) | no, 0.32 | no |
| lno_plr10 | 1 | 0.181 | 1/128, -0.163+0.991i (0.09) | -0.260+1.017i (0.05), 0.73–0.86 | 0.52, 0.42 | -0.120+0.846i (0.18) | no, 0.05 | no |
| lno_plr10 | 2 | 0.165 | 0/128, -0.141+0.507i (0.47) | -71.286-0.103i (71.04), 0.84–0.85 | 0.63, 0.49 | -0.401+1.039i (0.17) | no, 0.53 | no |
| lno_plr10 | 3 | 0.119 | 0/128, -0.242+0.480i (0.49) | -22.654-0.013i (22.43), 0.82–0.83 | 0.62, 0.47 | -0.282+0.788i (0.18) | no, 0.37 | no |
| lno_plr10 | 4 | 0.186 | 0/128, -0.384+0.981i (0.13) | -448.631-337.433i (561.75), 0.82–0.82 | 0.62, 0.43 | -0.708+0.000i (1.07) | no, 0.58 | no |
| lno_stable_plr10 | 0 | 0.578 | 2/128, -0.236+0.953i (0.02) | -1.844+11.784i (10.93), 0.62–0.65 | 0.15, 0.02 | -0.859+33.880i (32.92) | no, 0.01 | no |
| lno_stable_plr10 | 1 | 0.580 | 3/128, -0.253+0.918i (0.05) | -7.660+39.047i (38.79), 0.82–0.82 | 0.26, 0.02 | -0.332+7.583i (6.62) | no, 0.02 | no |
| lno_stable_plr10 | 2 | 0.434 | 1/128, -0.213+0.904i (0.07) | -90.248-0.000i (90.00), 0.60–0.60 | 0.17, 0.04 | -0.319+6.446i (5.48) | no, 0.02 | no |
| lno_stable_plr10 | 3 | 0.582 | 1/128, -0.238+0.955i (0.02) | -7.163-0.003i (6.98), 0.68–0.68 | 0.26, 0.03 | -0.916+30.250i (29.29) | no, 0.01 | no |
| lno_stable_plr10 | 4 | 0.574 | 1/128, -0.242+1.027i (0.06) | -0.833+1.894i (1.09), 0.97–0.97 | 0.47, 0.12 | -0.344+7.359i (6.39) | no, 0.02 | no |
| tied4_stable_plr10 | 0 | 0.084 | 1/4, -0.256+0.946i (0.02) | -0.252+0.942i (0.03), 0.03–0.03 | 0.00, 0.01 | -0.255+0.944i (0.02) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 1 | 0.086 | 1/4, -0.255+0.944i (0.02) | -0.252+0.941i (0.03), 0.04–0.04 | 0.01, 0.01 | -0.260+0.949i (0.02) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 2 | 0.928 | 0/4, -2.698+6.109i (5.69) | -2.476+5.863i (5.38), 0.11–0.11 | 0.02, 0.01 | -0.744+20.114i (19.15) | yes, 0.06 | no |
| tied4_stable_plr10 | 3 | 0.956 | 0/4, -1.277+3.561i (2.79) | -30.928+0.000i (30.69), 0.52–0.52 | 0.02, 0.01 | -1.374+27.927i (26.98) | yes, 0.03 | no |
| tied4_stable_plr10 | 4 | 0.083 | 1/4, -0.257+0.949i (0.02) | -0.250+0.940i (0.03), 0.04–0.04 | 0.02, 0.01 | -0.261+0.946i (0.02) | yes, 0.01 | **yes** |
| lnotb_stable_plr10 | 0 | 0.066 | 1/128, -0.247+0.971i (0.00) | -0.237+0.956i (0.02), 0.10–0.11 | 0.00, 0.02 | -0.242+0.957i (0.01) | yes, 0.02 | **yes** |
| lnotb_stable_plr10 | 1 | 0.071 | 2/128, -0.214+0.985i (0.04) | -0.243+0.956i (0.01), 0.10–0.12 | 0.00, 0.01 | -0.239+0.943i (0.03) | yes, 0.05 | **yes** |
| lnotb_stable_plr10 | 2 | 0.066 | 2/128, -0.182+1.004i (0.08) | -0.244+0.958i (0.01), 0.09–0.10 | 0.01, 0.03 | -0.246+0.948i (0.02) | yes, 0.09 | **yes** |
| lnotb_stable_plr10 | 3 | 0.079 | 1/128, -0.218+0.968i (0.03) | -0.216+0.967i (0.03), 0.21–0.21 | 0.01, 0.03 | -0.244+0.999i (0.03) | yes, 0.04 | **yes** |
| lnotb_stable_plr10 | 4 | 0.091 | 1/128, -0.295+0.891i (0.09) | -0.215+0.976i (0.04), 0.22–0.23 | 0.01, 0.03 | -0.223+1.016i (0.05) | yes, 0.08 | **yes** |
| lnots_stable_plr10 | 0 | 0.438 | 0/8, -0.379+4.174i (3.21) | -1.016+0.000i (1.23), 0.67–0.97 | 0.00, 0.29 | -0.149+16.284i (15.32) | no, 0.01 | no |
| lnots_stable_plr10 | 1 | 0.404 | 0/8, -0.174+5.908i (4.94) | -159.962+327.086i (363.13), 0.81–0.86 | 0.00, 0.02 | -0.104+10.383i (9.42) | yes, 0.04 | **yes** |
| lnots_stable_plr10 | 2 | 0.371 | 0/8, -0.193+5.878i (4.91) | -291.899+368.933i (469.53), 0.86–0.89 | 0.00, 0.03 | -0.185+10.366i (9.40) | yes, 0.04 | **yes** |
| lnots_stable_plr10 | 3 | 0.419 | 0/8, -0.178+5.887i (4.92) | -48.648+317.345i (320.06), 0.87–0.89 | 0.00, 0.02 | -0.036+10.104i (9.14) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 4 | 0.068 | 1/8, -0.242+0.952i (0.02) | -0.239+0.949i (0.02), 0.08–0.09 | 0.00, 0.02 | -0.241+0.947i (0.02) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 0 | 0.685 | 0/4, -0.248+9.156i (8.19) | -132.846-335.250i (361.42), 0.84–0.89 | 0.00, 0.02 | -0.152+10.639i (9.67) | yes, 0.03 | no |
| lnots4_stable_plr10 | 1 | 0.745 | 0/4, -0.196+10.615i (9.65) | -0.812+8.365i (7.42), 0.61–0.69 | 0.00, 0.01 | -0.180+10.869i (9.90) | yes, 0.01 | no |
| lnots4_stable_plr10 | 2 | 0.082 | 1/4, -0.257+0.947i (0.02) | -0.249+0.941i (0.03), 0.03–0.04 | 0.00, 0.01 | -0.258+0.947i (0.02) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 3 | 0.760 | 0/4, -0.170+10.784i (9.82) | -0.872+8.389i (7.45), 0.65–0.70 | 0.00, 0.01 | -0.161+10.887i (9.92) | yes, 0.01 | no |
| lnots4_stable_plr10 | 4 | 0.071 | 1/4, -0.256+0.950i (0.02) | -0.249+0.943i (0.03), 0.09–0.10 | 0.00, 0.02 | -0.262+0.957i (0.02) | yes, 0.01 | **yes** |
| lnots2_stable_plr10 | 0 | 0.713 | 0/2, -0.187+9.227i (8.26) | -53.923+316.926i (320.48), 0.79–0.82 | 0.00, 0.03 | -0.139+10.772i (9.80) | yes, 0.03 | no |
| lnots2_stable_plr10 | 1 | 0.769 | 0/2, -0.202+10.899i (9.93) | -1.449+8.604i (7.73), 0.66–0.67 | 0.00, 0.01 | -0.203+10.900i (9.93) | yes, 0.00 | no |
| lnots2_stable_plr10 | 2 | 0.514 | 0/2, -0.055+4.096i (3.13) | -72.080+0.181i (71.83), 0.31–0.34 | 0.00, 0.18 | 0.060+22.494i (21.53) | no, 0.00 | no |
| lnots2_stable_plr10 | 3 | 0.827 | 0/2, -0.045+20.927i (19.96) | -50.037+0.063i (49.80), 0.11–0.13 | 0.00, 0.10 | -0.042+20.924i (19.96) | yes, 0.00 | no |
| lnots2_stable_plr10 | 4 | 0.771 | 0/2, -0.166+10.910i (9.94) | -0.033+10.770i (9.80), 0.60–0.64 | 0.00, 0.02 | -0.168+10.911i (9.94) | yes, 0.00 | no |

Trusted runs (rule v1): 14 of 40. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): tied4_stable_plr10 s0 0.02 (0.02), tied4_stable_plr10 s1 0.02 (0.03), tied4_stable_plr10 s4 0.02 (0.02), lnotb_stable_plr10 s0 0.01 (0.04), lnotb_stable_plr10 s1 0.03 (0.03), lnotb_stable_plr10 s2 0.02 (0.03), lnotb_stable_plr10 s3 0.03 (0.08), lnotb_stable_plr10 s4 0.05 (0.10), lnots_stable_plr10 s1 9.42 (9.87), lnots_stable_plr10 s2 9.40 (9.85), lnots_stable_plr10 s3 9.14 (9.58), lnots_stable_plr10 s4 0.02 (0.03), lnots4_stable_plr10 s2 0.02 (0.02), lnots4_stable_plr10 s4 0.02 (0.03)

Rule v2 linearity gate: the data's best linear pair leaves 0.121 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
