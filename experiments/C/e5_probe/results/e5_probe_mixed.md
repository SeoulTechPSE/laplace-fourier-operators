# E5: linear-probe poles of Duffing models, `mixed` data set

Simulated damped Duffing x'' + 0.5 x' + x + x³ = f, `e5_data.py` set `mixed` (diverse E1 waveforms), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.893+1.753i, residual 0.44; A <= 1 (39): -0.425+1.161i, residual 0.27; 1 < A <= 3 (91): -0.644+1.720i, residual 0.35; A > 3 (70): -0.808+2.137i, residual 0.34.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.338 | 0/256, -0.161+0.464i (0.51) | -7.148-0.000i (6.97), 0.54–0.55 | 0.04, 0.05 | -0.335+0.335i (0.64) | yes, 0.46 | no |
| base | 1 | 0.415 | 0/256, -0.061+0.163i (0.83) | -130.237+0.000i (129.99), 0.26–0.27 | 0.03, 0.14 | -0.084+0.144i (0.84) | no, 0.18 | no |
| base | 2 | 0.414 | 0/256, -0.060+0.118i (0.87) | -200.202+0.000i (199.95), 0.20–0.24 | 0.02, 0.13 | -0.075+0.134i (0.85) | no, 0.14 | no |
| base | 3 | 0.385 | 0/256, -0.140+0.221i (0.76) | -23.836+0.000i (23.61), 0.48–0.51 | 0.04, 0.25 | -0.228+0.183i (0.79) | no, 0.33 | no |
| base | 4 | 0.415 | 0/256, -0.040+0.185i (0.81) | -78.618-0.000i (78.37), 0.31–0.32 | 0.02, 0.19 | -0.105+0.205i (0.78) | no, 0.29 | no |
| lno_plr10 | 0 | 0.257 | 1/256, -0.152+0.967i (0.10) | -2.220+0.001i (2.19), 0.29–0.32 | 0.04, 0.19 | -0.607+0.967i (0.36) | no, 0.22 | no |
| lno_plr10 | 1 | 0.284 | 1/256, -0.155+0.946i (0.10) | -0.513+0.495i (0.54), 0.25–0.26 | 0.08, 0.24 | -0.559+0.742i (0.38) | no, 0.26 | no |
| lno_plr10 | 2 | 0.249 | 1/256, -0.314+0.921i (0.08) | -2.318+0.001i (2.28), 0.28–0.31 | 0.03, 0.10 | -0.615+0.928i (0.37) | no, 0.23 | no |
| lno_plr10 | 3 | 0.254 | 0/256, -0.177+0.886i (0.11) | -2.420+0.001i (2.38), 0.34–0.40 | 0.02, 0.30 | -0.562+0.964i (0.31) | no, 0.35 | no |
| lno_plr10 | 4 | 0.266 | 0/256, -0.158+0.899i (0.12) | -0.615+0.851i (0.38), 0.16–0.19 | 0.07, 0.31 | -0.590+0.855i (0.36) | no, 0.42 | no |
| lno_stable_plr10 | 0 | 0.159 | 1/256, -0.261+1.006i (0.04) | -0.279+1.201i (0.23), 0.44–0.49 | 0.40, 0.52 | -0.336+1.291i (0.33) | no, 0.09 | no |
| lno_stable_plr10 | 1 | 0.142 | 4/256, -0.254+0.999i (0.03) | -0.304-1.126i (2.10), 0.28–0.32 | 0.17, 0.05 | -0.307+1.155i (0.20) | no, 0.06 | no |
| lno_stable_plr10 | 2 | 0.160 | 1/256, -0.336+0.978i (0.09) | -0.259-1.198i (2.17), 0.42–0.57 | 0.10, 0.02 | -0.361+1.300i (0.35) | no, 0.08 | no |
| lno_stable_plr10 | 3 | 0.150 | 2/256, -0.280+1.057i (0.09) | -0.243+1.097i (0.13), 0.40–0.51 | 0.13, 0.05 | -0.290+1.164i (0.20) | no, 0.07 | no |
| lno_stable_plr10 | 4 | 0.161 | 0/256, -0.208+0.858i (0.12) | -0.297+1.193i (0.23), 0.36–0.47 | 0.15, 0.09 | -0.358+1.290i (0.34) | no, 0.09 | no |
| tied4_stable_plr10 | 0 | 0.433 | 0/4, -0.946+1.723i (1.03) | -0.824+1.621i (0.87), 0.08–0.08 | 0.01, 0.01 | -0.854+1.585i (0.86) | yes, 0.09 | **yes** |
| tied4_stable_plr10 | 1 | 0.437 | 0/4, -0.754+0.651i (0.60) | -0.897+1.276i (0.72), 0.42–0.42 | 0.04, 0.01 | -2.220+0.978i (1.97) | yes, 0.62 | no |
| tied4_stable_plr10 | 2 | 0.434 | 0/4, -0.700+1.250i (0.53) | -0.655+1.473i (0.65), 0.44–0.44 | 0.03, 0.01 | -0.604+1.075i (0.37) | yes, 0.16 | no |
| tied4_stable_plr10 | 3 | 0.436 | 0/4, -0.676+1.227i (0.50) | -0.740+1.414i (0.66), 0.33–0.33 | 0.02, 0.01 | -0.614+1.202i (0.43) | yes, 0.05 | **yes** |
| tied4_stable_plr10 | 4 | 0.427 | 0/4, -0.936+0.364i (0.91) | -0.560+1.386i (0.52), 0.41–0.41 | 0.03, 0.01 | -0.726+1.777i (0.94) | yes, 0.36 | no |
| lnotb_stable_plr10 | 0 | 0.121 | 1/256, -0.238+0.883i (0.09) | -0.298+1.087i (0.13), 0.28–0.30 | 0.00, 0.01 | -0.250+1.127i (0.16) | yes, 0.04 | **yes** |
| lnotb_stable_plr10 | 1 | 0.137 | 4/256, -0.216+0.965i (0.03) | -0.293+1.076i (0.12), 0.30–0.33 | 0.01, 0.01 | -0.323+1.153i (0.20) | yes, 0.06 | **yes** |
| lnotb_stable_plr10 | 2 | 0.133 | 2/256, -0.218+0.903i (0.07) | -0.311+1.128i (0.17), 0.30–0.33 | 0.01, 0.02 | -0.307+1.187i (0.23) | yes, 0.02 | **yes** |
| lnotb_stable_plr10 | 3 | 0.127 | 3/256, -0.235+0.940i (0.03) | -0.278+1.072i (0.11), 0.26–0.31 | 0.01, 0.01 | -0.292+1.119i (0.16) | yes, 0.14 | no |
| lnotb_stable_plr10 | 4 | 0.143 | 3/256, -0.217+0.962i (0.03) | -0.261+1.108i (0.14), 0.32–0.37 | 0.01, 0.02 | -0.316+1.168i (0.21) | yes, 0.03 | **yes** |
| lnots_stable_plr10 | 0 | 0.130 | 0/16, -0.199+1.179i (0.22) | -0.299+1.104i (0.14), 0.26–0.30 | 0.01, 0.01 | -0.302+1.152i (0.19) | yes, 0.09 | **yes** |
| lnots_stable_plr10 | 1 | 0.132 | 1/16, -0.312+1.032i (0.09) | -0.328+1.129i (0.18), 0.27–0.30 | 0.01, 0.01 | -0.288+1.190i (0.23) | yes, 0.13 | no |
| lnots_stable_plr10 | 2 | 0.151 | 0/16, -0.323+1.075i (0.13) | -0.273+1.194i (0.23), 0.44–0.45 | 0.01, 0.01 | -0.319+1.266i (0.31) | yes, 0.12 | no |
| lnots_stable_plr10 | 3 | 0.142 | 0/16, -0.306+1.055i (0.10) | -0.226+1.086i (0.12), 0.36–0.51 | 0.01, 0.02 | -0.280+1.175i (0.21) | yes, 0.10 | no |
| lnots_stable_plr10 | 4 | 0.153 | 0/16, -0.303+1.124i (0.16) | -0.265+1.153i (0.19), 0.46–0.52 | 0.01, 0.01 | -0.321+1.248i (0.29) | yes, 0.10 | **yes** |

Trusted runs (rule v1): 8 of 30. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): tied4_stable_plr10 s0 0.86 (0.09), tied4_stable_plr10 s3 0.43 (0.31), lnotb_stable_plr10 s0 0.16 (0.46), lnotb_stable_plr10 s1 0.20 (0.42), lnotb_stable_plr10 s2 0.23 (0.41), lnotb_stable_plr10 s4 0.21 (0.42), lnots_stable_plr10 s0 0.19 (0.43), lnots_stable_plr10 s4 0.29 (0.39)

Rule v2 linearity gate: the data's best linear pair leaves 0.439 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
