# E5: linear-probe poles of Duffing models, `medium` data set

Simulated damped Duffing x'' + 0.5 x' + x + x³ = f, `e5_data.py` set `medium` (diverse E1 waveforms), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.280+1.044i, residual 0.10.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.359 | 0/256, -0.030+0.322i (0.68) | -15.734-0.000i (15.51), 0.50–0.76 | 0.04, 0.19 | -0.219+0.327i (0.64) | no, 0.48 | no |
| base | 1 | 0.438 | 0/256, -0.005+0.130i (0.87) | -71.193+0.000i (70.95), 0.38–0.44 | 0.04, 0.12 | -0.013+0.108i (0.89) | no, 0.04 | no |
| base | 2 | 0.361 | 0/256, -0.031+0.304i (0.70) | -16.970-0.000i (16.75), 0.55–0.64 | 0.04, 0.18 | -0.189+0.289i (0.68) | no, 0.46 | no |
| base | 3 | 0.383 | 0/256, -0.053+0.194i (0.80) | -19.329-0.000i (19.10), 0.53–0.79 | 0.04, 0.13 | -0.118+0.218i (0.76) | no, 0.25 | no |
| base | 4 | 0.451 | 0/256, 0.002+0.130i (0.87) | -35.264+0.000i (35.03), 0.54–0.58 | 0.08, 0.30 | -0.188+0.172i (0.80) | no, 0.69 | no |
| lno_plr10 | 0 | 0.144 | 0/256, -0.038+0.966i (0.21) | -0.361+0.903i (0.13), 0.40–0.64 | 0.05, 0.17 | -0.327+0.796i (0.19) | no, 0.20 | no |
| lno_plr10 | 1 | 0.195 | 0/256, -0.022+0.887i (0.24) | -0.371+0.723i (0.27), 0.25–0.49 | 0.12, 0.38 | -0.353+0.686i (0.30) | no, 0.34 | no |
| lno_plr10 | 2 | 0.232 | 0/256, -0.073+0.875i (0.20) | -0.253+0.592i (0.38), 0.52–0.65 | 0.25, 0.35 | -0.286+0.654i (0.32) | no, 0.23 | no |
| lno_plr10 | 3 | 0.110 | 0/256, 0.027+1.072i (0.30) | -0.312+0.933i (0.07), 0.41–0.73 | 0.01, 0.12 | -0.282+0.847i (0.13) | no, 0.21 | no |
| lno_plr10 | 4 | 0.206 | 0/256, -0.167+0.677i (0.30) | -0.375+0.693i (0.30), 0.29–0.44 | 0.11, 0.39 | -0.366+0.642i (0.35) | no, 0.27 | no |
| lno_stable_plr10 | 0 | 0.043 | 4/256, -0.247+0.947i (0.02) | -0.252+1.004i (0.04), 0.11–0.40 | 0.08, 0.02 | -0.263+1.019i (0.05) | yes, 0.06 | **yes** |
| lno_stable_plr10 | 1 | 0.053 | 4/256, -0.239+0.996i (0.03) | -0.250+1.015i (0.05), 0.15–0.38 | 0.11, 0.03 | -0.254+1.025i (0.06) | no, 0.03 | no |
| lno_stable_plr10 | 2 | 0.044 | 2/256, -0.183+0.993i (0.07) | -0.244+0.994i (0.03), 0.18–0.41 | 0.12, 0.03 | -0.267+1.019i (0.05) | no, 0.08 | no |
| lno_stable_plr10 | 3 | 0.047 | 3/256, -0.218+0.933i (0.05) | -0.231+0.998i (0.04), 0.19–0.56 | 0.16, 0.06 | -0.256+1.010i (0.04) | no, 0.07 | no |
| lno_stable_plr10 | 4 | 0.086 | 0/256, -0.358+0.958i (0.11) | -0.255+1.031i (0.06), 0.36–0.41 | 0.30, 0.09 | -0.304+1.078i (0.12) | no, 0.02 | no |
| tied4_stable_plr10 | 0 | 0.080 | 1/4, -0.266+1.041i (0.07) | -0.224-1.011i (1.98), 0.19–0.19 | 0.00, 0.01 | -0.257+1.023i (0.06) | yes, 0.02 | **yes** |
| tied4_stable_plr10 | 1 | 0.083 | 1/4, -0.276+1.014i (0.05) | -0.246+1.026i (0.06), 0.23–0.23 | 0.01, 0.01 | -0.270+1.025i (0.06) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 2 | 0.081 | 1/4, -0.269+1.029i (0.06) | -0.247+1.034i (0.07), 0.19–0.19 | 0.01, 0.01 | -0.260+1.022i (0.06) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 3 | 0.080 | 1/4, -0.269+1.028i (0.06) | -0.258+1.034i (0.07), 0.20–0.20 | 0.00, 0.01 | -0.263+1.023i (0.06) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 4 | 0.080 | 1/4, -0.329+1.020i (0.09) | -0.263+1.018i (0.05), 0.06–0.06 | 0.01, 0.01 | -0.265+1.024i (0.06) | yes, 0.06 | **yes** |
| lnotb_stable_plr10 | 0 | 0.031 | 1/256, -0.213+0.924i (0.06) | -0.246+0.990i (0.02), 0.05–0.36 | 0.01, 0.01 | -0.253+0.999i (0.03) | yes, 0.08 | **yes** |
| lnotb_stable_plr10 | 1 | 0.034 | 0/256, -0.204+0.865i (0.11) | -0.247+0.993i (0.02), 0.05–0.43 | 0.00, 0.01 | -0.250+0.998i (0.03) | yes, 0.12 | no |
| lnotb_stable_plr10 | 2 | 0.037 | 3/256, -0.277+0.926i (0.05) | -0.254+1.002i (0.03), 0.07–0.42 | 0.01, 0.03 | -0.266+1.016i (0.05) | yes, 0.07 | **yes** |
| lnotb_stable_plr10 | 3 | 0.032 | 0/256, -0.194+1.075i (0.12) | -0.250+0.994i (0.03), 0.07–0.40 | 0.01, 0.01 | -0.259+1.009i (0.04) | yes, 0.09 | **yes** |
| lnotb_stable_plr10 | 4 | 0.045 | 2/256, -0.208+0.970i (0.04) | -0.246+0.995i (0.03), 0.10–0.50 | 0.02, 0.01 | -0.277+1.026i (0.06) | yes, 0.08 | **yes** |
| lnots_stable_plr10 | 0 | 0.046 | 0/16, -0.216+0.859i (0.11) | -0.242+1.008i (0.04), 0.06–0.36 | 0.00, 0.01 | -0.253+1.015i (0.05) | yes, 0.12 | no |
| lnots_stable_plr10 | 1 | 0.050 | 1/16, -0.241+0.936i (0.03) | -0.252+1.007i (0.04), 0.05–0.35 | 0.00, 0.02 | -0.257+1.014i (0.05) | yes, 0.08 | **yes** |
| lnots_stable_plr10 | 2 | 0.065 | 1/16, -0.227+0.954i (0.03) | -0.247+1.021i (0.05), 0.08–0.22 | 0.01, 0.01 | -0.267+1.043i (0.08) | yes, 0.09 | **yes** |
| lnots_stable_plr10 | 3 | 0.058 | 1/16, -0.218+0.962i (0.03) | -0.245+1.016i (0.05), 0.08–0.45 | 0.01, 0.01 | -0.262+1.034i (0.07) | yes, 0.08 | **yes** |
| lnots_stable_plr10 | 4 | 0.075 | 1/16, -0.189+0.999i (0.07) | -0.255+1.025i (0.06), 0.10–0.13 | 0.02, 0.01 | -0.288+1.060i (0.10) | yes, 0.11 | no |

Trusted runs (rule v1): 13 of 30. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): lno_stable_plr10 s0 0.05 (0.03), tied4_stable_plr10 s0 0.06 (0.03), tied4_stable_plr10 s1 0.06 (0.02), tied4_stable_plr10 s2 0.06 (0.03), tied4_stable_plr10 s3 0.06 (0.02), tied4_stable_plr10 s4 0.06 (0.02), lnotb_stable_plr10 s0 0.03 (0.05), lnotb_stable_plr10 s2 0.05 (0.03), lnotb_stable_plr10 s3 0.04 (0.04), lnotb_stable_plr10 s4 0.06 (0.02), lnots_stable_plr10 s1 0.05 (0.03), lnots_stable_plr10 s2 0.08 (0.01), lnots_stable_plr10 s3 0.07 (0.02)

Rule v2 linearity gate: the data's best linear pair leaves 0.095 of the output (gate 0.1): **passed**, v2 trusts the same runs as v1.
