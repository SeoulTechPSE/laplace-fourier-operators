# E5: linear-probe poles of Duffing models, `lz_wide` data set

Forced Lorenz ρ = 5 (z' = xy − 8/3 z − f, output x), `e5_data.py` set `lz_wide` (diverse E1 waveforms, started at the equilibrium C+), Lorenz LNO architecture (grid input, tanh, 16 modes), 5 seeds per variant; the tied layer has a learned output offset. Reference: the dominant pole of the linearisation at C+, **−0.9287 ± 4.1476i** (third pole −11.81). Equivalent poles from centred data.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -1.115+4.023i, residual 0.18; A <= 1 (21): -1.007+4.083i, residual 0.03; 1 < A <= 3 (97): -1.070+4.071i, residual 0.09; A > 3 (82): -1.193+3.981i, residual 0.16.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.046 | 0/256, -0.090+0.226i (0.94) | -214.268-0.000i (50.20), 0.09–0.10 | 0.00, 0.06 | -0.140+0.173i (0.95) | yes, 0.32 | no |
| base | 1 | 0.046 | 0/256, -0.128+0.218i (0.94) | -387.864-0.000i (91.04), 0.05–0.05 | 0.00, 0.02 | -0.183+0.178i (0.95) | yes, 0.27 | no |
| base | 2 | 0.046 | 0/256, -0.116+0.224i (0.94) | -341.385-0.000i (80.11), 0.07–0.07 | 0.00, 0.05 | -0.105+0.143i (0.96) | yes, 0.38 | no |
| base | 3 | 0.046 | 0/256, -0.115+0.164i (0.96) | -382.844-0.000i (89.86), 0.06–0.06 | 0.00, 0.02 | -0.115+0.168i (0.96) | yes, 0.02 | **yes** |
| base | 4 | 0.044 | 0/256, -0.120+0.342i (0.92) | -169.008+0.000i (39.56), 0.08–0.10 | 0.00, 0.01 | -0.246+0.277i (0.92) | yes, 0.38 | no |
| lno_plr10 | 0 | 0.022 | 0/256, -0.632+2.740i (0.34) | -2.204+3.618i (0.32), 0.22–0.22 | 0.04, 0.06 | -1.416+2.463i (0.41) | yes, 0.29 | no |
| lno_plr10 | 1 | 0.032 | 0/256, -0.339+1.832i (0.56) | -3.290+0.000i (1.12), 0.30–0.30 | 0.03, 0.34 | -1.076+1.644i (0.59) | no, 0.39 | no |
| lno_plr10 | 2 | 0.024 | 0/256, -0.562+2.357i (0.43) | -1.675+2.937i (0.33), 0.18–0.18 | 0.06, 0.05 | -1.435+2.161i (0.48) | yes, 0.34 | no |
| lno_plr10 | 3 | 0.032 | 0/256, -0.265+1.829i (0.57) | -1.359+0.521i (0.86), 0.21–0.22 | 0.07, 0.23 | -1.151+1.419i (0.64) | no, 0.43 | no |
| lno_plr10 | 4 | 0.023 | 0/256, -0.437+2.568i (0.39) | -2.177+3.381i (0.34), 0.20–0.21 | 0.03, 0.10 | -1.357+2.236i (0.46) | no, 0.37 | no |
| lno_stable_plr10 | 0 | 0.014 | 2/256, -0.632+4.172i (0.07) | -1.109+3.962i (0.06), 0.10–0.10 | 0.17, 0.05 | -0.982+3.818i (0.08) | no, 0.13 | no |
| lno_stable_plr10 | 1 | 0.013 | 1/256, -0.577+4.373i (0.10) | -1.096+4.022i (0.05), 0.08–0.09 | 0.12, 0.03 | -0.975+3.895i (0.06) | no, 0.15 | no |
| lno_stable_plr10 | 2 | 0.012 | 0/256, -0.512+4.481i (0.13) | -1.097+3.977i (0.06), 0.13–0.13 | 0.20, 0.04 | -0.971+3.802i (0.08) | no, 0.13 | no |
| lno_stable_plr10 | 3 | 0.010 | 0/256, -0.582+4.528i (0.12) | -1.071+4.043i (0.04), 0.11–0.11 | 0.20, 0.03 | -0.974+3.939i (0.05) | no, 0.13 | no |
| lno_stable_plr10 | 4 | 0.011 | 1/256, -0.639+4.311i (0.08) | -1.071+3.956i (0.06), 0.12–0.12 | 0.17, 0.03 | -0.977+3.828i (0.08) | no, 0.11 | no |
| tied4_stable_plr10 | 0 | 0.533 | 0/4, -0.824+3.345i (0.19) | -0.960+3.162i (0.23), 0.18–0.18 | 0.01, 0.01 | -2.435+1.442i (0.73) | yes, 0.88 | no |
| tied4_stable_plr10 | 1 | 0.536 | 0/4, -1.372+6.112i (0.47) | -0.889+1.585i (0.60), 0.43–0.43 | 0.03, 0.01 | -0.848+26.538i (5.27) | yes, 0.02 | no |
| tied4_stable_plr10 | 2 | 0.535 | 0/4, -0.799+5.803i (0.39) | -0.850+2.701i (0.34), 0.50–0.50 | 0.03, 0.01 | -1.764+17.298i (3.10) | yes, 0.17 | no |
| tied4_stable_plr10 | 3 | 0.534 | 0/4, -1.192+2.508i (0.39) | -0.881+2.559i (0.37), 0.54–0.54 | 0.03, 0.01 | -0.758+2.214i (0.46) | yes, 0.22 | no |
| tied4_stable_plr10 | 4 | 0.535 | 0/4, -0.728+1.543i (0.61) | -1.580+2.557i (0.40), 0.14–0.14 | 0.02, 0.01 | -1.302+39.524i (8.32) | yes, 0.05 | no |
| lnotb_stable_plr10 | 0 | 0.010 | 0/256, -0.331+4.397i (0.15) | -1.102+3.998i (0.05), 0.08–0.08 | 0.01, 0.04 | -0.980+3.838i (0.07) | yes, 0.15 | no |
| lnotb_stable_plr10 | 1 | 0.009 | 0/256, -0.212+4.206i (0.17) | -1.079+4.071i (0.04), 0.07–0.07 | 0.01, 0.02 | -0.973+3.959i (0.05) | yes, 0.20 | no |
| lnotb_stable_plr10 | 2 | 0.008 | 0/256, -0.405+4.024i (0.13) | -1.095+4.070i (0.04), 0.07–0.07 | 0.02, 0.02 | -0.957+3.925i (0.05) | yes, 0.14 | no |
| lnotb_stable_plr10 | 3 | 0.008 | 0/256, -0.310+4.267i (0.15) | -1.045+4.138i (0.03), 0.08–0.08 | 0.01, 0.03 | -0.940+4.029i (0.03) | yes, 0.15 | no |
| lnotb_stable_plr10 | 4 | 0.008 | 0/256, -0.357+4.064i (0.14) | -1.084+4.105i (0.04), 0.08–0.08 | 0.01, 0.02 | -0.941+3.908i (0.06) | yes, 0.15 | no |
| lnots_stable_plr10 | 0 | 0.008 | 1/16, -0.538+4.219i (0.09) | -1.083+4.064i (0.04), 0.07–0.07 | 0.00, 0.03 | -1.023+4.004i (0.04) | yes, 0.13 | no |
| lnots_stable_plr10 | 1 | 0.008 | 1/16, -0.566+3.953i (0.10) | -1.084+4.091i (0.04), 0.06–0.06 | 0.00, 0.01 | -1.042+4.051i (0.03) | yes, 0.12 | no |
| lnots_stable_plr10 | 2 | 0.009 | 0/16, -0.404+4.134i (0.12) | -1.105+4.061i (0.05), 0.07–0.07 | 0.00, 0.01 | -1.050+4.018i (0.04) | yes, 0.16 | no |
| lnots_stable_plr10 | 3 | 0.008 | 1/16, -0.855+4.101i (0.02) | -1.058+4.148i (0.03), 0.08–0.08 | 0.00, 0.03 | -0.985+4.086i (0.02) | yes, 0.03 | **yes** |
| lnots_stable_plr10 | 4 | 0.009 | 1/16, -0.523+4.020i (0.10) | -1.105+4.078i (0.04), 0.05–0.06 | 0.00, 0.02 | -1.021+3.965i (0.05) | yes, 0.12 | no |
| lnots4_stable_plr10 | 0 | 0.013 | 2/4, -1.103+4.193i (0.04) | -1.138+4.002i (0.06), 0.04–0.04 | 0.01, 0.02 | -1.143+4.023i (0.06) | yes, 0.04 | **yes** |
| lnots4_stable_plr10 | 1 | 0.009 | 1/4, -0.763+4.075i (0.04) | -1.090+4.005i (0.05), 0.05–0.06 | 0.00, 0.02 | -1.057+3.972i (0.05) | yes, 0.08 | **yes** |
| lnots4_stable_plr10 | 2 | 0.013 | 0/4, -0.397+4.248i (0.13) | -1.111+3.968i (0.06), 0.05–0.05 | 0.00, 0.02 | -1.068+3.942i (0.06) | yes, 0.10 | **yes** |
| lnots4_stable_plr10 | 3 | 0.011 | 1/4, -0.922+3.907i (0.06) | -1.060+3.923i (0.06), 0.05–0.05 | 0.00, 0.03 | -1.048+3.898i (0.07) | yes, 0.03 | **yes** |
| lnots4_stable_plr10 | 4 | 0.009 | 1/4, -0.790+4.198i (0.03) | -1.080+4.028i (0.05), 0.04–0.05 | 0.00, 0.02 | -1.049+3.989i (0.05) | yes, 0.08 | **yes** |
| lnots2_stable_plr10 | 0 | 0.013 | 1/2, -1.164+3.988i (0.07) | -1.104+3.993i (0.06), 0.03–0.03 | 0.01, 0.03 | -1.118+4.020i (0.05) | yes, 0.01 | **yes** |
| lnots2_stable_plr10 | 1 | 0.013 | 1/2, -1.112+4.009i (0.05) | -1.096+3.978i (0.06), 0.02–0.02 | 0.01, 0.02 | -1.112+4.009i (0.05) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 2 | 0.013 | 1/2, -1.132+4.037i (0.05) | -1.108+4.006i (0.05), 0.03–0.03 | 0.01, 0.01 | -1.114+4.029i (0.05) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 3 | 0.045 | 0/2, -1.791+13.024i (2.10) | -11.382-0.006i (2.65), 0.06–0.06 | 0.01, 0.02 | -0.368+41.643i (8.82) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 4 | 0.013 | 1/2, -1.138+4.033i (0.06) | -1.124+4.008i (0.06), 0.02–0.02 | 0.01, 0.04 | -1.137+4.032i (0.06) | yes, 0.00 | **yes** |

Trusted runs (rule v1): 12 of 40. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): base s3 0.96 (0.95), lnots_stable_plr10 s3 0.02 (0.03), lnots4_stable_plr10 s0 0.06 (0.01), lnots4_stable_plr10 s1 0.05 (0.02), lnots4_stable_plr10 s2 0.06 (0.02), lnots4_stable_plr10 s3 0.07 (0.03), lnots4_stable_plr10 s4 0.05 (0.02), lnots2_stable_plr10 s0 0.05 (0.00), lnots2_stable_plr10 s1 0.05 (0.00), lnots2_stable_plr10 s2 0.05 (0.00), lnots2_stable_plr10 s3 8.82 (9.01), lnots2_stable_plr10 s4 0.06 (0.01)

Rule v2 linearity gate: the data's best linear pair leaves 0.183 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
