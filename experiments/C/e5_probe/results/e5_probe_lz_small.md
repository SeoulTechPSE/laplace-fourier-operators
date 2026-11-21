# E5: linear-probe poles of Duffing models, `lz_small` data set

Forced Lorenz ρ = 5 (z' = xy − 8/3 z − f, output x), `e5_data.py` set `lz_small` (diverse E1 waveforms, started at the equilibrium C+), Lorenz LNO architecture (grid input, tanh, 16 modes), 5 seeds per variant; the tied layer has a learned output offset. Reference: the dominant pole of the linearization at C+, **−0.9287 ± 4.1476i** (third pole −11.81). Equivalent poles from centered data.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.999+4.055i, residual 0.03.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.005 | 0/256, -0.005+0.099i (0.98) | -144.792+572.790i (138.00), 0.11–0.12 | 0.00, 0.08 | -0.057+0.145i (0.96) | yes, 0.44 | no |
| base | 1 | 0.005 | 0/256, 0.007+0.096i (0.98) | -330.985+0.000i (77.66), 0.07–0.15 | 0.00, 0.03 | -0.069+0.148i (0.96) | yes, 0.56 | no |
| base | 2 | 0.005 | 0/256, 0.005+0.108i (0.98) | -324.465-0.000i (76.13), 0.13–0.26 | 0.01, 0.08 | -0.048+0.147i (0.96) | yes, 0.43 | no |
| base | 3 | 0.005 | 0/256, 0.060+0.143i (0.97) | -623.063+662.709i (213.15), 0.06–0.35 | 0.00, 0.02 | -0.033+0.120i (0.97) | yes, 0.27 | no |
| base | 4 | 0.005 | 0/256, 0.021+0.082i (0.98) | -524.266-0.000i (123.13), 0.06–0.34 | 0.00, 0.04 | -0.037+0.072i (0.98) | yes, 0.38 | no |
| lno_plr10 | 0 | 0.004 | 0/256, -0.262+0.320i (0.91) | -82.672-0.000i (19.26), 0.26–0.27 | 0.02, 0.22 | -0.215+0.251i (0.93) | no, 0.25 | no |
| lno_plr10 | 1 | 0.004 | 0/256, -0.141+0.595i (0.86) | -33.238+0.000i (7.66), 0.23–0.26 | 0.01, 0.05 | -0.544+0.609i (0.84) | yes, 0.39 | no |
| lno_plr10 | 2 | 0.004 | 0/256, -0.142+0.621i (0.85) | -47.399-0.000i (10.98), 0.19–0.21 | 0.01, 0.06 | -0.493+0.562i (0.85) | yes, 0.48 | no |
| lno_plr10 | 3 | 0.005 | 0/256, -0.122+0.313i (0.92) | -146.340-0.000i (34.23), 0.27–0.30 | 0.01, 0.24 | -0.136+0.219i (0.94) | no, 0.25 | no |
| lno_plr10 | 4 | 0.004 | 0/256, -0.038+0.323i (0.92) | -83.378+0.000i (19.42), 0.17–0.23 | 0.01, 0.10 | -0.250+0.286i (0.92) | yes, 0.40 | no |
| lno_stable_plr10 | 0 | 0.001 | 0/256, -0.246+4.385i (0.17) | -0.907+3.711i (0.10), 0.18–0.18 | 0.26, 0.14 | -0.827+3.551i (0.14) | no, 0.09 | no |
| lno_stable_plr10 | 1 | 0.001 | 0/256, -0.235+3.846i (0.18) | -0.933+3.825i (0.08), 0.15–0.15 | 0.17, 0.05 | -0.847+3.619i (0.13) | no, 0.16 | no |
| lno_stable_plr10 | 2 | 0.001 | 0/256, -0.335+4.121i (0.14) | -0.894+3.965i (0.04), 0.12–0.12 | 0.07, 0.05 | -0.820+3.845i (0.08) | yes, 0.13 | no |
| lno_stable_plr10 | 3 | 0.001 | 0/256, -0.414+3.739i (0.15) | -0.851+3.881i (0.07), 0.13–0.13 | 0.14, 0.03 | -0.778+3.696i (0.11) | no, 0.10 | no |
| lno_stable_plr10 | 4 | 0.002 | 0/256, -0.238+4.179i (0.16) | -1.006+3.447i (0.17), 0.27–0.27 | 0.32, 0.24 | -0.875+3.469i (0.16) | no, 0.13 | no |
| tied4_stable_plr10 | 0 | 0.540 | 0/4, -0.370+6.603i (0.59) | -0.575+6.666i (0.60), 0.57–0.57 | 0.01, 0.01 | -2.349+0.452i (0.93) | yes, 0.77 | no |
| tied4_stable_plr10 | 1 | 0.539 | 0/4, -0.342+2.418i (0.43) | -0.798+0.000i (0.98), 0.29–0.29 | 0.02, 0.01 | -0.958+0.000i (0.98) | yes, 0.49 | no |
| tied4_stable_plr10 | 2 | 0.539 | 0/4, -0.259+4.171i (0.16) | -0.481+0.000i (0.98), 0.40–0.40 | 0.02, 0.01 | -0.414+0.000i (0.98) | yes, 0.18 | no |
| tied4_stable_plr10 | 3 | 0.539 | 0/4, -0.215+5.617i (0.38) | -0.366-0.000i (0.98), 0.44–0.44 | 0.01, 0.01 | -0.323+0.000i (0.99) | yes, 0.44 | no |
| tied4_stable_plr10 | 4 | 0.539 | 0/4, -0.330+1.024i (0.75) | -0.889-0.000i (0.98), 0.28–0.28 | 0.02, 0.01 | -0.047+0.000i (1.00) | yes, 8.23 | no |
| lnotb_stable_plr10 | 0 | 0.001 | 0/256, -0.189+3.969i (0.18) | -0.912+3.657i (0.12), 0.15–0.15 | 0.01, 0.10 | -0.812+3.530i (0.15) | yes, 0.17 | no |
| lnotb_stable_plr10 | 1 | 0.001 | 0/256, -0.209+4.013i (0.17) | -0.876+3.913i (0.06), 0.08–0.09 | 0.01, 0.03 | -0.838+3.842i (0.08) | yes, 0.17 | no |
| lnotb_stable_plr10 | 2 | 0.000 | 0/256, -0.185+3.833i (0.19) | -0.896+3.993i (0.04), 0.09–0.09 | 0.01, 0.03 | -0.826+3.892i (0.06) | yes, 0.16 | no |
| lnotb_stable_plr10 | 3 | 0.001 | 0/256, -0.167+4.362i (0.19) | -0.875+3.774i (0.09), 0.12–0.12 | 0.01, 0.07 | -0.827+3.668i (0.12) | yes, 0.14 | no |
| lnotb_stable_plr10 | 4 | 0.001 | 0/256, -0.296+3.745i (0.18) | -0.882+3.629i (0.12), 0.14–0.14 | 0.01, 0.12 | -0.775+3.458i (0.17) | no, 0.16 | no |
| lnots_stable_plr10 | 0 | 0.000 | 0/16, -0.198+4.034i (0.17) | -0.868+3.941i (0.05), 0.11–0.11 | 0.01, 0.04 | -0.812+3.800i (0.09) | yes, 0.15 | no |
| lnots_stable_plr10 | 1 | 0.000 | 0/16, -0.393+4.366i (0.14) | -0.913+3.949i (0.05), 0.09–0.09 | 0.01, 0.01 | -0.824+3.826i (0.08) | yes, 0.11 | no |
| lnots_stable_plr10 | 2 | 0.000 | 1/16, -0.521+4.059i (0.10) | -0.906+3.955i (0.05), 0.09–0.09 | 0.01, 0.02 | -0.836+3.862i (0.07) | yes, 0.09 | **yes** |
| lnots_stable_plr10 | 3 | 0.000 | 0/16, -0.448+3.755i (0.15) | -0.887+3.934i (0.05), 0.09–0.09 | 0.01, 0.11 | -0.820+3.827i (0.08) | no, 0.10 | no |
| lnots_stable_plr10 | 4 | 0.001 | 0/16, -0.341+4.414i (0.15) | -0.892+3.901i (0.06), 0.10–0.10 | 0.01, 0.11 | -0.829+3.792i (0.09) | no, 0.11 | no |
| lnots4_stable_plr10 | 0 | 0.000 | 1/4, -1.008+3.923i (0.06) | -0.931+4.019i (0.03), 0.04–0.04 | 0.01, 0.02 | -0.931+4.029i (0.03) | yes, 0.03 | **yes** |
| lnots4_stable_plr10 | 1 | 0.001 | 0/4, -0.398+3.770i (0.15) | -0.950+3.837i (0.07), 0.12–0.12 | 0.00, 0.01 | -0.886+3.684i (0.11) | yes, 0.13 | no |
| lnots4_stable_plr10 | 2 | 0.000 | 1/4, -0.537+4.047i (0.10) | -0.902+3.963i (0.04), 0.08–0.08 | 0.00, 0.02 | -0.844+3.877i (0.07) | yes, 0.09 | **yes** |
| lnots4_stable_plr10 | 3 | 0.000 | 1/4, -0.526+4.037i (0.10) | -0.906+4.010i (0.03), 0.05–0.05 | 0.01, 0.02 | -0.902+4.012i (0.03) | yes, 0.09 | **yes** |
| lnots4_stable_plr10 | 4 | 0.001 | 0/4, -0.567+3.759i (0.12) | -0.929+3.852i (0.07), 0.10–0.11 | 0.00, 0.05 | -0.861+3.710i (0.10) | yes, 0.08 | **yes** |
| lnots2_stable_plr10 | 0 | 0.000 | 1/2, -0.985+3.993i (0.04) | -0.933+4.029i (0.03), 0.04–0.04 | 0.01, 0.02 | -0.940+4.053i (0.02) | yes, 0.02 | **yes** |
| lnots2_stable_plr10 | 1 | 0.000 | 1/2, -0.958+4.083i (0.02) | -0.938+4.040i (0.03), 0.03–0.03 | 0.01, 0.01 | -0.959+4.083i (0.02) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 2 | 0.000 | 1/2, -0.961+4.086i (0.02) | -0.939+4.043i (0.02), 0.04–0.04 | 0.01, 0.02 | -0.959+4.084i (0.02) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 3 | 0.005 | 0/2, -0.159+13.197i (2.14) | -594.823+6.686i (139.73), 0.03–0.03 | 0.00, 0.00 | -0.160+13.195i (2.14) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 4 | 0.000 | 1/2, -0.954+4.074i (0.02) | -0.935+4.034i (0.03), 0.03–0.03 | 0.01, 0.01 | -0.955+4.074i (0.02) | yes, 0.00 | **yes** |

Trusted runs (rule v1): 10 of 40. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): lnots_stable_plr10 s2 0.07 (0.06), lnots4_stable_plr10 s0 0.03 (0.02), lnots4_stable_plr10 s2 0.07 (0.06), lnots4_stable_plr10 s3 0.03 (0.03), lnots4_stable_plr10 s4 0.10 (0.09), lnots2_stable_plr10 s0 0.02 (0.01), lnots2_stable_plr10 s1 0.02 (0.01), lnots2_stable_plr10 s2 0.02 (0.01), lnots2_stable_plr10 s3 2.14 (2.20), lnots2_stable_plr10 s4 0.02 (0.01)

Rule v2 linearity gate: the data's best linear pair leaves 0.027 of the output (gate 0.1): **passed**, v2 trusts the same runs as v1.
