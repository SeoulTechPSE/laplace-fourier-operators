# E5: linear-probe poles of Duffing models, `pend_medium` data set

Driven pendulum x'' + 0.5 x' + sin x = f, `e5_data.py` set `pend_medium` (diverse E1 waveforms), pendulum LNO architecture (grid input, 8 modes), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.245+0.947i, residual 0.04.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.228 | 0/128, -0.150+0.418i (0.56) | -1.649+0.000i (1.70), 0.36–0.40 | 0.04, 0.06 | -0.311+0.393i (0.58) | yes, 0.32 | no |
| base | 1 | 0.345 | 0/128, -0.136+0.223i (0.75) | -17.479+0.000i (17.26), 0.58–0.62 | 0.03, 0.30 | -0.155+0.218i (0.76) | no, 0.07 | no |
| base | 2 | 0.397 | 0/128, -0.036+0.111i (0.88) | -42.595+0.000i (42.36), 0.37–0.52 | 0.02, 0.15 | -0.080+0.151i (0.83) | no, 0.34 | no |
| base | 3 | 0.334 | 0/128, -0.065+0.275i (0.72) | -14.754+0.000i (14.54), 0.54–0.55 | 0.03, 0.17 | -0.196+0.290i (0.68) | no, 0.34 | no |
| base | 4 | 0.372 | 0/128, -0.070+0.197i (0.79) | -40.751-0.000i (40.51), 0.38–0.45 | 0.01, 0.18 | -0.107+0.187i (0.79) | no, 0.17 | no |
| lno_plr10 | 0 | 0.114 | 0/128, -0.039+0.896i (0.22) | -1.818-0.000i (1.84), 0.48–0.48 | 0.04, 0.03 | -0.289+0.731i (0.24) | yes, 0.20 | no |
| lno_plr10 | 1 | 0.106 | 0/128, -0.077+0.904i (0.18) | -1.736+0.000i (1.77), 0.48–0.50 | 0.03, 0.18 | -0.284+0.748i (0.22) | no, 0.18 | no |
| lno_plr10 | 2 | 0.058 | 0/128, 0.038+0.888i (0.30) | -0.255+0.903i (0.07), 0.39–0.43 | 0.01, 0.07 | -0.231+0.848i (0.12) | yes, 0.27 | no |
| lno_plr10 | 3 | 0.113 | 0/128, -0.056+0.895i (0.21) | -1.825+0.000i (1.85), 0.47–0.49 | 0.04, 0.13 | -0.294+0.732i (0.24) | no, 0.26 | no |
| lno_plr10 | 4 | 0.099 | 0/128, -0.071+0.901i (0.19) | -1.685-0.000i (1.73), 0.48–0.49 | 0.03, 0.12 | -0.277+0.767i (0.20) | no, 0.30 | no |
| lno_stable_plr10 | 0 | 0.020 | 7/128, -0.244+0.949i (0.02) | -0.255+0.960i (0.01), 0.08–0.08 | 0.10, 0.02 | -0.254+0.964i (0.01) | yes, 0.02 | **yes** |
| lno_stable_plr10 | 1 | 0.021 | 5/128, -0.228+0.934i (0.04) | -0.254+0.958i (0.01), 0.11–0.11 | 0.10, 0.01 | -0.258+0.969i (0.01) | no, 0.05 | no |
| lno_stable_plr10 | 2 | 0.021 | 9/128, -0.268+0.972i (0.02) | -0.253+0.960i (0.01), 0.07–0.07 | 0.12, 0.02 | -0.249+0.954i (0.01) | no, 0.02 | no |
| lno_stable_plr10 | 3 | 0.021 | 6/128, -0.247+0.954i (0.01) | -0.254+0.958i (0.01), 0.09–0.09 | 0.11, 0.02 | -0.255+0.965i (0.01) | no, 0.01 | no |
| lno_stable_plr10 | 4 | 0.023 | 1/128, -0.245+0.952i (0.02) | -0.252+0.957i (0.01), 0.17–0.18 | 0.13, 0.02 | -0.263+0.973i (0.01) | no, 0.03 | no |
| tied4_stable_plr10 | 0 | 0.018 | 1/4, -0.248+0.956i (0.01) | -0.250+0.954i (0.01), 0.07–0.07 | 0.00, 0.01 | -0.250+0.958i (0.01) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 1 | 0.019 | 1/4, -0.250+0.951i (0.02) | -0.243+0.960i (0.01), 0.11–0.11 | 0.01, 0.01 | -0.250+0.958i (0.01) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 2 | 0.018 | 1/4, -0.249+0.952i (0.02) | -0.252+0.962i (0.01), 0.07–0.07 | 0.00, 0.01 | -0.249+0.957i (0.01) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 3 | 0.018 | 1/4, -0.249+0.952i (0.02) | -0.256+0.965i (0.01), 0.11–0.11 | 0.00, 0.01 | -0.249+0.957i (0.01) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 4 | 0.017 | 2/4, -0.261+0.949i (0.02) | -0.255+0.958i (0.01), 0.03–0.03 | 0.01, 0.01 | -0.251+0.959i (0.01) | yes, 0.01 | **yes** |
| lnotb_stable_plr10 | 0 | 0.019 | 3/128, -0.279+1.010i (0.05) | -0.257+0.959i (0.01), 0.06–0.06 | 0.01, 0.01 | -0.260+0.973i (0.01) | yes, 0.04 | **yes** |
| lnotb_stable_plr10 | 1 | 0.019 | 2/128, -0.169+0.970i (0.08) | -0.258+0.960i (0.01), 0.06–0.07 | 0.01, 0.01 | -0.257+0.969i (0.01) | yes, 0.09 | **yes** |
| lnotb_stable_plr10 | 2 | 0.019 | 1/128, -0.205+1.043i (0.09) | -0.254+0.959i (0.01), 0.03–0.04 | 0.00, 0.01 | -0.250+0.959i (0.01) | yes, 0.10 | **yes** |
| lnotb_stable_plr10 | 3 | 0.020 | 2/128, -0.197+0.947i (0.06) | -0.255+0.959i (0.01), 0.05–0.05 | 0.01, 0.02 | -0.257+0.969i (0.01) | yes, 0.06 | **yes** |
| lnotb_stable_plr10 | 4 | 0.022 | 1/128, -0.250+0.901i (0.07) | -0.260+0.961i (0.01), 0.08–0.11 | 0.01, 0.01 | -0.270+0.984i (0.03) | yes, 0.08 | **yes** |
| lnots_stable_plr10 | 0 | 0.018 | 1/8, -0.248+0.954i (0.01) | -0.253+0.958i (0.01), 0.03–0.04 | 0.00, 0.01 | -0.251+0.960i (0.01) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 1 | 0.019 | 1/8, -0.248+0.953i (0.02) | -0.253+0.958i (0.01), 0.05–0.06 | 0.00, 0.01 | -0.253+0.962i (0.01) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 2 | 0.019 | 2/8, -0.254+0.994i (0.03) | -0.255+0.959i (0.01), 0.04–0.05 | 0.00, 0.01 | -0.251+0.960i (0.01) | yes, 0.03 | **yes** |
| lnots_stable_plr10 | 3 | 0.020 | 1/8, -0.248+0.954i (0.01) | -0.253+0.959i (0.01), 0.03–0.05 | 0.00, 0.02 | -0.250+0.959i (0.01) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 4 | 0.019 | 1/8, -0.248+0.955i (0.01) | -0.253+0.959i (0.01), 0.02–0.04 | 0.00, 0.01 | -0.249+0.958i (0.01) | yes, 0.00 | **yes** |

Trusted runs (rule v1): 16 of 30. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): lno_stable_plr10 s0 0.01 (0.02), tied4_stable_plr10 s0 0.01 (0.01), tied4_stable_plr10 s1 0.01 (0.01), tied4_stable_plr10 s2 0.01 (0.01), tied4_stable_plr10 s3 0.01 (0.01), tied4_stable_plr10 s4 0.01 (0.01), lnotb_stable_plr10 s0 0.01 (0.03), lnotb_stable_plr10 s1 0.01 (0.03), lnotb_stable_plr10 s2 0.01 (0.01), lnotb_stable_plr10 s3 0.01 (0.03), lnotb_stable_plr10 s4 0.03 (0.05), lnots_stable_plr10 s0 0.01 (0.01), lnots_stable_plr10 s1 0.01 (0.02), lnots_stable_plr10 s2 0.01 (0.01), lnots_stable_plr10 s3 0.01 (0.01), lnots_stable_plr10 s4 0.01 (0.01)

Rule v2 linearity gate: the data's best linear pair leaves 0.036 of the output (gate 0.1): **passed**, v2 trusts the same runs as v1.
