# E5: linear-probe poles of Duffing models, `pend_wide` data set

Driven pendulum x'' + 0.5 x' + sin x = f, `e5_data.py` set `pend_wide` (diverse E1 waveforms), pendulum LNO architecture (grid input, 8 modes), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.260+0.909i, residual 0.24.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.275 | 0/128, 0.032+0.325i (0.70) | -0.240+0.481i (0.49), 0.33–0.35 | 0.04, 0.06 | -0.206+0.314i (0.66) | yes, 0.30 | no |
| base | 1 | 0.359 | 0/128, -0.127+0.244i (0.73) | -9.898-0.000i (9.70), 0.67–0.67 | 0.07, 0.25 | -0.152+0.193i (0.78) | no, 0.23 | no |
| base | 2 | 0.366 | 0/128, 0.071+0.223i (0.81) | -26.967+0.000i (26.73), 0.49–0.55 | 0.02, 0.20 | -0.076+0.179i (0.81) | no, 0.34 | no |
| base | 3 | 0.350 | 0/128, -0.001+0.208i (0.80) | -20.271+0.000i (20.04), 0.49–0.51 | 0.02, 0.09 | -0.106+0.167i (0.81) | yes, 0.30 | no |
| base | 4 | 0.357 | 0/128, -0.057+0.302i (0.69) | -0.519+0.000i (1.00), 0.55–0.63 | 0.06, 0.12 | -0.138+0.231i (0.75) | no, 0.40 | no |
| lno_plr10 | 0 | 0.167 | 0/128, 0.029+0.824i (0.31) | -1.798-0.000i (1.83), 0.46–0.48 | 0.02, 0.01 | -0.271+0.674i (0.30) | yes, 0.16 | no |
| lno_plr10 | 1 | 0.165 | 0/128, -0.027+0.855i (0.25) | -1.557+0.000i (1.63), 0.43–0.43 | 0.03, 0.04 | -0.281+0.680i (0.29) | yes, 0.22 | no |
| lno_plr10 | 2 | 0.160 | 0/128, -0.064+0.899i (0.20) | -1.597-0.000i (1.66), 0.47–0.48 | 0.04, 0.09 | -0.269+0.705i (0.26) | yes, 0.34 | no |
| lno_plr10 | 3 | 0.147 | 0/128, -0.051+0.877i (0.22) | -0.289+0.884i (0.09), 0.40–0.42 | 0.06, 0.13 | -0.250+0.815i (0.15) | no, 0.25 | no |
| lno_plr10 | 4 | 0.182 | 0/128, -0.049+0.821i (0.25) | -1.708-0.000i (1.75), 0.41–0.43 | 0.04, 0.22 | -0.311+0.661i (0.31) | no, 0.31 | no |
| lno_stable_plr10 | 0 | 0.133 | 3/128, -0.215+0.940i (0.05) | -0.263+0.944i (0.03), 0.12–0.12 | 0.14, 0.02 | -0.264+0.955i (0.02) | no, 0.05 | no |
| lno_stable_plr10 | 1 | 0.133 | 5/128, -0.265+1.026i (0.06) | -0.267+0.943i (0.03), 0.13–0.13 | 0.11, 0.02 | -0.264+0.954i (0.02) | no, 0.07 | no |
| lno_stable_plr10 | 2 | 0.132 | 6/128, -0.246+0.926i (0.04) | -0.260+0.942i (0.03), 0.10–0.11 | 0.11, 0.02 | -0.250+0.935i (0.03) | no, 0.01 | no |
| lno_stable_plr10 | 3 | 0.134 | 9/128, -0.217+0.966i (0.03) | -0.268+0.945i (0.03), 0.19–0.19 | 0.18, 0.03 | -0.267+0.956i (0.02) | no, 0.04 | no |
| lno_stable_plr10 | 4 | 0.132 | 2/128, -0.259+0.924i (0.04) | -0.259+0.942i (0.03), 0.13–0.13 | 0.10, 0.03 | -0.257+0.946i (0.02) | no, 0.02 | no |
| tied4_stable_plr10 | 0 | 0.129 | 1/4, -0.250+0.941i (0.03) | -0.252+0.937i (0.03), 0.10–0.10 | 0.00, 0.01 | -0.253+0.944i (0.02) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 1 | 0.129 | 1/4, -0.255+0.925i (0.04) | -0.252+0.944i (0.02), 0.13–0.13 | 0.01, 0.01 | -0.255+0.942i (0.03) | yes, 0.02 | **yes** |
| tied4_stable_plr10 | 2 | 0.129 | 1/4, -0.251+0.934i (0.03) | -0.268+0.950i (0.03), 0.07–0.07 | 0.00, 0.01 | -0.253+0.942i (0.03) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 3 | 0.129 | 1/4, -0.251+0.934i (0.03) | -0.269+0.950i (0.03), 0.10–0.10 | 0.00, 0.01 | -0.253+0.942i (0.03) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 4 | 0.128 | 1/4, -0.286+0.914i (0.07) | -0.261+0.940i (0.03), 0.06–0.06 | 0.01, 0.01 | -0.254+0.944i (0.02) | yes, 0.05 | **yes** |
| lnotb_stable_plr10 | 0 | 0.131 | 2/128, -0.229+0.910i (0.06) | -0.265+0.943i (0.03), 0.07–0.07 | 0.01, 0.01 | -0.264+0.957i (0.02) | yes, 0.06 | **yes** |
| lnotb_stable_plr10 | 1 | 0.131 | 5/128, -0.279+0.973i (0.03) | -0.267+0.944i (0.03), 0.08–0.09 | 0.01, 0.01 | -0.260+0.954i (0.02) | yes, 0.03 | **yes** |
| lnotb_stable_plr10 | 2 | 0.131 | 6/128, -0.230+0.975i (0.02) | -0.261+0.943i (0.03), 0.04–0.05 | 0.00, 0.01 | -0.252+0.943i (0.03) | yes, 0.04 | **yes** |
| lnotb_stable_plr10 | 3 | 0.132 | 3/128, -0.189+0.940i (0.07) | -0.265+0.944i (0.03), 0.07–0.07 | 0.01, 0.01 | -0.260+0.954i (0.02) | yes, 0.07 | **yes** |
| lnotb_stable_plr10 | 4 | 0.092 | 0/128, -0.159+0.892i (0.12) | -0.254+0.968i (0.00), 0.07–0.15 | 0.01, 0.01 | -0.258+0.984i (0.02) | yes, 0.13 | no |
| lnots_stable_plr10 | 0 | 0.130 | 1/8, -0.252+0.933i (0.03) | -0.262+0.943i (0.03), 0.05–0.05 | 0.00, 0.01 | -0.254+0.942i (0.03) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 1 | 0.131 | 1/8, -0.252+0.932i (0.04) | -0.260+0.943i (0.03), 0.07–0.07 | 0.01, 0.01 | -0.257+0.948i (0.02) | yes, 0.02 | **yes** |
| lnots_stable_plr10 | 2 | 0.131 | 1/8, -0.254+0.931i (0.04) | -0.263+0.944i (0.03), 0.06–0.06 | 0.00, 0.01 | -0.256+0.944i (0.02) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 3 | 0.131 | 1/8, -0.251+0.935i (0.03) | -0.262+0.944i (0.03), 0.05–0.05 | 0.00, 0.01 | -0.253+0.942i (0.03) | yes, 0.01 | **yes** |
| lnots_stable_plr10 | 4 | 0.131 | 1/8, -0.251+0.935i (0.03) | -0.263+0.946i (0.03), 0.04–0.04 | 0.00, 0.01 | -0.254+0.943i (0.03) | yes, 0.01 | **yes** |
| lnots4_stable_plr10 | 0 | 0.127 | 1/4, -0.250+0.933i (0.04) | -0.266+0.945i (0.03), 0.06–0.07 | 0.00, 0.01 | -0.254+0.942i (0.03) | yes, 0.01 | **yes** |
| lnots4_stable_plr10 | 1 | 0.132 | 1/4, -0.251+0.932i (0.04) | -0.264+0.945i (0.03), 0.09–0.10 | 0.01, 0.01 | -0.260+0.951i (0.02) | yes, 0.02 | **yes** |
| lnots4_stable_plr10 | 2 | 0.131 | 1/4, -0.250+0.937i (0.03) | -0.260+0.944i (0.03), 0.05–0.06 | 0.00, 0.01 | -0.250+0.938i (0.03) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 3 | 0.135 | 1/4, -0.248+0.941i (0.03) | -0.271+0.945i (0.03), 0.14–0.15 | 0.01, 0.04 | -0.258+0.951i (0.02) | yes, 0.01 | **yes** |
| lnots4_stable_plr10 | 4 | 0.131 | 1/4, -0.251+0.934i (0.03) | -0.262+0.944i (0.03), 0.05–0.05 | 0.00, 0.01 | -0.254+0.942i (0.03) | yes, 0.01 | **yes** |
| lnots2_stable_plr10 | 0 | 0.128 | 1/2, -0.250+0.936i (0.03) | -0.263+0.943i (0.03), 0.05–0.06 | 0.00, 0.01 | -0.250+0.937i (0.03) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 1 | 0.132 | 1/2, -0.250+0.938i (0.03) | -0.261+0.943i (0.03), 0.06–0.06 | 0.00, 0.01 | -0.251+0.938i (0.03) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 2 | 0.131 | 1/2, -0.250+0.938i (0.03) | -0.262+0.943i (0.03), 0.06–0.06 | 0.00, 0.01 | -0.250+0.938i (0.03) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 3 | 0.506 | 0/2, -0.804+13.215i (12.26) | -252.770+0.000i (252.52), 0.03–0.10 | 0.00, 0.01 | -0.218+41.666i (40.70) | yes, 0.00 | no |
| lnots2_stable_plr10 | 4 | 0.110 | 1/2, -0.253+0.933i (0.04) | -0.275+0.943i (0.03), 0.11–0.14 | 0.00, 0.01 | -0.253+0.933i (0.04) | yes, 0.00 | **yes** |

Trusted runs (rule v1): 23 of 40. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): tied4_stable_plr10 s0 0.02 (0.04), tied4_stable_plr10 s1 0.03 (0.04), tied4_stable_plr10 s2 0.03 (0.04), tied4_stable_plr10 s3 0.03 (0.04), tied4_stable_plr10 s4 0.02 (0.04), lnotb_stable_plr10 s0 0.02 (0.05), lnotb_stable_plr10 s1 0.02 (0.05), lnotb_stable_plr10 s2 0.03 (0.04), lnotb_stable_plr10 s3 0.02 (0.05), lnots_stable_plr10 s0 0.03 (0.04), lnots_stable_plr10 s1 0.02 (0.04), lnots_stable_plr10 s2 0.02 (0.04), lnots_stable_plr10 s3 0.03 (0.04), lnots_stable_plr10 s4 0.03 (0.04), lnots4_stable_plr10 s0 0.03 (0.04), lnots4_stable_plr10 s1 0.02 (0.04), lnots4_stable_plr10 s2 0.03 (0.03), lnots4_stable_plr10 s3 0.02 (0.04), lnots4_stable_plr10 s4 0.03 (0.04), lnots2_stable_plr10 s0 0.03 (0.03), lnots2_stable_plr10 s1 0.03 (0.03), lnots2_stable_plr10 s2 0.03 (0.03), lnots2_stable_plr10 s4 0.04 (0.03)

Rule v2 linearity gate: the data's best linear pair leaves 0.244 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
