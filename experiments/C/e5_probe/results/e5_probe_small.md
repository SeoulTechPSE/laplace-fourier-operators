# E5: linear-probe poles of Duffing models, `small` data set

Simulated damped Duffing x'' + 0.5 x' + x + x³ = f, `e5_data.py` set `small` (diverse E1 waveforms), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.251+0.971i, residual 0.01.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.408 | 0/256, -0.052+0.187i (0.81) | -62.364-0.000i (62.12), 0.53–0.99 | 0.03, 0.46 | -0.250+0.210i (0.76) | no, 0.61 | no |
| base | 1 | 0.471 | 0/256, 0.072+0.139i (0.89) | -19.555+0.000i (19.33), 0.88–0.93 | 0.36, 0.56 | -0.107+0.127i (0.85) | no, 0.70 | no |
| base | 2 | 0.531 | 0/256, 0.007+0.071i (0.93) | -0.031-0.000i (0.99), 0.85–0.96 | 0.40, 0.40 | -0.033+0.000i (0.99) | no, 1.03 | no |
| base | 3 | 0.456 | 0/256, 0.024+0.100i (0.91) | -14.953-0.000i (14.73), 0.88–0.92 | 0.24, 0.79 | -0.204+0.098i (0.87) | no, 0.84 | no |
| base | 4 | 0.481 | 0/256, 0.014+0.080i (0.93) | -21.321-0.000i (21.09), 0.91–0.96 | 0.43, 0.49 | -0.058+0.145i (0.85) | no, 0.61 | no |
| lno_plr10 | 0 | 0.141 | 0/256, -0.015+0.884i (0.25) | -0.291+0.770i (0.20), 0.34–0.99 | 0.12, 0.36 | -0.285+0.721i (0.25) | no, 0.34 | no |
| lno_plr10 | 1 | 0.374 | 0/256, -0.073+0.340i (0.65) | -0.043+0.320i (0.68), 0.80–0.96 | 0.35, 0.53 | -0.197+0.330i (0.64) | no, 0.32 | no |
| lno_plr10 | 2 | 0.345 | 0/256, 0.009+0.348i (0.67) | -0.068+0.394i (0.60), 0.68–0.97 | 0.48, 0.46 | -0.204+0.406i (0.56) | no, 0.37 | no |
| lno_plr10 | 3 | 0.344 | 0/256, -0.046+0.321i (0.68) | -0.135-0.344i (1.32), 0.55–0.98 | 0.25, 0.63 | -0.192+0.325i (0.65) | no, 0.26 | no |
| lno_plr10 | 4 | 0.950 | 0/256, -0.182+0.453i (0.52) | -0.064+0.491i (0.51), 0.96–1.00 | 0.41, 1.00 | -0.120+0.281i (0.70) | no, 0.06 | no |
| lno_stable_plr10 | 0 | 0.009 | 6/256, -0.279+0.914i (0.06) | -0.252+0.973i (0.00), 0.07–0.56 | 0.05, 0.01 | -0.256+0.980i (0.01) | yes, 0.07 | **yes** |
| lno_stable_plr10 | 1 | 0.020 | 1/256, -0.213+0.917i (0.06) | -0.255+0.977i (0.01), 0.21–0.47 | 0.13, 0.02 | -0.275+1.003i (0.04) | no, 0.10 | no |
| lno_stable_plr10 | 2 | 0.018 | 7/256, -0.244+0.964i (0.01) | -0.257+0.973i (0.01), 0.14–0.24 | 0.17, 0.03 | -0.273+0.999i (0.04) | no, 0.04 | no |
| lno_stable_plr10 | 3 | 0.013 | 1/256, -0.155+0.992i (0.10) | -0.252+0.975i (0.01), 0.14–0.36 | 0.16, 0.02 | -0.265+0.992i (0.03) | no, 0.10 | no |
| lno_stable_plr10 | 4 | 0.102 | 0/256, -0.223+1.102i (0.14) | -0.295+0.985i (0.05), 0.41–0.46 | 0.21, 0.04 | -0.418+1.088i (0.21) | no, 0.17 | no |
| tied4_stable_plr10 | 0 | 0.005 | 1/4, -0.250+0.971i (0.00) | -0.248+0.969i (0.00), 0.01–0.01 | 0.00, 0.01 | -0.250+0.970i (0.00) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 1 | 0.007 | 1/4, -0.252+0.970i (0.00) | -0.241+0.971i (0.01), 0.09–0.09 | 0.01, 0.01 | -0.250+0.971i (0.00) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 2 | 0.005 | 1/4, -0.250+0.971i (0.00) | -0.250+0.970i (0.00), 0.01–0.01 | 0.00, 0.01 | -0.250+0.970i (0.00) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 3 | 0.005 | 1/4, -0.250+0.971i (0.00) | -0.249+0.970i (0.00), 0.01–0.01 | 0.00, 0.01 | -0.250+0.970i (0.00) | yes, 0.00 | **yes** |
| tied4_stable_plr10 | 4 | 0.005 | 2/4, -0.259+0.974i (0.01) | -0.250+0.970i (0.00), 0.00–0.00 | 0.00, 0.01 | -0.250+0.970i (0.00) | yes, 0.01 | **yes** |

Trusted runs: 6 of 20. Their J impulse-response pole vs the reference: lno_stable_plr10 s0 0.01, tied4_stable_plr10 s0 0.00, tied4_stable_plr10 s1 0.00, tied4_stable_plr10 s2 0.00, tied4_stable_plr10 s3 0.00, tied4_stable_plr10 s4 0.00
