# E5: linear-probe poles of Duffing models, `moderate` data set

Simulated damped Duffing x'' + 0.5 x' + x + x³ = f, `e5_data.py` set `moderate` (diverse E1 waveforms), 5 seeds per variant. Reference: the exact small-signal pole **−0.25 ± 0.9682i**.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.359+1.176i, residual 0.20.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.264 | 0/256, -0.152+0.618i (0.36) | -0.461+0.590i (0.43), 0.19–0.28 | 0.12, 0.49 | -0.427+0.576i (0.43) | no, 0.39 | no |
| base | 1 | 0.404 | 0/256, 0.003+0.237i (0.77) | -35.420-0.000i (35.18), 0.41–0.61 | 0.01, 0.16 | -0.105+0.196i (0.79) | no, 0.16 | no |
| base | 2 | 0.391 | 0/256, -0.122+0.334i (0.65) | -24.317-0.000i (24.09), 0.51–0.73 | 0.03, 0.35 | -0.216+0.313i (0.66) | no, 0.25 | no |
| base | 3 | 0.374 | 0/256, -0.017+0.315i (0.69) | -0.647+0.000i (1.05), 0.60–0.63 | 0.05, 0.53 | -0.228+0.263i (0.71) | no, 0.53 | no |
| base | 4 | 0.539 | 0/256, -0.003+0.096i (0.91) | -834.001+0.000i (833.75), 0.55–0.55 | 0.21, 0.31 | -0.070+0.000i (0.98) | no, 0.61 | no |
| lno_plr10 | 0 | 0.156 | 0/256, -0.031+1.073i (0.24) | -0.400+1.003i (0.15), 0.40–0.46 | 0.01, 0.15 | -0.333+0.895i (0.11) | no, 0.35 | no |
| lno_plr10 | 1 | 0.160 | 0/256, 0.003+1.056i (0.27) | -0.351+0.949i (0.10), 0.34–0.41 | 0.03, 0.17 | -0.322+0.887i (0.11) | no, 0.35 | no |
| lno_plr10 | 2 | 0.178 | 0/256, -0.153+0.875i (0.13) | -1.862+0.001i (1.88), 0.40–0.46 | 0.03, 0.12 | -0.354+0.805i (0.19) | no, 0.24 | no |
| lno_plr10 | 3 | 0.193 | 0/256, -0.060+0.978i (0.19) | -2.019+0.001i (2.02), 0.40–0.45 | 0.03, 0.16 | -0.380+0.797i (0.22) | no, 0.28 | no |
| lno_plr10 | 4 | 0.208 | 0/256, 0.049+0.932i (0.30) | -0.361+0.803i (0.20), 0.27–0.35 | 0.08, 0.45 | -0.348+0.776i (0.22) | no, 0.40 | no |
| lno_stable_plr10 | 0 | 0.075 | 4/256, -0.282+0.935i (0.05) | -0.244+1.029i (0.06), 0.16–0.27 | 0.14, 0.06 | -0.255+1.053i (0.09) | no, 0.05 | no |
| lno_stable_plr10 | 1 | 0.089 | 6/256, -0.255+0.961i (0.01) | -0.258+1.083i (0.11), 0.27–0.37 | 0.17, 0.08 | -0.272+1.095i (0.13) | no, 0.06 | no |
| lno_stable_plr10 | 2 | 0.081 | 4/256, -0.293+0.914i (0.07) | -0.249+1.047i (0.08), 0.19–0.33 | 0.11, 0.03 | -0.269+1.069i (0.10) | no, 0.01 | no |
| lno_stable_plr10 | 3 | 0.082 | 3/256, -0.275+0.921i (0.05) | -0.236+1.037i (0.07), 0.23–0.56 | 0.18, 0.07 | -0.245+1.052i (0.08) | no, 0.02 | no |
| lno_stable_plr10 | 4 | 0.094 | 0/256, -0.151+1.026i (0.11) | -0.240+1.071i (0.10), 0.33–0.37 | 0.12, 0.03 | -0.330+1.143i (0.19) | no, 0.10 | no |
| tied4_stable_plr10 | 0 | 0.173 | 0/4, -0.322+1.185i (0.23) | -0.227+1.123i (0.16), 0.42–0.42 | 0.01, 0.01 | -0.291+1.135i (0.17) | yes, 0.05 | **yes** |
| tied4_stable_plr10 | 1 | 0.198 | 0/4, -0.371+0.990i (0.12) | -0.346+1.099i (0.16), 0.38–0.38 | 0.02, 0.01 | -0.365+1.090i (0.17) | yes, 0.09 | **yes** |
| tied4_stable_plr10 | 2 | 0.178 | 0/4, -0.326+1.114i (0.16) | -0.307+1.146i (0.19), 0.30–0.30 | 0.01, 0.01 | -0.309+1.109i (0.15) | yes, 0.02 | **yes** |
| tied4_stable_plr10 | 3 | 0.178 | 0/4, -0.322+1.104i (0.15) | -0.328+1.133i (0.18), 0.27–0.27 | 0.01, 0.01 | -0.312+1.102i (0.15) | yes, 0.01 | **yes** |
| tied4_stable_plr10 | 4 | 0.190 | 0/4, -0.278+1.410i (0.44) | -0.286+1.049i (0.09), 0.33–0.33 | 0.02, 0.01 | -0.285+1.141i (0.18) | yes, 0.23 | no |
| lnotb_stable_plr10 | 0 | 0.045 | 2/256, -0.212+0.948i (0.04) | -0.233+1.010i (0.05), 0.12–0.29 | 0.01, 0.01 | -0.240+1.020i (0.05) | yes, 0.07 | **yes** |
| lnotb_stable_plr10 | 1 | 0.071 | 11/256, -0.268+0.996i (0.03) | -0.248+1.028i (0.06), 0.10–0.24 | 0.00, 0.01 | -0.248+1.040i (0.07) | yes, 0.02 | **yes** |
| lnotb_stable_plr10 | 2 | 0.068 | 2/256, -0.228+0.940i (0.04) | -0.251+1.032i (0.06), 0.10–0.28 | 0.01, 0.01 | -0.251+1.044i (0.08) | yes, 0.05 | **yes** |
| lnotb_stable_plr10 | 3 | 0.066 | 6/256, -0.219+0.942i (0.04) | -0.244+1.024i (0.06), 0.13–0.37 | 0.01, 0.01 | -0.253+1.040i (0.07) | yes, 0.02 | **yes** |
| lnotb_stable_plr10 | 4 | 0.079 | 1/256, -0.185+0.973i (0.07) | -0.267+1.061i (0.09), 0.23–0.30 | 0.02, 0.06 | -0.289+1.098i (0.14) | yes, 0.10 | no |
| lnots_stable_plr10 | 0 | 0.073 | 0/16, -0.263+0.852i (0.12) | -0.258+1.042i (0.07), 0.14–0.26 | 0.01, 0.01 | -0.262+1.061i (0.09) | yes, 0.12 | no |
| lnots_stable_plr10 | 1 | 0.080 | 1/16, -0.252+0.962i (0.01) | -0.242+1.052i (0.08), 0.16–0.28 | 0.00, 0.01 | -0.254+1.077i (0.11) | yes, 0.10 | no |
| lnots_stable_plr10 | 2 | 0.085 | 1/16, -0.166+0.972i (0.08) | -0.257+1.082i (0.11), 0.13–0.29 | 0.00, 0.02 | -0.275+1.097i (0.13) | yes, 0.14 | no |
| lnots_stable_plr10 | 3 | 0.078 | 1/16, -0.260+0.898i (0.07) | -0.231+1.053i (0.09), 0.12–0.34 | 0.00, 0.01 | -0.254+1.072i (0.10) | yes, 0.16 | no |
| lnots_stable_plr10 | 4 | 0.087 | 1/16, -0.189+0.986i (0.06) | -0.237+1.068i (0.10), 0.22–0.35 | 0.01, 0.02 | -0.274+1.099i (0.13) | yes, 0.13 | no |

Trusted runs (rule v1): 8 of 30. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): tied4_stable_plr10 s0 0.17 (0.06), tied4_stable_plr10 s1 0.17 (0.07), tied4_stable_plr10 s2 0.15 (0.07), tied4_stable_plr10 s3 0.15 (0.07), lnotb_stable_plr10 s0 0.05 (0.16), lnotb_stable_plr10 s1 0.07 (0.14), lnotb_stable_plr10 s2 0.08 (0.14), lnotb_stable_plr10 s3 0.07 (0.14)

Rule v2 linearity gate: the data's best linear pair leaves 0.200 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
