# E5: linear-probe poles of Duffing models, `lz_r10_wide` data set

Forced Lorenz ρ = 10 (z' = xy − 8/3 z − f, output x), `e5_data.py` set `lz_r10_wide` (diverse E1 waveforms, started at the equilibrium C+), Lorenz ρ = 10 LNO architecture (grid input, tanh, 84 modes), 5 seeds per variant; the tied layer has a learned output offset started at the training mean. Reference: the dominant pole of the linearisation at C+, **−0.5955 ± 6.1742i** (third pole −12.48). Equivalent poles from centred data.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `tied4o_stable_plr10` the same with no weight decay on the output offset (tenth-pass addendum).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.784+6.051i, residual 0.16; A <= 1 (26): -0.714+6.096i, residual 0.05; 1 < A <= 3 (81): -0.739+6.081i, residual 0.07; A > 3 (93): -0.823+6.061i, residual 0.14.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.019 | 0/1344, -0.059+0.298i (0.95) | -425.588-0.000i (68.52), 0.05–0.06 | 0.00, 0.04 | -0.212+0.266i (0.95) | yes, 0.46 | no |
| base | 1 | 0.019 | 0/1344, -0.029+0.354i (0.94) | -298.043-0.000i (47.96), 0.06–0.07 | 0.00, 0.03 | -0.296+0.350i (0.94) | yes, 0.58 | no |
| base | 2 | 0.019 | 0/1344, -0.110+0.437i (0.93) | -169.118-0.000i (27.19), 0.09–0.10 | 0.00, 0.01 | -0.377+0.394i (0.93) | yes, 0.39 | no |
| base | 3 | 0.019 | 0/1344, -0.128+0.295i (0.95) | -173.250+1197.126i (194.01), 0.07–0.08 | 0.00, 0.04 | -0.199+0.153i (0.97) | yes, 0.07 | **yes** |
| base | 4 | 0.019 | 0/1344, -0.123+0.158i (0.97) | -572.449-0.000i (92.20), 0.03–0.07 | 0.00, 0.01 | -0.166+0.201i (0.97) | yes, 0.23 | no |
| lno_plr10 | 0 | 0.018 | 0/1344, -0.168+0.945i (0.85) | -46.286+0.000i (7.43), 0.16–0.17 | 0.01, 0.05 | -0.765+0.828i (0.86) | yes, 0.54 | no |
| lno_plr10 | 1 | 0.019 | 0/1344, -0.138+0.755i (0.88) | -67.812+0.000i (10.88), 0.13–0.13 | 0.01, 0.06 | -0.560+0.666i (0.89) | yes, 0.48 | no |
| lno_plr10 | 2 | 0.018 | 0/1344, -0.141+0.976i (0.84) | -27.212+0.000i (4.40), 0.21–0.21 | 0.01, 0.05 | -0.760+0.967i (0.84) | yes, 0.43 | no |
| lno_plr10 | 3 | 0.018 | 0/1344, -0.268+1.235i (0.80) | -28.590+0.000i (4.62), 0.20–0.21 | 0.01, 0.03 | -0.878+1.100i (0.82) | yes, 0.44 | no |
| lno_plr10 | 4 | 0.019 | 0/1344, -0.057+0.661i (0.89) | -60.783+0.000i (9.75), 0.16–0.16 | 0.01, 0.03 | -0.549+0.576i (0.90) | yes, 0.43 | no |
| lno_stable_plr10 | 0 | 0.006 | 17/1344, -0.115+6.111i (0.08) | -0.681+5.953i (0.04), 0.15–0.15 | 0.05, 0.04 | -0.616+5.958i (0.04) | yes, 0.08 | **yes** |
| lno_stable_plr10 | 1 | 0.007 | 29/1344, -0.293+6.303i (0.05) | -0.749+6.001i (0.04), 0.18–0.18 | 0.23, 0.12 | -0.689+5.982i (0.03) | no, 0.08 | no |
| lno_stable_plr10 | 2 | 0.006 | 24/1344, -0.232+6.173i (0.06) | -0.710+6.005i (0.03), 0.11–0.11 | 0.14, 0.04 | -0.665+5.993i (0.03) | no, 0.06 | no |
| lno_stable_plr10 | 3 | 0.007 | 19/1344, -0.388+6.004i (0.04) | -0.726+5.949i (0.04), 0.20–0.20 | 0.21, 0.06 | -0.653+5.950i (0.04) | no, 0.05 | no |
| lno_stable_plr10 | 4 | 0.007 | 24/1344, -0.183+6.177i (0.07) | -0.794+5.991i (0.04), 0.26–0.26 | 0.32, 0.09 | -0.707+5.960i (0.04) | no, 0.09 | no |
| tied4_stable_plr10 | 0 | 0.643 | 1/4, -0.271+6.687i (0.10) | -0.310+6.610i (0.08), 0.57–0.57 | 0.01, 0.01 | -0.259+34.056i (4.50) | yes, 0.00 | no |
| tied4_stable_plr10 | 1 | 0.645 | 0/4, -1.454+3.831i (0.40) | -1.649-0.001i (1.01), 0.47–0.47 | 0.02, 0.01 | -0.868+26.390i (3.26) | yes, 0.03 | no |
| tied4_stable_plr10 | 2 | 0.641 | 0/4, -1.518+5.725i (0.17) | -0.481+1.926i (0.69), 0.76–0.76 | 0.03, 0.01 | -1.097+17.506i (1.83) | yes, 0.17 | no |
| tied4_stable_plr10 | 3 | 0.639 | 0/4, -1.082+7.132i (0.17) | -0.463+7.392i (0.20), 0.78–0.78 | 0.02, 0.01 | -0.323+23.684i (2.82) | yes, 0.03 | no |
| tied4_stable_plr10 | 4 | 0.648 | 0/4, -0.881+2.043i (0.67) | -1.145+0.000i (1.00), 0.50–0.50 | 0.02, 0.01 | -1.208+38.819i (5.26) | yes, 0.07 | no |
| tied4o_stable_plr10 | 0 | 0.006 | 1/4, -0.732+6.098i (0.03) | -0.753+6.045i (0.03), 0.02–0.02 | 0.01, 0.01 | -0.730+6.037i (0.03) | yes, 0.01 | **yes** |
| tied4o_stable_plr10 | 1 | 0.007 | 1/4, -0.584+6.295i (0.02) | -0.714+6.073i (0.03), 0.15–0.15 | 0.01, 0.01 | -0.632+6.069i (0.02) | yes, 0.04 | **yes** |
| tied4o_stable_plr10 | 2 | 0.006 | 1/4, -0.723+6.518i (0.06) | -0.753+6.002i (0.04), 0.11–0.11 | 0.00, 0.01 | -0.708+5.996i (0.03) | yes, 0.09 | **yes** |
| tied4o_stable_plr10 | 3 | 0.009 | 0/4, -0.567+6.852i (0.11) | -0.931+5.978i (0.06), 0.32–0.32 | 0.01, 0.01 | -0.551+6.080i (0.02) | yes, 0.13 | no |
| tied4o_stable_plr10 | 4 | 0.014 | 0/4, -1.421+2.228i (0.65) | -2.148+3.580i (0.49), 0.09–0.09 | 0.02, 0.01 | -1.492+39.370i (5.35) | yes, 0.05 | **yes** |

Trusted runs (rule v1): 6 of 25. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): base s3 0.97 (0.97), lno_stable_plr10 s0 0.04 (0.03), tied4o_stable_plr10 s0 0.03 (0.01), tied4o_stable_plr10 s1 0.02 (0.03), tied4o_stable_plr10 s2 0.03 (0.02), tied4o_stable_plr10 s4 5.35 (5.46)

Rule v2 linearity gate: the data's best linear pair leaves 0.164 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
