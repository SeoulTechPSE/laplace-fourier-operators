# E5: linear-probe poles of Duffing models, `c0_orig` data set

Undamped Duffing x'' + x + x³ = f, the original LNO data (`1D_Duffing_c0`), 5 seeds per variant. Reference: the exact small-signal pole **± i** (on the imaginary axis).
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.229+1.260i, residual 0.88; A <= 1 (20): -0.000+1.006i, residual 0.06; 1 < A <= 3 (40): -0.010+1.067i, residual 0.35; A > 3 (140): -0.143+1.348i, residual 0.85.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.214 | 0/256, -0.081+0.159i (0.84) | -238.180-0.000i (238.18), 0.93–0.94 | 0.62, 0.97 | 0.032+1.025i (0.04) | no, 0.85 | no |
| base | 1 | 0.264 | 0/256, -0.057+0.102i (0.90) | -405.088-344.569i (532.46), 0.96–0.97 | 0.56, 0.98 | -0.059+0.865i (0.15) | no, 0.88 | no |
| base | 2 | 0.199 | 0/256, -0.008+0.121i (0.88) | -85.406-140.761i (165.50), 0.96–0.97 | 0.58, 0.97 | -0.154+0.527i (0.50) | no, 0.77 | no |
| base | 3 | 0.232 | 0/256, -0.010+0.110i (0.89) | -224.947-361.492i (426.62), 0.94–0.95 | 0.68, 0.97 | 0.231+0.965i (0.23) | no, 0.90 | no |
| base | 4 | 0.593 | 0/256, 0.005+0.090i (0.91) | 0.224+0.146i (0.88), 0.99–0.99 | 0.57, 0.97 | -0.008+0.000i (1.00) | no, 1.73 | no |
| lno_plr10 | 0 | 0.202 | 0/256, -0.181+0.409i (0.62) | 0.081+0.000i (1.00), 0.99–0.99 | 0.45, 0.96 | -0.103+0.370i (0.64) | no, 0.09 | no |
| lno_plr10 | 1 | 0.222 | 0/256, -0.207+0.713i (0.35) | -112.105-1175.844i (1182.17), 0.96–0.96 | 0.46, 0.92 | 0.170+0.981i (0.17) | no, 0.46 | no |
| lno_plr10 | 2 | 0.177 | 0/256, 0.131+0.812i (0.23) | -101.173-84.435i (132.42), 0.96–0.96 | 0.40, 0.90 | 0.331+0.000i (1.05) | no, 0.64 | no |
| lno_plr10 | 3 | 0.299 | 0/256, 0.122+0.474i (0.54) | -166.765-0.000i (166.77), 0.98–0.98 | 0.48, 0.94 | -0.109+0.609i (0.41) | no, 0.28 | no |
| lno_plr10 | 4 | 0.661 | 0/256, 0.008+0.332i (0.67) | -397.405+559.475i (685.44), 0.98–0.98 | 0.54, 0.97 | -0.043+0.718i (0.29) | no, 0.54 | no |
| lno_stable_plr10 | 0 | 0.908 | 2/256, 0.014+1.034i (0.04) | -0.039+1.083i (0.09), 0.95–1.00 | 0.62, 0.96 | -0.032+1.127i (0.13) | no, 0.05 | no |
| lno_stable_plr10 | 1 | 0.884 | 3/256, -0.029+1.071i (0.08) | -10.045-0.000i (10.09), 0.97–0.98 | 0.49, 0.95 | -0.283+30.540i (29.54) | no, 0.00 | no |
| lno_stable_plr10 | 2 | 0.814 | 2/256, -0.019+0.948i (0.06) | 0.031+1.149i (0.15), 0.97–0.98 | 0.57, 0.82 | 0.029+9.522i (8.52) | no, 0.01 | no |
| lno_stable_plr10 | 3 | 0.925 | 3/256, -0.023+1.037i (0.04) | 0.016+3.139i (2.14), 0.98–0.99 | 0.69, 0.71 | -0.348+11.240i (10.25) | no, 0.02 | no |
| lno_stable_plr10 | 4 | 0.989 | 2/256, 0.033+0.989i (0.04) | 0.052+2.988i (1.99), 0.96–0.98 | 0.68, 0.89 | 0.007+7.381i (6.38) | no, 0.00 | no |
| tied4_stable_plr10 | 0 | 0.880 | 0/4, -0.213+1.294i (0.36) | -0.207+1.287i (0.35), 0.04–0.04 | 0.00, 0.01 | -0.211+1.288i (0.36) | yes, 0.00 | no |
| tied4_stable_plr10 | 1 | 0.886 | 0/4, -0.173+1.112i (0.21) | -0.184+1.169i (0.25), 0.20–0.20 | 0.02, 0.01 | -0.308+1.291i (0.42) | yes, 0.17 | no |
| tied4_stable_plr10 | 2 | 0.981 | 0/4, -1.763+2.863i (2.56) | -2.124+2.892i (2.84), 0.20–0.20 | 0.00, 0.01 | -0.491+20.863i (19.87) | yes, 0.01 | no |
| tied4_stable_plr10 | 3 | 0.995 | 0/4, -1.126+3.581i (2.82) | -3.449+0.000i (3.59), 0.59–0.59 | 0.01, 0.01 | -0.787+25.594i (24.61) | yes, 0.03 | no |
| tied4_stable_plr10 | 4 | 0.933 | 0/4, -0.434+0.307i (0.82) | -0.397+1.352i (0.53), 0.48–0.48 | 0.01, 0.01 | -0.678+40.951i (39.96) | yes, 0.01 | no |
| lnotb_stable_plr10 | 0 | 0.378 | 2/256, 0.037+0.982i (0.04) | 0.005+1.017i (0.02), 0.48–0.88 | 0.00, 0.37 | -0.008+1.031i (0.03) | no, 0.06 | no |
| lnotb_stable_plr10 | 1 | 0.487 | 4/256, -0.023+1.016i (0.03) | 0.021+0.997i (0.02), 0.45–0.93 | 0.00, 0.26 | 0.029+0.995i (0.03) | no, 0.05 | no |
| lnotb_stable_plr10 | 2 | 0.412 | 3/256, -0.008+0.984i (0.02) | -0.004+0.975i (0.03), 0.60–0.83 | 0.01, 0.32 | -0.006+1.040i (0.04) | no, 0.05 | no |
| lnotb_stable_plr10 | 3 | 0.299 | 0/256, 0.103+0.998i (0.10) | -0.004+1.017i (0.02), 0.38–0.90 | 0.00, 0.25 | 0.018+1.025i (0.03) | no, 0.09 | no |
| lnotb_stable_plr10 | 4 | 0.637 | 4/256, 0.028+1.042i (0.05) | 0.030+1.179i (0.18), 0.88–0.96 | 0.00, 0.90 | -0.030+0.831i (0.17) | no, 0.07 | no |

Trusted runs (rule v1): 0 of 25. 

Rule v2 linearity gate: the data's best linear pair leaves 0.884 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
