# E5: linear-probe poles of Duffing models

Damped Duffing (`1D_Duffing_c05`), original split, 5 seeds per variant. Linearised pole from the 4 smallest-amplitude training samples (|f| ≤ 0.20): **-0.2500 ± 0.9684i** (theory −0.25 ± 0.9682i). Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.304+1.137i, residual 0.31; A <= 1 (20): -0.250+0.971i, residual 0.01; 1 < A <= 3 (40): -0.251+0.997i, residual 0.04; A > 3 (140): -0.298+1.199i, residual 0.28.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1).

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d |
|---|---|---|---|---|---|---|---|
| base | 0 | 0.135 | 0/256, 0.005+0.076i (0.93) | -26.539-0.017i (26.30), 0.86–0.87 | 0.88, 0.59 | -0.169+0.968i (0.08) | no, 0.92 |
| base | 1 | 0.132 | 0/256, 0.010+0.087i (0.92) | -627.936+539.449i (826.91), 0.83–0.84 | 0.79, 0.62 | -0.246+1.050i (0.08) | no, 0.92 |
| base | 2 | 0.127 | 0/256, 0.028+0.092i (0.92) | -549.849-358.315i (656.53), 0.82–0.82 | 0.84, 0.58 | -0.150+1.072i (0.14) | no, 0.92 |
| base | 3 | 0.224 | 0/256, 0.028+0.086i (0.93) | -298.285-532.815i (611.27), 0.98–0.98 | 0.90, 0.95 | -0.189+0.757i (0.22) | no, 0.90 |
| base | 4 | 0.373 | 0/256, 0.007+0.110i (0.90) | -260.978-341.811i (430.61), 0.97–0.98 | 0.76, 0.97 | -0.159+0.987i (0.09) | no, 0.89 |
| lno_plr10 | 0 | 0.149 | 0/256, -0.212+0.650i (0.32) | -11.163-0.004i (10.95), 0.85–0.86 | 0.55, 0.44 | -0.423+0.000i (0.98) | no, 0.34 |
| lno_plr10 | 1 | 0.301 | 0/256, 0.124+0.395i (0.68) | -61.437+0.088i (61.19), 0.99–0.99 | 0.52, 0.98 | -0.163+1.006i (0.09) | no, 0.66 |
| lno_plr10 | 2 | 0.240 | 0/256, -0.179+0.296i (0.68) | -398.015+952.390i (1031.09), 0.98–0.99 | 0.65, 0.97 | -0.283+1.033i (0.07) | no, 0.69 |
| lno_plr10 | 3 | 0.200 | 0/256, -0.262+0.652i (0.32) | -67.609-0.089i (67.36), 0.88–0.89 | 0.52, 0.64 | -0.142+0.604i (0.38) | no, 0.21 |
| lno_plr10 | 4 | 0.289 | 0/256, 0.121+0.567i (0.55) | -0.194+0.734i (0.24), 0.99–0.99 | 0.54, 0.96 | -0.171+0.604i (0.37) | no, 0.43 |
| lno_stable_plr10 | 0 | 0.338 | 2/256, -0.267+1.013i (0.05) | -0.216+0.989i (0.04), 0.74–0.77 | 0.68, 0.32 | -0.125+1.349i (0.40) | no, 0.16 |
| lno_stable_plr10 | 1 | 0.397 | 0/256, -0.257+1.113i (0.14) | -2.956-0.001i (2.87), 0.93–0.94 | 0.76, 0.38 | -0.464+8.360i (7.39) | no, 0.02 |
| lno_stable_plr10 | 2 | 0.326 | 1/256, -0.248+0.980i (0.01) | -47.841+0.060i (47.59), 0.60–0.61 | 0.21, 0.04 | -0.372+2.457i (1.49) | no, 0.14 |
| lno_stable_plr10 | 3 | 0.468 | 0/256, -0.154+1.058i (0.13) | -0.430+1.055i (0.20), 0.70–0.70 | 0.37, 0.14 | -0.225+0.986i (0.03) | no, 0.10 |
| lno_stable_plr10 | 4 | 0.504 | 2/256, -0.289+0.942i (0.05) | -1.224+6.470i (5.59), 0.78–0.79 | 0.72, 0.21 | -1.331+1.293i (1.13) | no, 0.49 |
| tied4_stable_plr10 | 0 | 0.269 | 0/4, -0.265+1.087i (0.12) | -0.279+1.103i (0.14), 0.09–0.09 | 0.01, 0.01 | -0.268+1.091i (0.12) | yes, 0.00 |
| tied4_stable_plr10 | 1 | 0.267 | 0/4, -0.255+1.072i (0.10) | -0.274+1.101i (0.13), 0.10–0.10 | 0.02, 0.01 | -0.273+1.105i (0.14) | yes, 0.03 |
| tied4_stable_plr10 | 2 | 0.899 | 0/4, -2.803+7.910i (7.40) | -2.638-7.861i (9.15), 0.08–0.08 | 0.02, 0.01 | -0.951+18.105i (17.15) | yes, 0.18 |
| tied4_stable_plr10 | 3 | 0.929 | 0/4, -1.323+3.614i (2.85) | -42.280-0.000i (42.04), 0.46–0.46 | 0.02, 0.01 | -1.524+28.632i (27.69) | yes, 0.03 |
| tied4_stable_plr10 | 4 | 0.258 | 1/4, -0.252+1.045i (0.08) | -0.286+1.123i (0.16), 0.11–0.11 | 0.02, 0.01 | -0.274+1.111i (0.14) | yes, 0.06 |
