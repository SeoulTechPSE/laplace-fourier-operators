# E5: linear-probe poles of Duffing models, `lz_r10_orig` data set

Forced Lorenz ρ = 10 (z' = xy − 8/3 z − f, output x), the original LNO data (`1D_Lorenz_rho10`, from x(0) = 1, y(0) = z(0) = 0, inputs A sin 5t), Lorenz ρ = 10 LNO architecture (grid input, tanh, 84 modes), 5 seeds per variant; the tied layer has a learned output offset started at the training mean. Reference: the dominant pole of the linearisation at C+, **−0.5955 ± 6.1742i** (third pole −12.48). Equivalent poles from centred data.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `tied4o_stable_plr10` the same with no weight decay on the output offset (tenth-pass addendum); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.215+6.577i, residual 0.99; A <= 1 (20): -0.147+5.848i, residual 0.50; 1 < A <= 3 (40): -0.105+5.725i, residual 0.66; A > 3 (140): -0.241+5.842i, residual 0.97.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.191 | 0/1344, -0.388+1.203i (0.80) | -0.027-0.000i (1.00), 0.94–0.96 | 0.96, 0.96 | 0.099+0.000i (1.00) | no, 0.03 | no |
| base | 1 | 0.267 | 0/1344, -0.337+1.242i (0.80) | -0.023+0.000i (1.00), 0.94–0.95 | 0.98, 0.89 | 0.207+0.000i (1.00) | no, 0.19 | no |
| base | 2 | 0.165 | 0/1344, -0.203+0.720i (0.88) | -0.278-0.000i (1.00), 0.92–0.97 | 0.95, 1.00 | 0.380+0.555i (0.92) | no, 0.59 | no |
| base | 3 | 0.181 | 0/1344, 0.134+0.488i (0.92) | -0.114-0.000i (1.00), 0.87–0.90 | 0.93, 0.89 | -0.316+0.104i (0.98) | no, 0.42 | no |
| base | 4 | 0.281 | 0/1344, -0.228+0.766i (0.87) | -0.029+0.000i (1.00), 0.92–0.96 | 0.96, 0.91 | 0.219+0.357i (0.95) | no, 0.12 | no |
| lno_plr10 | 0 | 0.187 | 0/1344, -0.410+2.031i (0.67) | -0.880+0.000i (1.00), 0.94–0.95 | 0.95, 0.85 | -0.781+0.000i (1.00) | no, 0.16 | no |
| lno_plr10 | 1 | 0.223 | 0/1344, -0.084+2.001i (0.68) | -0.273+0.000i (1.00), 0.90–0.93 | 0.73, 0.87 | -0.092+0.000i (1.00) | no, 0.09 | no |
| lno_plr10 | 2 | 0.540 | 0/1344, 0.034+2.163i (0.65) | -0.414+11.487i (0.86), 0.99–1.00 | 0.86, 0.98 | -0.890+0.699i (0.88) | no, 0.44 | no |
| lno_plr10 | 3 | 0.584 | 0/1344, 0.430+1.240i (0.81) | -0.128+0.612i (0.90), 0.99–1.00 | 0.95, 0.98 | -1.048+1.003i (0.84) | no, 0.32 | no |
| lno_plr10 | 4 | 0.201 | 0/1344, -0.317+1.986i (0.68) | -0.096-0.076i (1.01), 0.86–0.92 | 0.87, 0.94 | -0.286+0.225i (0.96) | no, 0.04 | no |
| lno_stable_plr10 | 0 | 0.536 | 52/1344, -0.598+6.125i (0.01) | -0.113+5.491i (0.13), 0.93–0.94 | 0.75, 0.92 | -0.244+5.135i (0.18) | no, 0.00 | no |
| lno_stable_plr10 | 1 | 0.503 | 49/1344, -0.536+6.287i (0.02) | -0.148+5.522i (0.13), 0.99–1.00 | 0.51, 1.00 | -0.248+5.890i (0.07) | no, 0.01 | no |
| lno_stable_plr10 | 2 | 0.520 | 69/1344, -0.648+6.081i (0.02) | -0.185+5.501i (0.13), 0.98–0.99 | 0.64, 1.00 | -0.321+3.072i (0.50) | no, 0.01 | no |
| lno_stable_plr10 | 3 | 0.483 | 67/1344, -0.532+6.247i (0.02) | 0.145-0.641i (1.11), 0.99–0.99 | 0.69, 1.00 | -0.219+36.081i (4.82) | no, 0.00 | no |
| lno_stable_plr10 | 4 | 0.351 | 46/1344, -0.597+6.020i (0.02) | -0.246-0.278i (1.04), 1.00–1.00 | 0.79, 0.90 | -0.022+0.621i (0.90) | no, 0.01 | no |
| tied4_stable_plr10 | 0 | 0.932 | 0/4, -0.265+6.826i (0.12) | -0.178+5.308i (0.15), 0.73–0.73 | 0.01, 0.01 | -0.038+34.065i (4.50) | yes, 0.00 | no |
| tied4_stable_plr10 | 1 | 0.931 | 1/4, -0.822+6.144i (0.04) | -0.865+6.385i (0.06), 0.30–0.30 | 0.02, 0.01 | -0.083+27.070i (3.37) | yes, 0.00 | no |
| tied4_stable_plr10 | 2 | 0.934 | 0/4, -0.698+5.165i (0.16) | -0.532+5.148i (0.17), 0.63–0.63 | 0.01, 0.02 | 0.055+10.106i (0.64) | yes, 0.00 | no |
| tied4_stable_plr10 | 3 | 0.933 | 0/4, -0.842+4.333i (0.30) | -0.809+3.803i (0.38), 0.43–0.43 | 0.01, 0.01 | -0.527+0.000i (1.00) | yes, 8.25 | no |
| tied4_stable_plr10 | 4 | 0.928 | 0/4, -0.344+4.878i (0.21) | -0.437+1.470i (0.76), 0.35–0.35 | 0.01, 0.01 | -0.564+0.000i (1.00) | yes, 2.49 | no |
| tied4o_stable_plr10 | 0 | 0.503 | 0/4, -1.656+5.573i (0.20) | -0.446+7.942i (0.29), 0.26–0.26 | 0.01, 0.01 | -0.062+33.936i (4.48) | yes, 0.00 | no |
| tied4o_stable_plr10 | 1 | 0.501 | 1/4, -0.160+6.446i (0.08) | -0.210+0.560i (0.91), 0.47–0.47 | 0.02, 0.01 | -0.204+0.571i (0.91) | yes, 0.01 | no |
| tied4o_stable_plr10 | 2 | 0.502 | 1/4, -0.430+5.766i (0.07) | -3.086+0.001i (1.07), 0.41–0.41 | 0.01, 0.01 | -0.216+20.929i (2.38) | yes, 0.00 | no |
| tied4o_stable_plr10 | 3 | 0.508 | 0/4, -0.753+4.027i (0.35) | -0.559+3.413i (0.45), 0.74–0.74 | 0.01, 0.01 | -0.339+23.841i (2.85) | yes, 0.03 | no |
| tied4o_stable_plr10 | 4 | 0.533 | 0/4, 0.064+0.339i (0.95) | -0.088+0.023i (1.00), 0.16–0.16 | 0.00, 0.02 | 0.153+0.000i (1.00) | yes, 0.99 | no |
| lnotb_stable_plr10 | 0 | 0.336 | 8/1344, -0.061+6.367i (0.09) | -3.790+4.706i (0.57), 0.96–0.99 | 0.02, 0.97 | -1.131+0.000i (1.00) | no, 0.93 | no |
| lnotb_stable_plr10 | 1 | 0.212 | 23/1344, -0.174+5.888i (0.08) | -1.052+2.037i (0.67), 0.88–0.92 | 0.01, 0.90 | -0.738+2.348i (0.62) | no, 0.16 | no |
| lnotb_stable_plr10 | 2 | 0.214 | 20/1344, -0.161+6.159i (0.07) | -1.078+2.279i (0.63), 0.88–0.95 | 0.01, 0.93 | -0.799+2.422i (0.61) | no, 0.15 | no |
| lnotb_stable_plr10 | 3 | 0.206 | 11/1344, -0.242+6.249i (0.06) | -0.891+2.027i (0.67), 0.91–0.96 | 0.01, 0.98 | -0.408+0.773i (0.87) | no, 0.24 | no |
| lnotb_stable_plr10 | 4 | 0.216 | 13/1344, -0.490+6.190i (0.02) | -0.925+2.287i (0.63), 0.89–0.95 | 0.01, 0.91 | -0.681+2.406i (0.61) | no, 0.18 | no |
| lnots_stable_plr10 | 0 | 0.223 | 7/84, -0.225+6.031i (0.06) | -2.207-0.000i (1.03), 0.78–0.89 | 0.01, 0.76 | 0.126+10.045i (0.63) | no, 0.01 | no |
| lnots_stable_plr10 | 1 | 0.215 | 3/84, -0.194+5.948i (0.07) | -1.237+5.842i (0.12), 0.94–0.97 | 0.01, 0.95 | -1.217+5.386i (0.16) | no, 0.18 | no |
| lnots_stable_plr10 | 2 | 0.416 | 7/84, -0.690+5.990i (0.03) | -3.402+3.817i (0.59), 0.84–0.92 | 0.01, 0.92 | 0.174+9.981i (0.63) | no, 0.00 | no |
| lnots_stable_plr10 | 3 | 0.203 | 1/84, -0.274+5.692i (0.09) | -0.629+2.556i (0.58), 0.92–0.98 | 0.01, 0.98 | -0.262+10.073i (0.63) | no, 0.01 | no |
| lnots_stable_plr10 | 4 | 0.429 | 6/84, -0.524+6.394i (0.04) | -29.441+0.029i (4.75), 0.92–0.94 | 0.01, 0.92 | -0.258+9.991i (0.62) | no, 0.01 | no |
| lnots4_stable_plr10 | 0 | 0.484 | 1/4, -0.607+5.790i (0.06) | -2.116+7.983i (0.38), 0.51–0.55 | 0.01, 0.37 | -1.542+8.055i (0.34) | no, 0.13 | no |
| lnots4_stable_plr10 | 1 | 0.465 | 1/4, -0.635+5.780i (0.06) | -1.586+8.365i (0.39), 0.70–0.70 | 0.00, 0.44 | -0.004+10.283i (0.67) | no, 0.02 | no |
| lnots4_stable_plr10 | 2 | 0.326 | 0/4, -1.455+6.235i (0.14) | 0.438+1.793i (0.73), 0.62–0.98 | 0.01, 0.79 | -1.074+11.911i (0.93) | no, 0.09 | no |
| lnots4_stable_plr10 | 3 | 0.485 | 1/4, -1.093+6.128i (0.08) | -3.171-0.001i (1.08), 0.77–0.78 | 0.02, 0.46 | -1.018+10.472i (0.70) | no, 0.03 | no |
| lnots4_stable_plr10 | 4 | 0.521 | 0/4, -0.536+5.223i (0.15) | -2.040+7.811i (0.35), 0.70–0.72 | 0.01, 0.03 | 0.017+10.003i (0.63) | yes, 0.00 | no |
| lnots2_stable_plr10 | 0 | 0.484 | 1/2, -0.731+5.734i (0.07) | -1.372+7.461i (0.24), 0.59–0.60 | 0.00, 0.23 | -0.423+14.433i (1.33) | no, 0.10 | no |
| lnots2_stable_plr10 | 1 | 0.556 | 0/2, 0.008+10.012i (0.63) | 0.002+10.036i (0.63), 0.46–0.48 | 0.01, 0.32 | -0.009+10.003i (0.62) | no, 0.00 | no |
| lnots2_stable_plr10 | 2 | 0.508 | 0/2, -0.922+4.472i (0.28) | -0.640+4.834i (0.22), 0.79–0.81 | 0.00, 0.87 | -0.468+12.113i (0.96) | no, 0.02 | no |
| lnots2_stable_plr10 | 3 | 0.539 | 0/2, 0.023+15.106i (1.44) | -0.484+14.697i (1.37), 0.32–0.64 | 0.01, 0.50 | -0.038+15.104i (1.44) | no, 0.00 | no |
| lnots2_stable_plr10 | 4 | 0.505 | 1/2, -0.742+5.715i (0.08) | -1.554+8.236i (0.37), 0.56–0.57 | 0.00, 0.18 | -0.838+6.090i (0.04) | no, 0.06 | no |

Trusted runs (rule v1): 0 of 45. 

Rule v2 linearity gate: the data's best linear pair leaves 0.992 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
