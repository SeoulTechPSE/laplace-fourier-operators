# E5: linear-probe poles of Duffing models, `lz_orig` data set

Forced Lorenz ρ = 5 (z' = xy − 8/3 z − f, output x), the original LNO data (`1D_Lorenz_rho5`, from x(0) = 1, y(0) = z(0) = 0, inputs A sin 5t), Lorenz LNO architecture (grid input, tanh, 16 modes), 5 seeds per variant; the tied layer has a learned output offset. Reference: the dominant pole of the linearisation at C+, **−0.9287 ± 4.1476i** (third pole −11.81). Equivalent poles from centred data.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.661+3.883i, residual 0.53; A <= 1 (20): -1.410+4.854i, residual 0.14; 1 < A <= 3 (40): -1.515+4.441i, residual 0.23; A > 3 (140): -0.487+3.804i, residual 0.57.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.020 | 0/256, -0.095+0.261i (0.94) | -12.245-0.004i (2.84), 0.79–0.82 | 0.59, 0.33 | -0.025+0.174i (0.96) | no, 0.28 | no |
| base | 1 | 0.016 | 0/256, -0.037+0.253i (0.94) | -8.696-0.003i (2.07), 0.79–0.80 | 0.54, 0.30 | -0.110+0.000i (0.99) | no, 0.23 | no |
| base | 2 | 0.022 | 0/256, -0.123+0.147i (0.96) | -11.402-0.004i (2.65), 0.81–0.82 | 0.58, 0.31 | 0.096+0.278i (0.94) | no, 0.55 | no |
| base | 3 | 0.022 | 0/256, -0.007+0.138i (0.97) | -12.568-0.005i (2.91), 0.80–0.83 | 0.62, 0.38 | -0.149+0.145i (0.96) | no, 0.47 | no |
| base | 4 | 0.035 | 0/256, -0.023+0.119i (0.97) | -13.681-0.005i (3.16), 0.81–0.83 | 0.67, 0.39 | -0.044+0.000i (1.00) | no, 0.61 | no |
| lno_plr10 | 0 | 0.033 | 0/256, -0.545+2.189i (0.47) | -1.557-0.000i (0.99), 0.74–0.79 | 0.49, 0.21 | -0.408+0.000i (0.98) | no, 0.28 | no |
| lno_plr10 | 1 | 0.098 | 0/256, -0.153+0.604i (0.85) | -2.631+0.001i (1.05), 0.96–0.98 | 0.65, 0.91 | -0.191+0.345i (0.91) | no, 0.04 | no |
| lno_plr10 | 2 | 0.027 | 0/256, -0.053+0.973i (0.77) | -11.831-0.004i (2.74), 0.86–0.87 | 0.41, 0.39 | 0.106+0.601i (0.87) | no, 0.20 | no |
| lno_plr10 | 3 | 0.015 | 0/256, -0.007+0.931i (0.79) | -14.145-0.005i (3.26), 0.79–0.82 | 0.43, 0.24 | -0.019+0.116i (0.97) | no, 0.07 | no |
| lno_plr10 | 4 | 0.021 | 0/256, -0.321+1.718i (0.59) | -0.652+0.000i (0.98), 0.70–0.74 | 0.42, 0.24 | -0.159+0.045i (0.98) | no, 0.06 | no |
| lno_stable_plr10 | 0 | 0.030 | 6/256, -0.690+4.206i (0.06) | -5.058+9.801i (1.65), 0.51–0.51 | 0.16, 0.07 | -0.114+11.086i (1.64) | no, 0.02 | no |
| lno_stable_plr10 | 1 | 0.042 | 1/256, -1.171+4.120i (0.06) | -66.981-0.000i (15.57), 0.67–0.73 | 0.17, 0.13 | -1.671+1.123i (0.73) | no, 0.47 | no |
| lno_stable_plr10 | 2 | 0.025 | 2/256, -0.928+3.920i (0.05) | -5.720+0.002i (1.49), 0.68–0.68 | 0.25, 0.09 | -0.513+0.368i (0.89) | no, 0.59 | no |
| lno_stable_plr10 | 3 | 0.028 | 0/256, -0.474+4.276i (0.11) | -6.696+16.228i (3.15), 0.59–0.59 | 0.11, 0.05 | -0.897+22.341i (4.28) | no, 0.03 | no |
| lno_stable_plr10 | 4 | 0.034 | 2/256, -0.910+3.739i (0.10) | -19.131+0.011i (4.39), 0.73–0.73 | 0.23, 0.11 | -0.100+20.113i (3.76) | no, 0.01 | no |
| tied4_stable_plr10 | 0 | 0.512 | 0/4, -1.224+5.247i (0.27) | -2.241+5.787i (0.49), 0.14–0.14 | 0.02, 0.01 | -0.142+34.046i (7.04) | yes, 0.00 | no |
| tied4_stable_plr10 | 1 | 0.363 | 0/4, -2.167+7.649i (0.87) | -0.081+0.000i (1.00), 0.37–0.37 | 0.01, 0.02 | 0.040+0.000i (1.00) | yes, 1.60 | no |
| tied4_stable_plr10 | 2 | 0.512 | 0/4, -1.340+4.366i (0.11) | -0.842+5.590i (0.34), 0.25–0.25 | 0.02, 0.01 | -0.165+20.916i (3.95) | yes, 0.00 | no |
| tied4_stable_plr10 | 3 | 0.515 | 0/4, -0.651+3.581i (0.15) | -323.494-276.513i (100.60), 0.61–0.61 | 0.01, 0.01 | -0.524+24.382i (4.76) | yes, 0.04 | no |
| tied4_stable_plr10 | 4 | 0.365 | 0/4, -0.669+1.596i (0.60) | -0.117-0.000i (0.99), 0.46–0.46 | 0.01, 0.02 | 0.027+0.000i (1.00) | yes, 2.32 | no |
| lnotb_stable_plr10 | 0 | 0.015 | 3/256, -0.776+4.126i (0.04) | -5.081-7.939i (3.01), 0.87–0.89 | 0.01, 0.32 | 0.023+0.603i (0.86) | no, 0.21 | no |
| lnotb_stable_plr10 | 1 | 0.024 | 2/256, -0.952+3.836i (0.07) | -7.680-0.003i (1.86), 0.81–0.85 | 0.00, 0.61 | -0.293+0.680i (0.83) | no, 0.25 | no |
| lnotb_stable_plr10 | 2 | 0.013 | 3/256, -0.546+4.105i (0.09) | -0.789+1.049i (0.73), 0.85–0.85 | 0.01, 0.10 | -0.026+1.730i (0.61) | no, 0.20 | no |
| lnotb_stable_plr10 | 3 | 0.010 | 2/256, -0.783+4.406i (0.07) | -9.995-19.625i (5.99), 0.65–0.66 | 0.00, 0.15 | -1.594+2.983i (0.32) | no, 0.28 | no |
| lnotb_stable_plr10 | 4 | 0.009 | 6/256, -0.702+4.317i (0.07) | -2.802+6.010i (0.62), 0.74–0.75 | 0.00, 0.23 | -0.329+7.830i (0.88) | no, 0.01 | no |
| lnots_stable_plr10 | 0 | 0.007 | 0/16, -0.629+3.793i (0.11) | -1.947+0.000i (1.00), 0.60–0.61 | 0.00, 0.10 | -0.168+0.000i (0.99) | no, 7.35 | no |
| lnots_stable_plr10 | 1 | 0.035 | 1/16, -0.768+3.965i (0.06) | -1.160+1.977i (0.51), 0.59–0.65 | 0.00, 0.06 | -0.841+3.830i (0.08) | yes, 0.04 | **yes** |
| lnots_stable_plr10 | 2 | 0.007 | 0/16, -0.810+4.586i (0.11) | -1.301+5.403i (0.31), 0.44–0.44 | 0.00, 0.09 | -0.827+10.259i (1.44) | yes, 0.10 | no |
| lnots_stable_plr10 | 3 | 0.008 | 1/16, -0.528+4.135i (0.09) | -1.978+0.000i (1.01), 0.68–0.68 | 0.01, 0.10 | -0.524+33.090i (6.81) | no, 0.01 | no |
| lnots_stable_plr10 | 4 | 0.009 | 1/16, -0.561+4.183i (0.09) | -1.783+5.033i (0.29), 0.51–0.51 | 0.00, 0.07 | -0.300+40.497i (8.55) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 0 | 0.022 | 0/4, -0.737+6.345i (0.52) | -448.127+338.138i (131.32), 0.44–0.48 | 0.00, 0.14 | -0.824+8.902i (1.12) | no, 0.02 | no |
| lnots4_stable_plr10 | 1 | 0.052 | 0/4, -1.061+6.808i (0.63) | -0.196+8.468i (1.03), 0.66–0.92 | 0.00, 0.39 | -0.928+8.649i (1.06) | no, 0.06 | no |
| lnots4_stable_plr10 | 2 | 0.018 | 0/4, -0.637+4.533i (0.11) | -1.792+2.371i (0.46), 0.11–0.11 | 0.01, 0.06 | -1.089+1.987i (0.51) | yes, 0.27 | no |
| lnots4_stable_plr10 | 3 | 0.057 | 0/4, -0.752+8.091i (0.93) | -2.394+22.084i (4.23), 0.83–0.86 | 0.00, 0.22 | -0.997+8.785i (1.09) | no, 0.05 | no |
| lnots4_stable_plr10 | 4 | 0.030 | 0/4, -0.555+4.524i (0.12) | -1.106+1.884i (0.53), 0.66–0.69 | 0.00, 0.06 | -1.616+3.839i (0.18) | yes, 0.12 | no |
| lnots2_stable_plr10 | 0 | 0.026 | 0/2, -1.107+7.357i (0.76) | -448.299-325.245i (130.71), 0.44–0.47 | 0.00, 0.13 | -0.911+8.849i (1.11) | no, 0.02 | no |
| lnots2_stable_plr10 | 1 | 0.043 | 0/2, -1.689+3.300i (0.27) | -1.082+1.820i (0.55), 0.61–0.64 | 0.00, 0.09 | -1.656+3.296i (0.26) | yes, 0.01 | **yes** |
| lnots2_stable_plr10 | 2 | 0.031 | 0/2, -1.593+3.134i (0.29) | -1.027-0.000i (0.98), 0.76–0.76 | 0.00, 0.14 | -0.413+4.845i (0.20) | no, 0.43 | no |
| lnots2_stable_plr10 | 3 | 0.122 | 0/2, -1.092+13.132i (2.11) | -45.672+48.421i (14.81), 0.15–0.15 | 0.01, 0.01 | -0.458+41.490i (8.79) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 4 | 0.070 | 0/2, -1.011+8.960i (1.13) | -21.683+68.925i (16.00), 0.79–0.86 | 0.00, 0.13 | -1.028+8.966i (1.13) | no, 0.00 | no |

Trusted runs (rule v1): 4 of 40. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): lnots_stable_plr10 s1 0.08 (0.05), lnots_stable_plr10 s4 8.55 (9.30), lnots2_stable_plr10 s1 0.26 (0.29), lnots2_stable_plr10 s3 8.79 (9.55)

Rule v2 linearity gate: the data's best linear pair leaves 0.527 of the output (gate 0.1): **failed**, v2 reads no linear pole as physical on this set.
