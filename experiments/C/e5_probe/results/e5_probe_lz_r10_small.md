# E5: linear-probe poles of Duffing models, `lz_r10_small` data set

Forced Lorenz ρ = 10 (z' = xy − 8/3 z − f, output x), `e5_data.py` set `lz_r10_small` (diverse E1 waveforms, started at the equilibrium C+), Lorenz ρ = 10 LNO architecture (grid input, tanh, 84 modes), 5 seeds per variant; the tied layer has a learned output offset started at the training mean. Reference: the dominant pole of the linearization at C+, **−0.5955 ± 6.1742i** (third pole −12.48). Equivalent poles from centered data.
Errors in brackets: relative distance to it.

- **layer**: stored poles within 10 % / all, and the closest.
- **LTI probe**: odd part (G(εu) − G(−εu)) / 2ε on 16 smooth random inputs, one complex pair fitted (first-order hold); pole at ε = 0.01, and the relative fit residual over all ε = 0.01, 0.1, 0.3, 1, 3 (how well a causal LTI model describes G).
- **Jacobian** J = ∂G/∂x at x = 0: share of ‖J‖ above the diagonal (non-causal), residual of a shift-invariant fit of its causal part, and the order-2 matrix-pencil pole of its average impulse response (mean subdiagonals h_1 … h_799).

Variants: `base` original LNO (original init, pole lr × 1; first pass); `lno_plr10` pole lr × 10; `lno_stable_plr10` E1 stable init + pole lr × 10; `tied4_stable_plr10` E1 tied single layer, 4 slots (linear), stable init + pole lr × 10 (`e5_train.py`); `tied4o_stable_plr10` the same with no weight decay on the output offset (tenth-pass addendum); `lnotb_stable_plr10` the LNO with its pole-residue layer in the textbook wiring (eleventh pass).; `lnots_stable_plr10` the same with poles shared by all channel pairs (twelfth pass).

Equivalent linear pole of the training data (best linear pair, per-sample relative weights) by amplitude band: all (200): -0.709+6.097i, residual 0.05.

RQ4 indicator: d = distance from the J impulse-response pole to the nearest stored pole (relative); valid = J causal and shift-invariant (both shares < 0.1). Trusted (rule fixed before the `small` and `mixed` runs): valid ∧ d < 0.1 ∧ test rel. L2 < 0.5. Rule v2 (fixed before the `medium` and `moderate` runs) adds a gate on the data: its best linear pair must leave < 0.1.

| variant | seed | test rel. L2 | layer: within 10 %, closest | LTI probe: pole, residual | J: non-causal, Toeplitz residual | J impulse-response pole | valid, d | trusted |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.003 | 0/1344, 0.005+0.083i (0.99) | -47.665-0.044i (7.65), 0.97–0.97 | 0.53, 0.27 | -0.003+0.015i (1.00) | no, 0.10 | no |
| base | 1 | 0.003 | 0/1344, 0.004+0.053i (0.99) | -70.885+0.092i (11.37), 0.96–0.96 | 0.51, 0.26 | 0.009+0.032i (0.99) | no, 0.03 | no |
| base | 2 | 0.003 | 0/1344, 0.028+0.072i (0.99) | 0.104+0.000i (1.00), 0.40–0.41 | 0.30, 0.12 | 0.031+0.065i (0.99) | no, 0.06 | no |
| base | 3 | 0.003 | 0/1344, 0.015+0.061i (0.99) | -137.214-0.000i (22.05), 0.48–0.50 | 0.11, 0.04 | 0.018+0.024i (1.00) | no, 0.02 | no |
| base | 4 | 0.003 | 0/1344, 0.041+0.061i (0.99) | -112.643-0.186i (18.09), 0.93–0.93 | 0.52, 0.28 | 0.013+0.000i (1.00) | no, 0.18 | no |
| lno_plr10 | 0 | 0.003 | 0/1344, -0.019+0.418i (0.93) | -0.058-0.304i (1.05), 0.72–0.75 | 0.54, 0.28 | -0.123+0.232i (0.96) | no, 0.37 | no |
| lno_plr10 | 1 | 0.002 | 0/1344, 0.012+0.448i (0.93) | -61.113+0.000i (9.81), 0.57–0.58 | 0.08, 0.44 | -0.174+0.321i (0.95) | no, 0.35 | no |
| lno_plr10 | 2 | 0.003 | 0/1344, 0.254+0.423i (0.94) | -0.024-0.338i (1.05), 0.77–0.78 | 0.58, 0.37 | -0.089+0.229i (0.96) | no, 0.20 | no |
| lno_plr10 | 3 | 0.002 | 0/1344, -0.034+0.353i (0.94) | -0.025+0.326i (0.95), 0.82–0.82 | 0.45, 0.43 | -0.138+0.283i (0.95) | no, 0.11 | no |
| lno_plr10 | 4 | 0.003 | 0/1344, -0.035+0.352i (0.94) | -21.739+0.000i (3.55), 0.74–0.75 | 0.26, 0.39 | -0.075+0.167i (0.97) | no, 0.25 | no |
| lno_stable_plr10 | 0 | 0.002 | 13/1344, -0.117+6.100i (0.08) | -3.783-1.695i (1.37), 0.66–0.68 | 0.32, 0.14 | -0.191+5.921i (0.08) | no, 0.01 | no |
| lno_stable_plr10 | 1 | 0.002 | 24/1344, -0.153+6.207i (0.07) | -1.701+1.953i (0.70), 0.73–0.83 | 0.43, 0.17 | -0.168+5.880i (0.08) | no, 0.01 | no |
| lno_stable_plr10 | 2 | 0.002 | 21/1344, -0.122+6.081i (0.08) | -3.357+2.692i (0.72), 0.64–0.64 | 0.32, 0.14 | -0.332+5.680i (0.09) | no, 0.05 | no |
| lno_stable_plr10 | 3 | 0.002 | 11/1344, -0.065+6.197i (0.09) | -1.861+3.095i (0.54), 0.44–0.45 | 0.25, 0.11 | -1.258+3.752i (0.40) | no, 0.28 | no |
| lno_stable_plr10 | 4 | 0.002 | 13/1344, -0.102+6.124i (0.08) | -3.053+3.163i (0.63), 0.62–0.63 | 0.28, 0.12 | -0.635+5.563i (0.10) | no, 0.10 | no |
| tied4_stable_plr10 | 0 | 0.653 | 1/4, -0.260+6.585i (0.09) | -0.252+6.559i (0.08), 0.66–0.66 | 0.00, 0.01 | -0.019+0.000i (1.00) | yes, 19.47 | no |
| tied4_stable_plr10 | 1 | 0.652 | 0/4, -0.508+2.462i (0.60) | -0.070+0.236i (0.96), 0.65–0.65 | 0.00, 0.01 | -0.051+0.000i (1.00) | yes, 6.21 | no |
| tied4_stable_plr10 | 2 | 0.652 | 0/4, -0.463+4.143i (0.33) | -0.450+4.102i (0.33), 0.65–0.65 | 0.00, 0.01 | -0.043+0.000i (1.00) | yes, 7.53 | no |
| tied4_stable_plr10 | 3 | 0.651 | 1/4, -0.384+5.626i (0.09) | -0.355+5.609i (0.10), 0.68–0.68 | 0.00, 0.01 | -0.018+0.000i (1.00) | yes, 20.42 | no |
| tied4_stable_plr10 | 4 | 0.652 | 0/4, -0.147+0.653i (0.89) | -0.361+0.316i (0.95), 0.29–0.29 | 0.00, 0.01 | -0.121+0.000i (1.00) | yes, 2.75 | no |
| tied4o_stable_plr10 | 0 | 0.000 | 2/4, -0.701+6.068i (0.02) | -0.675+6.046i (0.02), 0.02–0.02 | 0.01, 0.01 | -0.662+6.042i (0.02) | yes, 0.01 | **yes** |
| tied4o_stable_plr10 | 1 | 0.000 | 1/4, -0.601+6.074i (0.02) | -0.627+6.061i (0.02), 0.04–0.04 | 0.01, 0.01 | -0.615+6.055i (0.02) | yes, 0.00 | **yes** |
| tied4o_stable_plr10 | 2 | 0.000 | 1/4, -0.532+6.074i (0.02) | -0.591+6.046i (0.02), 0.07–0.07 | 0.00, 0.01 | -0.568+6.037i (0.02) | yes, 0.01 | **yes** |
| tied4o_stable_plr10 | 3 | 0.000 | 1/4, -0.377+6.194i (0.04) | -0.512+6.073i (0.02), 0.23–0.23 | 0.00, 0.01 | -0.435+6.058i (0.03) | yes, 0.02 | **yes** |
| tied4o_stable_plr10 | 4 | 0.002 | 0/4, -0.974+1.103i (0.82) | -1.808+1.753i (0.74), 0.06–0.06 | 0.03, 0.01 | -1.694+1.490i (0.78) | yes, 0.36 | no |
| lnotb_stable_plr10 | 0 | 0.001 | 8/1344, -0.029+6.259i (0.09) | -1.184+4.925i (0.22), 0.26–0.27 | 0.01, 0.45 | -0.891+4.630i (0.25) | no, 0.18 | no |
| lnotb_stable_plr10 | 1 | 0.001 | 19/1344, -0.079+6.315i (0.09) | -1.441+4.914i (0.24), 0.39–0.39 | 0.01, 0.25 | -1.063+5.069i (0.19) | no, 0.19 | no |
| lnotb_stable_plr10 | 2 | 0.001 | 20/1344, -0.075+6.189i (0.08) | -0.928+5.562i (0.11), 0.27–0.27 | 0.00, 0.23 | -0.772+5.599i (0.10) | no, 0.13 | no |
| lnotb_stable_plr10 | 3 | 0.002 | 11/1344, -0.063+6.250i (0.09) | -1.512+3.921i (0.39), 0.28–0.28 | 0.00, 0.12 | -1.360+4.356i (0.32) | no, 0.29 | no |
| lnotb_stable_plr10 | 4 | 0.005 | 13/1344, -0.067+6.147i (0.09) | -2.471+4.454i (0.41), 0.88–0.88 | 0.01, 0.91 | -1.666+4.671i (0.30) | no, 0.33 | no |
| lnots_stable_plr10 | 0 | 0.002 | 0/84, -0.048+6.540i (0.11) | -1.941+4.299i (0.37), 0.27–0.27 | 0.01, 0.38 | -1.443+3.413i (0.47) | no, 0.35 | no |
| lnots_stable_plr10 | 1 | 0.002 | 0/84, 0.011+5.941i (0.10) | -1.415+1.525i (0.76), 0.53–0.53 | 0.00, 0.29 | -1.038+3.353i (0.46) | no, 0.33 | no |
| lnots_stable_plr10 | 2 | 0.000 | 4/84, -0.063+6.228i (0.09) | -0.665+6.008i (0.03), 0.10–0.10 | 0.01, 0.16 | -0.606+5.963i (0.03) | no, 0.08 | no |
| lnots_stable_plr10 | 3 | 0.002 | 0/84, -0.100+5.707i (0.11) | -2.965+3.502i (0.58), 0.34–0.34 | 0.00, 0.10 | -1.369+4.046i (0.37) | no, 0.25 | no |
| lnots_stable_plr10 | 4 | 0.001 | 1/84, -0.114+5.918i (0.09) | -0.844+5.763i (0.08), 0.19–0.20 | 0.02, 0.45 | -0.745+5.523i (0.11) | no, 0.11 | no |
| lnots4_stable_plr10 | 0 | 0.000 | 1/4, -0.368+5.921i (0.05) | -0.712+5.921i (0.04), 0.21–0.21 | 0.01, 0.02 | -0.564+5.776i (0.06) | yes, 0.04 | **yes** |
| lnots4_stable_plr10 | 1 | 0.000 | 1/4, -0.597+6.060i (0.02) | -0.619+6.042i (0.02), 0.04–0.04 | 0.01, 0.04 | -0.611+6.051i (0.02) | yes, 0.00 | **yes** |
| lnots4_stable_plr10 | 2 | 0.001 | 0/4, -0.857+4.422i (0.29) | -1.117+4.982i (0.21), 0.19–0.19 | 0.01, 0.02 | -1.047+4.672i (0.25) | yes, 0.07 | **yes** |
| lnots4_stable_plr10 | 3 | 0.001 | 1/4, -0.389+5.821i (0.07) | -0.738+5.944i (0.04), 0.23–0.23 | 0.00, 0.01 | -0.558+5.818i (0.06) | yes, 0.03 | **yes** |
| lnots4_stable_plr10 | 4 | 0.002 | 0/4, -0.946+2.153i (0.65) | -1.607+2.347i (0.64), 0.10–0.10 | 0.02, 0.18 | -1.413+1.973i (0.69) | no, 0.21 | no |
| lnots2_stable_plr10 | 0 | 0.001 | 0/2, -0.342+5.293i (0.15) | -0.919+5.738i (0.09), 0.15–0.15 | 0.00, 0.01 | -0.740+5.588i (0.10) | yes, 0.09 | **yes** |
| lnots2_stable_plr10 | 1 | 0.000 | 1/2, -0.630+6.109i (0.01) | -0.633+6.063i (0.02), 0.05–0.05 | 0.01, 0.01 | -0.631+6.109i (0.01) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 2 | 0.002 | 0/2, -0.252+8.837i (0.43) | -5.332+0.000i (1.25), 0.20–0.20 | 0.00, 0.02 | -0.238+8.549i (0.39) | yes, 0.03 | **yes** |
| lnots2_stable_plr10 | 3 | 0.002 | 0/2, -0.090+13.201i (1.14) | -595.913+596.091i (135.11), 0.02–0.02 | 0.00, 0.00 | -0.092+13.202i (1.14) | yes, 0.00 | **yes** |
| lnots2_stable_plr10 | 4 | 0.000 | 1/2, -0.634+6.082i (0.02) | -0.637+6.041i (0.02), 0.05–0.05 | 0.01, 0.09 | -0.634+6.083i (0.02) | yes, 0.00 | **yes** |

Trusted runs (rule v1): 13 of 45. Their J impulse-response pole vs the reference (and vs the data's equivalent pole): tied4o_stable_plr10 s0 0.02 (0.01), tied4o_stable_plr10 s1 0.02 (0.02), tied4o_stable_plr10 s2 0.02 (0.03), tied4o_stable_plr10 s3 0.03 (0.05), lnots4_stable_plr10 s0 0.06 (0.06), lnots4_stable_plr10 s1 0.02 (0.02), lnots4_stable_plr10 s2 0.25 (0.24), lnots4_stable_plr10 s3 0.06 (0.05), lnots2_stable_plr10 s0 0.10 (0.08), lnots2_stable_plr10 s1 0.01 (0.01), lnots2_stable_plr10 s2 0.39 (0.41), lnots2_stable_plr10 s3 1.14 (1.16), lnots2_stable_plr10 s4 0.02 (0.01)

Rule v2 linearity gate: the data's best linear pair leaves 0.045 of the output (gate 0.1): **passed**, v2 trusts the same runs as v1.
