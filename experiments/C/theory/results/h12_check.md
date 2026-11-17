# H1 / H2: numerical checks (`h12_check.py`)

Single-channel pole-residue layer in float64, true parameters of the E1 systems (minimal form), E1 time grid (T = 20, n = 1024). See `H1_H2.md` for the statements.

## Lemma 1: tied layer = zero-initial-state response of the trigonometric interpolant

Relative difference between the tied layer and an independent first-order-hold solve (`lfo.data.lti.response`) of the trigonometric interpolant on a 32× finer grid, 4 E1 inputs: p4 1.6e-05, p6 4.9e-06 (the residual is the hold's O(h²) error).

## Theorem 2: exact recovery from a sparse grid-line input

Tied-layer output for a multisine on DFT lines S, matrix pencil of order |S| + N_tot; nodes on the unit circle (the input lines) discarded, the rest are the poles.

| system | lines | order | poles found / N_tot | max relative pole error | singular-value gap at the order |
|---|---|---|---|---|---|
| p4 | [3, 11] | 8 | 4 / 4 | 1.4e-13 | 2.5e+12 |
| p6 | [3, 11] | 10 | 6 / 6 | 3.5e-12 | 2.7e+10 |
| p6 | [7] | 8 | 6 / 6 | 4.9e-13 | 1.8e+10 |

## Proposition 3: local identifiability of the tied layer from one input

Jacobian of the output for one input, columns scaled to unit norm; smallest singular value and condition number. Full rank = locally identifiable.

| system | parameters | input | smallest singular value | condition number |
|---|---|---|---|---|
| p4 | 8 | grid cosine (line 5) | 1.6e-02 | 1.3e+02 |
| p4 | 8 | E1 input 0 | 8.6e-02 | 2.0e+01 |
| p4 | 8 | E1 input 1 | 2.9e-02 | 7.6e+01 |
| p4 | 8 | white noise | 7.9e-02 | 2.2e+01 |
| p6 | 12 | grid cosine (line 5) | 4.7e-04 | 5.3e+03 |
| p6 | 12 | E1 input 0 | 1.6e-03 | 1.3e+03 |
| p6 | 12 | E1 input 1 | 7.7e-04 | 3.5e+03 |
| p6 | 12 | white noise | 2.1e-03 | 9.9e+02 |

## Theorem 4 / Proposition 5: a free steady branch

Information on each Re μ (1 / (F⁻¹)_kk, other parameters profiled; per-sample weights 1 / ‖y‖², as the E1 loss). Tied: the tied layer. Free R: transient branch tied to (μ, β), steady branch a free symbol R_l on every DFT line, profiled out. Ratio = free R / tied.

### p4

| inputs | -0.4+3i: tied, ratio | -1: tied, ratio | -0.2: tied, ratio |
|---|---|---|---|
| 1 E1 input | 2.11e-01, 7.5e-34 | 4.91e-03, 3.5e-32 | 1.89e-01, 1.6e-32 |
| 5 circular shifts of it | 1.10e+00, 1.2e-01 | 3.11e-02, 9.2e-02 | 1.11e+00, 2.3e-01 |
| 2 E1 inputs | 2.79e-01, 4.1e-02 | 1.38e-02, 5.0e-02 | 8.72e-01, 7.4e-02 |
| 1 white-noise input | 2.50e-01, 7.3e-33 | 8.04e-03, 2.7e-32 | 3.28e-01, 7.2e-32 |
| 2 white-noise inputs | 8.40e-01, 2.6e-02 | 1.10e-02, 3.2e-02 | 3.78e-01, 8.9e-02 |
| E1 training set (200) | 1.24e+01, 5.2e-01 | 7.86e-01, 4.8e-01 | 8.07e+01, 7.2e-01 |

### p6

| inputs | -0.5+1.5i: tied, ratio | -0.3+4i: tied, ratio | -1: tied, ratio | -2: tied, ratio |
|---|---|---|---|---|
| 1 E1 input | 1.06e-02, 2.7e-31 | 8.99e-02, 4.1e-32 | 1.85e-05, 7.0e-30 | 7.42e-07, 1.8e-29 |
| 5 circular shifts of it | 6.51e-02, 2.1e-01 | 4.58e-01, 2.8e-01 | 1.36e-04, 1.6e-01 | 4.86e-06, 1.3e-01 |
| 2 E1 inputs | 1.31e-02, 7.6e-02 | 1.03e-01, 1.2e-01 | 5.06e-05, 8.5e-02 | 1.36e-06, 6.2e-02 |
| 1 white-noise input | 1.73e-02, 1.1e-32 | 2.46e-01, 4.4e-32 | 4.68e-05, 2.7e-30 | 2.31e-06, 1.2e-30 |
| 2 white-noise inputs | 3.45e-02, 4.5e-02 | 9.08e-01, 1.0e-01 | 6.23e-05, 2.9e-02 | 4.22e-06, 1.5e-02 |
| E1 training set (200) | 9.19e-01, 3.1e-01 | 9.02e+00, 2.8e-01 | 4.76e-03, 3.4e-01 | 1.31e-04, 2.3e-01 |

