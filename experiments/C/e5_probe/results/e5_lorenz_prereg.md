# E5 ninth pass (forced Lorenz, ρ = 5): pre-registration (written 2026-11-18, before any Lorenz training)

**System.** LNO paper Eq. 22: x′ = 10(y − x), y′ = x(5 − z) − y, z′ = xy − (8/3)z − f; output x. The LNO data
`1D_Lorenz_rho5` follow it from x(0) = 1, y(0) = z(0) = 0, with inputs A sin 5t (checked: a simulation reproduces
stored samples to 3.0e-4 … 7.8e-4). Unforced, the state settles to the stable equilibrium C+ = (3.266, 3.266, 4)
within about 5 s. The linearisation at C+ from f to x is third order: **−0.9287 ± 4.1476i** (reference) and −11.81.

**What changes for a nonzero operating point** (all fixed here, before training):

- Data side: the zero-input response is common to all samples, so the gates fit centred data (u − mean u to
  f − mean f, within each band). The fit model is still one complex pair.
- Tied layer: a learned output offset b(t) (`e5_train.Offset`). It does not depend on the input, so the probes
  (odd part, Jacobian at f = 0) see only the layer.
- Probes: unchanged. On the true impulse response the order-2 pencil on h_1 … h_799 gives −0.889 + 4.035i, 2.8 %
  from the reference, because of the fast pole. Correct = J pole within 10 % of the reference.
- The rule is v3, unchanged (`LOW_FRAC, LOW_RESID, SHIFT = 0.2, 0.1, 0.1`), on top of v1.

**Sets.**

- `lz_orig`: the LNO data.
- `lz_small` (peak |f| 0.05–0.5) and `lz_wide` (0.5–5): new, started at C+, E1 input families, data seed 5, bands
  chosen from a data-only scan.
- Same 4 variants × 5 seeds per set (`e5_train.py --data lz_*`), 60 runs. The LNO uses the Lorenz architecture (grid
  input, tanh, 16 modes); training is the E1 `common` protocol, as before.

Gate values on the actual sets:

| set | all: pole, resid | low fifth: pole, resid | shift | v2 | v3 |
|---|---|---|---|---|---|
| `lz_orig` | −0.661 + 3.883i, 0.527 | −1.484 + 4.722i, 0.179 | 0.238 | fail | fail |
| `lz_small` | −0.999 + 4.055i, 0.027 | −1.009 + 4.066i, 0.016 | 0.004 | pass | pass |
| `lz_wide` | −1.115 + 4.023i, 0.183 | −1.027 + 4.083i, 0.041 | 0.025 | fail | pass |

**Predictions.**

1. `lz_small`: every v1-trusted run has its J pole within 10 % of the reference.
2. `lz_wide`: v3 passes where v2 fails, and every v1-trusted run is correct. The all-sample equivalent pole is
   5 % from the reference, so trusted poles that track it are still correct.
3. `lz_orig`: v3 abstains. The operating point moves during the first ~5 s, so the Jacobian at f = 0 is a
   linearisation about a time-varying trajectory. Expect the Toeplitz residual to be large, so the probes are
   invalid and v1 trusts few or no runs. Any run v1 trusts there is checked against the reference.
4. Not predicted, reported: whether any model represents the fast pole −11.81, and how the offset affects the
   tied layer's fit.
