# E5 tenth pass (rule v4, forced Lorenz ρ = 10): pre-registration (written 2026-11-18, before any ρ = 10 training)

**Why a rule v4.** In the ninth pass (Lorenz ρ = 5) rule v3 trusted one wrong pole (`lz_wide`, `base` s3, 96 % off).
That model's Jacobian and an unmoved stored pole agreed on a slow pole that is not the system's. The data gate cannot
catch this. The fit criterion did not catch it either: on Lorenz the output has a large mean, so even the mean
trajectory scores rel. L2 0.011–0.110.

**Rule v4** (`e5_v4.py`). It keeps v1's validity check, d < 0.1 and the v3 data gate unchanged, and changes the rest:

- **Probe agreement**: a = |μ_LTI − μ_J| / |μ_J| < 0.1. Here μ_LTI is the one-pair pole of the odd-part LTI probe at
  ε = 0.01, and μ_J is the Jacobian impulse-response pole. Both probe the same linearisation, so they must agree.
- **Centred fit**: e_c < 0.5. This is the test error relative to the test outputs minus the training-mean
  trajectory, per sample, averaged. It replaces rel. L2 < 0.5.

A run is correct if its J pole is within 10 % of the reference, as before.

**Retrospective check (seen data, not a test).** Applied to the 15 earlier sets (`results/e5_v4.md`):

- v4 trusts 44 runs, with 0 false trusts and 5 missed (`c0_medium`, the gate's known cost).
- v3 trusted 45, with 1 false and 5 missed.
- The 44 correct trusts have a ≤ 0.04 and e_c ≤ 0.18.
- The ninth-pass false trust has a ≈ 1900 and e_c = 0.30. So the agreement condition removes it, not the centred
  fit.
- The thresholds 0.1 and 0.5 were fixed by analogy with d and the old fit criterion, not tuned on these values.

**Tied-layer offset.** On `lz_r10_*` the tied layer's offset starts at the training-mean trajectory. In the ninth
pass, a zero-initialised offset stopped at about 1.5 of 3.27: the `common` lr schedule bounds how far any parameter
can move.

**System.** LNO Example 6: the same equations with ρ = 10, output x.

- The LNO data `1D_Lorenz_rho10` start from x(0) = 1 with inputs A sin 5t, A 0.05–10. A simulation reproduces the
  stored samples to 5.1e-4 … 2.7e-3.
- Equilibrium C+ = (4.899, 4.899, 9). The linearisation from f to x has poles **−0.5955 ± 6.1742i** (reference)
  and −12.48. It is less damped than ρ = 5 (ζ ≈ 0.10 vs 0.22).
- On the true impulse response the order-2 pencil gives −0.589 + 6.104i, 1.1 % from the reference.

**Sets.**

- `lz_r10_orig`: the LNO data.
- `lz_r10_small` (peak |f| 0.05–0.5) and `lz_r10_wide` (0.5–5): new, started at C+, E1 input families, data seed
  6. The bands are the ninth pass's bands, unchanged; gate values below are from the data only.
- Same 4 variants × 5 seeds per set, 60 runs. The LNO uses the ρ = 10 architecture (grid input, tanh, 84 modes);
  training follows the E1 `common` protocol, as before.
- The LTI probe adds two higher-frequency starts (−0.5 + 6i, −1 + 8i) on these sets. The gate fit adds one start
  (−0.6 + 6.2i). On the ρ = 5 sets the gate values are unchanged with it.

| set | all: pole, resid | low fifth: pole, resid | shift | v2 | v3 |
|---|---|---|---|---|---|
| `lz_r10_orig` | −0.190 + 4.753i, 0.991 | −0.128 + 5.813i, 0.562 | 0.183 | fail | fail |
| `lz_r10_small` | −0.709 + 6.097i, 0.045 | −0.708 + 6.101i, 0.039 | 0.001 | pass | pass |
| `lz_r10_wide` | −0.784 + 6.051i, 0.164 | −0.719 + 6.107i, 0.053 | 0.014 | fail | pass |

**Predictions.**

1. `lz_r10_small` and `lz_r10_wide`: v4 makes no false trust (every v4-trusted run within 10 % of the reference).
2. `lz_r10_orig`: the gate fails, so v4 (and v3) trust nothing.
3. With the offset started at the mean, the tied layers fit `lz_r10_small` (e_c < 0.5 in at least 3 of 5 seeds),
   and at least 3 of 5 are v4-trusted and correct.
4. Reported, not predicted:
   - v3 on the same runs (does a v3-only false trust recur?);
   - whether any model stores the fast pole −12.48;
   - whether the stable-init LNOs again have correct J poles but non-causal Jacobians.

If prediction 1 fails, v4 is not adopted as the RQ4 rule.
