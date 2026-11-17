# E5 eighth pass (undamped Duffing, c = 0): pre-registration (written 2026-11-18, before any c0 training)

**System.** x'' + x + x³ = f, zero initial state. The LNO data `1D_Duffing_c0` follow it (c = 0, k3 = 1 reproduce
stored samples to 6.6e-4 at peak |f| = 0.05 and 1.2e-1 at 10; same input amplitudes, 0.05–10, as `1D_Duffing_c05`).
Small-signal poles **± i**, on the imaginary axis: no decay, so a phase error accumulates over the 20.47 s window and
the response resonates near ω = 1. This is the boundary case for the trust rule. Correct = J pole within 10 % of i.

**Rule.** Unchanged from the seventh pass (v3; `e5_gate.py`, `LOW_FRAC, LOW_RESID, SHIFT = 0.2, 0.1, 0.1`) on top of
v1 (J valid ∧ d(J pole, stored pole) < 0.1 ∧ test < 0.5). v2 is reported alongside.

**Sets** (`e5_data.py`, data seed 4, E1 input families, 200 / 50 / 130), bands chosen from a data-only scan (200
samples, same seed, before any training). Gate values on the actual sets:

| set | all: pole, resid | low fifth: pole, resid | shift | v2 | v3 |
|---|---|---|---|---|---|
| `c0_orig` (LNO data, peak |f| 0.05–10) | −0.229 + 1.260i, 0.884 | −0.004 + 1.022i, 0.208 | 0.321 | fail | fail |
| `c0_small` (0.02–0.1) | −0.002 + 1.010i, 0.078 | −0.000 + 1.002i, 0.025 | 0.008 | pass | pass |
| `c0_medium` (0.05–0.3) | −0.032 + 1.049i, 0.356 | −0.006 + 1.014i, 0.154 | 0.043 | fail | fail |

Same 4 variants × 5 seeds per set (`e5_train.py --data c0_*`), 60 runs.

**Predictions.**

1. `c0_small`: v3 passes; every v1-trusted run has its J pole within 10 % of i (no false trust).
2. `c0_orig`: v3 abstains. If v1 trusts runs there, their poles are the data's equivalent pole (≈ 26 % from i),
   i.e. false as small-signal statements, and the gate is what prevents the error.
3. `c0_medium`: v3 abstains on a set whose equivalent pole is within 5 % of i. If v1 trusts runs here and they are
   correct, this is a cost of the gate on undamped data: the low band is not linear by the residual criterion
   (0.154) even though its pole is right, because an undamped resonance turns a small frequency error into a large
   output residual. This would be recorded as a missed set, not used to change the threshold.
4. Undamped poles may also be harder to learn: models may end with Re μ > 0 or fit poorly; reported, not
   predicted.
