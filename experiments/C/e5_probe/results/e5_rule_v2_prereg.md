# E5 rule v2: pre-registration (written 2026-11-17T17:28Z, before any medium / moderate training)

**Rule v2.** A stored pole is read as physical (small-signal) only if

1. **gate (data):** the best linear pair of the training data (FOH VarPro, all samples) leaves < 0.1 of the output;
2. **rule v1 (model):** J valid (non-causal and Toeplitz shares < 0.1) ∧ d(J pole, nearest stored pole) < 0.1 ∧
   test rel. L2 < 0.5.

The gate threshold 0.1 was proposed after the fourth pass (where it passes on `small`, 0.005, and fails on
`mixed`, 0.44), so it needs new data.

**New sets** (`e5_data.py`, data seed 1, E1 input families). The bands were picked from a data-only scan of the
gate residual (80 samples per band, no models): 0.1–0.5 → 0.094, 0.2–1 → 0.18, 0.5–2 → 0.29, 1–3 → 0.34.

- `medium`: peak |f| ~ U[0.1, 0.5]; expected gate residual ≈ 0.09 (passes, close to the threshold).
- `moderate`: peak |f| ~ U[0.2, 1]; expected ≈ 0.18 (fails).

**Predictions.**

- `medium`: if the gate passes, every run trusted by v2 has a J pole within 10 % of the small-signal pole
  −0.25 ± 0.9682i. No false trust.
- `moderate`: the gate fails and v2 trusts nothing. Rule v1 alone may still trust runs (expected: tied layers),
  and their poles are expected to be > 10 % from the small-signal pole (the scan's equivalent pole is 23 % off);
  that would show the gate is needed.
