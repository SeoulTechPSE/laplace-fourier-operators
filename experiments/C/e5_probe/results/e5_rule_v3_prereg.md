# E5 rule v3 (pole-shift gate): pre-registration (written 2026-11-17T18:36Z, before any wide / pend_wide training)

**Gates** (`e5_gate.py`), each combined with rule v1 (J valid ∧ d(J pole, stored pole) < 0.1 ∧ test < 0.5):

- v2: all training samples' best linear pair leaves < 0.1.
- v3: the lowest-amplitude fifth of the training samples is linear (residual < 0.1), and the equivalent pole of
  all samples is within 10 % of that band's (shift < 0.1).

**History of the threshold.** v3 was first run on the seven earlier sets with shift < 0.05. That gave no false
trust but left 11 correct runs untrusted (v2: 8). The threshold 0.1 equals the 10 % accuracy the rule claims and
leaves none untrusted on those sets, but it was set after seeing them. These two new sets test it.

**New sets** (`e5_data.py`, data seed 3, E1 input families), chosen from a data-only scan (100 samples, scan seed
9) where v2 and v3 disagree:

- `wide`: Duffing, peak |f| ~ U[0.05, 0.6]; scan: residual 0.117 (v2 fails), shift 0.082 (v3 passes); the
  all-sample equivalent pole is 9.7 % from the small-signal pole, so this is a near-boundary test.
- `pend_wide`: pendulum, peak |f| ~ U[0.05, 1]; scan: residual 0.204 (v2 fails), shift 0.057 (v3 passes).

**Predictions.**

1. On both sets v3 passes and v2 fails (data-side; to be confirmed on the actual sets).
2. v3: every trusted run has a J pole within 10 % of −0.25 ± 0.9682i (no false trust).
3. Hence v2 abstains on runs that are correct. If prediction 2 fails on `wide`, the shift gate is too loose
   for a hardening spring, and v2's caution was warranted.
