# E5 twelfth pass: shared poles against under-damped, dephasing stored modes (written 2026-11-18, before training)

**Motivation** (eleventh-pass follow-up, `results/e5_cluster.md`). Some textbook-wired LNOs have a correct
small-signal (Jacobian) pole but no stored pole within 10 % of it: 12 of 47 such runs on the gated sets, ρ = 10
excluded. In those runs the stored modes that carry the oscillation decay at a median 36 % of the true rate, and
several of them at slightly different frequencies dephase. Across all 42 runs with a non-zero true decay rate
(`c0_*` excluded), the share-weighted damping ratio of the stored modes has median 0.71 (IQR 0.40–0.87). With
16 × 16 = 256 poles (one set per channel pair), a mode can be built from many near-copies.

**Variant** `lnots_stable_plr10`. It is the textbook-wired LNO with one set of poles shared by all channel pairs:
16 poles (ρ = 10: 84) instead of width² × modes. Residues stay per channel pair. Everything else matches
`lnotb_stable_plr10`: stable initialisation, pole lr × 10, `common` protocol. 5 seeds × 18 sets, 90 runs; probed
with `e5_probe.py --only`; rule v4 unchanged.

**Predictions** (baseline `lnotb_stable_plr10` in brackets).

1. v4 makes no false trust among these runs and trusts at least 45 (37).
2. Among runs with a correct J pole on the gated sets (ρ = 10 excluded), at most 10 % have d ≥ 0.1 (12 / 47 =
   26 %).
3. On `lz_small` + `lz_wide`, v4 trusts at least 3 of 10 (0 of 10).
4. The fit stays comparable: on at least 15 of 18 sets, the median test error is within 2× of
   `lnotb_stable_plr10`'s.
5. Reported: the share-weighted damping ratio of the stored modes (`e5_cluster.py`), against 0.71 (IQR
   0.40–0.87).
