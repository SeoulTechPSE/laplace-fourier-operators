# E5 eleventh pass: a causal LNO layer (written 2026-11-18; part B before any training of the new variant)

## Part A (diagnostic, done before this file's part B; `e5_causal.py`, `results/e5_causal.md`)

Predictions were committed with the script (4bd17c1) before it ran.

- **Predicted: the textbook wiring is causal.** For every trained LNO, the same weights in the textbook wiring
  (`PoleResidueTS1d`: each transient coefficient on its own pole) give a non-causal share below 0.05. Held in 268
  of 270 runs (0.005–0.025). The other 2 runs overflow (NaN), from strongly unstable poles.
- **Predicted: the layer as coded is non-causal.** Its share was above 0.1 in all 268 finite runs (0.12–0.82).
  The original code attaches each transient coefficient to another channel's pole, so the periodic wrap-around of
  the steady term is not canceled.
- **Predicted: the full model's share tracks the layer's.** Failed (Spearman 0.00). The pointwise path `w0` is
  causal, so the full share depends on how much a model uses the pole layer. Original-init models with unmoved
  poles show 0.002–0.05; stable-init models with pole lr × 10 show 0.11–0.32.

## Part B: train the LNO with its pole-residue layer in the textbook wiring

**Variant** `lnotb_stable_plr10`. It is the original LNO for each benchmark, with `conv0` replaced by
`PoleResidueTS1d` (same shapes, same residue initialization distribution), stable pole initialization and pole
lr × 10. The `common` protocol is unchanged. 5 seeds × all 18 sets, 90 runs; probed with `e5_probe.py --only`.
Rule v4 is unchanged. The comparison is `lno_stable_plr10`, the same variant in the original wiring.

**Predictions.**

1. The full-model Jacobian is causal (non-causal share < 0.1) in at least 80 % of runs.
2. It is valid (causal and Toeplitz residual < 0.1) in at least 60 % of runs on the 15 simulated sets.
3. The fit is comparable: on at least 15 of 18 sets, the median test error is within 2× of `lno_stable_plr10`'s.
4. v4 makes no false trust among these runs, and trusts at least 20 of them. `lno_stable_plr10`: 7 of 90.
5. Reported: stored poles within 10 % of the small-signal pole; Toeplitz residuals; probe agreement.
