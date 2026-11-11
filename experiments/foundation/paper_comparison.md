# Reproduction vs. published values

Test relative L2 error. Ours: 3 seeds, mean ± std (`results/summary.md`,
`results/diag_duffing/`). Published values were read from the papers in
`~/references/` (not committed).

## Duffing c = 0 (original out-of-distribution split: train A·sin 5t, test A·e^{-0.05t}·sin 5t)

| Source | Model | Params | Protocol | Test rel. L2 |
|---|---|---|---|---|
| **Ours** | LNO (port) | 1,821 | original, 1000 ep | **0.216 ± 0.068** |
| Ours | original code, 1 run | 1,821 | original | 0.315 |
| LNO paper (arXiv 2303.10528v2, Fig. 3, 5 runs) | LNO | — | original | ≈ 0.24 ± 0.03 (read from plot) |
| SSM paper (arXiv 2409.03231v2, Table 3) | LNO (Small) | 1,309 | original | 0.260 ± 0.033 |
| SSM paper | LNO (Large: 16 ch, 8 modes) | 6,737 | 10k ep, MSE | 0.999 ± 0.099 |
| Ours | FNO, budget-matched | 1,804 | original | 0.895 ± 0.037 |
| Ours | FNO, 20k, no pos. emb., padding | 19,974 | original | 0.785 ± 0.070 |
| LNO paper (Fig. 3) | FNO (4 layers, width 128, 1025 modes) | not given | original | ≈ 0.77 (read from plot) |
| SSM paper | FNO (3 layers, 32 ch, 8 modes) | 27,873 | 10k ep, MSE, linear LR decay | 0.590 ± 0.065 |
| SSM paper | Mamba | 19,233 | 10k ep | 0.445 ± 0.230 |

**Verdict:** the LNO port reproduces the published LNO error (0.216 vs ≈0.24 and
0.260; within one standard deviation). FNO's poor result is also published (≈0.77,
0.59): the gap is a property of the out-of-distribution split.

## Diffusion (2D)

| Source | Model | Test rel. L2 |
|---|---|---|
| **Ours** | LNO (port) | **1.33e-3 ± 1.5e-4** |
| Ours | original code, 1 run | 1.42e-3 |
| LNO paper (Fig. 3) | LNO | ≈ 0 on the plot (< 0.005; text: "two orders of magnitude smaller than FNO") |
| Ours | FNO, budget-matched (15.9k) | 4.55e-3 ± 1.4e-3 |
| LNO paper (Fig. 3) | FNO (4 layers, width 64, modes 41×13) | ≈ 0.12 (read from plot) |

**Verdict:** LNO reproduced. Our budget-matched FNO is about 25× better than the
paper's FNO. The LNO-over-FNO factor is about 3× in our setup, not 100×.

## Not comparable

- **LFNO** (arXiv 2606.07601v1) does *not* use the original data. It regenerates
  ODE datasets with one forcing family (A·e^{-0.05t}·sin 5t, 380 samples, random
  split), so train and test are in-distribution. Its Duffing c=0 numbers
  (FNO 0.073, LNO 0.201, LFNO 0.052) are therefore not comparable to the table above.
  The LFNO "Heat" PDE (LNO 0.0227) is also its own dataset.
- **ALNO/PILNO** (arXiv 2602.12706v2) reports LNO only on its own PDE benchmarks
  (e.g. Darcy), not on Duffing or the LNO diffusion data.

## Caveats

- Published values for the LNO paper come from a plot (±0.01).
- The published version of the LNO paper (and its SI) has not been checked yet.
- Our FNO uses the original LNO training protocol (1000 epochs, relative-L2 loss);
  the SSM paper trained FNO for 10k epochs with MSE, which may explain 0.59 vs 0.79.
