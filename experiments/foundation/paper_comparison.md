# Reproduction vs. published values

Test relative L2 error. Ours: 3 seeds, mean ± std (`results/ood/summary.md`,
`results/ood/diag_duffing/`). Published values were read from the papers in
`~/references/` (not committed).

LNO paper versions: **NMI** = published version, Cao, Goswami & Karniadakis,
Nature Machine Intelligence 6, 631–640 (2024), doi:10.1038/s42256-024-00844-4
(Fig. 2, Table 3; 5 runs); **arXiv** = 2303.10528v2 (Fig. 3). Both report errors
only as plots; values below are read from the plots (NMI: log axis, about ±5%;
arXiv: linear axis, about ±0.01). The code archive matching the paper
(commit `c61c8fc`) has the same code and data as our reference `78c64ef`
(only README/note.txt differ).

## Duffing c = 0 (original out-of-distribution split: train A·sin 5t, test A·e^{-0.05t}·sin 5t)

| Source | Model | Params | Protocol | Test rel. L2 |
|---|---|---|---|---|
| **Ours** | LNO (port) | 1,821 | original, 1000 ep | **0.216 ± 0.068** |
| Ours | original code, 1 run | 1,821 | original | 0.315 |
| LNO paper, NMI (Fig. 2) | LNO | 797 as reported (1,821 incl. complex) | original | ≈ 0.24 (0.22–0.26) |
| LNO paper, arXiv (Fig. 3) | LNO | — | original | ≈ 0.24 ± 0.03 |
| SSM paper (arXiv 2409.03231v2, Table 3) | LNO (Small) | 1,309 | original | 0.260 ± 0.033 |
| SSM paper | LNO (Large: 16 ch, 8 modes) | 6,737 | 10k ep, MSE | 0.999 ± 0.099 |
| Ours | FNO, budget-matched | 1,804 | original | 0.895 ± 0.037 |
| Ours | FNO, 20k, no pos. emb., padding | 19,974 | original | 0.785 ± 0.070 |
| LNO paper, NMI (Fig. 2, Table 3) | FNO | 5,505 as reported | original | ≈ 0.52 |
| LNO paper, arXiv (Fig. 3) | FNO (4 layers, width 128, 1025 modes) | not given | original | ≈ 0.77 |
| SSM paper | FNO (3 layers, 32 ch, 8 modes) | 27,873 | 10k ep, MSE, linear LR decay | 0.590 ± 0.065 |
| SSM paper | Mamba | 19,233 | 10k ep | 0.445 ± 0.230 |

**Verdict:** the LNO port reproduces the published LNO error (0.216 vs ≈0.24 in
both versions and 0.260 in the SSM paper; within one standard deviation). FNO's
poor result is also published (≈0.52 NMI, ≈0.77 arXiv, 0.59 SSM paper): the gap is
a property of the out-of-distribution split. Our budget-matched FNO (0.79–0.90)
is worse than the NMI FNO (0.52).

## Diffusion (2D)

| Source | Model | Params | Test rel. L2 |
|---|---|---|---|
| **Ours** | LNO (port) | 14,929 | **1.33e-3 ± 1.5e-4** |
| Ours | original code, 1 run | 14,929 | 1.42e-3 |
| LNO paper, NMI (Fig. 2) | LNO | 2,641 as reported (14,929 incl. complex) | ≈ 1.2e-3 (1.0–1.4e-3) |
| LNO paper, arXiv (Fig. 3) | LNO | — | ≈ 0 on the plot (< 0.005) |
| Ours | FNO, budget-matched | 15,877 | 4.55e-3 ± 1.4e-3 |
| LNO paper, NMI (Fig. 2, Table 3) | FNO | 2,641 as reported | ≈ 7.5e-3 |
| LNO paper, NMI | CNO / U-Net | — | ≈ 6.9e-3 / ≈ 4.7e-3 |
| LNO paper, arXiv (Fig. 3) | FNO (4 layers, width 64, modes 41×13) | — | ≈ 0.12 |

**Verdict:** LNO reproduced (1.33e-3 vs ≈1.2e-3 NMI). Our FNO (4.5e-3) is close to
the NMI FNO (≈7.5e-3); the LNO-over-FNO factor is ≈3× for us and ≈6× in NMI.
The arXiv FNO value (≈0.12) was replaced in the published version, so the earlier
"25× better than the paper's FNO" remark applies to arXiv only.

## Parameter counts in the LNO paper

NMI Table 3 lists the LNO at 797 (Duffing) and 2,641 (Diffusion) parameters. These
equal the counts of the **real-valued** weights only (fc0, w0, fc1, fc2); the complex
pole and residue weights (512 and 6,144 complex entries) are left out. Counting a
complex entry as two real parameters (`lfo.training.count_params`) gives 1,821 and
14,929. The published "LNO is smaller than FNO" comparison therefore understates
the LNO size by 2.3× (Duffing) and 5.7× (Diffusion). Whether the FNO counts in
Table 3 include FNO's complex spectral weights cannot be told from the paper.

## Both splits (ours, 3 seeds, test rel. L2)

`ood` = original split; `iid` = same 380 samples re-split at random (`lfo.data.lno_bench`).

| case | split | LNO | FNO | TFNO |
|---|---|---|---|---|
| Duffing c=0 | ood | **0.216 ± 0.068** | 0.895 ± 0.037 | 1.007 ± 0.054 |
| Duffing c=0 | iid | 0.243 ± 0.038 | **0.181 ± 0.006** | 0.219 ± 0.015 |
| Diffusion | ood | **1.33e-3 ± 1.5e-4** | 4.55e-3 ± 1.4e-3 | 9.45e-3 ± 9.5e-3 |
| Diffusion | iid | 1.78e-3 ± 2.2e-4 | 7.02e-4 ± 1.9e-5 | **6.42e-4 ± 2.3e-4** |

LNO's error is nearly the same on both splits; FNO/TFNO improve by 5× (Duffing)
and 6–15× (Diffusion) when the test inputs come from the training distribution.
The LNO advantage reported in the original paper is an out-of-distribution
advantage; in-distribution, the budget-matched FNO/TFNO are better. This agrees
in direction with LFNO's in-distribution results.

## Not comparable

- **LFNO** (arXiv 2606.07601v1) does *not* use the original data. It regenerates
  ODE datasets with one forcing family (A·e^{-0.05t}·sin 5t, 380 samples, random
  split), so train and test are in-distribution. Its Duffing c=0 numbers
  (FNO 0.073, LNO 0.201, LFNO 0.052) are therefore not comparable to the table above.
  The LFNO "Heat" PDE (LNO 0.0227) is also its own dataset.
- **ALNO/PILNO** (arXiv 2602.12706v2) reports LNO only on its own PDE benchmarks
  (e.g. Darcy), not on Duffing or the LNO diffusion data.

## Caveats

- Published values for the LNO paper come from plots.
- The NMI Supplementary Information (architectures, data generation) has not been
  checked yet.
- Our FNO uses the original LNO training protocol (1000 epochs, relative-L2 loss);
  the SSM paper trained FNO for 10k epochs with MSE, which may explain 0.59 vs 0.79.
