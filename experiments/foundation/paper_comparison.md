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
| LNO paper, NMI (Fig. 2, Table 3) | FNO (1 layer, width 32, 16 modes) | 5,505 as reported (38,273 incl. complex) | original | ≈ 0.52 |
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
| LNO paper, NMI (Fig. 2, Table 3) | FNO (1 layer, width 16, 4×4 modes) | 2,641 as reported (19,025 incl. complex) | ≈ 7.5e-3 |
| LNO paper, NMI | CNO / U-Net | — | ≈ 6.9e-3 / ≈ 4.7e-3 |
| LNO paper, arXiv (Fig. 3) | FNO (4 layers, width 64, modes 41×13) | — | ≈ 0.12 |

**Verdict:** LNO reproduced (1.33e-3 vs ≈1.2e-3 NMI). Our FNO (4.5e-3) is close to
the NMI FNO (≈7.5e-3); the LNO-over-FNO factor is ≈3× for us and ≈6× in NMI.
The arXiv FNO value (≈0.12) was replaced in the published version, so the earlier
"25× better than the paper's FNO" remark applies to arXiv only.

## Parameter counts in the LNO paper

NMI Table 3 counts **real-valued weights only**, for both LNO and FNO; complex
weights (LNO poles/residues, FNO spectral weights) are left out. Verified by
recomputing the reported numbers from the architectures in SI Table S8
(the published FNO is a 1-layer classic FNO with sin activation, not the
4-layer FNO of arXiv Table A.1):

| case | model (SI Table S8) | reported | real-only (recomputed) | complex entries | total real dof |
|---|---|---|---|---|---|
| Duffing c=0 | LNO: 1 layer, width 4, 16 poles | 797 | 797 | 512 | **1,821** |
| Duffing c=0 | FNO: 1 layer, width 32, 16 modes, input f + grid | 5,505 | 5,505 | 16,384 | **38,273** |
| Diffusion | LNO: 1 layer, width 16, 4×4 poles | 2,641 | 2,641 | 6,144 | **14,929** |
| Diffusion | FNO: 1 layer, width 16, 4×4 modes | 2,641 | 2,641 | 8,192 | **19,025** |

So the published Duffing FNO is about 21× larger than the LNO, not 7×; on
Diffusion the FNO is 1.3× larger, not equal. We count a complex entry as two real
parameters (`lfo.training.count_params`) and match budgets on that count.

Our FNO/TFNO are 4-layer `neuraloperator` models (CLAUDE.md: baselines come from
`neuraloperator`), so they differ from the published 1-layer FNO in architecture
as well as size.

## Burgers (v2)

SI Table S2 (Burgers, low-to-high resolution: input u0(x) on 32 of 64 points,
output u(x, t) on 64 × 50; 5 runs): LNO 9.15e-2 ± 6.99e-4, **advanced LNO (v2)
5.31e-2 ± 1.79e-4**, FNO with IDFT 5.32e-2 ± 3.05e-4. SI Table S5: v2 uses
4 layers, width 16, modes 4×4, lr 0.01, batch 10, sin, 1000 epochs, which matches
the original `2D_Burger/main.py`.

Ours (`reproduce_lno.py --case burgers`, 3 seeds, test rel. L2, mean ± std):

| split | LNO v2 (120,514) | FNO (120,444) | TFNO (119,614) |
|---|---|---|---|
| ood | 5.39e-2 ± 4.5e-4 | 2.59e-2 ± 3.7e-4 | **1.88e-2 ± 6.5e-5** |
| iid | 5.31e-2 ± 1.0e-3 | 2.69e-2 ± 2.0e-4 | **1.78e-2 ± 8.2e-4** |

- **LNO v2 reproduced**: 5.39e-2 vs published 5.31e-2 (+1.5%, about 4 published
  standard deviations; within our seed spread on iid).
- Protocol as the original: `torch.optim.Adam` (not `ComplexAdam`), weight decay
  1e-4, relative-L2 loss, StepLR 100/0.5. Input points with both indices odd are
  zeroed as in the original; the models read every 2nd point in x and t only
  (u0 is constant along t), so this has no effect.
- FNO/TFNO (4-layer `neuraloperator`, budget-matched) read the same 32 × 25
  input and evaluate the last inverse DFT on the full 64 × 50 grid
  (`output_shape`), i.e. an "FNO with IDFT". They are 2.0× (FNO) and 2.9× (TFNO)
  better than LNO v2 and than the published FNO with IDFT (5.32e-2). The SI does not
  give the published FNO-with-IDFT architecture, so the gap cannot be attributed.
- The Burgers data has one initial-condition family (Gaussian random field) in
  all splits: the original split is already in-distribution (nearest-neighbour
  distance test→train 0.64 vs train→train 0.62). As expected, ood and iid agree.
- LNO v1 on Burgers (9.15e-2) is not reproduced: the original code has no v1
  Burgers script and the SI gives no v1 hyperparameters for it.
- Cost: LNO v2 ≈ 2.1 s/epoch alone on the RTX 4090 (≈ 35 min per run), FNO
  ≈ 0.6 s, TFNO ≈ 0.9 s; with 8 runs sharing the GPU, LNO took 2.8 h per run.

## Both splits (ours, 3 seeds, test rel. L2)

`ood` = original split; `iid` = same samples re-split at random (`lfo.data.lno_bench`).

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
- NMI Supplementary Information checked (Tables S2, S5, S8).
- Our FNO uses the original LNO training protocol (1000 epochs, relative-L2 loss);
  the SSM paper trained FNO for 10k epochs with MSE, which may explain 0.59 vs 0.79.

## LFNO: implementation details in the two versions

Sources: NeurIPS 2025 ML4PS workshop paper ("Best of Both Worlds", Appendix C,
Table 2) and arXiv:2606.07601 v1 (Appendix A, Tables 6–7). Neither gives the
initialisation or the exact layer arrangement.

| setting (Duffing / ODE) | workshop Table 2 | arXiv Table 6 | our sanity runs |
|---|---|---|---|
| layers | 6 | 6 | 4 pole-residue + 2 R (= 6) |
| width / modes | 4 / 16 | 4 / 16 | 4 / 16 |
| batch | 64 | 64 | 20 (LNO protocol) / 64 |
| lr | 0.002 | 0.0025 | 0.002 / paper value |
| weight decay | 0.01 | 0.02 | 1e-4 / paper value |
| StepLR step, gamma | 100, 0.8 | 100, 0.85 | 100, 0.5 / paper value |
| activation | relu | relu | relu |
| optimizer | Adam | Adam | ComplexAdam / torch Adam |
| epochs | 6,200 (Duffing c=0) | 5,000, early stopping | 300 / 1,000 |

PDE (Beam) in the workshop: width 16, 8 modes, batch 200, other settings as Duffing.
arXiv PDEs: width 16, 4 modes, batch 50, lr 0.002, wd 1e-4, gamma 0.8. In the arXiv
version FNO and LNO are trained with the same settings as LFNO (so LNO uses wd 0.02,
not the original 1e-4).

Observations:

- The 6 layers are consistent with our reconstruction (4 transient + 2 steady), and the
  arXiv parameter count (Table 7: 3,417, ODE setting) is matched exactly.
- Order inside the steady layer: the workshop text applies R first and then computes
  the residues lambda_l; the arXiv text and Fig. 1(b) compute lambda_l first and then
  apply R (Eq. 5). With channel matrices the two differ. We follow the arXiv version.
- The workshop version calls R "non-linear" and mentions "a learnable nonlinearity" in
  the steady component; Eqs. (5)-(8) are linear in lambda. We apply the activation
  between layers only.
- Duffing c=0 numbers differ between versions: LFNO 0.1288 (workshop) vs 0.0518 (arXiv);
  FNO 0.9998 (workshop) vs 0.0726 (arXiv). The workshop used the original LNO data
  (FNO fails on the ood split, as in our runs: 0.895); the arXiv version regenerated
  aligned data (f = A e^{-0.05t} sin 5t in all splits). LNO is 0.2014 in both tables,
  although the data changed.
- Paper protocol with our model (Duffing c=0, iid, 1,000 epochs, 1 seed, test rel. L2):

  | protocol | `pole_init="lno"` | `pole_init="stable"` |
  |---|---|---|
  | workshop (lr 0.002, wd 0.01, gamma 0.8) | 1.000 (no training) | 0.185 |
  | arXiv (lr 0.0025, wd 0.02, gamma 0.85) | 0.995 (no training) | 0.260 |

  The large weight decay does not rescue the original LNO initialisation, so the
  authors' code must differ from our reconstruction in initialisation, normalisation or
  layer arrangement. With the stable init, about half of the learned poles end with
  Re mu > 0 (all start with Re mu < 0): coupled L2 decay pulls poles toward the
  origin and nothing keeps them stable. Relevant to the (C) audit.
