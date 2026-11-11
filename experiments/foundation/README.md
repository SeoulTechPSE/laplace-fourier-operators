# Shared foundation (months 1–4)

Plan: `docs/integrated_schedule_ACD.qmd` (shared foundation row; gate "reproduce the
original LNO results", month 4).

## S1: reproduce the original LNO results

| Item | Where |
|---|---|
| Original code (reference only, not imported) | `runs/ref/LNO`, github.com/qianyingcao/Laplace-Neural-Operator @ `78c64ef` |
| Original data | `/srv/data/lno_drive` (shared) or `runs/data/lno_drive` |
| Ported models | `lfo.models.lno` (`LNO1d`, `LNO2d`, `LNO2dV2`) |
| Original optimizer and loss | `lfo.training` (`ComplexAdam`, `relative_l2_loss`) |
| Port = original, bit for bit | `tests/test_lno.py` (needs `runs/ref/LNO` and CUDA) |
| Reproduction script | `reproduce_lno.py` |
| Per-run results, summary | `results/*.json`, `results/summary.md` |

```bash
python experiments/foundation/reproduce_lno.py --case duffing_c0 diffusion --model lno fno tfno --seeds 0 1 2
python experiments/foundation/reproduce_lno.py --summarize
```

Protocol: every model is trained with the original case settings (epochs, batch
size, learning rate, StepLR, `ComplexAdam` with weight decay 1e-4, relative-L2
loss). FNO/TFNO (`neuraloperator`, 4 layers) get the hidden width and mode count
whose parameter count is closest to the LNO's. Test error is the per-sample
relative L2 averaged over the test set (`lfo.metrics.relative_l2`), as in the
original scripts.

Notes on the original code:

- The original validation loop reshapes with the *training* batch size, so for
  Duffing (50 validation samples, batch 20) the last validation batch is
  reshaped incorrectly. This only affects the printed validation error; the
  script here computes validation error per sample.
- The original scripts set no random seed.
