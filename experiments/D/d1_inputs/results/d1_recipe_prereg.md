# D1 third pass: Fourier vs FOH time axis with the E2 recipe, fine-grid truth (written 2026-11-18, before the runs)

**Why.** In the second pass (`d1_train.py`) the published protocol's training error swamped the representation
error. The E2 recipe reaches the representation floor:

- variant `damp8_ms`: 8 prior-free slots with a damping bound, multi-start;
- `e2_polish.py`: L-BFGS polish;
- `e2_rescue.py`: truth-free single-slot restarts.

With it, (C) E4 found the FOH layer far better than the band (Fourier) layer on E1 data. On `p6_e1_T20` vs
`p6_e1_T20_foh` (20 runs each, after restarts) the test error was 6.7e-4 vs 4.7e-6, and −2 was recovered in 70 %
vs 80 % of runs. But those data were generated with a first-order hold, which is exactly the FOH layer's input
model.

**Conditions** (`e2_train.py`). `p6_e1fine_T20` (band / Fourier layer) and `p6_e1fine_T20_foh` (FOH layer). The
E1 families have inputs and truth on a 32× finer grid, sampled to n = 1024 (`e2_richness.make`, family `e1fine`).
Seeds 10–17 (8 runs each), E2 recipe as above.

**Oracle floors** (true poles in each layer, test split; computed before writing this): band 3.7e-3, FOH 3.4e-3.
On the sample truth they were about 1e-3 and 2e-14. With the fine truth, both paths are limited by the sampling of
steps and PRBS (D1 first pass).

**Predictions.**

1. The FOH layer's median test error after restarts is within 20 % of the band layer's (on the sample truth: 140×
   lower).
2. Recovery of p6's −2 is at most 1 run in 8 apart between the two layers, and below the sample-truth rates
   (70 % / 80 %) for both.
3. The complex pairs are recovered in all runs by both layers, as on the sample truth.

If 1 and 2 hold, the E4 advantage of the FOH layer on E1 data is a model-class match with the data's hold. The D1
time-axis choice must then be argued from the oracle on smooth inputs and ramps, and from causality and cost, not
from pole recovery on sampled data.
