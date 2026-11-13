# (C) Identifiability theory: E2, data richness

Plan: `docs/proposal_C_identifiability.qmd`, RQ2 / H2 / E2. Builds on the E1 audit
(`../e1_audit/`, especially `identifiability.py` and `E1_report.md`).

Question: does richer data (more input frequencies, a longer window, other input
families) make the poles practically identifiable, and does training then recover them?

## Scripts

| script | what | output |
|---|---|---|
| `e2_richness.py` | phase 1: local identifiability of the tied layer at the truth, over 46 data configurations (p4, p6) | `results/e2_richness.md` / `.json` |
| `e2_grid.py` | phase 1b: the phase-1 analysis over the time step dt = 20/N, N = 256–8192 | `results/e2_grid.md` / `.json` |
| `e2_select.py` | best of 5 by validation loss on independent seeds 10–59 | `results/e2_select.md` / `.json` |
| `e2_rescue.py` | truth-free single-slot restarts of polished runs | `results/e2_rescue/*.json`, `results/e2_rescue.md` |
| `e2_floor.py` | floor certificate without the truth: estimated floor + consensus poles | `results/e2_floor.md` / `.json` |
| `e2_polish.py` | L-BFGS-B polish of saved Adam checkpoints (paired before/after), floor-certificate table | `results/e2_polish/*.json`, `results/e2_polish.md` |
| `e2_noise.py` | E4: pole error under output noise vs the Cramér–Rao bound of the minimal-form layer | `results/e2_noise.md` / `.json` |
| `e2_figures.py` | figures: recovery by recipe stage; noise vs CRB | `results/figures/fig_e2_stages.png`, `fig_e4_noise.png` |
| `e3_deep.py` | E3 deep first pass: bounded slots in full ALNO / LFNO stacks | `results/e3_deep/*.json`, `results/e3_deep.md` |
| `e2_train.py` | phase 2: train the E1 tied layer on selected configurations, 10 seeds; `--variant` controls (warm, no spare slot, real-constrained slots) | `results/e2_train/*.json`, `results/e2_train.md` |

```
python experiments/C/theory/e2_richness.py                  # < 1 min (CPU)
python experiments/C/theory/e2_train.py --cond <name> --seeds 0 1 2 3 4   # ≈ 105 s per run on the 4090
python experiments/C/theory/e2_train.py --summarize
```
Phase 2 was run as 16 parallel processes (8 conditions × 2 seed halves), ≈ 9 min.

## Phase 1: local identifiability vs data

Setup: the float64 tied layer in minimal form from `e1_audit/identifiability.py`
(one slot per conjugate pair, **real poles constrained real**), at the true parameters;
dt = 20/1024 fixed, T changes n_t; 200 samples, data seed 0. Per pole: ratio =
profiled error of a 10 % change in Re μ (others re-fitted, linearised) / representation
floor; ratio ≥ 10 = practically identifiable. The Jacobian is computed in forward mode
(few parameters, many outputs); this needed `.conj()` → `torch.conj_physical` in
`identifiability.py` (same values; the view form fails under forward AD). The E1
numbers are reproduced (p6, E1 data: ratio 0.6 / 0.19 for −1 / −2).

Results (`results/e2_richness.md`):

- **p4: richer data make all poles identifiable.** E1 data (T = 20): 7.0 / 4.6 / 9.5
  (pair, −1, −0.2). Multisine with K ≥ 6 frequencies: 11–24 / 10–15 / 18–26;
  E1 data at T = 40: 18 / 11 / 30; multisine16 at T = 80: 46 / 30 / 65. PRBS alone:
  19.5 / 9.7 / 13. Ramps alone: nothing (≤ 1.5). Short windows (T ≤ 10): real poles
  not identifiable.
- **p6: the fast real poles −1, −2 stay non-identifiable for every data change tried**
  (max ratio 1.8 / 0.6, multisine16 at T = 80). The complex pairs become identifiable
  with rich data (±4i from K ≥ 6; both at multisine16, T ≥ 40).
- The floor (5e-4 to 2e-3) is the layer's DFT discretisation error and does not fall
  with richer data; for p6 it, not the data, is what hides −1 and −2. Grid resolution
  (dt, number of modes) is the untested variable.

## Phase 2: training on the phase-1 configurations

Setup: E1 `Layer1("tied")`, n_poles = 4 (p4) / 6 (p6) complex slots, `band` init with
its T = 20 ranges for every T (same initial poles in rad/s), `common` + pole lr × 10,
1000 epochs. Train = the 200 analysed samples (seed 0), validation 50 (seed 1), test
100 (seed 2), same family. 8 conditions × 10 seeds = 80 runs, none diverged.

Results (`results/e2_train.md`; ratio / recovered fraction / pole error):

| condition | test rel. L2 | pair −0.4+3i | −1 | −0.2 |
|---|---|---|---|---|
| p4, E1 data, T = 20 | 4.5e-2 ± 2.3e-2 | 7.0 / 70 % / 0.09 | 4.6 / 0 % / 0.84 | 9.5 / 0 % / 0.97 |
| p4, ramps | 1.1e-2 ± 3.3e-3 | 0.3 / 0 % / 0.28 | 0.4 / 0 % / 0.97 | 1.5 / 0 % / 1.26 |
| p4, PRBS | 5.6e-3 ± 3.9e-3 | 19.5 / 100 % / 0.00 | 9.7 / 0 % / 0.62 | 13.0 / 0 % / 0.64 |
| p4, multisine16, T = 20 | 4.1e-3 ± 1.6e-3 | 24.1 / 100 % / 0.00 | 13.9 / 0 % / 0.61 | 23.8 / 0 % / 0.60 |
| p4, E1 data, T = 40 | 3.1e-2 ± 2.2e-2 | 18.2 / 70 % / 0.07 | 11.3 / 0 % / 0.76 | 30.0 / 0 % / 0.67 |
| p4, multisine16, T = 40 | 7.4e-3 ± 8.1e-3 | 34.7 / 100 % / 0.01 | 21.9 / 0 % / 0.58 | 44.7 / 0 % / 0.48 |

p6: E1 data: pairs 60 % / 100 %, −1 / −2 0 %; multisine16 at T = 80: pairs 80 % / 100 %,
−1 10 %, −2 0 %; test error 9.4e-3 → 1.9e-3.

Findings:

1. **Where the data identify a complex pair, training finds it**: p4's pair goes from
   70 % to 100 % recovered (error 0.09 → 0.00) with PRBS or multisine; test error falls
   up to 10×.
2. **Predicted-identifiable real poles are still not recovered (0 % in every p4
   condition)**, although their error falls (0.84 / 0.97 → 0.58 / 0.48). The learned slots
   that stand in for −1 and −0.2 stay complex: e.g. −0.82 + 0.61i and −0.23 + 0.13i
   (multisine16, T = 20/40). Most of the pole error is the spurious Im part. Test error
   stays 4–10× above the oracle, so training ends in a local minimum, not at the truth.
3. The mismatch is in the parameterisation: phase 1 analyses the minimal form in which a
   real pole has no Im parameter; the trained layer gives every slot a free Im μ (and p4
   has one spare slot, 4 slots for 3 folded poles). A real decay can be approximated by a
   slowly oscillating pair, and at Im = 0 the loss is flat in Im to first order (E1),
   so nothing pulls the slot onto the real axis.
4. p6 behaves as phase 1 predicts: pairs found, −1 / −2 not.

So for p4, data richness is necessary but not sufficient: it removes the data
limitation, and what remains is an optimisation / parameterisation limitation of
complex slots for real poles. This qualifies E1's part (iii) ("the data cannot identify
them") as stated for p6 only.

## Phase 2 controls: optimisation or parameterisation?

`e2_train.py --variant` (100 runs, 10 seeds each, none diverged): `warm` (E1 warm init),
`free_min` (one complex slot per folded pole, no spare), `real_spare` (as base, two slots
constrained real), `real_min` (the minimal form of phase 1: one complex + real-constrained
slots). Real constraint = parametrisation of `pr.mu` with Im × 0/1 mask (pole lr × 10 still
applied). Recovered fraction / pole error per true pole:

| data | variant | slots | test rel. L2 | pair | −1 | −0.2 |
|---|---|---|---|---|---|---|
| p4 multisine16 | base | 4 | 4.1e-3 ± 1.6e-3 | 100 % / 0.00 | 0 % / 0.61 | 0 % / 0.60 |
| p4 multisine16 | **warm** | 4 | **5.7e-4 ± 7e-5** (oracle 5.9e-4) | 100 % / 0.00 | **100 % / 0.01** | **100 % / 0.01** |
| p4 multisine16 | free_min | 3 | 2.0e-2 ± 1.9e-2 | 90 % / 0.02 | 0 % / 1.26 | 0 % / 0.69 |
| p4 multisine16 | real_spare | 4 | 1.2e-2 ± 2.2e-2 | 100 % / 0.01 | 20 % / 0.18 | 20 % / 0.25 |
| p4 multisine16 | real_min | 3 | 2.9e-1 ± 9.4e-2 | 10 % / 1.09 | 10 % / 1.30 | 50 % / 0.18 |
| p4 E1 data | warm | 4 | 2.4e-3 ± 2.4e-3 | 100 % / 0.01 | 80 % / 0.05 | 80 % / 0.04 |
| p6 multisine16 T = 80 | warm | 6 | 4.3e-4 ± 2.4e-4 (oracle 4.0e-4) | 100 % / 100 % | 100 % / 0.02 | −2: 70 % / 0.08 |
| p6 multisine16 T = 80 | real_min | 4 | 4.1e-2 ± 9.0e-2 | 90 % / 90 % | 30 % / 0.17 | −2: 0 % / 0.53 |

(p4 E1-data rows for free_min / real_spare / real_min: all worse than base; `results/e2_train.md`.)

Findings:

1. **The truth is a stable, reachable-from-nearby minimum where phase 1 says so.**
   Warm start on multisine16: all p4 poles recovered in 10/10 seeds, test error at the
   oracle. On E1 data (ratios < 10) the warm start drifts away in 2/10 seeds. p6 even
   recovers −1 (100 %) and −2 (70 %) from a warm start although their ratios are < 2:
   locally stable but flat.
2. **The failure from the band init is global: slots get the wrong assignment.** Typical
   minima: the slow real pole −0.2 is taken by a complex slot (−0.25 + 0.05i) and the
   remaining real slots coalesce (−1.15, −1.13: two slots on one decay). In `real_min`
   the only complex slot goes to −0.2 in every seed and the ±3i pair is left unrepresented
   (test 0.3); the inits differ, the wrong minimum is the same.
3. **Real constraint helps, removing the spare slot hurts.** `real_spare` lowers the
   real-pole error 0.6 → 0.2 (20 % recovered; one seed reaches the oracle);
   the minimal forms (`free_min`, `real_min`) are worse than base: the spare slot is
   what lets training escape wrong assignments.

Revised E2 message: data richness makes the poles locally identifiable (phase 1, confirmed
by the warm start), but from a generic init the tied layer's landscape has wrong-assignment
and pole-coalescence minima, so recovery is an optimisation problem, not a data problem,
for p4. Candidate remedies for E3/E4: inits on the real axis plus complex slots, spare
slots with pruning, multi-start.

## Remedies (first round): mixed init with real-constrained slots, multi-start

`e2_train.py --variant` (160 runs, 10 seeds each, none diverged):
`wide` (pairs + 1 + 2 × real slots, all band init, all free: slot-count control, same
parameter count as `mixed`), `mixed` (same slots; 2 × real slots started on a stratified
log grid of Re in [−5, −0.05] and **constrained real**, the rest band init), `mixed_off`
(grid shifted by half a log step: 0.11 / 0.50 / 2.3 / 10.7, a control because the
unshifted grid 0.05 / 0.23 / 1.08 / 5 lies near p4's −0.2 and −1), `*_ms` (multi-start:
8 inits × 100 epochs, best on validation continues with the full 1000-epoch protocol;
≈ 1.8 × the compute).

Why constrained: a slot started on the real axis but left free leaves it within a few
epochs, because d/d(Im μ) = −t Im(β) e^{Re μ t} ≠ 0 for a complex residue and Adam turns
any small gradient into a step of ≈ lr. (E1's "zero first-order sensitivity to Im μ"
holds only for a real residue.)

Recovered fraction / pole error (multisine16 data):

| system | variant | slots | test rel. L2 | −1 | −0.2 or −2 |
|---|---|---|---|---|---|
| p4 | base | 4 | 4.1e-3 ± 1.6e-3 | 0 % / 0.61 | −0.2: 0 % / 0.60 |
| p4 | wide | 6 | 3.1e-3 ± 1.3e-3 | 0 % / 0.68 | 0 % / 0.76 |
| p4 | base_ms | 4 | 3.8e-3 ± 1.9e-3 | 0 % / 0.56 | 0 % / 0.57 |
| p4 | mixed | 6 | 4.2e-3 ± 3.5e-3 | 40 % / 0.21 | 0 % / 0.24 |
| p4 | mixed_ms | 6 | 1.1e-3 ± 9.1e-4 | 60 % / 0.11 | 80 % / 0.06 |
| p4 | mixed_off | 6 | 3.3e-3 ± 1.8e-3 | 40 % / 0.18 | 20 % / 0.23 |
| p4 | **mixed_off_ms** | 6 | **1.4e-3 ± 1.2e-3** (oracle 5.9e-4) | **60 % / 0.11** | **70 % / 0.09** |
| p6 T = 80 | base | 6 | 1.9e-3 ± 1.0e-3 | 10 % / 0.31 | −2: 0 % / 0.79 |
| p6 T = 80 | wide | 7 | 1.8e-3 ± 1.1e-3 | 0 % / 0.41 | 0 % / 0.79 |
| p6 T = 80 | base_ms | 6 | 6.2e-4 ± 1.6e-4 | 30 % / 0.24 | 0 % / 0.68 |
| p6 T = 80 | mixed | 7 | 2.6e-3 ± 2.6e-3 | 40 % / 0.15 | 40 % / 0.18 |
| p6 T = 80 | mixed_off | 7 | 2.9e-3 ± 2.7e-3 | 60 % / 0.12 | 60 % / 0.12 |
| p6 T = 80 | **mixed_off_ms** | 7 | **5.6e-4 ± 1.9e-4** (oracle 4.0e-4) | **50 % / 0.11** | **40 % / 0.14** |

Complex pairs: 90–100 % in all these rows. E1 data (p4): no remedy helps (real poles
≤ 20 %, `mixed` loses the pair); the data limit applies there.

Findings:

1. **Real-constrained, stratified real slots are the active ingredient.** More slots
   alone (`wide`, same parameter count as `mixed`) or multi-start alone (`base_ms`) recover
   no real pole. `mixed` gives 40–60 %; with multi-start p4 reaches 60 / 70 % and test error
   within 2–3 × of the oracle.
2. **Not an artefact of the grid.** With the grid shifted away from the true poles
   (`mixed_off`) the results are the same within seed noise. In p6 the slot that ends at
   −2 travels there from −4 to −6.
3. **p6's −2 is recovered in 40–60 % of runs from a generic init**, although its
   phase-1 ratio is 0.6: phase 1 (ratio of a 10 % change) is conservative; with the
   real constraint the remaining minima are shallow but the truth is still the best one.
4. Remaining failure: real slots coalescing (e.g. three slots at −1.17) instead of one
   moving to −0.2 or −2. Candidates: repulsion / minimum-separation penalty, prune-and-split.

## Remedies (second round): against coalescing real slots

Variants on top of `mixed_off` (shifted grid), with and without multi-start; 80 runs,
10 seeds, none diverged. `*_rep`: repulsion between real slots in log|Re μ|,
λ Σ exp(−Δ²/2σ²), σ = 0.15, λ = 1e-2 annealed linearly to 0 at epoch 600. `*_split`: at
epochs 150 and 300, real slots within 10 % are merged (residues added) and the freed slot
is moved, residue 0, to the grid point farthest from the other real slots (0–3 splits per
run). `e1_audit.train` got optional `penalty` / `hook` arguments for this (unused by E1).

| system | variant | test rel. L2 | −1 | −0.2 or −2 | slots coalesced at end (pairs, 10 seeds) |
|---|---|---|---|---|---|
| p4 | mixed_off_ms | 1.4e-3 ± 1.2e-3 | 60 % / 0.11 | 70 % / 0.09 | 11 |
| p4 | mixed_off_rep | 5.6e-2 ± 1.0e-1 | 60 % / 0.12 | 0 % / 0.42 | – |
| p4 | mixed_off_rep_ms | 2.0e-3 ± 1.3e-3 | 60 % / 0.11 | 40 % / 0.17 | 0 |
| p4 | mixed_off_split | 2.8e-3 ± 1.8e-3 | 40 % / 0.14 | 10 % / 0.33 | – |
| p4 | mixed_off_split_ms | 9.6e-4 ± 4.6e-4 | 40 % / 0.16 | 70 % / 0.15 | 2 |
| p6 | mixed_off_ms | 5.6e-4 ± 1.9e-4 | 50 % / 0.11 | 40 % / 0.14 | 9 |
| p6 | mixed_off_rep | 2.6e-3 ± 2.5e-3 | 70 % / 0.09 | 50 % / 0.10 | – |
| p6 | mixed_off_rep_ms | 8.3e-4 ± 3.1e-4 | 60 % / 0.09 | 50 % / 0.13 | 2 |
| p6 | mixed_off_split | 2.6e-3 ± 2.4e-3 | 20 % / 0.20 | 70 % / 0.12 | – |
| p6 | mixed_off_split_ms | 5.8e-4 ± 2.2e-4 | 30 % / 0.14 | 40 % / 0.15 | 2 |

Findings:

1. **Both remedies remove the coalescence** (11 / 9 coalesced pairs over 10 seeds →
   0–2), **but recovery does not improve**: all rows are within seed noise of
   `mixed_off_ms` (binomial s.e. ≈ 15 % at 10 seeds). Two p4 `rep` seeds (2, 5) end
   in a poor minimum (test 0.24 / 0.28, the ±3i pair lost; no divergence), hence the
   large mean; the other eight reach 2.7e-3 to 7.9e-3.
2. **The failure moves from coalescence to bracketing.** Failed runs now have the real
   slots spread out, with two slots on either side of a true pole, e.g. p4:
   −0.05 / −0.31 around −0.2 and −1.1 / −1.58 around −1; p6: −0.72 / −2.06 around −1 / −2.
   With 2 × as many real slots as real poles, the spare slots do not die out but share
   the decay with a neighbour. Test error stays near the oracle (9.6e-4 / 5.8e-4 with
   split + multi-start): these are good-fit, wrong-pole solutions.
3. Hence the redundant real slots are the remaining problem: a remedy needs to remove
   them (residue sparsity / pruning) or use fewer of them, not to separate them.

## Remedies (third round): removing redundant slots; what limits recovery

Variants (120 runs, 10 seeds, none diverged): `mixed_off_l1(_ms)` / `mixed_off_l1s_ms`
(L1 on each slot's impulse-response norm |β| ‖e^{μt}‖, λ = 1e-3 / 1e-2 to epoch 200, 0 at
300; slots with < 2 % of the output pruned at epoch 300, re-fit to 1000), `mixed_off_prune_ms`
(pruning without the penalty), `few(_ms)` (n_real + 1 real slots, grid with a random
log offset per seed). Pole metrics use the remaining slots.

Recovered (−1 / −0.2 or −2), test rel. L2:

| variant | p4 | p4 test | p6 | p6 test | pruned per run |
|---|---|---|---|---|---|
| mixed_off_ms (reference) | 60 / 70 % | 1.4e-3 | 50 / 40 % | 5.6e-4 | – |
| mixed_off_l1 | 40 / 10 % | 3.4e-3 | 50 / 70 % | 2.8e-3 | 0–1 |
| mixed_off_l1_ms | 40 / 70 % | 1.1e-3 | 60 / 40 % | 5.5e-4 | 0–2 |
| mixed_off_l1s_ms | 50 / 90 % | 9.7e-4 | 30 / 10 % | 1.0e-3 | 0–2 |
| mixed_off_prune_ms | 60 / 70 % | 1.4e-3 | 50 / 40 % | 5.6e-4 | 0–1 |
| few | 40 / 30 % | 3.5e-3 | 30 / 0 % | 3.6e-2 | – |
| few_ms | 60 / 40 % | 1.6e-3 | 50 / 20 % | 7.2e-4 | – |

No variant is better than the reference beyond seed noise; pruning rarely fires
(spare slots keep > 2 % of the output even under L1); fewer real slots hurt p6's −2.

**What limits recovery: fit quality of the failed runs** (all `mixed_off*_ms` runs pooled):

| system | runs with all poles recovered | runs missing a real pole |
|---|---|---|
| p4 | 24 runs, test median 6.0e-4 = 1.0 × oracle | 36 runs, 1.5e-3 = 2.6 × oracle |
| p6 | 8 runs, 5.8e-4 = 1.5 × oracle | 52 runs, 5.9e-4 = 1.5 × oracle |

- **p4: the wrong-pole solutions fit worse**, so they are local minima that the loss
  can tell apart. Selecting the lowest validation loss among k independent runs
  (bootstrap from the 10 seeds, 2,000 draws): `mixed_off_ms` all poles recovered in
  50 % (k = 1), 84 % (k = 3), 95 % (k = 5); `mixed_off_l1s_ms` 50 / 88 / 98 %;
  `mixed_off` (no multi-start) 24 / 45 / 61 %.
- **p6: the wrong-pole solutions fit as well or better** on these 10 seeds; validation
  selection made recovery worse (10 → 0 %). *Not confirmed by independent runs (next
  section): this was a 10-seed artefact.*

(The E2 conclusion drawn here from 10 seeds is revised in the next section.)

## Best of 5 by validation loss: independent check

`e2_select.py` → `results/e2_select.md` / `.json`. `mixed_off_ms`, new seeds 10–59 (multi-start
candidates also new), 10 disjoint groups of 5 consecutive seeds per system; per group the
lowest final validation loss is selected. 100 runs, none diverged.

| system | single run: all poles recovered | best of 5: all recovered | per real pole, single → selected | Spearman(val loss, max pole error) |
|---|---|---|---|---|
| p4 | 38 % (19/50) | **80 % (8/10)** | −1 44 → 80 %, −0.2 58 → 100 % | +0.84 (seeds 0–9: +0.79) |
| p6 | 32 % (16/50) | **70 % (7/10)** | −1 70 → 100 %, −2 32 → 70 % | +0.24 (seeds 0–9: +0.14) |

Complex pairs: 100 % single and selected in both systems.

- **p4: confirmed, but lower than the bootstrap estimate (95 %).** In every group with at
  least one good run (8/10) the selection picked a good one. The two misses are groups
  with no good run; their selected runs have −1 errors of 0.136 / 0.111 at oracle-level
  test error (5.6–5.7e-4), with two real slots bracketing −1 (−1.14 / −0.79).
- **p6: the 10-seed conclusion does not hold.** On seeds 0–9 1/10 runs recovered all
  poles and selection picked a wrong one; on seeds 10–59 16/50 did and selection
  recovers 7/10 groups. Validation loss carries less information about the poles than in
  p4 (Spearman 0.24 vs 0.84): misses (−2 errors 0.12–0.41) reach test errors of
  3.3–4.5e-4, overlapping the good runs (3.1–4.2e-4).
- With 10 groups the rates have wide intervals (8/10: about 44–97 %; 7/10: 35–93 %,
  Clopper–Pearson 95 %).

**E2 conclusion (revised).** Rich data make all poles locally identifiable for p4, and
the truth is a stable minimum (warm start). From generic inits, recovery is limited by
optimisation: mixed real-constrained init + multi-start + best of 5 by validation loss
recovers all p4 poles in 8/10 groups and all p6 poles in 7/10, from 0 % with the E1 recipe.
p6's −1, −2 remain the harder case: phase-1 ratios < 2, weak link between loss and pole
error, misses near the floor. Their identifiability is weak but not absent: we did not
show they cannot be recovered.

## Without the true number of real poles: fixed split and learned gate

The recipe above sizes the slots from the truth (pairs + 1 complex, 2 × real constrained).
Two prior-free variants, both with 8 slots for p4 and p6 (`e2_train.py`): `fixed8` (4 band
complex + 4 real-constrained on a randomly offset log grid) and `gate8` (no constraint;
Im μ = sigmoid(a) Im μ̃ with a learned gate logit per slot, a = −5 on the grid slots, +3 on
the band slots, pole lr). 100 runs (`_ms`: 20 seeds = 4 groups of 5; `gate8`: 10 seeds).
Reference `mixed_off_ms` on seeds 10–29 (same group structure).

| system | variant | test (median) | −1 | −0.2 or −2 | all poles | best of 5 | Spearman(val, error) |
|---|---|---|---|---|---|---|---|
| p4 | mixed_off_ms (truth-sized) | 1.3e-3 | 25 % | 65 % | 25 % | 2/4 | +0.87 |
| p4 | fixed8_ms | 1.5e-3 | 40 % | 25 % | 5 % | 1/4 | +0.13 |
| p4 | gate8_ms | 9.4e-4 | 0 % | 0 % | 0 % | 0/4 | +0.08 |
| p4 | gate8 | 1.4e-3 | 10 % | 0 % | 0 % | – | +0.05 |
| p6 | mixed_off_ms (truth-sized) | 5.6e-4 | 75 % | 30 % | 30 % | 3/4 | +0.40 |
| p6 | fixed8_ms | 8.4e-4 | 45 % | 15 % | 15 % | 1/4 | +0.31 |
| p6 | gate8_ms | 1.1e-3 | 10 % | 10 % | 0 % | 0/4 | +0.30 |
| p6 | gate8 | 3.2e-3 | 0 % | 0 % | 0 % | – | +0.60 |

Complex pairs 80–100 % everywhere.

- **The learned gate does not work.** Gates end partly open (0.06–0.2, 0.7–1.5 of 4
  closed per run) while Im μ̃ grows: only the product g · Im μ̃ matters, so the gate does
  not decide real vs complex, and the slots end like the base model (−0.24 + 0.11i for −0.2).
- **A fixed 4 + 4 split is worse than the truth-sized one.** The three spare complex slots
  take the slow real decays at low frequency (−0.42 + 0.17i, −0.36 + 0.18i): the
  wrong-assignment failure again. Validation loss also stops telling right from wrong
  poles in p4 (Spearman 0.13), so best-of-5 selection no longer helps.
- So the recipe's success depends on the slot sizing prior. A prior-free candidate that
  targets the mechanism: bound complex slots away from the real axis (|Im μ| ≥ ω_min,
  e.g. one period in the window, 2π/T), so slow decays can only be taken by real slots.

## Phase 1b: grid resolution

`e2_grid.py` → `results/e2_grid.md` / `.json`: the phase-1 analysis with dt = 20/N,
N = 256 … 8192, at fixed T (outputs exact, so only the layer's discretisation changes).

- The floor falls ∝ dt (first-order discretisation error): p6 multisine16 T = 80: 5.5e-3
  (N = 256) → 4.2e-4 (1024) → 3.1e-5 (8192). The profiled error of a 10 % pole change
  is set by the data and nearly constant, so ratios grow ∝ 1/dt.
- p6 −1 / −2 ratios, N = 1024 → 4096 → 8192: E1 data T = 20: 0.6 / 0.2 → 3.0 / 1.0 → 6.3 / 2.0;
  multisine16 T = 20: 1.0 / 0.4 → 4.7 / 1.7 → 9.6 / 3.4; multisine16 T = 80: 1.8 / 0.6 →
  **11.6** / 4.0 → **24.3** / 8.3. −1 becomes practically identifiable with rich data on
  a fine grid; −2 stays just below the threshold even at N = 8192, T = 80.
- p4: ratios double per halving of dt (E1 data: all ≥ 10 from N = 2048 except −1 at 9.5).
- Interpretation: "practically identifiable" is relative to the error level the model
  cannot remove (here the discretisation floor; with real data also noise). With exact
  data and an exact model every pole is identifiable; the ratio asks whether a 10 % pole
  change is visible above that level. Not tested: training on the fine grids
  (N = 4096 at T = 80 is n_t = 16,384, ≈ 4 × the run time).

## Prior-free sizing, second try: complex slots bounded away from the real axis

`bound8` (`e2_train.py`): as `fixed8` (8 slots: 4 real-constrained on a random-offset log
grid, 4 complex), with the complex slots parameterised as |Im μ| = 2π/T + |Im μ̃| (at
least one period in the window; 0.31 rad/s at T = 20, 0.079 at T = 80). 60 runs (`_ms` 20
seeds = 4 groups, plain 10). References on the same group structure.

| system | variant | test (median) | −1 | −0.2 or −2 | all poles | best of 5 | Spearman(val, error) |
|---|---|---|---|---|---|---|---|
| p4 | mixed_off_ms (truth-sized, seeds 10–29) | 1.3e-3 | 25 % | 65 % | 5/20 | 2/4 | +0.87 |
| p4 | fixed8_ms | 1.5e-3 | 40 % | 25 % | 1/20 | 1/4 | +0.13 |
| p4 | **bound8_ms** | 2.2e-3 | 30 % | 60 % | 4/20 | **3/4** | +0.69 |
| p4 | bound8 | 6.8e-3 | 40 % | 10 % | 0/10 | – | +0.84 |
| p6 | mixed_off_ms (truth-sized, seeds 10–29) | 5.6e-4 | 75 % | 30 % | 6/20 | 3/4 | +0.40 |
| p6 | fixed8_ms | 8.4e-4 | 45 % | 15 % | 3/20 | 1/4 | +0.31 |
| p6 | bound8_ms | 6.9e-4 | 50 % | 10 % | 2/20 | 0/4 | +0.10 |
| p6 | bound8 | 2.5e-3 | 80 % | 0 % | 0/10 | – | +0.32 |

- **p4: the bound restores the truth-sized result without the prior** (4/20 vs 5/20 single,
  3/4 vs 2/4 best of 5) and brings back the link between validation loss and pole error
  (Spearman 0.69 vs 0.13). Complex slots still sit at the bound with strong damping
  (−0.75 + 0.31i) in some runs.
- **p6: no gain.** The fast decay −2 is taken by strongly damped complex slots
  (−1.10 + 1.93i, −2.21 + 1.38i, |Re| ≈ |Im|), while the real slots coalesce near −0.9.
  A bound on |Im| alone does not stop a heavily damped oscillation from mimicking a fast
  decay; the bound is also 4× lower at T = 80.
- Candidate next: bound the damping of complex slots as well (|Re μ| ≤ c |Im μ|, a
  minimum quality factor), so that fast decays can only be real.

## Training on a finer grid (p6)

`p6_ms16_T80_N4096` (multisine16, T = 80, dt = 20/4096, n_t = 16,384), `mixed_off_ms`,
seeds 10–29 (20 runs, ≈ 6 min each), compared with the same seeds at dt = 20/1024.

| dt | −1 | −2 | all poles | best of 5 | test (median) | oracle (floor) |
|---|---|---|---|---|---|---|
| 20/1024 | 75 % | 30 % | 6/20 | 3/4 | 5.6e-4 | 4.0e-4 |
| 20/4096 | 85 % | 35 % | 7/20 | 4/4 | 4.2e-4 | 5.6e-5 |

- Direction as phase 1b predicts (−1 ratio 11.6: slightly better; −2 ratio 4.0: unchanged),
  but the differences are within seed noise.
- **Training does not reach the lower floor**: test error stays at 4.2e-4, 7.5 × the
  fine-grid oracle and near the coarse floor. The finer grid makes −1 distinguishable in
  principle; the training protocol (1000 epochs, StepLR) does not converge to that
  precision, so the gain is not realised.

## Prior-free sizing, third try: damping-bounded complex slots

`damp8` (`e2_train.py`): as `bound8`, and the complex slots are lightly damped,
Re μ = −c |Im μ| σ(a) (a learned), so |Re μ| < c |Im μ|; c = 0.5 (damping ratio < 0.45),
`damp8w`: c = 1. The true complex poles have |Re|/|Im| ≤ 0.33; the p6 absorbers seen with
`bound8` had 0.57 and 1.6. Still no use of the true pole count. `_ms`, seeds 10–29
(4 groups), 80 runs. Adam only (before the polish below):

| system | variant | all poles | best of 5 | −1 | −0.2 or −2 |
|---|---|---|---|---|---|
| p4 | mixed_off_ms (truth-sized, seeds 10–59) | 38 % | 8/10 | 44 % | 58 % |
| p4 | bound8_ms (seeds 0–19) | 20 % | 3/4 | 30 % | 60 % |
| p4 | **damp8_ms** | **65 %** | **4/4** | 70 % | 75 % |
| p4 | damp8w_ms | 35 % | 4/4 | 45 % | 50 % |
| p6 | mixed_off_ms (truth-sized, seeds 10–59) | 32 % | 7/10 | 70 % | 32 % |
| p6 | bound8_ms (seeds 0–19) | 10 % | 0/4 | 50 % | 10 % |
| p6 | damp8_ms | 15 % | 1/4 | 55 % | 20 % |
| p6 | damp8w_ms | 5 % | 1/4 | 55 % | 5 % |

- **p4: the damping bound beats the truth-sized recipe** (65 % vs 38 % single). c = 1 is
  too loose (35 %).
- **p6 (dt = 20/1024): the absorbers are gone, but −2 is still missed.** No complex slot
  holds −2 any more. The failures are either near misses (−2 at −2.2 to −2.6, error
  0.1–0.3) at the floor, or local minima where the real slots coalesce near −0.8 (test
  ≈ 2e-3). The near misses reach the floor (see the certificate table below), so at this
  dt the data do not pin −2 (phase 1: ratio 0.6). With 8 slots the polished test error
  is *below* the oracle (3.0e-4 vs 4.0e-4): the extra slots fit the discretisation error
  of the truth.

## L-BFGS polish after Adam

`e2_polish.py`: every saved Adam checkpoint is polished with L-BFGS-B (scipy, full
training batch, data loss only, ≤ 3000 iterations, restarted after a line-search stall;
seconds per run). Before and after are the same run (`results/e2_polish.md`, 280 runs).

| condition | variant | test, Adam → polish | oracle | all poles | best of 5 |
|---|---|---|---|---|---|
| p4 | mixed_off_ms | 1.5e-3 → 7.3e-4 | 5.9e-4 | 38 % → 74 % | 8/10 → 9/10 |
| p4 | damp8_ms | 1.7e-3 → 7.4e-4 | 5.9e-4 | 65 % → **85 %** | 4/4 → 4/4 |
| p6, dt 20/1024 | mixed_off_ms | 6.0e-4 → 4.7e-4 | 4.0e-4 | 32 % → 38 % | 7/10 → 6/10 |
| p6, dt 20/1024 | damp8_ms | 1.6e-3 → 6.9e-4 | 4.0e-4 | 15 % → 35 % | 1/4 → 2/4 |
| p6, dt 20/4096 | mixed_off_ms | 4.4e-4 → 3.0e-4 | 5.6e-5 | 35 % → 45 % | 4/4 → 4/4 |
| p6, dt 20/4096 | **damp8_ms** (prior-free) | 1.6e-3 → 5.6e-4 | 5.6e-5 | 15 % → **50 %** | 1/4 → **3/4** |

Does reaching the floor certify the poles? All polished p6 runs (seeds 10–29, all
variants), binned by final training loss / oracle error:

| grid | train/oracle < 1.5 | 1.5–3 | ≥ 3 |
|---|---|---|---|
| dt = 20/1024 | 20/53 all recovered | 0/11 | 0/6 |
| dt = 20/4096 | **8/8** | 5/6 | 6/26 |

- **Adam was the bottleneck at the end of training**: the polish halves the p4 test error
  to near the oracle and doubles the p4 recovery (38 → 74 %, truth-sized; 65 → 85 %,
  prior-free). The Adam-stage pole errors of 0.1–0.2 were partly unconverged runs, not
  wrong basins.
- **Fine grid + polish confirms phase 1b.** At dt = 20/4096 every run that reaches the
  floor recovers all four p6 poles (8/8), and the floor is reached only near the truth.
  At dt = 20/1024 most runs reach the oracle level (53/70) but only 20 of them recover the poles.
  *Corrected by the restarts below:* the oracle is not the floor of the 8-slot model there.
  Restarted runs go down to 0.7 × the oracle, and at that level `damp8` recovers −2 in
  85 % of runs. The coarse grid biases −2 (median error 0.054, against 0.024 at
  dt = 20/4096) rather than hiding it.
- The remaining fine-grid failures are local minima (train/oracle 3–12): a damped spare
  complex slot holding −2 (truth-sized `mixed_off`, which has no damping bound), or
  coalesced real slots. A run's distance to the floor flags them without the truth.
- **Prior-free p6 now matches truth-sized p6** on the fine grid after the polish (50 % vs
  45 % single, 3/4 vs 4/4 best of 5; 4 groups, within noise). Full prior-free recipe:
  8 slots (4 real on a random log grid, 4 complex with |Im μ| ≥ 2π/T and |Re μ| < 0.5 |Im μ|),
  multi-start, Adam, L-BFGS polish, a dt fine enough for the poles to be identifiable, and
  the floor ratio as a check.

## Truth-free restarts after the polish

`e2_rescue.py`: each polished run gets up to 3 rounds of single-slot moves. One slot is
moved, with residue 0, to a grid point: real slots to Re ∈ −geomspace(0.05, 5, 8),
complex slots to Im ∈ geomspace(0.3, 5, 8) with Re = −0.1 Im. Each candidate gets a
300-iteration L-BFGS screen. The best candidate is accepted if training loss falls by
≥ 1 %, then fully polished. Every run gets the same treatment and only the training loss
is used. 120 runs (seeds 10–29), 5–7 min each with 12 processes in parallel.
Before = after the polish.

| condition | variant | train/oracle | test (median) | all poles | best of 5 | −1 | −2 (or −0.2) |
|---|---|---|---|---|---|---|---|
| p4 | mixed_off_ms | 0.90 → 0.86 | 5.7e-4 → 5.5e-4 | 65 → **95 %** | 3/4 → 4/4 | 65 → 95 % | 95 → 100 % |
| p4 | damp8_ms | 1.11 → 0.86 | 6.9e-4 → 5.4e-4 | 85 → 90 % | 4/4 → 4/4 | 90 → 100 % | 90 → 90 % |
| p6, dt 20/1024 | mixed_off_ms | 1.08 → 0.73 | 4.3e-4 → 2.9e-4 | 40 → 35 % | 2/4 → 0/4 | 80 → 95 % | 40 → 40 % |
| p6, dt 20/1024 | damp8_ms | 0.93 → 0.72 | 3.7e-4 → 2.9e-4 | 35 → **80 %** | 2/4 → 4/4 | 60 → 90 % | 45 → 85 % |
| p6, dt 20/4096 | mixed_off_ms | 5.03 → 0.92 | 2.8e-4 → 5.2e-5 | 45 → 80 % | 4/4 → 3/4 | 90 → 100 % | 45 → 80 % |
| p6, dt 20/4096 | **damp8_ms** | 3.50 → 0.93 | 2.0e-4 → 5.2e-5 | 50 → **90 %** | 3/4 → **4/4** | 65 → 90 % | 50 → **100 %** |

- **The restarts remove almost all local minima.** On the fine grid, training loss falls
  from 3.5–5 × to 0.93 × the oracle (median), and test error reaches the oracle
  (5.2e-5 vs 5.6e-5). With the damping bound, prior-free p6 now recovers every pole in
  90 % of single runs and −2 in all 20.
- **The damping bound matters most once the optimiser is good.** Without it
  (`mixed_off_ms`, the truth-sized recipe) the restarts lower the loss further on the
  coarse grid, but a damped spare complex slot (−2.40 + 0.73i, −1.27 + 0.31i) takes part
  of −2 (35 % all poles). With it, the fast decay can only be real (80 %).
- **On the coarse grid every restarted run ends at the same level** (0.67–0.77 × the
  oracle). The extra slots fit the discretisation error of the truth (e.g. a slot at
  −0.50 + 5.0i, at the band edge), so the minimum is below the oracle and shifted from it.
  −2 is biased (median error 0.054 vs 0.024 on the fine grid) rather than unidentifiable.
  The phase-1 ratio (0.6) measures this bias relative to a 10 % shift; it does not mean
  the minimum is far from the truth.
- Cost: 64 candidates × up to 3 rounds per run, 5–7 min with 12 processes on the 4090,
  about as much as the Adam phase.

## Floor certificate without the truth

`e2_floor.py`, applied to groups of 5 runs (seeds 10–29). The floor estimate is the
lowest training loss in the group, and runs within 1.5 × of it count as certified.
Effective poles are the folded learned poles, with slots below 2 % of the summed
impulse-response norm dropped and slots within 10 % merged. The consensus is the set of
effective poles found, within 10 %, in ≥ 2/3 of the certified runs (≥ 2 runs needed). The
truth is used only to score the consensus: per true pole, how many groups contain it, and
how many consensus poles match no true pole ("false").

| stage | condition | variant | estimate / oracle | certified: all recovered | complex pairs | −1 | −2 | false |
|---|---|---|---|---|---|---|---|---|
| polish | p6, dt 20/1024 | damp8_ms | 0.76 | 6/15 | 4/4, 4/4 | 4/4 | 3/4 | 1 |
| polish | p6, dt 20/4096 | mixed_off_ms | 1.11 | 4/4 | 0/4, 0/4 | 0/4 | 0/4 | 0 |
| polish | p6, dt 20/4096 | damp8_ms | 1.68 | 6/8 | 3/4, 3/4 | 3/4 | 2/4 | 1 |
| restart | p4 | mixed_off_ms | 0.86 | 19/20 | 4/4 | 4/4 | 4/4 (−0.2) | 0 |
| restart | p4 | damp8_ms | 0.85 | 18/20 | 4/4 | 4/4 | 3/4 (−0.2) | 0 |
| restart | p6, dt 20/1024 | mixed_off_ms | 0.69 | 7/20 | 4/4, 4/4 | 4/4 | 1/4 | 0 |
| restart | p6, dt 20/1024 | damp8_ms | 0.67 | 16/20 | 4/4, 4/4 | 4/4 | 3/4 | 2 |
| restart | p6, dt 20/4096 | mixed_off_ms | 0.91 | 16/20 | 4/4, 4/4 | 4/4 | 4/4 | 1 |
| restart | p6, dt 20/4096 | **damp8_ms** | 0.91 | 18/20 | 4/4, 4/4 | 4/4 | **4/4** | **0** |

(full table: `results/e2_floor.md`)

- **After the polish alone the certificate cannot be applied on the fine grid.** In a group
  of 5, typically only the lowest-loss run is near the floor, so there is no consensus
  (truth-sized: 0/4 groups). The estimate is also off (1.1–1.7 × the oracle).
- **After the restarts it works without the truth.** All runs of a group reach the
  same floor, the estimate is within 10 % of the oracle (0.91), and on the fine grid every
  group's consensus contains all four p6 poles with no false pole (`damp8_ms`). The
  consensus is the reported model: the poles that independent runs at the floor share.
- On the coarse grid the estimate falls below the oracle (0.67–0.69). The consensus misses
  −2 in one of four groups (`damp8_ms`) and adds false poles (pooled: −1.08 next to −0.98).
  A truth-free way to choose the grid is to shrink dt until the consensus stops changing.
  Here it changes from 20/1024 to 20/4096: −2 is found in every group and the false poles
  vanish. A third grid would be needed to show that it has settled.
- Tolerances (1.5 ×, 10 %, 2 %, 2/3) were set once and not tuned. The p4 and p6 results
  come from the same settings.

**Prior-free recipe, final form:** 8 slots (4 real on a random log grid, 4 complex with
|Im μ| ≥ 2π/T and |Re μ| < 0.5 |Im μ|) → multi-start → Adam → L-BFGS polish →
single-slot restarts → in groups of 5, floor estimate and consensus poles → refine dt
until the consensus is stable (checked here on two grids only). On rich data this recovers all poles of p4 and p6, with no
use of the true poles.

## E3, layer setting: the recipe with a steady branch (ALNO / LFNO)

Same recipe (`damp8_ms` → polish → restarts, seeds 10–29), but the tied layer gets the
steady branch of ALNO (FFT branch, its own spectral weights) or LFNO (R per mode). Modes:
64 on p4 (dt = 20/1024) and 256 on p6 (dt = 20/4096), i.e. a steady band of about 20 rad/s,
twice the multisine band. Oracle = true poles and residues, with the steady weights set to
the tied layer's steady coefficients (ALNO) or R = 1 (LFNO). The oracle error is set by the
mode truncation: 1.7e-3 (p4) and 9.8e-4 (p6), against 5.9e-4 and 5.6e-5 for the tied layer.

| condition | model | train/oracle (median) | test (median) | all poles (single) | per pole | runs at floor (< 1.5 ×): all recovered |
|---|---|---|---|---|---|---|
| p4 | tied | 0.86 | 5.4e-4 | 90 % | −0.4+3i 100, −1 100, −0.2 90 | 18/20 |
| p4 | ALNO | 0.90 | 1.6e-3 | 75 % | 100, 80, 95 | 15/20 |
| p4 | LFNO | 0.98 | 1.8e-3 | 65 % | 100, 70, 65 | 11/15 |
| p6 fine | tied | 0.93 | 5.2e-5 | 90 % | −0.5+1.5i 100, −0.3+4i 100, −1 90, −2 100 | 18/20 |
| p6 fine | ALNO | 2.60 | 2.5e-3 | 15 % | 100, 100, 35, 15 | 0/3 |
| p6 fine | LFNO | 6.74 | 6.4e-3 | 15 % | 100, 100, 85, 15 | 1/4 |

(full tables: `results/e2_rescue.md`, `results/e2_floor.md`)

- **The steady branch costs the real poles, not the complex ones.** The complex pairs are
  recovered in every run with either branch. On p4 the real poles drop from 90–100 % to
  65–95 %; on p6 −2 drops from 100 % to 15 %.
- **On p4 the failures sit at the floor (H1, weak form).** All 20 ALNO runs are within
  1.5 × of the oracle and 5 of them still miss a real pole. Loss alone cannot tell
  these runs from the correct ones: the truncation error (1.7e-3) is larger than what a
  shifted slow decay costs.
- **On p6 it is both optimisation and identifiability.** Most runs stay 2–7 × above the
  oracle (restarts move 1/20 ALNO runs, 16/20 LFNO runs, but do not close the gap). The few
  runs at the floor also miss −1 / −2: ALNO seed 10 is at 1.07 × the oracle with its four
  real slots at −0.55 to −0.69 instead of −1 and −2.
- **The certificate does not report wrong poles here, but reports little.** On p6 the
  consensus never contains −1 or −2 (0/4 groups, both models), because few runs are
  certified. On p4 ALNO it contains −1 in 3/4 groups and −0.2 in 4/4, with no false pole.
- Reading: the steady branch's truncation error sets a floor above which slow real
  decays are traded with the steady part. This is H1 in the layer setting. Remedies to
  test: a partial tie (R = K + R_res with a small R_res), more modes, or a weight on
  the transient branch.

## E4: pole error under output noise vs the Cramér–Rao bound

`e2_noise.py`. Gaussian noise on the training and validation outputs, per sample with
relative L2 norm η; the test set stays clean. One noise draw per group of 5 seeds. CRB:
trace bound for the minimal-form tied layer at the truth (Fisher information from the
Jacobian of the normalised residual), as a relative error.

![noise vs CRB](results/figures/fig_e4_noise.png)

| system | η | all poles (single) | complex poles | real poles recovered | certified runs | consensus: real poles found | false |
|---|---|---|---|---|---|---|---|
| p4 | 0 | 90 % | 100 % | −1 100, −0.2 90 | 20/20 | 4/4, 3/4 | 0 |
| p4 | 1e-3 | 100 % | 100 % | 100, 100 | 20/20 | 4/4, 4/4 | 0 |
| p4 | 1e-2 | 70 % | 100 % | 90, 80 | 20/20 | 4/4, 4/4 | 2 |
| p4 | 1e-1 | 45 % | 100 % | 60, 80 | 20/20 | 2/4, 3/4 | 2 |
| p4 | 3e-1 | 45 % | 100 % | 70, 60 | 20/20 | 2/4, 3/4 | 2 |
| p6 fine | 0 | 90 % | 100 % | −1 90, −2 100 | 20/20 | 4/4, 4/4 | 0 |
| p6 fine | 1e-3 | 85 % | 100 % | 100, 85 | 20/20 | 4/4, 2/4 | 0 |
| p6 fine | 1e-2 | 70 % | 100 % | 85, 70 | 20/20 | 4/4, 3/4 | 1 |
| p6 fine | 1e-1 | 5 % | 100 % | 20, 5 | 20/20 | 0/4, 0/4 | 5 |
| p6 fine | 3e-1 | 0 % | 100 % | 10, 0 | 20/20 | 0/4, 0/4 | 5 |

(errors per pole with the CRB: `results/e2_noise.md`)

- **Errors are 10–2400 × the CRB.** The bound is far below the errors at every η. At
  η ≤ 1e-3 the error equals the noise-free error (the bias of the 8-slot fit), so it
  does not grow with η. The gap is smallest for −2 at η = 0.3 (0.72 vs 0.064).
- **The information is there; the fit does not use it.** At η = 0.1 the CRB of −2 is 2 %,
  but only 1 of 20 runs is within 10 %. At η ≥ 0.1 the p6 consensus merges −1 and −2 into
  one decay at about −0.68 (4/4 groups).
- **The complex poles are robust:** 100 % at every η up to 0.3, with errors ≤ 5 %.
- **The truth-free steps break down under noise.**
  - Certificate: every run is certified at every η > 0, because the training loss of every
    run is at the noise level. The 1.5 × threshold then separates nothing, and false poles
    enter the consensus (5 on p6 at η ≥ 0.1).
  - Restarts: they accept a move only if it lowers the loss by ≥ 1 %. With the loss at
    the noise level no move qualifies (0/20 runs moved at η ≥ 0.1).
  - Both need thresholds on the excess loss over the noise level (estimated, for example,
    from the validation residual), not on the ratio.
- Caveats:
  - The CRB is for the minimal form with known order and an unbiased estimator. The
    8-slot model has spare slots and a relative-L2 loss, so the bound is optimistic for it.
  - There are 4 noise draws per η.

## E3, deep setting: first pass

`e3_deep.py`. The E1 deep models (ALNO / LFNO, 5,977 parameters, four layers) are trained
on the E2 rich data (p4, multisine16, T = 20) with E1's best protocol (`common` + pole lr
× 10), 5 seeds. The `plr10` variant is the E1 setting. In `bounded`, every layer and
channel pair has half of its slots real (log grid on [−5, −0.05]) and the other half
complex with the `damp8` bounds, followed by an L-BFGS polish; all poles keep a decay of
≥ 0.01. Weighted = slots with ≥ 2 % of their layer's impulse-response weight.

| model | variant | NaN / dead | test rel. L2 (final) | slots weighted / all | −0.4+3i pooled / weighted | −1 | −0.2 |
|---|---|---|---|---|---|---|---|
| ALNO | plr10 | 0 / 0 | 9.9e-3 ± 4.8e-4 | 22 / 768 | 100 / 40 % | 0 / 0 % | 0 / 0 % |
| ALNO | bounded | 0 / 0 | 9.5e-3 ± 1.1e-3 | 9 / 768 | 100 / 80 % | 0 / 0 % | 100 / 0 % |
| LFNO | plr10 | 2 / 0 | 9.6e-3 ± 7.9e-4 | 4 / 1024 | 60 / 0 % | 0 / 0 % | 0 / 0 % |
| LFNO | bounded | 0 / 3 | 1.7e-2 ± 7.1e-3 | 29 / 1024 | 40 / 40 % | 40 / 0 % | 40 / 0 % |

(full table: `results/e3_deep.md`; test error is the mean over runs that are not NaN or
dead; NaN and dead runs count as not recovered.)

- **In the deep stack the recipe does not recover the real poles.** With the weighted
  metric no real pole is found in any run, with either design. The bounded design only
  makes the complex pair more reliable (ALNO 40 → 80 %). The pooled 100 % for −0.2 is the
  real grid covering [−5, −0.05], not recovery (the pooled metric is vacuous with dense real
  slots).
- **The deep models are far from the floor.** Their test errors are about 1e-2, against
  5.9e-4 for the tied-layer oracle, and the polish halves the ALNO bounded error but does
  not close the gap.
- **LFNO has a singular failure mode.** The steady response β / (iω − μ) is singular at
  μ = iω.
  - In 2 of 5 `plr10` runs a pole reached μ ≈ 0 (|μ| ~ 1e-21 in layer 4 by epoch 121) and
    training went to NaN.
  - The first `bounded` version allowed decay 0 on real slots and went to NaN in 5/5 runs.
    The decay floor of 0.01 removed that, but 3 of 5 runs then stayed at loss 1.0 (output
    near 0) from the start.
- Reading: poles of a deep stack are not the system's poles one-to-one (E1 finding 2); the
  slot-level recipe does not change that. The next step for the deep setting is reading the
  poles off a linear probe of the trained network (RQ4), not off its slots.

## Next

- E3 layer: a partial tie R = K + R_res (small R_res) or more modes, to remove the trade
  between the steady branch and slow real decays (H3); re-run p6 ALNO / LFNO.
- E4: a noise-aware certificate and restart rule. Estimate the noise level, then use
  thresholds on the excess loss (or a χ² test) in place of the 1.5 × ratio and the 1 %
  acceptance. Check whether the restarts then separate −1 and −2 at η = 0.1.
- E4: why the errors are far above the CRB at small η (bias of the spare slots?); compare
  with a minimal-form fit started at the consensus.
- E3 deep: linear-probe poles of trained deep models (RQ4); the dead-start LFNO runs
  (bounded, loss 1.0 from epoch 1).
- Back to the E1 data (step / ramp / PRBS / smooth, T = 20): which poles the consensus
  reports there, as the truth-free version of E1 finding 3.
- More groups for the best-of-5 rates; c sensitivity between 0.35 and 0.7.
