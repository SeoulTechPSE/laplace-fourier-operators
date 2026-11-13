# STATUS

Running log of progress, decisions, and next steps. Newest entry first.
Each session appends one entry using the template at the bottom.

## Current state

| Work package | Months | Status | Gate |
|---|---|---|---|
| Shared foundation | 1–4 | nearly done: LNO v1/v2 reproduced (gate met), ALNO/LFNO 1D re-implemented; LTI/PDE/process generators partly | reproduce original LNO results (month 4) — **met** |
| (C) Pole audit of existing models | 3–5 | **in progress**: E1 first full run done (585 runs) | — |
| (D) Non-periodic input representation | 4–8 | not started | differs from padded FNO? (month 8) |
| (A) Pole layer and linear validation | 5–10 | not started | dispersion relations recovered (month 10) |
| (C) Identifiability theory and remedies | 6–12 | **started**: E2 phases 1–2, controls, first remedies done (`experiments/C/theory/`) | — |
| (A) Application 1: process dynamics | 11–16 | not started | — |
| (D) Initial conditions and rollout | 13–19 | not started | — |
| (A) Application 2: Cahn–Hilliard | 17–22 | not started | recover k_c, k_max (month 20) |
| Consolidation and code release | 23–24 | not started | — |

## Open decisions

- [ ] Staffing (affects schedule adjustments in `docs/integrated_schedule_ACD.qmd`, Section 4)
- [x] Whether ALNO / LFNO code is public: **not public** (2026-11-12); both re-implemented in `lfo.models.hybrid`
- [ ] E1 paper: reported set; deep "closest pooled pole" in main text; waive the budget rule for the minimal layer setting? (`experiments/C/e1_audit/E1_report.md`)
- [x] Ask the LFNO authors? **No** (2026-11-12): paper not formally published; use the reconstruction and document it

## Handoff (2026-11-12, after E1 first run) — start here in the next session

Work package continues: **(C) E1 audit**. Script, conventions and results:
`experiments/C/e1_audit/` (`README.md`, `results/summary.md`). Reproduce with the
commands in the README (≈ 1 h on the RTX 4090 with 6–12 parallel processes).

Suggested next steps (in order):

1. ~~Per-pole breakdown~~, ~~pole-lr ×10 control~~, ~~practical identifiability~~ done
   (README sections). E1 now has its three-part message (see the 2026-11-13 log entries).
2. ~~E1 summary~~ drafted: `experiments/C/e1_audit/E1_report.md` (3 figures). Open:
   the reported set, deep "closest pooled pole" in the main text, budget rule for the
   layer setting (report section "Open decisions").
3. E2 phases 1–2, controls and first remedies done (`experiments/C/theory/README.md`):
   mixed init with real-constrained slots + multi-start recovers 40–70 % of the real
   poles; repulsion / split / sparsity do not add. p4 = optimisation limit (best of 5 by
   validation ≈ 95 %), p6 −1/−2 = identifiability limit. Next: confirm best-of-5
   independently, learned real gate, p6 grid-resolution sweep.
4. Optional E1 extras: log-parameterised decay rate; untied models with 64 steady modes
   (H1); weight decay excluded from poles; deep LNO v1 at its published size.

Things to know:

- Environment: `.venv` (Python 3.12, torch 2.6, neuraloperator 2.0), RTX 4090; `pytest` 34/34.
- Data: `/srv/data/lno_drive` (LNO originals); reference code `runs/ref/LNO`; papers in
  `~/references/` (not committed).
- Every result: matched parameter budget, ≥ 3 seeds, mean ± std, `ood` and `iid` on LNO data.
- Commits as Kee-Youn Yoo, no Co-Authored-By trailers (CLAUDE.md).

## Log

### 2026-11-13 — (C) E2: third remedies (sparsity, pruning, fewer slots); two kinds of limit

- Done: `e2_train.py` variants `mixed_off_l1(_ms)`, `mixed_off_l1s_ms`, `mixed_off_prune_ms`,
  `few(_ms)`; 120 runs (p4 multisine16, p6 multisine16 T = 80; 10 seeds). Fit-quality and
  best-of-k selection analysis of all `mixed_off*_ms` runs. README section "Remedies (third round)".
- Results:
  - No variant beats `mixed_off_ms` beyond seed noise; pruning fires in 0–2 slots per run;
    fewer real slots hurt p6's −2 (0–20 %).
  - p4: runs missing a real pole fit worse (test median 2.6 × oracle vs 1.0 ×). Selecting
    the lowest validation loss among k runs: 50 / 84 / 95 % all poles recovered for k = 1 / 3 / 5
    (`mixed_off_ms`, bootstrap from 10 seeds).
  - p6: failed and successful runs fit equally (1.5 × oracle); validation selection lowers
    recovery (10 → 0 %): practical non-identifiability, as in phase 1.
- Decisions: E2 conclusion: two separable limits, optimisation (p4: fixed by mixed
  real-constrained init + multi-start + selection) and identifiability (p6 fast real poles:
  no training remedy). Candidate E3/E4 recipe: `mixed_off_ms` + best of ≥ 5 by validation.
  Sparsity / pruning / repulsion / split not adopted.
- Problems / open questions: the best-of-k numbers are a bootstrap over 10 seeds, not
  independent repeats; the real-slot count still uses the true number of real poles.
- Next: confirm best-of-5 with independent runs; learned real gate (no prior on the
  number of real poles); p6 grid-resolution sweep (phase 1, CPU).

### 2026-11-13 — (C) E2: second remedies (repulsion, split) against coalescing real slots

- Done: `e2_train.py` variants `mixed_off_rep(_ms)` (annealed log-space repulsion between
  real slots) and `mixed_off_split(_ms)` (merge slots within 10 %, move the freed slot);
  `e1_audit.train` gained optional `penalty` / `hook` (E1 behaviour unchanged). 80 runs
  (p4 multisine16, p6 multisine16 T = 80; 10 seeds). README section "Remedies (second round)".
- Results:
  - Coalescence removed (11 / 9 coalesced pairs over 10 seeds → 0–2), but recovery not
    improved: p4 −1 / −0.2 40–60 % / 40–70 %, p6 −1 / −2 30–60 % / 40–50 %, all within
    seed noise of `mixed_off_ms` (60 / 70, 50 / 40).
  - Failures now bracket the true pole with two real slots (p4: −0.05 / −0.31 around
    −0.2) at near-oracle test error (split + multi-start: 9.6e-4 / 5.8e-4).
- Decisions: neither remedy adopted; the remaining problem is redundancy of the spare
  real slots, not their separation.
- Next: residue sparsity + pruning with re-fit; fewer real slots; p6 grid-resolution sweep.

### 2026-11-13 — (C) E2: first remedies for the unrecovered real poles

- Done: `e2_train.py` variants `wide`, `mixed`, `mixed_off`, `base_ms`, `mixed_ms`,
  `mixed_off_ms` (definitions in the script docstring); 160 runs (p4 E1 data and
  multisine16, p6 multisine16 T = 80; 10 seeds). README section "Remedies (first round)".
- Results:
  - Mixed init (2 × real-pole count slots on a stratified log grid of Re, constrained
    real, plus spare band slots) + multi-start: p4 −1 / −0.2 recovered 60 % / 70 %
    (from 0 %), test 1.4e-3 (oracle 5.9e-4); p6 −1 / −2 50 % / 40 % (−2 from 0 %), test 5.6e-4.
  - Same slot count without the constraint (`wide`) or multi-start alone: 0 % for the
    real poles. A grid shifted away from the true poles gives the same results.
  - Correction to E1: a real pole's loss is flat in Im μ only for a real residue;
    with complex β a free slot leaves the real axis within a few Adam steps.
  - E1 data (p4): no remedy helps (data limit).
- Decisions: candidate remedy for E3/E4 = mixed init with real-constrained slots
  (+ multi-start). Parameter count differs from base (6 vs 4 slots); `wide` is the
  matched control.
- Next: coalescence of real slots (separation penalty, prune-and-split); a learned
  real gate instead of a fixed number of real slots; p6 grid-resolution sweep.

### 2026-11-13 — (C) E2: controls for the unrecovered real poles of p4

- Done: `e2_train.py --variant` (`warm`, `free_min`, `real_spare`, `real_min`; real
  constraint = Im mask parametrisation of `pr.mu`). 100 runs: p4 on E1 data and
  multisine16 × 4 variants, p6 multisine16 T = 80 × (`warm`, `real_min`), 10 seeds each.
  README section "Phase 2 controls".
- Results:
  - Warm start on multisine16: p4 all poles 100 % recovered, test 5.7e-4 = oracle; p6 −1 100 %,
    −2 70 %. On E1 data the warm start keeps the real poles in 8/10 seeds.
  - From the band init the minima have wrong slot assignments: −0.2 taken by a complex
    slot, the real slots coalescing (−1.15 / −1.13). Minimal form (`real_min`): the only
    complex slot goes to −0.2 in every seed, the ±3i pair is lost (test 0.3).
  - Real-constrained slots with a spare (`real_spare`): real-pole error 0.6 → 0.2, 20 %
    recovered. Removing the spare slot makes things worse.
- Decisions: E2 message: rich data make the poles locally identifiable (phase 1, warm
  start), but recovery from generic inits fails at wrong-assignment / coalescence
  minima, an optimisation problem (p4). p6's −1 / −2 remain data/floor-limited from
  generic inits. This feeds E3/E4 remedies.
- Next: remedies (mixed real/band init with spare slots, multi-start, pruning);
  p6 grid-resolution sweep.

### 2026-11-13 — (C) E2: data richness, local identifiability and training check

- Done: `experiments/C/theory/e2_richness.py` (phase 1: local identifiability of the
  tied layer at the truth, 46 data configurations for p4 and p6: multisine K = 1–32,
  window T = 5–80, six input families) and `e2_train.py` (phase 2: E1 tied layer,
  band init, `common` + pole lr × 10, trained on 8 of these configurations, 10 seeds,
  80 runs). `README.md` there. `e1_audit/identifiability.py`: `.conj()` →
  `torch.conj_physical` so the Jacobian can use forward mode (same values; minutes → 0.4 s
  per configuration).
- Results:
  - Phase 1, p4: richer data make all poles identifiable (ratio ≥ 10: multisine K ≥ 6,
    E1 data at T ≥ 40, mostly PRBS); E1 data at T = 20 do not (7.0 / 4.6 / 9.5).
  - Phase 1, p6: −1 / −2 stay non-identifiable under every data change (max ratio
    1.8 / 0.6); the pairs become identifiable with rich data. The floor (DFT
    discretisation) is the limit there.
  - Phase 2: complex pairs follow the prediction (p4 pair 70 % → 100 % recovered,
    test error up to 10× lower). p4's real poles: 0 % recovered in every condition,
    although predicted identifiable; their slots stay complex (e.g. −0.82 + 0.61i for −1)
    and test error stays 4–10× above the oracle (local minimum).
- Decisions: E1 part (iii) holds for p6 (data/floor limit); for p4 the remaining
  limitation is optimisation / parameterisation (free Im μ on every slot, spare slot),
  not data. To be reflected in `E1_report.md` when the paper set is decided.
- Problems / open questions: the first phase-1 run (reverse-mode Jacobian) timed out after
  1 h without output; fixed as above, results saved incrementally now.
- Next: p4 controls (warm start on rich data; real-constrained slots; n_poles = folded
  count); p6 grid-resolution sweep.

### 2026-11-13 — (C) E1 audit: draft summary and figures

- Done: `experiments/C/e1_audit/E1_report.md` (question, setup, three-part message,
  main-number table, limitations, open decisions); `figures.py` → `figures/fig1_travel`,
  `fig2_recovery`, `fig3_profile` (.png, .pdf).
- Results: no new runs; 900 reported runs (720 layer, 180 deep).
- Decisions: none final; three decisions for the paper are listed in the report and
  under "Open decisions".
- Next: settle the open decisions; E2 from `identifiability.py`.

### 2026-11-13 — (C) E1 audit: practical identifiability of the true poles

- Done: `experiments/C/e1_audit/identifiability.py` → `results/identifiability.md` / `.json`:
  tied layer in minimal form, float64 (= `PoleResidueTS1d` to 2e-7); Gauss–Newton
  sensitivity at the truth (fixed vs profiled) and nonlinear profiles of Re mu (L-BFGS).
- Results:
  - p6: a 10 % change of the real poles −1 / −2, with the rest re-fitted, costs
    e = 4.9e-4 / 1.6e-4, below the layer's own floor (8.2e-4). Profile of −2: e ≤ 2.1e-3
    for Re mu anywhere in [−6, −1]. The fast decays are practically non-identifiable.
  - p4: real poles weakly identifiable (profiled 6.3e-3 for −1), second basin in the profile.
  - Frequencies (Im mu) are well identified in all systems; cond(F) 47 / 3.9e4 / 6.8e6.
  - A real pole has zero first-order sensitivity to Im mu (explains spurious Im ≈ 0.2).
- Decisions: E1 message, three parts: (i) published protocols: poles limited by the
  learning-rate travel bound; (ii) with the bound lifted, frequencies are found;
  (iii) fast real poles remain unidentified because the data do not determine them
  (practical identifiability), not because of optimisation. (iii) hands over to E2.
- Problems / open questions: one profile point failed (p4, −0.2, c = 3: nan).
  Local analysis is at the truth only; the good-fit/wrong-pole minima found in training
  were not analysed individually.
- Next: E1 summary for the workshop paper; E2 starting from `identifiability.py`.

### 2026-11-13 — (C) E1 audit: pole learning-rate control

- Done: `e1_audit.py --pole-lr-mult` (poles get m × lr; label `<protocol>+plr<m>`;
  `per_pole.py` budget accounts for it). 315 runs with m = 10: layer (`stable`, `band`,
  10 seeds) and deep (LNO v1 `lno`, ALNO `lno`/`stable`, LFNO `stable` both protocols, 5 seeds).
- Results (`experiments/C/e1_audit/README.md`, section "Control"):
  - Travel saturation 35–92 % → 0 %; poles move ΔIm ≈ −15 to −20.
  - Layer `stable` tied pole error 5.4/14.8/6.4 → 0.04/1.6/0.83, test 0.29/0.64/0.23 →
    0.021/0.065/0.023. Deep ALNO pole error 0.12/0.35/0.38 → 0.05/0.23/0.32, test unchanged.
  - Complex pairs found (deep 80–100 %); **fast real poles −1, −2 still 0 %** everywhere
    except warm start; deep decay rates do not move (ΔRe ≈ +0.02).
  - LNO v1 deep (`lno` init) starts to train (test 0.5–0.8); LFNO becomes less stable.
- Decisions: E1 message is two-part — (i) under published protocols the poles are
  limited by the learning-rate travel bound (frequency); (ii) independently of that,
  fast real poles are not identified although the fit is good (practical
  identifiability). Part (ii) is the bridge to H2/E2.
- Next: per-pole Fisher information / profile likelihood for the tied layer; log decay
  parameterisation.

### 2026-11-12 — (C) E1 audit: per-pole breakdown and pole travel

- Done: `experiments/C/e1_audit/per_pole.py` → `results/per_pole.md` / `.json`: per true
  pole median error, recovery fraction (error < 0.1), signed ΔRe/ΔIm; pole travel
  |mu_final − mu_init| from initial poles rebuilt from the seeds and the checkpoints.
- Results (README section "Per-pole breakdown and pole travel"):
  - **Pole travel is bounded by the learning-rate schedule**: Adam moves ≈ lr per step,
    Σ lr × steps = 4.0 (`common`), 3.6 (`lfno_paper`). With the `stable` init 35–92 % of
    poles (layer) and 63–82 % (deep) reach ≥ 90 % of that bound; deep poles move
    ΔIm ≈ −3.9 with ΔRe ≈ 0. The `stable` init starts at |mu| ≈ 20, the band is Im ≤ 4,
    so the poles cannot get there. This explains the published-init failure.
  - Deep poles keep their initial decay rates (< 0.3 % below Re = −0.5); fast real poles
    −1, −2 are never recovered. Lightly damped pairs are recovered (100 %).
  - Band init (travel 0.3–0.9, not budget-limited): real poles still fail (0 %) — local
    minima, matched poles ≈ −0.5 with spurious Im.
  - Weight decay 0.01 (LFNO protocol) shrinks real poles toward 0 from a warm start.
- Decisions: E1 conclusions under the published protocols are stated as "poles ≈ init
  + bounded drift"; any interpretability claim needs a pole parameterisation or
  learning rate that lets poles move (also a design input for (A)).
- Problems / open questions: the travel bound is approximate (Adam's early steps can
  exceed lr slightly: layer max 5.0 vs 4.0).
- Next: E1 control with pole-specific learning rate / log parameterisation.

### 2026-11-12 — (C) E1 audit: first full run, layer and deep settings

- Done:
  - `experiments/C/e1_audit/e1_audit.py`: three fixed LTI systems (`p2`, `p4`, `p6`),
    T = 20, n_t = 1024, inputs step/ramp/PRBS/smooth (200/50/100, data seed 0).
    Settings `layer` (one single-channel pole-residue layer: tied = LNO v1, alno, lfno;
    n_poles = number of true poles) and `deep` (LNO1d port, ALNO1d, LFNO1d at the LFNO
    paper budget, 5,927–5,977 params). Protocols `common` (ComplexAdam, original LNO 1D)
    and `lfno_paper` (Adam, wd 0.01; LFNO only). Inits `lno`, `stable`; layer setting
    also `band` and `warm` (E1-only diagnostics, defined in the script).
  - 585 runs: layer 10 seeds (480), deep 5 seeds (105). `results/summary.md`, README.
- Results (details and numbers: `experiments/C/e1_audit/README.md`):
  - Oracle floors: tied layer ≈ 1e-3; ALNO/LFNO layer with true K(iω) on 16 steady modes
    4.4e-2–6.1e-2 (mode truncation; 4e-3 with 64 modes).
  - With the published inits **no model recovers the poles, the tied layer included**:
    `lno` init keeps 90–100 % of poles unstable (outputs blow up); `stable` init leaves
    poles far away (relative error 5–15). Deep models with the `lno` init do not train.
  - Deep, `stable`: ALNO/LFNO are accurate (test 1.5–3e-2) but even the closest pooled
    poles are off by 0.08–0.12 (p2) and 0.35–0.41 (p4, p6), with small seed spread.
  - LFNO deep diverges mid-training in 10/30 `stable` runs; with its paper protocol
    59–63 % of its poles end unstable.
  - Warm start: tied converges to the truth (0.007/0.024/0.042); ALNO/LFNO stay near
    (0.05–0.13) at the truncation floor; with LFNO's wd 0.01 the poles drift (up to 0.49).
  - Band init: good fit with wrong poles, also for the tied layer (p6: test 1.2e-2,
    pole error 0.37).
- Decisions:
  - Pole matching convention for E1 (script-level, `lfo.metrics` unchanged): fold learned
    and true poles to Im ≥ 0, one true pole per conjugate pair, complex-pair reference
    residue 2β (outputs are Re(...), so one learned pole can carry a pair).
  - No separate 1D LNO v2: one v2 layer equals the ALNO layer; deep v2 = ALNO stacks.
  - Unstable learned poles are reported as they are. Diverged runs (non-finite weights)
    are counted and excluded from the statistics; their pole metrics are stored as null.
  - Deep-setting pole error = closest pooled poles (optimistic bound), stated as such.
- Problems / open questions:
  - First obstacle is optimisation (init, weight decay, instability), not yet
    dual-branch non-identifiability. The audit therefore needs the warm/band diagnostics
    to separate the two; this shapes the E1 paper message.
  - My first `warm` init (complex factor on mu) made lightly damped poles unstable;
    fixed (Re and Im jittered separately) and re-run before reporting.
  - LNO v1 at the 5.9k matched budget does not train from its original init, although
    the published 1.8k Duffing configuration does; check at the published size.
- Next: per-pole breakdown; tied-layer multi-start; ALNO/LFNO with 64 steady modes;
  pole-excluded weight decay; then E2.

### 2026-11-12 — Shared foundation (S1): LFNO implementation details reviewed

- Done: compared the LFNO workshop paper (NeurIPS 2025 ML4PS, Appendix C Table 2) with
  arXiv:2606.07601 (Appendix A Tables 6–7); LFNO trained with both papers' protocols.
  Details in `experiments/foundation/paper_comparison.md` (section "LFNO").
- Results:
  - Both versions: 6 layers, width 4, 16 modes, relu, batch 64 for ODEs; consistent with
    our 4 + 2 reconstruction. Weight decay 0.01 (workshop) / 0.02 (arXiv), 100–200× LNO's.
  - The two versions order the steady layer differently (workshop: R then lambda; arXiv:
    lambda then R). We follow arXiv.
  - Paper protocols do not make the original LNO pole init trainable (test ≈ 1.0); with
    the stable init, test 0.185 (workshop protocol) and 0.260 (arXiv), 1,000 epochs, 1 seed.
  - With the stable init ≈ 50% of learned poles end unstable (Re mu > 0).
- Decisions: the LFNO authors will not be contacted (paper not formally published).
  Keep the reconstruction; state in any paper that LFNO is re-implemented, list
  the unspecified choices (init, layer arrangement, steady-layer order).
- Problems / open questions:
  - For (C) E1: train LFNO with its published protocol (Adam, wd 0.01–0.02) as well as the
    common protocol; the weight decay acts on the poles and may decide pole recovery.
  - Unstable learned poles: decide whether E1 reports them as-is (yes, for the audit) and
    whether (A) needs a stability parameterisation (Re mu = -softplus).
- Next: (C) E1 audit setup.

### 2026-11-12 — Shared foundation (S1): ALNO and LFNO re-implemented (1D)

- Done:
  - Code availability: no public code for ALNO/PILNO (arXiv:2602.12706) or LFNO
    (arXiv:2606.07601): no link in the papers; nothing on the authors' GitHub
    accounts, the LFNO lab page or GitHub search.
  - `lfo.models.hybrid`: `PoleResidueTS1d` (textbook pole-residue layer with separate
    transient and steady outputs, `poles()` for the audit), `SpectralConv1d`,
    `ALNO1d`, `LFNO1d`. `tests/test_hybrid.py` (7 tests; full suite 34/34 pass).
  - ALNO = PILNO Eq. (15) per layer: pole-residue transient + free FNO multiplier.
    It is the SI's "advanced LNO" (= our `LNO2dV2`), but written per layer instead
    of the original code's two parallel stacks.
  - LFNO = Eq. (7): steady coefficients from the same poles, then mixed by R on the
    lowest modes. The paper does not say how its 6 layers are arranged; with
    4 pole-residue layers, R on the first 2, and 4 W, our model has **exactly the
    published 3,417 parameters** (paper counts complex entries once). Reconstruction,
    not confirmed by the authors.
- Results:
  - Tests: one-channel layer = original LNO v1 layer; with the true poles the layer
    gives the exact LTI response for a periodic input (rel. error < 1e-2).
  - Initialisation matters. Duffing c=0, iid, 300 epochs, 1 seed, test rel. L2:

    | setting | ALNO (7,001) | LFNO (5,977) |
    |---|---|---|
    | original LNO pole init, no norm | 0.93 (no training) | 1.02 (diverges) |
    | stable pole init | 0.167 | 0.190 |
    | InstanceNorm (as `LNO2dV2`) | 0.221 | 0.359 |
    | both | 0.128 | 0.388 |

    (LNO v1 under the same 300-epoch run: 0.309.) Sanity checks only, not reported results.
- Decisions:
  - Default `pole_init="stable"` (Re mu = -exp(U[log 1/T, log 10/T]), Im mu ~ U[0, pi/(4 dt)]),
    `norm=False`: no normalisation between layers keeps learned poles interpretable for (C).
  - New re-implementations use the textbook pole-residue contraction (no second channel
    mixing); the LNO port keeps the original quirk.
- Problems / open questions:
  - Neither paper states its initialisation; the (C) audit should report pole recovery
    under both initialisations, since the init may decide which poles are found.
  - LFNO's published numbers use its own regenerated data (not comparable); ALNO has no
    published ODE results. So neither model can be checked against published values.
  - 2D versions of ALNO/LFNO not implemented (not needed for E1).
- Next: (C) E1 audit setup: LTI datasets from `lfo.data.lti`, train LNO v1, ALNO, LFNO
  (and LNO v2 as a 1D variant if needed) at matched budget, ≥ 3 seeds, compare learned poles
  with `lfo.metrics.match_poles` / `pole_error` / `seed_spread`.

### 2026-11-12 — Shared foundation (S1): Burgers with LNO v2, both splits

- Done:
  - `reproduce_lno.py`: case `burgers` (`LNO2dV2`, original `2D_Burger/main.py` settings:
    lr 0.01, batch 10, 1000 epochs, StepLR 100/0.5, `torch.optim.Adam`). New `Case` fields
    `optimizer`, `lno`, `input_stride`. Low-to-high resolution: every model reads every
    2nd input point (32 × 25) and predicts on 64 × 50; FNO/TFNO via neuraloperator
    `output_shape` ("FNO with IDFT").
  - `lfo.data.lno_bench`: docstring generalised (sizes per case; Burgers note);
    `tests/test_lno_bench.py` Burgers test (27/27 pass).
  - LNO v2, FNO, TFNO × 3 seeds × ood/iid (`results/<split>/burgers_*.json`).
- Results (test rel. L2, 3 seeds; `experiments/foundation/paper_comparison.md`):

  | split | LNO v2 (120,514) | FNO (120,444) | TFNO (119,614) |
  |---|---|---|---|
  | ood | 5.39e-2 ± 4.5e-4 | 2.59e-2 ± 3.7e-4 | **1.88e-2 ± 6.5e-5** |
  | iid | 5.31e-2 ± 1.0e-3 | 2.69e-2 ± 2.0e-4 | **1.78e-2 ± 8.2e-4** |

  - **LNO v2 reproduced** against SI Table S2 (5.31e-2 ± 1.79e-4): +1.5%.
  - Our budget-matched 4-layer FNO/TFNO are 2–3× better than LNO v2 and than the
    published FNO with IDFT (5.32e-2, architecture not given in the SI).
  - The Burgers data is in-distribution already (one random-field family for all
    splits); ood and iid agree, as expected.
- Decisions:
  - Month-4 gate "reproduce the original LNO results" is met for Duffing c=0,
    Diffusion and Burgers (v2).
  - Each case uses the original script's optimizer (`ComplexAdam` for v1 cases,
    `torch.optim.Adam` for Burgers).
- Problems / open questions:
  - LNO v1 on Burgers (SI 9.15e-2) not reproduced: no original script or hyperparameters.
  - LNO v2 is slow (≈ 2.1 s/epoch alone, 35 min per run; 3.5–4× FNO/TFNO).
  - For (A)/(C): on Burgers the free-Fourier models beat the dual-branch v2 at equal
    budget; v2's steady branch is itself FNO-like, so which branch carries the
    prediction is a direct question for the (C) audit.
- Next: ALNO/LFNO code availability and implementation (needed for (C) E1, hand-off
  month 4); remaining 1D/2D examples only as needed by (C) E1.

### 2026-11-12 — Shared foundation (S1): LNO paper Supplementary Information

- Done: read the NMI Supplementary Information (Tables S2, S5, S8);
  `experiments/foundation/paper_comparison.md` updated.
- Results:
  - Published FNO baselines (SI Table S8) are **1-layer** classic FNOs (Duffing:
    width 32, 16 modes; Diffusion: width 16, 4×4 modes), unlike arXiv Table A.1.
  - NMI Table 3 parameter counts omit complex weights for **both** LNO and FNO
    (recomputed exactly from Table S8). Total real dof: Duffing LNO 1,821 vs FNO
    38,273 (21×); Diffusion LNO 14,929 vs FNO 19,025 (1.3×).
  - Burgers targets (SI Table S2, low-to-high resolution): LNO v1 9.15e-2,
    advanced LNO (v2) 5.31e-2 ± 1.79e-4, FNO with IDFT 5.32e-2. v2 hyperparameters
    (Table S5) match the original `2D_Burger/main.py`.
- Decisions: closes the open question of the previous entry: published parameter
  comparisons understate model sizes; we keep counting complex entries twice.
- Next: reproduce Burgers with `LNO2dV2` against SI Table S2 (both splits);
  remaining 1D/2D examples.

### 2026-11-12 — Shared foundation (S1): check against the published LNO paper

- Done: compared with the published version (Nature Machine Intelligence 6, 631–640,
  2024; Fig. 2, Table 3) and the matching code archive (commit `c61c8fc`).
  `experiments/foundation/paper_comparison.md` updated.
- Results:
  - Code archive `c61c8fc` = our reference `78c64ef` in code and data (README and
    `note.txt` differ only). No re-runs needed.
  - LNO, published vs ours (ood): Duffing c=0 ≈0.24 vs 0.216 ± 0.068;
    Diffusion ≈1.2e-3 vs 1.33e-3 ± 1.5e-4. **Reproduced in both cases.**
  - FNO, published vs ours: Duffing ≈0.52 vs 0.895 (ours worse); Diffusion ≈7.5e-3
    vs 4.55e-3 (ours better). LNO/FNO factor on Diffusion: ≈6× published, ≈3× ours.
  - **Correction** to the earlier entry today: the "≈25× better than the paper's
    FNO on Diffusion" remark compared with the arXiv value (≈0.12), which the
    published version replaced. Against the published value our FNO is in line.
  - **Parameter counts:** the published LNO counts (797 Duffing, 2,641 Diffusion)
    are the real-valued weights only; the complex pole/residue weights are omitted.
    With complex entries counted twice: 1,821 and 14,929 (2.3× and 5.7× larger).
- Decisions: month-4 gate "reproduce the original LNO results" is **met** for
  Duffing c=0 and Diffusion, checked against the published version. We count
  complex parameters as two real parameters (`lfo.training.count_params`) and state
  this wherever we compare with published sizes.
- Problems / open questions: NMI Supplementary Information not yet checked
  (architectures for the FNO baselines, whether FNO counts include complex weights).
  The uploaded PDF file name says pages 231–240; the PDF itself says 631–640.
- Next: v2 on Burgers (both splits); remaining 1D/2D examples.

### 2026-11-12 — Shared foundation (S1): report both ood and iid splits

- Done:
  - `lfo.data.lno_bench`: shared loader for the original LNO data with
    `split="ood"` (original) and `split="iid"` (380 samples pooled, re-split
    200/50/130, stratified by forcing family, split seed 0). Tests in
    `tests/test_lno_bench.py` (26/26 pass).
  - `reproduce_lno.py --split`; results now under `results/ood/` and `results/iid/`
    (earlier results moved to `results/ood/`, checkpoints to `runs/foundation/ood/`).
  - iid runs: Duffing c=0 and Diffusion × LNO/FNO/TFNO × 3 seeds.
- Results (test rel. L2, 3 seeds; `experiments/foundation/paper_comparison.md`):

  | case | split | LNO | FNO | TFNO |
  |---|---|---|---|---|
  | Duffing c=0 | ood | **0.216 ± 0.068** | 0.895 ± 0.037 | 1.007 ± 0.054 |
  | Duffing c=0 | iid | 0.243 ± 0.038 | **0.181 ± 0.006** | 0.219 ± 0.015 |
  | Diffusion | ood | **1.33e-3 ± 1.5e-4** | 4.55e-3 ± 1.4e-3 | 9.45e-3 ± 9.5e-3 |
  | Diffusion | iid | 1.78e-3 ± 2.2e-4 | 7.02e-4 ± 1.9e-5 | **6.42e-4 ± 2.3e-4** |

  **The ranking flips with the split.** LNO is split-insensitive; FNO/TFNO gain
  5–15× in-distribution and beat LNO there. LNO's published advantage is an
  out-of-distribution advantage.
- Decisions:
  - **Every benchmark on the original LNO data is reported on both splits**
    (`ood` and `iid`); added to CLAUDE.md conventions. The iid split is a re-split
    of the original samples (no new data), so the two differ only in the split.
- Problems / open questions:
  - Implication for (A): the wavenumber-pole operator must be judged on both
    axes, OOD robustness (LNO's strength) and iid accuracy (FNO's strength).
  - Implication for (C): the dual-branch question (does the free steady branch
    absorb the poles?) may look different under ood vs iid training.
- Next: v2 on Burgers (both splits); remaining 1D/2D examples; published LNO version.

### 2026-11-12 — Shared foundation (S1): comparison with published values

- Done: read LNO (arXiv v2), SSM-operator, LFNO, PILNO papers (`~/references/`, not
  committed); comparison in `experiments/foundation/paper_comparison.md`.
- Results:
  - Duffing c=0 LNO: ours 0.216 ± 0.068 vs LNO paper ≈0.24 ± 0.03 (Fig. 3, read from
    plot) and SSM paper 0.260 ± 0.033. **Reproduced.**
  - Diffusion LNO: ours 1.33e-3 vs paper ≈0 on the plot (<0.005). **Reproduced.**
  - FNO on Duffing is also poor in the papers (≈0.77 LNO paper; 0.59 SSM paper, 10k
    epochs), confirming the out-of-distribution split as the cause.
  - On Diffusion, our budget-matched FNO (4.5e-3) is ≈25× better than the LNO
    paper's FNO (≈0.12). The published LNO-vs-FNO gap on this case is not reproduced.
  - LFNO regenerated its own in-distribution datasets; its numbers are not comparable.
- Decisions: month-4 gate "reproduce original LNO results" is met for Duffing c=0 and
  Diffusion (pending a check against the published version and SI).
- Next: check published version of the LNO paper when available; v2 on Burgers;
  remaining 1D/2D examples; decide whether (C)/(D) also use an in-distribution split
  like LFNO, in addition to the original OOD split.

### 2026-11-12 — Shared foundation (S1): why FNO fails on Duffing

- Done: data inspection; per-amplitude error analysis; FNO ablation on Duffing c=0
  (3 seeds each; `experiments/foundation/results/diag_duffing/`). `reproduce_lno.py`
  gained `--fno-kwargs`, `--budget`, `--tag`, and stores per-sample test errors.
- Results:
  - **The Duffing data is an extrapolation benchmark by design.** All 200 training
    inputs are one waveform scaled in amplitude (0.05–10; pairwise shape correlation
    1.000). All 130 test inputs are a *different* single waveform (decaying envelope;
    correlation 0.959 with the training shape). Validation = test family.
  - FNO ablation, test rel. L2 (mean ± std, 3 seeds):

    | variant | params | train L2 | test L2 |
    |---|---|---|---|
    | LNO (reference) | 1821 | 0.21 | 0.216 ± 0.068 |
    | FNO base | 1804 | 0.17 | 0.895 ± 0.037 |
    | no positional embedding | 1794 | 0.18 | 0.995 ± 0.14 |
    | domain padding 0.25 | 1804 | 0.20 | 0.979 ± 0.14 |
    | no pos. emb. + padding | 1794 | 0.20 | 0.828 ± 0.18 |
    | budget 20k | 20008 | 0.09 | 0.963 ± 0.14 |
    | budget 20k + no pos. emb. + padding | 19974 | 0.09 | 0.785 ± 0.070 |

    Budget, positional embedding, and padding are ruled out as the main cause;
    together they help only slightly. FNO fits the training family but has no
    structural prior that carries over to a new input waveform.
  - Error by input amplitude (3-seed mean; bins 0–1, 1–3, 3–6, 6–10):
    LNO test 0.15/0.12/0.22/0.31, LNO train 0.11/0.06/0.08/**0.49**;
    FNO test 0.39/0.60/0.96/1.22, FNO train 0.16/0.07/0.10/0.29.
    The original 1-layer LNO is nearly linear: it generalises across waveforms but
    cannot fit the strongly nonlinear regime even on training data.
- Decisions: FNO's Duffing gap is a property of the benchmark (single-waveform
  training set), not a bug in our setup. Report it as such; when FNO numbers are
  quoted for Duffing, state the train/test waveform split.
- Problems / open questions:
  - Relevant to (C) E5 and (D): the per-amplitude LNO train error shows where
    the linear pole layer stops explaining Duffing.
  - neuraloperator checkpoints need `torch.load(..., weights_only=False)` and the
    `_metadata` key removed before `load_state_dict` (torch 2.6).
- Next: paper values (papers to be uploaded); v2 on Burgers; remaining examples.

### 2026-11-12 — Shared foundation (S1): LNO port and first reproduction

- Done:
  - `lfo.models.lno`: LNO v1 (`LNO1d`, `LNO2d`) and v2 (`LNO2dV2`, Burgers) ported
    from the original (`78c64ef`); grids as constructor arguments, no hard-coded CUDA.
  - `lfo.training`: original `ComplexAdam`, `relative_l2_loss`, `seed_everything`, `count_params`.
  - `tests/test_lno.py`: port equals original **bit for bit** (atol=0) for 1D Duffing/
    Lorenz/Pendulum, 2D Diffusion/Reaction–diffusion/Beam, v2 Burgers, and the optimizer.
    The only discrepancy found during porting was the input grid: the original uses
    `np.linspace` (float64 → float32), not `torch.linspace` (1-ulp difference).
  - `experiments/foundation/reproduce_lno.py`: LNO/FNO/TFNO under the original protocol;
    FNO/TFNO (neuraloperator, 4 layers) matched to the LNO parameter count (within 6%).
- Results (test rel. L2, 3 seeds, mean ± std; `experiments/foundation/results/summary.md`):

  | case | LNO | FNO | TFNO | original code, 1 run |
  |---|---|---|---|---|
  | Duffing c=0 (≈1.8k params) | 2.16e-1 ± 6.8e-2 | 8.95e-1 ± 3.7e-2 | 1.01 ± 5.4e-2 | 3.15e-1 |
  | Diffusion (≈15k params) | 1.33e-3 ± 1.5e-4 | 4.55e-3 ± 1.4e-3 | 9.45e-3 ± 9.5e-3 | 1.42e-3 |

  The port reproduces the original code's error level in both cases.
- Decisions: shared-foundation work lives in `experiments/foundation/` (added to
  CLAUDE.md layout). All models use the original training protocol (ComplexAdam,
  lr 0.002, StepLR 100/0.5, 1000 epochs). FNO/TFNO size is set by matching parameter
  count over (n_modes, hidden_channels).
- Problems / open questions:
  - Paper values not yet compared: fill in the LNO paper's Duffing (Example 1) and
    Diffusion (Example 8) numbers to close the month-4 gate.
  - Duffing FNO/TFNO reach train L2 ≈ 0.17–0.20 but validation ≈ 0.9–1.1, i.e. they
    do not generalise. Candidate causes, untested: very small budget (1.8k params),
    default `positional_embedding="grid"`, periodicity assumption on transient
    signals. Must be understood before FNO numbers are used in any paper.
  - Diffusion TFNO has large seed spread (1.6e-3, 6.7e-3, 2.0e-2).
  - Original code quirks (kept in the port): second channel mixing in the transient
    term; transient residue sign −Hw (1D) vs +Hw (2D); original Duffing validation
    loop reshapes with the training batch size (printed vali error is off).
- Next: compare with paper values; diagnose the Duffing FNO gap (budget sweep,
  positional embedding off, padding); run v2 on Burgers; remaining 1D/2D examples.

### 2026-11-12 — Shared foundation: environment, LNO reference code and data

- Done:
  - Workstation: `.venv` (Python 3.12.6, torch 2.6.0+cu124, neuraloperator 2.0.0),
    RTX 4090; `lfo` installed editable, `pytest` 9/9 pass; Jupyter kernel `lfo` registered.
  - Original LNO code cloned to `runs/ref/LNO`
    (github.com/qianyingcao/Laplace-Neural-Operator, commit `78c64ef`).
  - All LNO data (Google Drive) in `runs/data/lno_drive` and shared copy in
    `/srv/data/lno_drive` (1D ×6, 2D Beam/Burgers/Diffusion/Reaction–diffusion,
    3D Brusselator, 3D shallow water 2×5.7 GB). Repo-included `.mat` files are
    byte-identical to the Drive copies.
  - Admin: TLJH `create_users: false`; `main` protected by ruleset `protect-main`
    (PR + 1 approval, no force push/deletion, repo admins bypass); unused classic
    PATs deleted; Claude Code remote control works.
- Results: original `1D_Duffing_c0/main.py` run unchanged, 1 seed, 1000 epochs, 38 s:
  test relative L2 = 3.146e-01 (vali 0.326). Log: `runs/logs/duffing_c0_orig.log`.
- Decisions: students push via PRs (Write role, not Admin); owner commits directly to `main`.
- Problems / open questions:
  - TLJH idle culler is disabled (`services.cull.enabled: false`). Enabling it may
    stop tmux training runs started from JupyterLab terminals; choose a long timeout
    and test before students arrive.
  - Compare the Duffing error above with the paper's reported value.
- Next: original Diffusion baseline run; port LNO v1/v2 to `src/lfo/models/` with tests;
  FNO/TFNO baselines from `neuraloperator` at matched parameter count; reproduce
  Duffing and Diffusion with ≥ 3 seeds.

### 2026-10-22 — Repository set up

- Done: repository skeleton, shared metrics (`lfo.metrics`), synthetic LTI generator
  (`lfo.data.lti`), research plans in `docs/`.
- Decisions: 24-month staggered schedule; (A) is the backbone, (C) first, (D) split in two.
- Next: shared foundation — port LNO v1/v2 to current PyTorch, set up FNO/TFNO
  baselines from `neuraloperator`, reproduce LNO results on Duffing and diffusion.

---

### Template

```
### YYYY-MM-DD — <work package>: <one-line summary>

- Done:
- Results: (numbers with seeds; link to experiments/<X>/results/...)
- Decisions:
- Problems / open questions:
- Next:
```
