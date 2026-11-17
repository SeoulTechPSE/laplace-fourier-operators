# STATUS

Running log of progress, decisions, and next steps. Newest entry first.
Each session appends one entry using the template at the bottom.

## Current state

| Work package | Months | Status | Gate |
|---|---|---|---|
| Shared foundation | 1–4 | nearly done: LNO v1/v2 reproduced (gate met), ALNO/LFNO 1D re-implemented; LTI/PDE/process generators partly | reproduce original LNO results (month 4) — **met** |
| (C) Pole audit of existing models | 3–5 | **in progress**: E1 done, draft report (`E1_report.md`, updated with E2); paper decisions made (2026-11-18); workshop paper draft written (`paper/e1_workshop.qmd`) | — |
| (D) Non-periodic input representation | 4–8 | not started | differs from padded FNO? (month 8) |
| (A) Pole layer and linear validation | 5–10 | not started | dispersion relations recovered (month 10) |
| (C) Identifiability theory and remedies | 6–12 | **started**: E2 phases 1–2, controls, remedies, prior-free recipe (damping bound, polish, restarts, truth-free certificate) done; E3 layer (ALNO / LFNO, partial tie, two-stage release), E4 noise vs CRB (noise-aware restarts, certificate, minimal refit with profile intervals, first-order-hold refit and layer, noise-free certificate with a precision term and bootstrap intervals, E1 data identifiable without noise and up to the CRB limit under noise, input noise as errors-in-variables with a bootstrap interval, EIV-weighted refits, a bootstrap-calibrated order, certificate and order test; reduction-path unresolved rules fail, a location-calibrated split test flags the merge, a combined post-report rule leaves 1 false pole in 40 groups but does not carry over to new p6 E1 groups; an EIVML refit of the reports removes the EIV bias; EIVML through the whole pipeline helps only up to η_u = 0.05; an EIVML elimination of the calibrated report leaves 0 false poles up to p4 30 η_y and p6 E1 0.05, with EIV-bootstrap intervals covering 33/36 at p4 30 η_y), E3 deep first pass done; H1–H3 theory draft (`experiments/C/theory/H1_H2.md`), E5 first pass (`experiments/C/e5_probe/`) | — |
| (A) Application 1: process dynamics | 11–16 | not started | — |
| (D) Initial conditions and rollout | 13–19 | not started | — |
| (A) Application 2: Cahn–Hilliard | 17–22 | not started | recover k_c, k_max (month 20) |
| Consolidation and code release | 23–24 | not started | — |

## Open decisions

- [ ] Staffing (affects schedule adjustments in `docs/integrated_schedule_ACD.qmd`, Section 4)
- [x] Whether ALNO / LFNO code is public: **not public** (2026-11-12); both re-implemented in `lfo.models.hybrid`
- [x] E1 paper: **decided** (2026-11-18): selected table in the main text, full grid in an appendix; deep "closest pooled pole" in the main text as an optimistic upper bound; budget rule waived for the layer setting only; workshop paper = E1 + an E2 paragraph, main (C) paper = E2–E5 + H1–H3 (`experiments/C/e1_audit/E1_report.md`, "Decisions for the paper")
- [x] E1 finding 3: **restated** (2026-11-16): "not found by the published recipes, but the E1 data identify them"; noise limit = CRB (p6 −2 to η ≈ 0.03, −1 ≈ 0.1, p4 real poles ≥ 0.3); `E1_report.md` finding 3, the E1 README and the web page updated
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
   poles; repulsion / split / sparsity do not add. Best of 5 by validation loss
   (independent check): all poles 8/10 (p4), 7/10 (p6). Learned gate and fixed 4 + 4
   sizing fail; |Im μ| ≥ 2π/T on complex slots (`bound8`) works for p4, not p6. Adding a
   damping bound (`damp8`, |Re μ| < 0.5 |Im μ|) and an L-BFGS polish gives a prior-free
   recipe. Truth-free single-slot restarts (`e2_rescue.py`) then bring the runs to the floor:
   p6 (dt = 20/4096) 90 % single, −2 in 20/20. A truth-free floor estimate + consensus over
   5 runs (`e2_floor.py`) reports all p6 poles in 4/4 groups with no false pole. Carried to
   ALNO / LFNO, noise and deep (2026-11-13 log): the steady branch costs the slow real poles
   (p6 −2 100 → 15 %), noise errors are 10–2400 × the CRB and the certificate certifies every
   noisy run, deep stacks recover no real pole. Partial tie R = K + R_res (2026-11-14) brings
   p4 back to the tied level (ALNO 95 %) but not p6 (≤ 30 %, runs above the floor);
   noise-aware restarts / χ² certificate give 100 % at η = 1e-2 and separate −1 / −2 at
   η = 0.1 (−2 still 9/20). Second pass: releasing δ from the tied run reaches the floor on
   p6 (90 %); a minimal refit + order test + profile intervals (`e4_profile.py`) gives
   CRB-level errors and honest intervals at η ≥ 1e-2. Third pass: the low-noise model
   error is the input hold (data: first-order hold; layer: band-limited). Refitting with
   the FOH response plus order reduction (`e4_profile.py --hold foh`) identifies every pole
   in 32/32 groups, η = 1e-3 … 0.3, errors at the CRB, no false pole. Release under noise:
   fine at η = 1e-2, worse than tied at η = 0.1. Hold vs reduction: at η ≥ 0.1 the
   reduction alone gives the gain, at η ≤ 1e-2 the hold is needed. FOH inside the layer:
   noise-free training reaches all poles in 95–100 % of runs (p6 coarse grid 100 % vs 80 %),
   test 7–225 × lower; no gain under noise; the ratio certificate then under-certifies.
   Precision term (σ² = float32 rounding) for noise-free data: FOH-refit certificate
   correct in 40/40 groups. On the E1 data the recipe recovers −1 / −2 (band p6 95 / 70 %,
   FOH 95 / 80 %) and the certificate reports every pole. Under noise the limit is the CRB,
   not η*: p6 −2 identified in 4/4 groups up to η = 0.03, lost near 0.1 (CRB limit 0.035);
   p4 real poles 4/4 at η = 0.3; see open decisions. The noise-free undercoverage is the
   float32 rounding of the inputs (not in σ²); a truth-free bootstrap covers 18/18.
   Input noise (errors-in-variables): the E4 intervals undercover (2–25 % at η_u = 10 η_y);
   a truth-free EIV bootstrap restores 88–98 % (`e4_eiv.py`, local diagnosis). In the full
   pipeline the certificate breaks for η_u > η_y; EIV-weighted refits (`--hold foh_eiv`)
   repair it up to η_u = 3 η_y; at 10 × a bootstrap-calibrated reduction (`--hold foh_eivb`)
   removes every false pole (4/4 groups end at the true order); report bootstrap intervals
   there (profile 5/12, bootstrap 11/12). Calibrating every run before the certificate with
   a pooled κ (`--hold foh_eivc`) certifies 5/5 runs in 8/8 groups at η_u = 0.1; with a
   bootstrap-calibrated order test (`--hold foh_eivo`) all 32 input-noise groups report the
   consensus, pass the order test and contain exactly the true poles; the test rejects all 108
   missing-pole models (ratio ≥ 112 vs ≤ 0.91). It holds on p4 up to η_u = 30 η_y (−1 biased
   8–11 %) but breaks on p6 E1 at 10 η_y (−1 / −2 merged or spurious, aliased poles).
   Truth-free rules on the reduction path (aliasing fold, "unresolved" flags, `--hold
   foh_eivu`, `e4_absorb.py`) do not fix it: a merged close pair and an EIV-biased isolated
   pole look the same there. A location-calibrated split test of each reported real pole
   (`e4_split.py`, 40 groups) flags the merged pole (ratio 2.9) and, among the controls, only
   an already-wrong pole (p4 30 η_y −0.892, ratio 1.14); the split refit does not resolve the
   pair, so it reads "unresolved". With a bias-corrected null, duplicate / alias checks and
   bias-corrected values (gated by the bootstrap width), the 40 reports reach every true pole
   at η_u ≤ 30 η_y on p4 (12/12, 0 false at 0.3) and leave 1 false pole on p6 E1 at 10 η_y
   (was 6); the gate was set after seeing the p6 data. Out of sample (12 new groups) the
   bias correction holds on p4 at 30 η_y (12/12, 0 false; report 8/12, 4 false), but on p6 E1
   at 0.1 the rule removes 1 of 7 false poles: spurious fast poles (−48 … −121) with narrow
   bootstrap intervals remain. An exact time-domain EIV cost degenerates (S / n → 0); the
   frequency-domain EIV ML refit of the reports removes the EIV bias without a gate (p4 30 η_y
   12/12 in 8/8 groups, p6 E1 0.05 15/16, 1 false) and its drop test removes part of the fast
   poles; p6 E1 at 0.1 stays unresolved. EIVML through the whole pipeline (plain χ², 12
   fresh groups) beats `foh_eivo` at p6 E1 0.05 (15/16, 1 false) but is worse at 0.1 / 0.3:
   undamped poles on the DFT grid remove lines from the cost; a finite-record variance
   (`EIVMLT`) degenerates too. Backward elimination under EIVML from the `foh_eivo` report
   (plain χ²) removes every spurious fast pole: p4 30 η_y 24/24 + 12/12 fresh, 0 false; p6 E1
   0.05 fresh 16/16, 0; p6 E1 0.1 fresh 8/16, 1 false, 6 unresolved (the −1 / −2 pair). Its
   profile intervals undercover at η_u ≥ 0.1 (17/36 at p4 0.3); an EIV bootstrap around the
   EIVML fit (denoised inputs) covers 33/36 there and 21/24 at p4 0.1; on p6 E1 0.1 misses
   follow the wrong −1 / −2 pair (20/31). Next: noise-dependent release; zero-order hold;
   joint intervals; RQ4.
4. Optional E1 extras: log-parameterised decay rate; untied models with 64 steady modes
   (H1); weight decay excluded from poles; deep LNO v1 at its published size.
5. Theory draft `experiments/C/theory/H1_H2.md` (checks `h12_check.py`): H2 proved for the
   tied layer (global uniqueness from one sparse input: every transient-excited pole; all poles
   with persistent excitation of order M + M′; generic local identifiability from one input);
   H1 proved in its exact form (per-line rank-one input set, e.g. one input) and shown false
   in the plan's "any ε" form for ≥ 2 inputs, with the profiled information formula (E1 set:
   23–72 % of the tied layer's). H3 (Theorem 6): the partial tie moves the poles from the
   tied estimate by −H⁻¹G R̂_δ ≤ C δ (checked to two digits; saturates at δ_sat ≈ 2–4e-3; with
   exact data no movement); C is truth-free (p6 −2: 711).
6. E5 / RQ4 first pass (`experiments/C/e5_probe/`): on damped Duffing the original LNO's 256
   layer poles stay at their initialisation (0/256 within 10 % of the linearised pole), while
   the average impulse response of its small-signal Jacobian gives the pole within 10 % in 3/5
   seeds; the operator is far from causal LTI (76–90 % non-causal Jacobian), so a causal LTI
   probe fails. Second pass (models whose poles move): only the linear tied layer gives a
   valid probe (J causal and shift-invariant), where probe and layer agree (d ≤ 0.06) on the
   amplitude-averaged equivalent pole (8–14 % from the small-signal one); a stable-init LNO
   with pole lr × 10 holds the physical pole in 3/5 seeds but no valid probe points to it.
   RQ4 rule: trust = valid probe ∧ agreement ∧ good fit. Third pass (secant probes along the
   amplitude): the original LNO, poles at init, reproduces the data's equivalent pole and its
   hardening drift (0.968 → 1.20 up to A = 7) within 0.1–1.5 %, out of sample too; the linear
   tied layer stays at 1.116. Fourth pass (new simulated sets, k₃ = 1, diverse E1 waveforms,
   rule fixed before training): `small` (|x| ≤ 0.23) 6 trusted, all within 1 % of the
   small-signal pole, no false trust; `mixed` (|f| ≤ 5) 2 trusted tied layers 43 / 86 % off
   (equivalent pole of data whose best linear fit leaves 44 %): the fit criterion (test < 0.5)
   is too lax. Stable init + pole lr × 10 gives LNO poles within 10 % in 4/5 seeds on both sets.
   Fifth pass (rule v2 = data linearity gate, best linear fit residual < 0.1, + v1; pre-registered):
   `medium` (gate 0.095) 6 trusted, 5–6 % from the small-signal pole; `moderate` (gate 0.20) v2
   abstains, where v1 alone trusts 4 tied layers 15–17 % off. Trusted poles track the data's
   equivalent linear pole (2–7 %). Sixth pass (driven pendulum, pre-registered): `pend_medium`
   gate passes, 6 trusted within 1 %; `pend_large` / `pend_orig` gate fails and v2 abstains,
   although the 8 runs v1 would trust are correct (2–5 %). Across 120 runs v2 has no false
   trust; its gate is sufficient but conservative for a softening spring. Seventh pass: rule v3
   (pole-shift gate: low-amplitude fifth linear, all-sample equivalent pole within 10 % of it),
   pre-registered on new sets where v2 and v3 disagree: v3 trusts 12 runs, all correct (2–7 %).
   Over 9 sets / 160 runs: v1 6 false; v2 0 false, 20 missed; v3 0 false, 0 missed. RQ4 rule =
   v3. E1 paper decisions made (2026-11-18); workshop paper draft
   `experiments/C/e1_audit/paper/e1_workshop.qmd`. Next: review the draft (venue, length,
   figures); Duffing c = 0; Lorenz (probe about its operating point).

Things to know:

- Environment: `.venv` (Python 3.12, torch 2.6, neuraloperator 2.0), RTX 4090; `pytest` 34/34.
- Data: `/srv/data/lno_drive` (LNO originals); reference code `runs/ref/LNO`; papers in
  `~/references/` (not committed).
- Every result: matched parameter budget, ≥ 3 seeds, mean ± std, `ood` and `iid` on LNO data.
- Commits as Kee-Youn Yoo, no Co-Authored-By trailers (CLAUDE.md).

## Log

### 2026-11-18 — (C) E1: workshop paper draft

- Done: `experiments/C/e1_audit/paper/e1_workshop.qmd` (Quarto; abstract, setup, three results, discussion,
  limitations, references, Appendix A = `results/summary.md`), following the 2026-11-18 decisions. Fig. 2's deep
  panel now says "closest pooled pole (optimistic upper bound)" (`figures.py`, figures regenerated). Fig. 3
  (band-layer profile) left out of the draft: its flat profile is the layer's input hold, so part 3 is told
  with the E2 numbers instead.
- Every number is taken from the E1 README / summaries and `experiments/C/theory/README.md` (E1-data recipe:
  damp8_ms + polish + restarts, −1 / −2 95 / 70 %; FOH-refit certificate 4/4 groups, 0 false; CRB limits).
- Not done: no Quarto / LaTeX in this environment, so the draft is not rendered; venue and page limit not
  chosen; references need full entries.
- Next: user review of the draft (venue, length, figure 3); Duffing c = 0; Lorenz.

### 2026-11-18 — (C) E1: paper decisions

- Done: the four open E1 paper decisions settled with the user; `experiments/C/e1_audit/E1_report.md`
  ("Decisions for the paper"; the E2 follow-up updated: the slot-count prior is removed by the prior-free
  recipe), `CLAUDE.md` (budget-rule exception for the E1 layer setting).
- Decisions:
  1. Main text: the selected table (published protocols, pole lr × 10, warm start); full grid in an appendix.
  2. Deep "closest pooled pole" in the main text, labelled as an optimistic upper bound.
  3. Same-budget rule waived for the layer setting only (minimal model, compares poles); deep setting
     keeps matched budgets.
  4. Workshop paper = E1 + one E2 paragraph (data identify the fast real poles; prior-free recipe; CRB
     limit under noise). Main (C) paper = E2–E5 and H1–H3.
- Next: workshop paper text from `E1_report.md`; Duffing c = 0; Lorenz.

### 2026-11-18 — (C) E5 / RQ4: a pole-shift gate (rule v3)

- Done: `experiments/C/e5_probe/e5_gate.py` → `results/e5_gate.md` (v2 and v3 gates from the data, against the
  v1-trusted runs of every set); pre-registration `results/e5_rule_v3_prereg.md`; `e5_data.py` sets `wide`
  (Duffing, |f| 0.05–0.6) and `pend_wide` (pendulum, 0.05–1), data seed 3, from a data-only scan for v2 / v3
  disagreement; 40 runs. `src/lfo/` unchanged.
- Results:
  - v3 with shift < 0.05 (first choice) on the seven earlier sets: 0 false, 11 correct runs missed. With shift
    < 0.1 (= the claimed 10 % accuracy, set after that run): 0 false, 0 missed.
  - New sets: v2 fails both (0.132, 0.244), v3 passes both (shift 0.098, 0.059); v3 trusts 7 runs on `wide`
    (5–7 %) and 5 on `pend_wide` (2–3 %), all correct.
  - Totals, 9 sets / 160 runs: v1 44 trusted, 6 false; v2 18 trusted, 0 false, 20 missed; v3 38 trusted,
    0 false, 0 missed.
- Decisions: the RQ4 trust rule is v3 (valid J ∧ probe–layer agreement ∧ fit ∧ data pole shift < 0.1).
  Limits: two oscillators with one small-signal pole; `wide` passed at 0.098; the gate needs low-amplitude
  training samples.
- Next: Duffing c = 0; Lorenz (probe about its operating point); E1 paper decisions.

### 2026-11-18 — (C) E5 / RQ4: rule v2 on the driven pendulum

- Done: `reproduce_lno.py` case `pendulum_c05` (original pendulum LNO: grid input, 8 modes); `e5_data.py`
  pendulum sets `pend_medium` (|f| 0.1–0.5) and `pend_large` (0.3–1), data seed 2; `e5_train.py` /
  `e5_probe.py` take `pend_*` tags; pre-registration
  `experiments/C/e5_probe/results/e5_rule_v2_pendulum_prereg.md`; 60 runs. `src/lfo/` unchanged.
- Results:
  - LNO pendulum data = x'' + 0.5x' + sin x = f (simulation 2.4e-4 … 1.4e-2); small-signal pole as Duffing.
  - `pend_medium`: gate 0.036, passes; 6 trusted (tied 5/5, stable LNO 1/5), all within 1 %.
  - `pend_large` (gate 0.236) and `pend_orig` (0.121): v2 abstains; v1 alone would trust 5 and 3 tied layers,
    all correct (2–5 %).
  - Grid input → invalid LNO Jacobian: only on `pend_orig`; several LNO probes are valid on simulated sets and
    the agreement check rejects the wrong ones (d 0.15–0.33).
- Decisions: rule v2 kept (120 runs, 18 trusted, no false trust). Its output-residual gate is conservative for
  softening systems; a gate on the pole shift (band-wise equivalent poles) is a post-hoc candidate.
- Next: pole-shift gate on new data of both systems; Duffing c = 0; Lorenz needs a probe about its own
  operating point; E1 paper decisions.

### 2026-11-18 — (C) E5 / RQ4: rule v2 with a data linearity gate

- Done: `e5_probe.py` `GATE` (rule v2 = data best-linear-pair residual < 0.1 ∧ rule v1), pre-registered in
  `experiments/C/e5_probe/results/e5_rule_v2_prereg.md`; `e5_data.py` sets `medium` (|f| 0.1–0.5) and
  `moderate` (0.2–1), data seed 1, bands from a data-only gate scan; 40 runs. `src/lfo/` unchanged.
- Results:
  - `medium`: gate 0.095, passes; 6 trusted (tied 5/5, stable-init LNO 1/5), J pole 5–6 % from
    −0.25 ± 0.9682i, 2–3 % from the data's equivalent pole −0.280 + 1.044i.
  - `moderate`: gate 0.20, fails, v2 trusts none; v1 alone trusts 4 tied layers 15–17 % off (6–7 % from
    the equivalent pole −0.359 + 1.176i).
  - Over the four simulated sets: v2 12 trusted, all ≤ 6 %, no false trust; v1 alone 6 false trusts.
- Decisions: the RQ4 rule is v2. A trusted stored pole is the equivalent linear pole of the training regime;
  the gate bounds its distance to the small-signal pole (< 10 % at gate residual < 0.1, on Duffing).
- Next: Duffing c = 0; pendulum / Lorenz secant probes; E1 paper decisions.

### 2026-11-17 — (C) E5 / RQ4: the trust rule on new Duffing data

- Done: `experiments/C/e5_probe/e5_data.py` (k₃ = 1 checked against the LNO data; sets `small`,
  peak |f| 0.02–0.1, and `mixed`, 0.05–5, E1 input families, 200 / 50 / 130); `e5_train.py
  --data` (+ `base` variant); `e5_probe.py <set>` with the trust rule fixed before the runs
  (valid J ∧ d < 0.1 ∧ test < 0.5). 40 runs. `src/lfo/` unchanged.
- Results:
  - `small`: 6 trusted (tied 5/5, stable-init LNO 1/5), all within 1 % of −0.25 ± 0.9682i; no
    false trust; 3 more LNO runs at 3–4 % excluded by a slightly non-causal J (0.13–0.21).
  - `mixed`: 2 trusted tied layers, 43 % and 86 % from the small-signal pole, near the data's
    equivalent linear pole (−0.89 + 1.75i, best linear fit residual 0.44); test 0.43.
  - Stable init + pole lr × 10 LNOs: poles within 10 % in 4/5 seeds on both sets and the best
    fits (0.009–0.16), but invalid probes.
- Decisions: the rule holds in the linear regime; its fit criterion is too lax for strongly
  nonlinear data. A stricter criterion (the data's best linear fit must itself fit) is a
  post-hoc candidate for a third set.
- Problems: two wait loops (`pgrep -f e5_train.py`) matched their own command line and never
  ended; replaced by a script without self-matching waits.
- Next: third set for the stricter fit criterion; Duffing c = 0; pendulum / Lorenz.

### 2026-11-17 — (C) E5 / RQ4: equivalent-linear (secant) probes along the amplitude

- Done: `experiments/C/e5_probe/e5_oppoint.py` → `results/e5_oppoint.md`: equivalent pole
  (best linear pair, FOH VarPro, one sample) at A = 0.1 … 10 for the data and for each model's
  own response, training and test waveforms, 4 variants × 5 seeds. `src/lfo/` unchanged.
- Results:
  - Data: Im 0.968 (A = 0.1) → 1.204 (A = 7) → 1.425 (A = 10), training waveform; test
    waveform similar.
  - Original LNO (layer poles at init): median error 0.1–1.5 % up to A = 7 (max over seeds
    3.8 %), 4.7 % at A = 10; test waveform 0.6–1.2 %. LNO variants 0.1–5 %.
  - Linear tied layer: constant 1.116, 1–15 % off (22 % at A = 10).
- Decisions: for nonlinear operators, poles are read from probes (secant along the operating
  regime; Jacobian at x = 0 when it is causal LTI), not from the stored layer poles.
- Next: small-amplitude / diverse Duffing training set; Duffing c = 0; pendulum / Lorenz.

### 2026-11-17 — (C) E5 / RQ4: probes on Duffing models whose poles move

- Done: `experiments/C/e5_probe/e5_train.py` (variants `lno_plr10`, `lno_stable_plr10`,
  `tied4_stable_plr10`, 5 seeds each, E1 protocol `common`); `e5_probe.py` over all 4 variants
  with the RQ4 indicator (probe valid = J non-causal share and Toeplitz residual < 0.1; d =
  impulse-response pole to nearest stored pole) and equivalent-linear fits of the training data
  by amplitude band. `src/lfo/` unchanged.
- Results:
  - Equivalent linear pole: A ≤ 1 −0.250 + 0.971i; A > 3 −0.298 + 1.199i; all −0.304 + 1.137i
    (hardening).
  - Tied layer: valid probe in 5/5, d ≤ 0.06 in the 3 good fits, pole −0.27 + 1.09 … 1.11i
    (8–14 % from the small-signal pole, between it and the equivalent pole); 2 failed fits
    (test 0.9) agree internally too.
  - LNO variants: probe invalid in 15/15 (non-causal 0.21–0.90). Stable init + pole lr × 10
    holds a pole within 5 % of the linearised one in 3/5 seeds, but the probe lands elsewhere.
- Decisions: RQ4 rule = valid probe ∧ probe–layer agreement ∧ good fit; a trusted pole is the
  amplitude-averaged equivalent linearisation, not the small-signal pole.
- Next: operating-point probes; a small-amplitude or diverse Duffing training set; Duffing c = 0.

### 2026-11-17 — (C) E5 / RQ4: linear-probe poles of the Duffing LNO (first pass)

- Done: `experiments/C/e5_probe/` (`e5_probe.py`, README). `reproduce_lno.py` gained the case
  `duffing_c05` (same configuration as c0); 5 LNO seeds trained (tag `e5`, test 0.13–0.37).
  Linearised pole from the 4 smallest-amplitude training samples: −0.2500 ± 0.9684i (theory
  −0.25 ± 0.9682i). Probes: stored layer poles; causal LTI fit of the odd small-signal response
  (ε = 0.01 … 3); Jacobian at x = 0 (non-causal share, Toeplitz residual, impulse-response
  pencil). `src/lfo/` unchanged.
- Results:
  - Layer poles stay at the initialisation (|μ| ≈ 0.1, 200–237 of 256 unstable); 0/256
    within 10 % of the linearised pole in every seed (E1 finding 1 in a nonlinear model).
  - The trained operator is not causal LTI: 76–90 % of ‖J‖ above the diagonal, Toeplitz
    residual 0.58–0.97; a causal LTI probe leaves 80–98 % unexplained at every scale.
  - The average impulse response of J gives −0.15 … −0.25 ± 0.76 … 1.07i: within 10 % in 3/5
    seeds.
- Decisions: the probe–layer disagreement is the RQ4 indicator; causality and shift invariance
  of J are checked before a probe is trusted.
- Next: probes on models whose poles move (pole lr × 10, damp8 recipe, tied single layer);
  operating-point probes; Duffing c = 0.

### 2026-11-17 — (C) Theory: H3, the partial tie (draft)

- Done: Theorem 6 in `experiments/C/theory/H1_H2.md` (now H1–H3) and `h3_check.py` →
  `results/h3_check.md` (partial-tie estimator fitted directly over δ = 0 … 3, 64 modes, E2
  training data p4 multisine and p6 E1). `src/lfo/` unchanged.
- Results:
  - Theorem 6: near a nondegenerate tied estimate θ̂₀, θ̂_δ − θ̂₀ = −H⁻¹G R̂_δ + O(δ²), so
    |Δ Re μ_k| ≤ C_k δ with C_k = Σ_l |(H⁻¹G)_{k,l}|, computable without the truth. With exact
    data the poles do not move; the deviation is driven by the mismatch and saturates at
    δ_sat = ‖R̂_∞‖_∞.
  - Check: deviation = first-order prediction to two digits for all δ (up to 16 %); bound
    10–50 × loose; δ_sat ≈ 2e-3 (p4), 4e-3 (p6).
  - Trade-off: p6 E1 fit 7.7e-4 → 6.1e-4 (−21 %) while −2's error goes 0.7 % → 9 % (C = 711;
    −1: 120; pairs 1.6–7); p4 multisine stays within 0.2 % even saturated.
- Decisions: H3 is stated as a bound on the deviation from the tied estimate (not from the
  truth); E3's δ = 0.3 is beyond saturation, so its p6 failures are optimisation.
- Next: E5 / RQ4 (Duffing linear-probe poles); E1 paper decisions.

### 2026-11-17 — (C) Theory: H1 and H2 for one pole-residue layer (draft)

- Done: `experiments/C/theory/H1_H2.md` (statements, proofs, mapping to the experiments) and
  `h12_check.py` → `results/h12_check.md` (numerical checks). `src/lfo/` unchanged.
- Results:
  - Lemma 1: the tied layer is the zero-initial-state response to the trigonometric interpolant
    of the input (checked to 1.6e-5 against an independent solve). The E1 / E2 floor is
    therefore the input hold.
  - Theorem 2 (H2): one input with DFT support S, |S| + M + M′ ≤ n, identifies every pole it
    excites in the transient; with |S| ≥ M + M′ (persistent excitation) all poles. Excitation
    is sufficient, not necessary (p6 recovered from one sinusoid by a matrix pencil, error
    5e-13). Spare slots of an exact fit are dead or merged.
  - Proposition 3: for a generic single input the tied layer is locally identifiable (full-rank
    Jacobian; checked on E1, white-noise and grid inputs).
  - Theorem 4 (H1): with a free steady symbol on all lines, the poles are completely
    non-identifiable iff the input set has rank one per DFT line (e.g. one input). With ≥ 2
    inputs they are locally identifiable; the plan's "for any ε" form does not hold in general.
  - Proposition 5: profiled information = per-line projection off the input spectra; on the
    E1 training set a free steady branch keeps 23–72 % of the tied information (p6 −2: 23 %).
- Decisions: H1 is restated (exact in the rank-one case; otherwise reduced information plus
  the truncation floor, the "weak form" of E3); H2 is restated with the transient route.
- Next: H3 (partial tie) bound and the E3 trade-off curve; E5 / RQ4.

### 2026-11-17 — (C) E4: EIV-bootstrap intervals around the EIVML fit

- Done: `e4_mlelim.py --boot` on the 64 groups: parametric EIV bootstrap around the final EIVML
  fit (32 draws; denoised stored inputs as clean inputs, x* = x̂ + e_u*, y* = G_fit x̂ + e_y*;
  Re μ ± 1.96 std), fixed before running. README section "E4, twenty-third pass". `src/lfo/`
  unchanged.
- Results (identified poles covered, profile / bootstrap):
  - p4 0.3: 17/36 / 33/36 (half-width 5.3e-3 / 1.9e-2); p4 0.1: 18/24 / 21/24.
  - η_u ≤ 0.05: both agree (p6 E1 0.05 29/31 each; p4 0.03 12/12).
  - p6 E1 0.1: 20/31 each; the complex poles are covered 7/8 when the −1 / −2 pair is
    identified, 8/16 when it is not (model error, not width).
- Decisions: the input-noise pipeline is `foh_eivo` → EIVML elimination → EIV-bootstrap
  intervals around the EIVML fit.
- Next: noise-dependent release; zero-order hold; joint intervals; RQ4.

### 2026-11-17 — (C) E4: backward elimination under EIVML from the calibrated report

- Done: `e4_mlelim.py`. The `foh_eivo` report of each of 64 groups (52 development, 12 fresh)
  refitted under EIVML and reduced by backward elimination with plain χ²_Δp(0.99); EIVML profile
  intervals; duplicates, aliased and undamped poles unresolved; rule fixed before running.
  README section "E4, twenty-second pass". `src/lfo/` unchanged.
- Results (identified / false, `foh_eivo` in brackets):
  - p4 0.3: 24/24, 0 (19/24, 5); fresh 12/12, 0 (9/12, 3). p4 ≤ 0.1, p6 ≤ 0.03: all, 0.
  - p6 E1 0.05: 15/16, 1 (13/16, 3); fresh 16/16, 0 (13/16, 3).
  - p6 E1 0.1: 23/32, 4, 6 unresolved (26/32, 13); fresh 8/16, 1, 6 unresolved (12/16, 6). The
    −1 / −2 pair is duplicated (5/12), merged (2/12), off > 10 % (2/12) or resolved (3/12).
  - Every spurious fast pole is removed; S / n 0.99–1.03.
  - Profile intervals undercover at η_u ≥ 0.1: 17/36 (p4 0.3), 18/24 (p4 0.1); η_u ≤ 0.05:
    29/31 – 16/16.
- Decisions: the input-noise pipeline is `foh_eivo` followed by the EIVML elimination;
  intervals at η_u ≥ 0.1 are not yet trusted.
- Next: EIV-bootstrap intervals around the EIVML fit; noise-dependent release; zero-order hold;
  joint intervals; RQ4.

### 2026-11-17 — (C) E4: EIVML through the whole pipeline, fresh groups

- Done: `--hold foh_ml` run through the pipeline (EIVML refits, backward elimination, certificate,
  order test, profile intervals, plain χ²; duplicates and aliased poles unresolved) on the 52
  development groups and 12 fresh groups trained after the rule was fixed (p4 0.3 and p6 E1 0.1:
  g50–65; p6 E1 0.05: g30–45), with `foh_eivo` on the fresh groups for comparison; summary
  `e4_mlpipe.py` → `results/e4_mlpipe.md`. New `EIVMLT` (finite-record variance) and
  `--hold foh_mlt`. README section "E4, twenty-first pass". `src/lfo/` unchanged.
- Results:
  - `foh_ml` vs `foh_eivo`, fresh groups: p6 E1 0.05 15/16, 1 false vs 13/16, 3; p6 E1 0.1
    10/16, 5 vs 12/16, 6; p4 0.3 7/12, 3 (9 unresolved) vs 9/12, 3. Development: p4 0.1 23/24,
    2 vs 24/24, 0; p4 0.3 20/24, 11 vs 19/24, 5; p6 0.1 26/32, 7 vs 26/32, 13. Clean and
    η_u ≤ 0.03 conditions: all true poles, 0 false, both.
  - Cause: undamped poles (Re μ ≈ 0) on DFT-grid frequencies; EIVML's stationary variance is
    unbounded there while the finite-record residual is not, so the line drops out (g40: gain
    845 vs χ²₈ 20.1).
  - Removing undamped poles with a fit bound fails (drift back, or the bound still removes the
    line). `EIVMLT` (exact marginal finite-record variance; closed form within 0.5 % of the
    exact matrix) cuts that gain to 25–47 but degenerates in the pipeline: a run reduces to two
    lightly damped high-frequency poles at S / n = 0.85.
- Decisions: whole-pipeline EIVML is not adopted (better only at η_u ≤ 0.05); `foh_eivo` plus
  an EIVML refit of its report stays the candidate; `EIVMLT` is not used.
- Next: backward elimination under EIVML started from the `foh_eivo` report (the refit, not the
  search); noise-dependent release; zero-order hold; joint intervals; RQ4.

### 2026-11-17 — (C) E4: exact time-domain EIV cost and an EIVML refit of the reports

- Done: `e4_eivtd.py`. `EIVTD` (exact time-domain EIV cost, covariance σ_y² I + σ_u² ratio M Mᵀ,
  batched Cholesky) and a truth-data diagnosis of what added fast poles gain under the FOH-EIV,
  EIVML and EIVTD costs (p6 E1 and p4 multisine, η_u = 0.1, 2 draws); `--drop`: the `foh_eivo`
  reports of 28 groups refitted under EIVML, each pole dropped in turn (supported if ΔS >
  χ²_Δp(0.99)). README section "E4, twentieth pass". `src/lfo/` unchanged.
- Results:
  - EIVTD is exact at the truth (S / n = 0.995) but degenerates when fitted (S / n → 0.00 on E1,
    0.25 on multisine): no log det term. Negative result.
  - EIVML gives a spurious −50 on E1 inputs a gain of 0.1–0.7 (FOH-EIV: 6.5–107; χ² 9.2).
  - EIVML refit of the reports: p4 −1 at 30 η_y → −0.97 … −1.03 (8/8 groups); p6 −2 at 0.05 →
    −1.96 … −2.08 (3/4). No true pole unsupported (min ΔS 570).
  - Report (supported poles, duplicates / aliased unresolved; fixed before scoring): p4 0.1 /
    0.3 / 0.3 new 12/12, 0 each; p6 0.03 16/16, 0; p6 0.05 15/16, 1 (combined rule 13/16, 3);
    p6 0.1 9/16, 2 and 11/16, 3 (combined rule 12/16, 1 and 14/16, 6).
  - Still supported spurious poles: −310, −108 ± 48i; the p6 pair at 0.1 refits to duplicates,
    a merged −0.73 or values 12 % off.
- Decisions: the EIVML refit replaces the gated bias correction as the candidate for input
  noise; `EIVTD` is not used.
- Next: EIVML through the whole pipeline (backward elimination with χ², certificate, intervals
  from its curvature or bootstrap) on fresh groups; noise-dependent release; zero-order hold;
  joint intervals; RQ4.

### 2026-11-17 — (C) E4: the combined input-noise rule out of sample

- Done: 12 new groups (60 runs, full recipe → `foh_eivo` → `e4_split.py --null bc`): p4
  η_u = 0.3 and p6 E1 η_u = 0.1 with seeds 30–49, and a new level p6 E1 η_u = 0.05
  (`p6_e1_T20_n1e-2_u5e-2`, registered in `e2_train` / `e2_polish` / `e2_floor` / `e2_rescue` /
  `e4_profile`). `e4_split.py` summary marks the new groups and adds a post-hoc width column.
  README section "E4, nineteenth pass". `src/lfo/` unchanged.
- Results (identified / false, new groups):
  - p4 η_u = 0.3: report 8/12, 4 false (−1 at −0.865 … −0.898) → combined rule 12/12, 0 false.
  - p6 E1 η_u = 0.1: 14/16, 7 false → 14/16, 6 false. Spurious fast poles (−48, −52,
    −108 ± 48i, −121 ± 1.2i; bootstrap spread 2.5–4 %) and −2 at −2.45 remain; one split flag
    (an unstable +0.30).
  - p6 E1 η_u = 0.05: 13/16, 3 false (−2 at −2.21 … −2.33, half-width 12 % > 10 % gate) →
    unchanged. A post-hoc width rule (half-width > 10 % → unresolved) gives 12/16, 0 false
    here, 9/16, 4 false at 0.1, and changes nothing elsewhere.
- Decisions: the bias correction (gated) is kept for multisine inputs; the split / duplicate /
  alias rules are not validated on E1 inputs; the width rule is recorded as post hoc.
- Next: a truth-free check of spurious fast poles (time constant vs input bandwidth and dt);
  then the width rule on fresh groups; calibrated profile intervals; exact time-domain EIV ML;
  coloured input noise; noise-dependent release; zero-order hold; joint intervals; RQ4.

### 2026-11-17 — (C) E4: bias-corrected split null and a combined report rule

- Done: `e4_split.py --null bc` (draws from the bias-corrected fit 2θ − mean θ*) on the 40
  `foh_eivo` reports; truth-free duplicate (two reported poles within 10 %) and alias
  (|Im μ| > π/dt) checks; bias-corrected values where the EIV bootstrap half-width is within
  10 % |z|; a combined score. README section "E4, eighteenth pass". `src/lfo/` unchanged.
- Results:
  - Split test, bc null: controls max ratio 0.80 (no flag; denoise null 1.14), merged
    −0.686 at 3.02; control G / mean G* at η_u = 0.3 is 1.1–3.6 (was 3.2–5.3).
  - Combined rule: p4 at η_u = 0.3 12/12, 0 false (was 11/12, 1); p6 E1 at 0.1 12/16, 1 false
    (was 6; left: a spurious −16.4 in g25); every other condition unchanged and exact; no true
    pole lost.
  - Ungated, the bias correction pushes the poorly identified p6 pair apart (8/16, 5 false);
    the gate (10 %, the identification tolerance) was chosen after seeing that.
- Decisions: the combined post-report rule (split flag with the bc null, duplicate / alias
  unresolved, gated bias-corrected values) is the candidate report for input noise; it stays
  in `e4_split.py` until checked on new groups.
- Next: new groups (other seeds, η_u between 0.03 and 0.1 on p6 E1) to check the rule out of
  sample; a check for a spurious isolated real pole; calibrated profile intervals; exact
  time-domain EIV ML; coloured input noise; noise-dependent release; zero-order hold; joint
  intervals; RQ4.

### 2026-11-17 — (C) E4: location-calibrated split test

- Done: `e4_split.py` on the `foh_eivo` reports of all 40 input-noise groups (p4 η_u = 0 …
  0.3, p6 E1 0 … 0.1). Per reported real pole z, a real pole held at w = f z (f = 0.25 … 4),
  gain G(w) against 16 EIV draws from the report's fit; flagged if G > max(1, mean G*/2)
  χ²_2(0.99). README section "E4, seventeenth pass". `src/lfo/` unchanged.
- Results:
  - p6 E1 η_u = 0.1: the merged −0.686 (g15) is flagged at every site (ratio 1.6–2.9,
    G / mean G* = 7–13). With w free it refits to a double pole at −0.44, not −1 / −2:
    unresolved, not resolved.
  - Controls: 35 of 36 groups have no flag (max ratio 0.97). The one flag (p4 η_u = 0.3 g10,
    −0.892, ratio 1.14) is on the EIV-biased −1, already false under the 10 % rule.
  - As a reporting rule (flag = open interval): false poles 6 → 5 (p6 0.1), 1 → 0 (p4 0.3),
    no true pole lost.
  - The null undershoots at strong input noise: the largest control G / mean G* is about 1 at
    η_u = 0.01, 4.5 at 0.1, 5.3 at 0.3. The margin over the merge at 30 η_y is about 2 ×.
  - Not covered: the duplicated pair of p6 g25 (redundancy, not missing structure); complex
    and aliased poles.
- Decisions: keep the split test as a post-report check; do not fold it into `e4_profile.py`
  yet.
- Next: a null that includes the input-noise effect on the real data (no denoising, or
  draws from the full reference model); a check for duplicated and aliased poles; the EIV
  bias at 30 η_y; calibrated profile intervals; exact time-domain EIV ML; coloured input
  noise; noise-dependent release; zero-order hold; joint intervals; RQ4.

### 2026-11-16 — (C) E4: aliasing fold and unresolved flags (negative result)

- Done: `e4_profile.py --hold foh_eivu` (= `foh_eivo` + aliasing fold of every run + an
  "unresolved" flag from calibration-only drops of a neighbouring pole) on the 20 groups of
  p6 E1 η_u = 0.1 / 0.03 and p4 η_u = 0.1 / 0.3; `e4_absorb.py` (a second flag rule, from the
  stored steps). README section "E4, sixteenth pass". `src/lfo/` unchanged.
- Results:
  - p6 E1 at 0.1 is not fixed. The folded aliased pole refits to a fast real pair
    (−127 ± 1.3i) and is kept; false poles 6 → 8. The −1 / −2 merge is never flagged (the
    dropped pole −5.7 is not a neighbour of the merged −0.69).
  - The flag fires on the controls: 5/8 p4 groups at 0.1 and 1/4 at 0.3 flag a
    well-identified true pole (near-duplicate or spurious fast pole dropped at ΔS > χ²).
  - The absorb rule (a slow pole moving > 10 % in a calibration-only drop) catches the p6
    merge in 3/4 groups but also fires in 4/4 p4 groups at 0.3 (EIV bias of −1: −1.12 → −0.89).
- Decisions: `foh_eivo` stays the reference pipeline; `foh_eivu` is kept as a recorded
  negative result.
- Next: an information-side unresolved rule (bootstrap spread of a merged pole vs the pair
  spacing, or the local EIV CRB); the EIV bias at 30 η_y; calibrated profile intervals;
  exact time-domain EIV ML; coloured input noise; noise-dependent release; zero-order hold;
  joint intervals; RQ4.

### 2026-11-16 — (C) E4: stronger input noise; where the calibrated pipeline breaks

- Done: conditions `p4_ms16_T20_n1e-2_u3e-1` and `p6_e1_T20_n1e-2_u1e-1` (4 groups each, 40
  runs, full recipe), `foh_eivo` on both, `e4_power.py` on p4. README section "E4, fifteenth
  pass". `src/lfo/` unchanged.
- Results:
  - p4 multisine, η_u = 30 η_y: 5/5 certified, consensus 4/4, order test 4/4, exactly 3
    poles. −1 biased to −0.89 … −0.92 (EIV bias). Missing-pole models all rejected, margin
    about 9 × (ratio ≥ 5.6 vs ≤ 0.63). Profile intervals cover 0/12, bootstrap 10/12.
  - p6 E1, η_u = 10 η_y: wrong in 4/4 groups. `calibrated_reduce` merges −1 / −2 into −0.69
    in 7/20 runs (the first true pole removed). Other runs keep spurious poles, including an
    aliased pair at −50 ± 325i (2π/dt = 322). −2 is beyond the information limit there (local
    spread 16 %), but the pipeline reports wrong values instead of "unresolved".
- Decisions: none.
- Next: report close pairs as unresolved under input noise (a merge stop rule); an aliasing
  check; the EIV bias at 30 η_y; calibrated profile intervals; exact time-domain EIV ML;
  coloured input noise; noise-dependent release; zero-order hold; joint intervals; RQ4.

### 2026-11-16 — (C) E4: power of the calibrated order test

- Done: `e4_power.py`. The `foh_eivo` report of each of the 32 input-noise groups minus one
  true pole, refitted and put through the order test (χ², pooled κ_g, calibrated κ_o). README
  section "E4, fourteenth pass". `src/lfo/` unchanged.
- Results:
  - All 108 missing-pole models are rejected by every κ.
  - Smallest calibrated ratio 112 (p6 E1 η_u = 0.03, a real pole of the −1 / −2 pair), against
    ≤ 0.91 for the correct reports.
  - The hard direction under input noise is false alarms, not power.
- Decisions: none.
- Next: stronger input noise and E1 inputs at η_u ≥ 10 η_y; the order test near the CRB limit;
  calibrated profile intervals; exact time-domain EIV ML; coloured input noise; close real
  pairs; noise-dependent release; zero-order hold; joint intervals; RQ4 linear probe.

### 2026-11-16 — (C) E4: a bootstrap-calibrated order test under input noise

- Done: `e4_profile.py`: `calibrated_order` and `--hold foh_eivo` (as `foh_eivc`, order test
  ΔS ≤ κ_o χ²_Δp(0.99) with κ_o from 16 EIV draws from the candidate refitted with the
  candidate and the reference's full FOH model, two starts). 32 groups (per-run calibrations
  reused from `foh_eivc`). README section "E4, thirteenth pass". `src/lfo/` unchanged.
- Results:
  - All 32 groups (p4 multisine η_u = 0–0.1, p6 E1 η_u = 0–0.03): 5/5 certified, report from
    the consensus, order test passed, exactly the true poles, 0 false.
  - η_u = 0.1 (8 groups): κ_o = 60–142, ratios 0.20–0.91 (`foh_eivc`: order test 5/8).
    Small input noise: κ_o = 1.0–3.7, ratios ≤ 0.51.
  - Coverage unchanged (η_u = 0.1: profile 8/24, bootstrap 21/24).
- Decisions: none.
- Next: power of the calibrated order test against a missed true pole; stronger input noise
  and E1 inputs at η_u ≥ 10 η_y; calibrated profile intervals; exact time-domain EIV ML;
  coloured input noise; close real pairs at the limit; noise-dependent release; zero-order
  hold; joint intervals; RQ4 linear probe.

### 2026-11-16 — (C) E4: calibrated certificate, fallback and more input-noise data

- Done: `foh_eivc` fallback: when every candidate fails the calibrated order test, report the
  least violating one (flagged; earlier results archived in `runs/archive/e4_profile_foh_eivc_v2`).
  New condition `p4_ms16_T20_n1e-2_u3e-2` (groups 10–25) and 4 more groups at η_u = 0.1
  (seeds 30–49): 40 runs, full recipe; `foh_eivb` and `foh_eivc` on them. README "E4, twelfth
  pass" extended. `src/lfo/` unchanged.
- Results:
  - η_u = 0.03: κ = 1, both holds 5/5 certified, consensus 4/4, 0 false.
  - η_u = 0.1, 8 groups: `foh_eivc` certifies 5.0 runs per group (`foh_eivb` 1.4); consensus
    formed 8/8 (reported 6/8, 2 ties with the identical best run), 0 false poles in both
    holds. 39/40 runs end at the three true poles.
  - With the fallback g20 reports the three true poles (before: 3 false).
  - The pooled-κ order test fails in 3/8 groups even where all runs agree (ratio 1.14–1.66).
    It is conservative, not calibrated.
  - Coverage at η_u = 0.1: profile 8/24, bootstrap 21/24.
- Decisions: none.
- Next: a calibrated order test (the calibrated elimination chain, or a bootstrap of the
  8-slot reference); calibrated profile intervals; exact time-domain EIV ML; coloured input
  noise; close real pairs at the limit; noise-dependent release; zero-order hold; joint
  intervals; RQ4 linear probe.

### 2026-11-16 — (C) E1: finding 3 restated

- Done: `E1_report.md` finding 3 rewritten ("not found by the published recipes, but the E1
  data identify them"; noise limit = CRB); the E1 band-layer analysis kept as a reference
  paragraph; the note removed. E1 README table row qualified; web page updated.
- Decisions: E1 finding 3 wording (option A, the user, 2026-11-16).
- Next: as the previous entry.

### 2026-11-16 — (C) E4: a calibrated certificate between runs under strong input noise

- Done: `e4_profile.py --hold foh_eivc`: `calibrated_reduce` on every run before the
  certificate (runs with the same poles share one calibration); certificate and order test
  with κ_g χ²(0.99), κ_g = max(1, median κ of the elimination steps taken in the group's
  runs); `--reuse` for earlier per-run calibrations. 24 groups (first version without κ_g
  archived in `runs/archive/e4_profile_foh_eivc_v1`). README section "E4, twelfth pass".
  `src/lfo/` unchanged.
- Results:
  - Small input noise (20 groups): κ_g = 1, identical to `foh_eivb` (5/5 certified,
    consensus 4/4, 0 false).
  - p4 multisine, η_u = 0.1: 18/20 runs end at the three true poles. Certified runs 5.0 per
    group (`foh_eivb` 1.8), consensus 3/4 (1/4), κ_g = 54–81.
  - g20 fails: one run's own bootstrap did not reproduce its spurious poles (κ = 1.4). The
    consensus misses the calibrated order test by 14 % (1618 vs 1416), and the fallback (best
    run) reports 3 false poles, one unstable. The group is flagged (order test failed);
    `foh_eivb` had it right. Total false poles 3 (`foh_eivb` 0).
  - Without κ_g (first version) two groups had a single certified run with spurious poles.
  - Coverage unchanged: profile 5/12, bootstrap 11/12.
- Decisions: none.
- Next: the fallback when every candidate fails the calibrated order test; more groups and
  input-noise levels; sensitivity to the κ_g summary (median / min / mean); calibrated
  profile intervals; exact time-domain EIV ML; coloured input noise; close real pairs at the
  limit; noise-dependent release; zero-order hold; joint intervals; RQ4 linear probe.

### 2026-11-16 — (C) E4: bootstrap-calibrated order under strong input noise

- Done: `e4_profile.py`: `EIVML` (frequency-domain EIV ML refit, poles and residues jointly;
  local tests only), `denoise_input`, `calibrated_reduce` and `--hold foh_eivb` (the
  `foh_eiv` report reduced further with bootstrap-calibrated thresholds; 16 draws per step;
  the current model refitted from two starts). 24 groups (first version archived in
  `runs/archive/e4_profile_foh_eivb_v1`). README section "E4, eleventh pass". `src/lfo/`
  unchanged.
- Results:
  - Truth data, p4 multisine, η_u = 0.1: two extra fast poles lower S by 1880 σ² (FOH) /
    179 (ML), against χ²₄(0.99) = 13.3; E1 inputs 5008 / 66. ML is not enough. The stationary
    noise model is off by −28 % to +23 % (start-up of the propagated input noise).
  - EIV-weighted tests are miscalibrated under strong input noise (an extra pole at −28 on
    truth data: ΔS 99–126 vs 9.2). The parametric bootstrap reproduces this (124–162).
  - `foh_eivb`: p4 multisine η_u = 0.1 goes from 5–7 to exactly the 3 true poles in 4/4
    groups (κ = 52–64 at the last spurious step, 21–25 at the first true-pole step), 0 false poles (foh_eiv: 8). κ = 1 at small
    input noise (reports unchanged), 2.2 for p6 E1 at η_u = 0.03. True poles are never removed
    (ΔS 6e3–8e7 vs thresholds 9–21).
  - At η_u = 0.1 the profile intervals cover 5/12, the bootstrap intervals 11/12.
  - The first version had one group fail (bootstrap fit in a local minimum, mean ΔS* < 0);
    the two-start refit fixed it.
- Decisions: none.
- Next: a calibrated certificate between runs (consensus 1/4 at η_u = 0.1); calibrated profile
  intervals; exact time-domain EIV ML; coloured input noise; resolving close real pairs at
  the limit; noise-dependent release; zero-order hold; joint intervals; RQ4 linear probe.

### 2026-11-16 — (C) E4: input noise in the full pipeline; EIV-weighted refits

- Done: `make_splits(in_noise=)` and conditions `p4_ms16_T20_n1e-2_u{1e-2,1e-1}`,
  `p6_e1_T20_n1e-2_u{1e-2,3e-2}` (`e2_train.py` and the case lists of polish / floor /
  rescue; full recipe, 80 runs). `e4_profile.py`: weighted VarPro (`w`, `whiten`, `coef`,
  `design`), `eiv_weights` (residual spectrum σ_y² + σ_u² (‖x‖/‖y‖)² |G|², G from the best
  run), `--hold foh_eiv` (all tests with σ² = 1), `noise_var_robust` (moved from `e4_eiv.py`),
  `eiv_bootstrap` per group and `--add-boot` for existing groups (η_u = 0 references); an
  "input noise" summary section. 48 groups (24 FOH, 24 foh_eiv). README section "E4, tenth
  pass". `src/lfo/` unchanged.
- Results:
  - Unweighted FOH refits: fine at η_u = η_y (certified 5/5; consensus 2–3/4). At η_u = 3 η_y
    (p6 E1) and 10 η_y (p4 multisine) only 1–2 runs are certified, consensus 0–1/4, 7 / 16
    false poles, profile intervals cover 5/16 and 0/12 (bootstrap 12/16, 9/12).
  - EIV-weighted: weighted S / n = 0.998 at the fit. η_u = 0.01 (both) and 0.03 (p6 E1):
    certified 5/5, consensus, order test and all poles 4/4 groups, 0 false, profile intervals
    cover 16/16 and 12/12; profile / bootstrap half-width 0.9–1.4 ×. With σ_u = 0 identical
    to the unweighted results.
  - p4 multisine at η_u = 0.1: all true poles 4/4 groups, but 8 false fast poles and
    consensus 1/4. Above the excited band the input is noise and the fit lowers the gain
    there (report |G| / true |G| = 0.92 at 5–25 Hz), as a Wiener filter would; plug-in
    weights do not remove this bias.
- Decisions: none.
- Next: a consistent EIV refit (θ-dependent weights or the excited band only); coloured input
  noise; resolving close real pairs at the limit; noise-dependent release; zero-order hold;
  joint intervals; RQ4 linear probe.

### 2026-11-15 — (C) E4: input noise as errors-in-variables under output noise

- Done: `e4_eiv.py` (minimal FOH refit started at the truth, 64 independent draws per row of
  input noise η_u ∈ {0, 0.01, 0.03, 0.1} on the stored inputs and output noise η_y = 0.01;
  4 data sets: p4 multisine / E1, p6 E1 / fine multisine). Three 95 % intervals per Re μ:
  the E4 one (output σ̂², Wald = profile), the residual σ², and a truth-free EIV parametric
  bootstrap (σ_u from the inputs by a robust MAD 6th-difference estimate, bias-corrected).
  README section "E4, ninth pass". `src/lfo/` unchanged.
- Results:
  - The pole spread grows ~ linearly with η_u: 1.1–1.8 × at η_u = η_y, 10–14 × at 10 × (p6
    fine −0.3+4i 5.7 ×).
  - E4 interval coverage (95 % nominal; 92–97 % at η_u = 0): 67–92 % at η_u = η_y, 30–58 %
    at 3 ×, 2–25 % at 10 ×. Residual σ²: at most 31–42 % at 10 ×.
  - EIV bootstrap: 88–98 % in every row except p6 E1, η_u = 0.1, −1 (83 %); std 0.86–1.23 ×
    the true spread. The MAD estimate reads η̂_u / η_u ≈ 1 also on the E1 step inputs
    (mean-square estimate: η ≈ 0.05 on clean E1 inputs).
  - Bias matters only at large η_u: |bias| / spread 0.53 (p4 E1 −0.2, η_u = 0.1), 0.44 / 2.5
    (p6 fine −0.3+4i, η_u = 0.03 / 0.1); after the bootstrap correction 0.15, 0.11, 0.08.
- Decisions: none.
- Next: the EIV bootstrap inside `e4_profile.py` (trained runs, certificate and order test
  under input noise); coloured input noise; resolving close real pairs at the limit;
  noise-dependent release; zero-order hold; joint intervals; RQ4 linear probe.

### 2026-11-15 — (C) E4: the E1 data under noise; the noise limit is the CRB, not η*

- Done: conditions `p4_e1_T20_n{1e-1,3e-1}`, `p6_e1_T20_n{1e-2,3e-2,1e-1}` in `e2_train.py`
  and the case lists of `e2_polish.py`, `e2_floor.py`, `e2_rescue.py`; full recipe (band
  layer, damp8_ms, seeds 10–29: Adam → polish → restarts → noise-aware restarts; 100 runs),
  then the FOH-refit certificate (`e4_profile.py --hold foh`, 20 groups; new "E1 data"
  sections with a CRB column). README section "E4, eighth pass"; sixth-pass η* statements
  corrected in README, the E1 report note, the handoff and the open decision. `src/lfo/`
  unchanged.
- Results:
  - η* (sixth pass) is per sample. It ignores the √n averaging over 200 × 1024 samples and is
    10³–10⁴ × too strict. The CRB limit (1.96 CRB = 0.1 relative) is, for the E1 data, p6
    −1 / −2: η ≈ 0.11 / 0.035; p4 −1 / −0.2: 1.4 / 3.
  - FOH certificate on the E1 data: p4 every pole in 4/4 groups at η = 0.1 and 0.3; p6
    every pole 4/4 at η = 0.01 and 0.03, 0 false poles, all intervals cover. Errors
    0.7–1.2 × CRB; real-pole half-widths ≈ 1.96 CRB.
  - p6 at η = 0.1 (above the −2 limit): order 4 in 4/4 groups and every interval covers, but
    the −1 / −2 intervals overlap; −2 within 0.1 in 2/4 groups (2 counted false).
  - Single runs (after noise-aware restarts): all poles 55 % (p4), 35–60 % (p6); p6 −2
    35–60 % (before them 0–15 %).
- Decisions: none. The open decision on finding 3 now proposes the CRB limit instead of η*.
- Next: input noise / rounding as errors-in-variables under noise; resolving close real pairs
  at the limit; noise-dependent release; zero-order hold; joint intervals; RQ4 linear probe.

### 2026-11-15 — (C) E4: the noise-free undercoverage is the input rounding; bootstrap intervals

- Done: `e4_bias.py` (minimal FOH refit on stored vs exact inputs / outputs, Monte Carlo over
  ±ulp/2 dithers, truth-free parametric bootstrap; 5 noise-free data sets). In `e4_profile.py`,
  noise-free conditions only: fits minimise S / (n σ²) and end with damped Newton steps;
  `bootstrap()` intervals per reported pole; `VarPro.pred` factored out of `S`. The 40
  noise-free groups rerun (old JSONs in `runs/archive/e4_profile_foh_clean_v1`). README
  section "E4, seventh pass". Noisy conditions and `src/lfo/` unchanged.
- Results:
  - Both exact: pole errors ≤ 2e-14 (generator and refit exact). Output-only dithers give the
    profile width; dithering the inputs as well widens the pole spread 1.2–2.0 × while S rises
    ≤ 1.3 ×. The missing term is the float32 rounding of the inputs (errors-in-variables). The
    stored-data errors are within 1.9 std of the full spread: no bias.
  - The groups of a condition share one data set, so the sixth-pass "0/4" misses are one draw
    each: the profile intervals cover 13 of 18 pole × data pairs, unchanged after the rerun.
  - Bootstrap (stored inputs + fitted outputs dithered, 32 draws): std 0.7–1.3 × the exact
    Monte Carlo; covers 18/18 poles, every pole in 40/40 groups.
  - L-BFGS on S ~ 1e-13 stalled (truth start 3.5 σ² above the minimum, ~2 std off); the Newton
    finish fixes it. The reports were already at the minimum (same digits).
- Decisions: none.
- Next: E1 data under output noise near η*; input noise / rounding as errors-in-variables
  under noise; noise-dependent release; zero-order hold; joint intervals; RQ4 linear probe.

### 2026-11-15 — (C) E4: noise-free certificate at float32 precision; the hold on the E1 data

- Done: `e4_profile.py` noise-free conditions (multisine p4 / p6 coarse / p6 fine, E1 data
  p4 / p6, band- and FOH-trained; σ² = float32 rounding of the stored outputs, since the
  sixth-difference σ̂² sees the signal and the E1 input steps; 40 groups). `e2_richness.py
  --hold foh` (phase 1 with the FOH layer: floor and noise limit η*). Conditions
  `p4_e1_T20{,_foh}`, `p6_e1_T20{,_foh}` through the full recipe (80 runs). README section
  "E4, sixth pass"; a note under E1 finding 3 in `E1_report.md`. `src/lfo/` unchanged.
- Results:
  - At the truth the FOH floor is 1e-15–3e-14 on every phase-1 configuration (band
    4e-4–3e-3). On the E1 data η* = 6e-4–1.3e-3 (p4 poles), 4.7e-5 / 1.5e-5 (p6 −1 / −2),
    all far above float32 rounding.
  - E1 data, E2 recipe (noise-free): all poles in a single run, band 90 % (p4) / 65 % (p6),
    FOH 85 / 80 %; p6 −2 70 / 80 % (E1 recipes: −1 / −2 in 0–10 %). Ratio certificate:
    band p4 4/4 groups, p6 −2 2/4 with 2 false; FOH runs 0–1/4.
  - Precision-term FOH-refit certificate: order test, consensus and all poles in 40/40
    groups (multisine and E1, both layers), no false pole, errors 1e-11–2.5e-8. The 95 %
    intervals undercover for some poles (truth 1–1.7 half-widths outside, same in every
    group): an unmodelled bias at 1e-9–1e-8 (input rounding? generator arithmetic?).
- Decisions: none taken. E1 finding 3 ("the E1 data cannot identify them") does not hold for
  noise-free data; it holds only under relative output noise above η*. Added to the open
  decisions (how to state finding 3 in the E1 paper).
- Next: the bias behind the undercoverage; the E1 data under noise near η*; noise-dependent
  release; zero-order-hold variant; joint intervals; linear-probe poles for deep (RQ4).

### 2026-11-15 — (C) E4: hold vs reduction; the first-order-hold response inside the layer

- Done: `e4_profile.py --hold band_red` (band-limited refits with the per-run reduction; 32
  groups). `FOHPoleResidue` in `e2_train.py` (tied layer with the exact first-order-hold
  response, complex128 inside) and 5 conditions `*_foh` (p4; p6 dt = 20/1024; p6 fine, clean,
  η = 1e-2, 0.1), 100 runs through the full recipe (Adam → polish → restarts, noise-aware on
  noisy data), registered in `e2_polish.py`, `e2_rescue.py`, `e2_floor.py`; FOH profile pass on
  the 8 noisy FOH-trained groups (own section in `results/e4_profile_foh.md`). README section
  "E4, fifth pass". `src/lfo/` unchanged.
- Results:
  - Hold vs reduction: with the band model and the reduction, p6 η = 0.1 / 0.3 report from the
    consensus in 4/4 groups with all poles (unreduced: 1/4), errors as FOH. At η = 1e-3 the
    band reduction is worse: consensus 1/4, 3–4 false poles, p6 −2 160 × the CRB with closed,
    wrong intervals. The hold is needed at η ≤ 1e-2, the reduction at η ≥ 0.1.
  - FOH layer, oracle 2.7–5.6e-7 (band 5.6e-5–5.9e-4). Noise-free, after restarts: all poles
    p4 95 % (band 90), p6 coarse 100 % (80), p6 fine 100 % (90); test 2.4–7.0e-6 (7–225 ×
    lower). After Adam / polish no gain (30–70 % vs 35–85 %). Runs stop 4–25 × above the FOH
    oracle, so the 1.5 × ratio certificate forms a consensus in only 1–3 of 4 groups.
  - Under noise no gain: η = 1e-2 95 % (band 100), η = 0.1 20 % (45; Fisher p = 0.18). The
    FOH refit of the FOH-trained runs gives the same report and intervals as of the
    band-trained runs. Restarts 1.3–7.6 × slower.
- Decisions: none.
- Next: a precision term for the noise-free certificate; the hold on the E1 data (non-sine
  inputs) and a zero-order-hold variant; noise-dependent release; joint intervals;
  linear-probe poles for deep (RQ4).

### 2026-11-14 — (C) E3 release under noise; E4 hold mismatch and first-order-hold refit; interval fix

- Done: two-stage partial tie under noise (`e2_rescue.py --start release --rule noise`,
  p6 fine, δ = 0.3, ALNO / LFNO, η = 1e-2 and 0.1, 80 runs; summary `e2_rescue_na_rel`,
  stage `rescue_na_rel` in `e2_floor.py`). Diagnosis of the E4 model error (scratch
  scripts). `e4_profile.py --hold foh`: refit, per-run backward order reduction, certificate,
  consensus and order test with the exact first-order-hold response; 32 groups. Fixed the
  profile-interval edge interpolation (now in √(S − min S)); band results rescored from the
  stored profiles. README sections "two-stage partial tie under noise" and "E4, fourth pass";
  third-pass table and bullets corrected.
- Results:
  - Release under noise: η = 1e-2 all poles 20/20 (both forms), certificate 4/4, no false
    pole, as tied. η = 0.1 worse than tied: all poles 25 % (tied 45 %), −2 30–35 % (45 %),
    test error 2.5–3.5 ×, best of 5 0/4 (3/4), 1–3 false poles.
  - The low-noise "discretisation floor" is the input hold: `lfo.data.lti.response` uses a
    piecewise-linear input, the layer a band-limited one. Bias ∝ dt² of the generator
    (p6 −2: 4.3e-2 at N = 1024, 3.0e-3 at 4096), unchanged by decimating the data; with the
    FOH model S at the truth is 1.3e-13 (float32), bias ≈ 1e-11.
  - FOH refit: 32/32 groups consensus + order test passed, all poles identified, no false
    pole, errors 0.5–1.3 × the CRB at every η (band at η = 1e-3: up to 100 ×), coverage
    100/112 (misses: complex poles, the same seeds at every η). 159/160 reduced runs have
    exactly the true poles.
  - Correction to the previous entry: the interval interpolation shrank intervals narrower
    than the first grid step ∝ η². Corrected band coverage: p4 η = 1e-3 0/4, 4/4, 4/4
    (was 0 everywhere); p4 η = 1e-2 3, 4, 4 (was 0, 2, 1); p6 η = 1e-2 4, 1, 4, 4 (was
    1, 0, 4, 4); p6 η = 1e-3 still 0/16. "0/28 cells at η = 1e-3" is 8/28; the complex
    half-width at p6 η = 1e-2 is 8.4e-6, not 4.7e-7. Half-widths are 1.4–2 × the CRB.
- Decisions: the hold of the input is treated as known measurement information (like the
  sample rate) in the refit; the neural layer is unchanged.
- Next: the FOH (or ZOH) response inside the layer, so training sees it; separate hold vs
  reduction (band refit with reduction); noise-dependent release; joint intervals;
  linear-probe poles for deep (RQ4); the report on the E1 data.

### 2026-11-14 — (C) E3 partial tie, second pass (release vs reset); E4 minimal refit and profile intervals

- Done: `e2_rescue.py --start reset` (restarts that also zero R_res) and `--start release`
  (tied run after restarts → release δ → polish → restarts) on the 6 partial-tie conditions
  (release 120 runs; reset stopped at 101 when release finished). New `e4_profile.py`:
  χ² certification → report (consensus, else best run) → minimal tied refit (variable
  projection, float64) → order test S_min − S_best ≤ χ²_Δp(0.99) σ̂² → 95 % profile
  interval per Re μ; 32 groups (p4, p6; η = 1e-3 … 0.3). README sections "partial tie,
  second pass" and "E4, third pass".
- Results:
  - Release brings p6 to the floor: all poles 90 % in all four p6 conditions (from 10–30 %),
    −2 20/20, train/oracle 0.83–0.92, certificate 4/4 groups, no false pole. It fails on
    the same seeds as the tied runs: the tied solution is a minimum of the partial-tie
    problem; Adam from scratch does not find it. Reset stays at 10–30 %, 1.25–1.66 × the
    oracle. Recipe: train tied, then release δ and polish.
  - The minimal refit removes the CRB gap: p6 η = 1e-2 errors 0.75–2.4 × the CRB (8-slot
    15–110 ×), p4 η = 0.1 0.7–1.1 ×. The order test rejects the p6 η = 0.1 consensus that
    lacks −2 (ΔS ≈ 1.8e4 σ̂² vs ≈ 29), and the fallback's real poles get open intervals
    (−1 / −2 unresolved) rather than a wrong pole. One false identified pole in 24 groups
    at η ≥ 1e-2.
  - Limit: at η = 1e-3 the model (discretisation) error exceeds the noise; order test 1/4,
    intervals cover the truth in 0/28 cells, 2 false poles per system.
- Decisions: the combined certificate is not "the looser of ratio / χ²" (the ratio rule
  certifies every noisy run); it is χ² certification + minimal refit + order test.
- Next: add a model-error term to σ̂² (low-noise validity); joint intervals; two-stage
  partial tie under noise; the report on the E1 data; linear-probe poles for deep (RQ4).

### 2026-11-14 — (C) E3 partial tie (H3) and E4 noise-aware restarts / certificate

- Done: `PartialTie` in `e2_train.py` (steady = K(x) over the full band + bounded untied
  residual |R_res| < δ on the low modes; ALNO and LFNO forms; oracle = tied layer) on p4
  (δ = 0.3) and p6 fine (δ = 0.03, 0.3); 120 runs (seeds 10–29, Adam → polish → restarts).
  Noise-aware rule (`e2_rescue.py --rule noise`): σ̂² from 6th-order differences of the
  training outputs, χ²₄(0.99) σ̂² acceptance on S in the noise regime; χ²_p(0.99) σ̂²
  certificate in `e2_floor.py`; 160 restarted noisy runs. `e2_noise.py` reports both
  rules. README sections "E3, layer setting: partial tie (H3)" and "E4, second pass".
- Results:
  - Partial tie removes the truncation floor (test p4 4.7–5.0e-4, p6 0.7–1.8e-4, at the
    tied level). p4 recovery back to tied level (ALNO 75 → 95 %, LFNO 65 → 85 %; tied 90 %),
    certificate 4/4 groups, no false pole. p6 not: all poles 10–30 % (untied 15 %, tied
    90 %); runs stop 1.2–3.2 × above the tied oracle and restarts close little. δ = 0.3
    better than 0.03 for ALNO (30 vs 10 %).
  - Noise-aware restarts move 14–20 of 20 runs at every η (ratio rule: 0 at η ≥ 0.1). All
    poles at η = 1e-2: 100 % on both systems (from 70 %); p6 η = 0.1: 5 → 45 %, −1 4/20 → 20/20,
    −2 1/20 → 9/20; −1 and −2 no longer merge in the consensus (−1.00 in 4/4 groups; −2
    missing). p6 η = 0.3: 0 → 15 %.
  - χ² certificate: η = 1e-2 certifies 19–20 runs, all correct, no false pole; η = 0.1
    certifies all 20 p6 runs (the data do not reject the wrong −2 at 99 %), consensus omits
    −2; η = 1e-3 certifies only 5–6 runs (optimiser spread > χ²_p σ̂²). Errors still
    15–110 × the CRB at η = 1e-2.
- Decisions: none.
- Next: p6 partial tie stuck above the floor (restarts that reset R_res, δ schedule from
  0); combined certificate (looser of ratio / χ²) and per-pole profile intervals; CRB gap;
  linear-probe poles for deep (RQ4); E1 data.

### 2026-11-13 — (C) E3 / E4: recipe with a steady branch, under noise, in deep stacks

- Done: `e2_train.py` conditions with the ALNO / LFNO steady branch (64 / 256 modes) and
  with output noise η = 1e-3 … 3e-1 (train / validation only, one draw per group of 5);
  `e2_noise.py` (CRB of the minimal-form layer, single-run and consensus errors);
  `e3_deep.py` (bounded slots in the E1 deep models); noise figure in `e2_figures.py`.
  240 layer runs (12 conditions × 20 seeds, Adam → polish → restarts) and 20 deep runs.
  README sections "E3, layer setting", "E4", "E3, deep setting"; web page updated (v5).
- Results:
  - Steady branch: complex pairs 100 % throughout. All poles p4 90 → 75 % (ALNO) / 65 %
    (LFNO), with the ALNO failures at the floor (oracle error 1.7e-3 from mode truncation);
    p6 fine 90 → 15 % (both), −2 100 → 15 %, most runs 2–7 × above the oracle.
  - Noise: all poles p6 85 / 70 / 5 / 0 % at η = 1e-3 / 1e-2 / 1e-1 / 3e-1; complex poles
    100 % at every η. Errors 10–2400 × the CRB (flat at the noise-free bias for η ≤ 1e-3).
    At η ≥ 0.1 the consensus merges −1 and −2 into ≈ −0.68; every noisy run is certified
    and restarts move no run (thresholds are ratios of a loss at the noise level).
  - Deep (p4, 5 seeds): no real pole recovered by the weighted metric with either design;
    bounded raises the complex pair 40 → 80 % (ALNO). Test ≈ 1e-2 vs tied oracle 5.9e-4.
- Bug found and fixed: in LFNO a slot pole at μ → 0 makes β / (iω − μ) singular at ω = 0
  (NaN in 2/5 `plr10` runs and in 5/5 runs of the first bounded version). Bounded slots now
  keep decay ≥ 0.01; 3/5 LFNO bounded runs then stay at loss 1.0 (open).
- Decisions: none. Deep pooled metric marked vacuous for dense real slots.
- Next: partial tie R = K + R_res on p6 ALNO / LFNO (H3); noise-aware certificate and
  restart rule (excess loss over an estimated noise level); linear-probe poles for deep
  (RQ4); LFNO dead starts; E1 data.

### 2026-11-13 — (C) E2: truth-free restarts and floor certificate; web page updated

- Done: `e2_rescue.py` (single-slot moves with residue 0 on a grid, 300-iteration L-BFGS
  screen, accept at ≥ 1 % lower training loss, ≤ 3 rounds; 120 polished runs, seeds 10–29)
  and `e2_floor.py` (floor estimate = lowest training loss in a group of 5; certified
  = < 1.5 ×; consensus = effective poles shared by ≥ 2/3 of certified runs). README
  sections added; E1/E2 web page republished (v3).
- Results:
  - Restarts: p6 dt = 20/4096 `damp8_ms` all poles 50 → 90 %, −2 50 → 100 %, train/oracle
    3.5 → 0.93, test 5.2e-5 (oracle 5.6e-5); p4 truth-sized 65 → 95 %; p6 dt = 20/1024
    `damp8_ms` 35 → 80 %, `mixed_off_ms` 40 → 35 % (a damped spare complex slot takes
    part of −2).
  - Certificate: after the restarts the estimate is 0.91 × the oracle on the fine grid and
    the consensus has all four p6 poles in 4/4 groups with no false pole (`damp8_ms`).
    After the polish alone it fails on the fine grid (usually one run per group near the floor).
  - Correction to the previous entry: at dt = 20/1024 the oracle is not the floor of the
    8-slot model (restarted runs reach 0.7 × the oracle). −2 is biased there (median error
    0.054 vs 0.024 on the fine grid), not unidentifiable.
- Decisions: the prior-free recipe is now 8 bounded slots + multi-start + Adam + polish +
  restarts + consensus over 5 runs + dt refinement. Tolerances were set once, not tuned.
- Next: ALNO / LFNO and the deep setting (E3/E4); output noise; the consensus on the E1 data.

### 2026-11-13 — (C) E2: damping-bounded complex slots and L-BFGS polish

- Done: `e2_train.py` variants `damp8(_ms)` / `damp8w(_ms)` (as `bound8`, plus
  Re μ = −c |Im μ| σ(a) on complex slots, c = 0.5 / 1); 100 runs (p4, p6 at dt = 20/1024,
  seeds 10–29; p6 at dt = 20/4096). New `e2_polish.py`: L-BFGS-B polish of saved Adam
  checkpoints (paired), 280 runs polished; README sections added.
- Results:
  - Adam only: p4 `damp8_ms` 65 % all poles (truth-sized `mixed_off_ms` 38 %); p6 absorbers
    gone, but −2 still missed at dt = 20/1024 (15 %).
  - Polish: p4 38 → 74 % (truth-sized), 65 → 85 % (`damp8_ms`); p6 coarse 32 → 38 %;
    p6 fine-grid `damp8_ms` 15 → 50 %, best of 5 1/4 → 3/4 (truth-sized fine grid 45 %, 4/4).
  - Floor certificate (p6): at dt = 20/4096 every run within 1.5 × the oracle floor
    recovers all poles (8/8); at dt = 20/1024 only 20/53 do, as phase 1 predicts.
  - With 8 slots on the coarse grid the test error falls below the oracle (3.0e-4 vs 4.0e-4).
- Decisions: the prior-free recipe is 8 slots (4 real, 4 complex with |Im μ| ≥ 2π/T and
  |Re μ| < 0.5 |Im μ|) + multi-start + Adam + L-BFGS polish + a dt at which the poles are
  identifiable; c = 1 is too loose. Polishing is cheap (seconds) and should follow Adam
  in all later pole-recovery runs.
- Next: estimate the floor without the truth (certificate in practice); restarts for runs
  far from the floor; more best-of-5 groups; carry the recipe to ALNO / LFNO / deep.

### 2026-11-13 — (C) E2: complex-slot bound (prior-free) and fine-grid training; web page updated

- Done: `e2_train.py` variant `bound8(_ms)` (8 slots, complex slots |Im μ| ≥ 2π/T) and
  condition `p6_ms16_T80_N4096` (dt = 20/4096; `make_splits` takes N, `predicted` reads
  `e2_grid.json`); 80 runs. README sections added. E1 web page republished with an E2 section.
- Results:
  - p4: `bound8_ms` matches the truth-sized recipe without the prior (all poles 4/20 single,
    best of 5 3/4; truth-sized 5/20, 2/4); validation–pole-error link restored (0.69).
  - p6: no gain (2/20, 0/4): strongly damped complex slots (|Re| ≈ |Im|) take −2.
  - Fine grid (p6, `mixed_off_ms`, seeds 10–29): −1 75 → 85 %, −2 30 → 35 %, best of 5
    3/4 → 4/4, within noise; test error 4.2e-4 stays 7.5 × above the fine-grid floor
    (5.6e-5): training does not reach the precision the finer grid allows.
- Decisions: the |Im| bound is a partial prior-free replacement for the slot-count
  prior (works for p4, not p6).
- Next: damping bound on complex slots; second-order polish (L-BFGS) to reach the fine-grid
  floor; more best-of-5 groups.

### 2026-11-13 — (C) E2: prior-free slot sizing, grid resolution; E1 report updated

- Done: `e2_train.py` variants `fixed8(_ms)` and `gate8(_ms)` (8 slots, no use of the true
  pole count; `gate8` = learned per-slot gate on Im μ); 100 runs (p4, p6; `_ms` 20 seeds =
  4 groups of 5). `e2_grid.py` (phase-1 analysis over dt = 20/N, N = 256–8192) →
  `results/e2_grid.md`. `E1_report.md`: part 3 qualified, new section "Follow-up: E2",
  limitations and open decision 4; `e1_audit/README.md`: Im μ flatness corrected, E2 note.
- Results:
  - Prior-free variants fail: all poles 0–15 % (truth-sized `mixed_off_ms` 25–30 % on the same
    seeds), best of 5: 0–1/4 groups (truth-sized 2/4, 3/4). Spare complex slots take the slow
    decays (−0.42 + 0.17i); the gate is compensated by Im μ̃ (only g · Im μ̃ matters).
    In p4 validation loss no longer tracks pole error (Spearman 0.13 vs 0.87).
  - Grid: floor ∝ dt, ratios ∝ 1/dt. p6 −1 identifiable with multisine16, T = 80 at
    N = 4096 (11.6) and 8192 (24.3); −2 reaches 8.3 at N = 8192.
- Decisions: the E2 recipe needs the real-pole count; removing that prior is open (learned
  gate rejected). "Practically identifiable" is relative to the model's error level
  (discretisation floor, in practice also noise); stated in the E1 report and the E2 README.
- Next: complex slots bounded away from the real axis (|Im μ| ≥ 2π/T) as a prior-free
  sizing; training on a fine grid for p6 (T = 80, N = 4096); more best-of-5 groups;
  E1/E2 paper decisions (open decision 4).

### 2026-11-13 — (C) E2: best-of-5 selection checked on independent runs; p6 claim revised

- Done: 100 new `mixed_off_ms` runs (seeds 10–59, p4 multisine16 and p6 multisine16
  T = 80); `experiments/C/theory/e2_select.py` (10 disjoint groups of 5 per system,
  lowest validation loss selected) → `results/e2_select.md` / `.json`. README section
  "Best of 5 by validation loss: independent check".
- Results:
  - p4: all poles recovered: single run 38 %, best of 5 **8/10** (bootstrap estimate was
    95 %); in all 8 groups with a good run the selection picked it. Spearman(val loss,
    pole error) +0.84.
  - p6: single 32 %, best of 5 **7/10**; Spearman +0.24. The previous entry's p6 claim
    (selection lowers recovery 10 → 0 %, "identifiability limit") was a 10-seed artefact.
- Decisions: E2 conclusion revised: from generic inits recovery is limited by optimisation
  in both systems; recipe = mixed real-constrained init + multi-start + best of 5 by
  validation loss. p6's fast real poles are weakly identifiable (phase-1 ratio < 2, weak
  loss–pole link), not shown to be non-identifiable. Previous log entry's p6 decision
  superseded.
- Problems / open questions: 10 groups give wide intervals (8/10: 44–97 %); the recipe
  still uses the true number of real poles.
- Next: learned real gate (no prior on the real-pole count); p6 grid-resolution sweep;
  then E1 report / paper text with the E2 results.

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
