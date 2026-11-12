# (C) E1: pole audit of existing models (RQ1)

Plan: `docs/proposal_C_identifiability.qmd`, RQ1 and Section 6.2 (E1).
Question: on LTI systems with known poles, how close are the poles learned by
LNO v1, ALNO (= LNO v2 per layer) and LFNO to the truth, and how much do they vary over seeds?

Report draft: `E1_report.md` (figures: `figures.py` → `figures/`).

Script: `e1_audit.py` (full description in its docstring). Results: `results/<setting>/*.json`
(one per run) and `results/summary.md`; checkpoints in `runs/C/e1_audit/`.

```bash
# layer setting: one single-channel pole-residue layer, 10 seeds
python experiments/C/e1_audit/e1_audit.py --setting layer --protocol common lfno_paper --seeds 0 1 2 3 4 5 6 7 8 9
# deep setting: full architectures at the LFNO paper's budget (~5.9k params), 5 seeds
python experiments/C/e1_audit/e1_audit.py --setting deep --protocol common lfno_paper --seeds 0 1 2 3 4
python experiments/C/e1_audit/e1_audit.py --summarize
```

| Item | Choice |
|---|---|
| Systems | `p2` (one complex pair), `p4` (pair + 2 real), `p6` (2 pairs + 2 real); all stable |
| Grid | T = 20, n_t = 1024, uniform |
| Inputs | equal mix of step, ramp, PRBS, smooth random (non-periodic), data seed 0 |
| Split | 200 / 50 / 100 (train / vali / test), same distribution |
| Oracle | true poles in one layer: test rel. L2 ≈ 1e-3 (floor of the representation) |
| Protocols | `common` (ComplexAdam, original LNO 1D); `lfno_paper` (Adam, wd 0.01, LFNO only) |
| Inits | `lno` (original) and `stable` (`lfo.models.hybrid` default); LNO1d port: `lno` only |
| Metrics | `lfo.metrics.pole_error`, `residue_error`, `seed_spread` after folding to Im ≥ 0 |

Pole conventions: outputs are `Re(...)`, so one learned complex pole can carry a
conjugate pair. Learned and true poles are folded to Im ≥ 0 and the truth reduced to
one pole per pair; for residues the reference of a complex pair is 2β.
Deep setting: all poles of all layers and channels pooled, closest match reported
(optimistic bound).

Layer-setting extra diagnostics: `transient_energy_fraction`
(‖transient‖² / (‖transient‖² + ‖steady‖²) on the test set) and `branch_cancellation`
(‖transient + steady‖ / (‖transient‖ + ‖steady‖); small = the branches cancel).
These are E1 diagnostics, not shared metrics (not in `lfo.metrics`).

## Results (2026-11-12, first full run)

Full table: `results/summary.md`. Layer setting: 10 seeds; deep: 5 seeds.
Oracle floors (test rel. L2, true poles loaded): tied layer ≈ 1e-3; ALNO/LFNO layer with
the true K(iω) on the 16 steady modes ≈ 4.4e-2 (p2) – 6.1e-2 (p6), falling to 4e-3
with 64 modes. The ALNO/LFNO floor is mode truncation, not identifiability.

| finding | evidence (pole error = relative, mean over seeds) |
|---|---|
| Published inits: no model recovers the poles, tied included | `lno`: 90–100 % of poles stay unstable, outputs blow up; `stable`: pole error 5–15 (Im init up to 40 rad/s, poles do not reach the band Im ≤ 4) |
| Deep, `lno` init: nothing trains | LNO v1, ALNO, LFNO test ≈ 1.0 |
| Deep, `stable` init: accurate but poles wrong, consistently | ALNO/LFNO test 1.5–3e-2; closest pooled poles 0.08–0.12 (p2), 0.35–0.41 (p4, p6); seed spread 0.02–0.05 |
| LFNO deep diverges mid-training | 10 of 30 `stable` runs (200–900 epochs, after reaching train ≈ 0.03–0.08) |
| Warm start, tied: truth is a stable point | pole error 0.007 / 0.024 / 0.042 (p2/p4/p6), test = oracle |
| Warm start, ALNO/LFNO (common): stay near the truth | 0.05–0.13, test at the truncation floor |
| Warm start, LFNO paper protocol (wd 0.01): poles drift | 0.05 / 0.24 / 0.49 — weight decay on the poles moves them off the truth |
| Band init: good fit, wrong poles even for tied | tied p6 test 1.2e-2, pole error 0.37; all models 0.14–1.7 |

Reading: in this audit the first obstacle is optimisation (initialisation, weight
decay, instability), before dual-branch non-identifiability. With a warm start the tied
layer identifies the truth (consistent with H2), the untied ones stay close but are
limited by truncation; from generic starts all models reach good-fit/wrong-pole minima.

## Per-pole breakdown and pole travel (2026-11-12)

`per_pole.py` → `results/per_pole.md` (per true pole: median error, fraction recovered
with error < 0.1, signed ΔRe/ΔIm; plus pole travel from the reconstructed initial poles).

| finding | evidence |
|---|---|
| **Pole travel is capped by the learning-rate schedule.** Adam moves a parameter ≈ lr per step, so Σ lr × steps bounds travel: 4.0 (`common`), 3.6 (`lfno_paper`) | `stable` init: median travel 3.1–3.9, 35–92 % of poles at ≥ 90 % of the budget; deep: 63–82 %, ΔIm ≈ −3.9 with ΔRe ≈ 0 — the poles move down in frequency at full speed for the whole run and stop where the budget ends |
| Hence the published-init failure: `stable` starts at \|μ\| ≈ 20 (Im up to 40 rad/s), the band is Im ≤ 4 | layer pole error 5–15; ΔIm of matched poles +2 to +22 |
| Deep poles keep their initial decay rates | of 768–1,024 pooled poles, < 0.3 % have Re < −0.5 (init range [−0.5, −0.05]); the fast real poles −1, −2 are never recovered (0 %) |
| Lightly damped complex pairs are the easy case | −0.3+4i, −0.4+3i: 100 % recovered in deep `stable` (closest pooled pole), 70–100 % with band init |
| Real poles fail in band init (not budget-limited: travel 0.3–0.9) | −1, −2: 0 % recovered, matched pole ≈ −0.5 ± 0.1–0.9i; slow −0.2: matched with spurious Im ≈ 0.2 |
| Weight decay shrinks poles toward 0 | warm + `lfno_paper`: −1 → ΔRe +0.57, −2 → +1.72 (0 % recovered), while `common` keeps them (50–70 %) |
| Untied warm start: fast real poles drift slower | ALNO/LFNO warm: −1, −2 recovered 20–70 % with ΔRe +0.1–0.2; tied warm 70–100 % |

Reading: under the published protocols the "learned" poles of the deep models are their
initial poles shifted by a bounded amount, so the audit mostly measures the
initialisation and the learning-rate schedule. Pole parameterisation (e.g. log decay
rate, pole-specific learning rate) is a prerequisite for any interpretability claim, and
a design input for (A).

## Control: pole learning rate × 10 (2026-11-13)

`--pole-lr-mult 10` (runs labelled `<protocol>+plr10`): the poles get 10 × lr, all else
unchanged, so the travel bound rises from 4.0 to 40 (3.6 → 36 for `lfno_paper`).
Layer 10 seeds, deep 5 seeds; same systems, data and inits (`stable`, `band`; deep LNO v1 `lno`).

| | baseline → plr10 |
|---|---|
| Travel saturation (≥ 90 % of the bound) | 35–92 % → **0 %**; poles now move ΔIm ≈ −15 to −20 (deep, layer `stable`) |
| Layer `stable`, pole error | tied 5.4 / 14.8 / 6.4 → **0.04 / 1.6 / 0.83** (p2/p4/p6); ALNO, LFNO similar trend |
| Layer `stable`, test rel. L2 | tied 0.29 / 0.64 / 0.23 → 0.021 / 0.065 / 0.023 |
| Deep ALNO `stable` | test unchanged (≈ 1.6e-2); pole error 0.12 / 0.35 / 0.38 → 0.05 / 0.23 / 0.32 |
| Deep complex pairs | recovered 80–100 % (closest pooled pole) |
| **Fast real poles −1, −2** | **still 0 %** in every setting except the warm start; matched poles stay at Re ≈ −0.5 (deep ΔRe of all poles ≈ +0.02: decay rates do not move even with budget) |
| Slow real pole −0.2 | deep 20–60 % recovered (was 20–25 %) |
| Deep LNO v1, `lno` init | starts to train: test 0.88 / 0.99 / 0.96 → 0.49 / 0.82 / 0.61 |
| Deep LFNO | less stable: test spread grows (p2 common 0.26 ± 0.34; p6 `lfno_paper` 0.86 ± 1.4), unstable fraction up |

Reading:

1. The travel bound explains the **frequency** failure of the published protocols: once
   the poles can move, oscillatory poles are found, in the layer and (as closest pooled
   poles) in the deep models.
2. It does **not** explain the **real-pole** failure. Fast decays (−1, −2) are never
   identified from generic starts, although the models fit the data (tied p6: test 8e-3
   with −2 missing; deep ALNO 1.6e-2). Their content is carried by other poles and, in the
   deep models, by the rest of the network. A fast decay contributes little output energy
   on [0, 20], so the loss is insensitive to it — a practical-identifiability question
   (Fisher information / profile likelihood; plan Section 4.3, E2).
3. Raising the pole learning rate alone destabilises LFNO; a parameterisation (e.g. log
   decay rate) is the cleaner remedy to test next.

## Practical identifiability of the true poles (2026-11-13)

`identifiability.py` → `results/identifiability.md` / `.json`. Tied layer in minimal form
(one slot per conjugate pair), float64 (agrees with `PoleResidueTS1d` to 2e-7), E1 training
inputs, error e = RMS relative L2. Local: Gauss–Newton at the truth, error caused by a
10 % change of one parameter with the others fixed or re-fitted ("profiled"). Profile:
Re μ of one pole fixed at c × truth, all else re-fitted with L-BFGS from the truth.

| finding | evidence |
|---|---|
| **Fast real poles are practically non-identifiable in p6** | 10 % change of −1 / −2, profiled: e = 4.9e-4 / 1.6e-4 — below the representation floor (8.2e-4); fixed: 7.6e-2 / 3.4e-2, i.e. the other poles compensate 150–200× |
| The loss is flat along Re of −2 over a factor 6 | profile, Re μ from −1 to −6: e ≤ 2.1e-3 (truth 8.2e-4); −1 from −0.25 to −1.25: e ≤ 3.3e-3. Both below the E1 test errors of good-fit runs (≈ 8e-3) |
| In p4 the real poles are weakly identifiable | −1 profiled 6.3e-3 (fixed 1.6e-1); profile not monotone (c = 0.25 lower than 0.5: second basin) |
| Frequencies are always well identified | Im μ of every complex pole: profiled e ≈ 1e-2 – 9e-2 for 10 %; profiles of complex Re μ are sharp |
| Conditioning degrades with the number of poles | cond(F) = 47 (p2), 3.9e4 (p4), 6.8e6 (p6) |
| A real pole sits at a flat point in Im μ | first-order sensitivity to Im μ and Im β is zero at Im μ = 0 — consistent with the spurious Im ≈ 0.2 of learned real poles |

Reading: the real-pole failure of E1 is not an optimisation artefact. With these inputs
and window, the data do not determine the fast decays: the other poles reproduce
their effect to within the discretisation floor of the layer itself. This is the
practical-identifiability side of H2 — unique in principle (warm start converges), flat
in practice. Which input/window changes restore it (fast-content inputs, more samples
near t = 0, finer grid) is the E2 question.
Note: one profile point failed (p4, −0.2, c = 3: L-BFGS returned nan).
