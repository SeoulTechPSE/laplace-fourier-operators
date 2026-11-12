# (C) E1: pole audit of existing models (RQ1)

Plan: `docs/proposal_C_identifiability.qmd`, RQ1 and Section 6.2 (E1).
Question: on LTI systems with known poles, how close are the poles learned by
LNO v1, ALNO (= LNO v2 per layer) and LFNO to the truth, and how much do they vary over seeds?

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
