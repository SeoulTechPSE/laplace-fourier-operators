# CLAUDE.md

Guidance for Claude (and humans) working in this repository.

## What this repository is

Research code for three linked projects on Laplace–Fourier neural operators.
The plans live in `docs/`; read the relevant section before starting work.

| ID | Project | Plan |
|---|---|---|
| A | Wavenumber-dependent pole operator (core) | `docs/proposal_A_wavenumber_pole_operator.qmd` |
| C | Identifiability of dual-branch operators | `docs/proposal_C_identifiability.qmd` |
| D | Non-periodic inputs and parametric initial conditions | `docs/proposal_D_nonperiodic_initial_conditions.qmd` |
| — | 24-month integrated schedule and hand-offs | `docs/integrated_schedule_ACD.qmd` |

## Session protocol

Every working session covers one work package from the integrated schedule
(for example "(C) audit E1" or "(D1) non-periodic input"), not a whole project.

1. **Start:** read `STATUS.md` (current state, open decisions, next steps) and the
   plan section for the work package.
2. **Work:** keep changes inside the package's folder unless the change belongs in
   shared code (`src/lfo/`); shared changes need a test.
3. **End:** append an entry to `STATUS.md` (what was done, decisions, results,
   next steps) and commit.

## Layout

```
src/lfo/            shared package (import as `lfo`)
  models/           operator implementations (LNO ports, ALNO, LFNO, proposed layers)
  data/             data generators (synthetic LTI, PDE solvers, process models)
  metrics/          shared metrics: relative L2, pole matching, dispersion error
experiments/A|C|D/  per-project experiment scripts, configs, and notes
experiments/foundation/  shared-foundation work (LNO reproduction, baselines)
tests/              pytest tests for shared code
docs/               research plans (Quarto .qmd) and figures
```

## Conventions

- English text (docs, papers, docstrings, comments, commit messages) uses American spelling
  (-ize, -ization, color, center, modeling, toward).
- Python ≥ 3.10, PyTorch. Baselines (FNO, TFNO) come from `neuraloperator`;
  do not re-implement them.
- **Metrics are defined once, in `src/lfo/metrics/`.** All three projects must use
  these functions so results are comparable. Change a definition only together with
  a `STATUS.md` decision entry.
- Poles are complex tensors; real-valued outputs require conjugate-symmetric poles
  and residues.
- Benchmarks built on the original LNO data are reported on both splits, `ood` (original)
  and `iid` (re-split); see `lfo.data.lno_bench`.
- Every reported number: same parameter budget across models, ≥ 3 seeds,
  mean ± standard deviation.
  Exception: the (C) E1 *layer* setting (one minimal pole-residue layer per model) compares
  poles, not accuracy, and does not match budgets (decision 2026-11-18 in `STATUS.md`).
- Experiment outputs (checkpoints, large arrays) go in `runs/` (git-ignored).
  Commit configs and small result summaries only.
- Run `pytest` before committing changes to `src/lfo/`.

## Git

- Commits are authored as **Kee-Youn Yoo**. Do not add Claude as author and do not
  add `Co-Authored-By` trailers.
- Commit messages: imperative mood, prefixed with the scope, e.g.
  `C: add E1 audit script`, `lfo.metrics: fix pole matching for N≠N*`.
