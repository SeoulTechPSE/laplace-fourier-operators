# laplace-fourier-operators

Code and reproduction scripts for three working papers on Laplace–Fourier neural operators: wavenumber-dependent
poles, whether learned poles are physical, and non-periodic inputs with initial conditions.

Kee-Youn Yoo, Department of Chemical and Biomolecular Engineering, SeoulTech. MIT License, version 0.0.1.

## Papers

| | paper | experiments |
|---|---|---|
| (A) | `experiments/A/paper/a_paper.qmd`: a neural operator with wavenumber-dependent poles μₙ(k) | `experiments/A/{layer,deep,process}/` |
| (C) | `experiments/C/paper/c_paper.qmd`: do Laplace neural operators learn physical poles? | `experiments/C/{e1_audit,e5_probe}/` |
| (D) | `experiments/D/paper/d_paper.qmd`: non-periodic inputs, initial conditions, boundaries | `experiments/D/{d1_inputs,d2_initial_conditions,d3_boundary}/` |

Render a paper with [Quarto](https://quarto.org): `quarto render experiments/A/paper/a_paper.qmd`. Every number in a
paper comes from the README and result files of its experiment folders.

## Pre-registration

Every experiment script was committed with its predictions before it ran, and every failed prediction is reported
in its folder's README. [`experiments/INDEX.md`](experiments/INDEX.md) lists each script with the commit that
added it (its pre-registration), its result files and recorded compute.

This repository is a path-filtered copy of the development repository: the history keeps every commit that touched
the released files, with its original date, so the order "predictions, then results" can be checked with
`git log`. Commit hashes changed in the filtering. The READMEs quote the development hashes;
[`HASHES.md`](HASHES.md) maps each one to its commit here.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Nine tests in `tests/test_lno.py` compare the ported LNO with the original code bit for bit; they need that
code in `runs/ref/LNO` and a CUDA GPU, and are skipped otherwise.

## Reproducing a result

```bash
python experiments/A/deep/a34_domain.py      # writes experiments/A/deep/results/a34_domain.{md,json}
```

- Scripts run from the repository root. Most skip runs already present in their JSON results; delete the JSON to
  rerun.
- Large data and checkpoints are cached in `runs/` (git-ignored) and regenerated when missing.
- Most deep experiments expect a CUDA GPU. Compute times per script are in the index.
- Seeds are fixed per run, so a rerun on the same hardware and library versions reproduces the numbers.
- The (C) nonlinear experiments use the original LNO benchmark data
  ([qianyingcao/Laplace-Neural-Operator](https://github.com/qianyingcao/Laplace-Neural-Operator)); place it in
  `runs/data/lno_drive` (see `experiments/foundation/README.md` and `lfo.data.lno_bench`).
- `experiments/C/theory/` holds only the modules that the released scripts import. The identification study they
  come from (the E2 paragraph of the (C) paper) is a separate paper and not part of this release.

## Using the layers

```python
from lfo.models import WavenumberPoleOperator1d, set_domain
model = WavenumberPoleOperator1d(t, width=24, n_layers=4, n_poles=6, modes=22, re_max=1.0, k_ref=16.0)
y = model(f)                       # f: (batch, n_x, n_t) forcing on a periodic grid of length 2 pi
set_domain(model, 4 * math.pi)     # the same model on a domain twice as long (inputs on 2 n_x points)
```

## Layout

```
src/lfo/        shared package: models, data generators, metrics, training utilities
experiments/    per-paper scripts, READMEs and result summaries (A, C, D)
tests/          tests for the shared package
```
