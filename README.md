# laplace-fourier-operators

Research code for Laplace–Fourier neural operators: wavenumber-dependent poles,
identifiability of dual-branch operators, and non-periodic inputs with parametric
initial conditions.

Kee-Youn Yoo, Department of Chemical and Biomolecular Engineering, SeoulTech.

## Projects

- **(A)** A Laplace–Fourier neural operator with wavenumber-dependent poles μₙ(k),
  applied to process dynamics with delay and Cahn–Hilliard phase separation.
- **(C)** Identifiability of dual-branch (pole–residue + Fourier multiplier) operators.
- **(D)** Non-periodic input representations and explicit initial conditions.

Plans and the 24-month schedule are in [`docs/`](docs/). Progress is tracked in
[`STATUS.md`](STATUS.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Render the plans (requires [Quarto](https://quarto.org)):

```bash
cd docs && quarto render proposal_A_wavenumber_pole_operator.qmd --to pdf
```

## Papers and results

Working papers (Quarto; render with `quarto render <file>.qmd`):

| project | paper | experiments |
|---|---|---|
| (A) 1 | `experiments/A/paper/a1_paper.qmd` (method; Appendix A: time axis, from D1) | `experiments/A/{layer,deep}/`, `experiments/D/d1_inputs/` |
| (A) 2 | `experiments/A/paper/a2_paper.qmd` (Cahn–Hilliard) | `experiments/A/deep/` |
| (C) | `experiments/C/paper/c_paper.qmd` | `experiments/C/` |
| (D) | `experiments/D/paper/d_paper.qmd` | `experiments/D/{d2_initial_conditions,d3_boundary}/` |

Every experiment script was committed with its predictions before it ran, and every failed prediction is reported
in its folder's README. [`experiments/INDEX.md`](experiments/INDEX.md) lists each script with its
pre-registration commit, result files and recorded compute (regenerate with `python experiments/make_index.py`).

## Reproducing a result

```bash
python experiments/A/deep/a34_domain.py      # writes experiments/A/deep/results/a34_domain.{md,json}
```

- Scripts run from the repository root. Most skip runs already present in their JSON results; delete the JSON to rerun.
- Large data and checkpoints are cached in `runs/` (git-ignored) and regenerated when missing.
- Most deep experiments expect a CUDA GPU. Compute times per script are in the index.
- Seeds are fixed per run, so a rerun on the same hardware and library versions reproduces the numbers.

## Using the layers

```python
from lfo.models import WavenumberPoleOperator1d, set_domain
model = WavenumberPoleOperator1d(t, width=24, n_layers=4, n_poles=6, modes=22, re_max=1.0, k_ref=16.0)
y = model(f)                       # f: (batch, n_x, n_t) forcing on a periodic grid of length 2 pi
set_domain(model, 4 * math.pi)     # the same model on a domain twice as long (inputs on 2 n_x points)
```

## Public release

`python release/make_release.py OUT_DIR --test` (needs `pip install git-filter-repo`) builds the public copy: MIT
license, `src/lfo`, its tests and the paper scripts with their history; see the script's docstring for what is left
out.

## Layout

```
src/lfo/        shared package: models, data generators, metrics
experiments/    per-project scripts and configs (A, C, D)
tests/          tests for shared code
docs/           research plans (Quarto) and figures
```
