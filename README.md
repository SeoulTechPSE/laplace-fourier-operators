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

## Layout

```
src/lfo/        shared package: models, data generators, metrics
experiments/    per-project scripts and configs (A, C, D)
tests/          tests for shared code
docs/           research plans (Quarto) and figures
```
