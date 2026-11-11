"""Operator implementations.

Planned modules (shared foundation, months 1-4):
  lno.py    LNO v1 (pole-residue) and v2 (pole-residue transient + FRF steady),
            ported from github.com/qianyingcao/Laplace-Neural-Operator
  alno.py   ALNO backbone (arXiv:2602.12706)
  lfno.py   LFNO (arXiv:2606.07601)
Project A adds the wavenumber-dependent pole layer here once validated.

FNO and TFNO baselines come from the `neuraloperator` package; do not re-implement them.
"""
from .lno import LNO1d, LNO2d, LNO2dV2, PoleResidue1d, PoleResidue2d, Steady2d, Transient2d

__all__ = ["LNO1d", "LNO2d", "LNO2dV2", "PoleResidue1d", "PoleResidue2d", "Steady2d", "Transient2d"]
