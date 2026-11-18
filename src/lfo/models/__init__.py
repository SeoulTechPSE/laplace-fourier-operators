"""Operator implementations.

  lno.py     LNO v1 (pole-residue) and v2 (pole-residue transient + FRF steady),
             ported from github.com/qianyingcao/Laplace-Neural-Operator
  hybrid.py  ALNO (arXiv:2602.12706) and LFNO (arXiv:2606.07601), 1D,
             re-implemented from the papers (no public code)
  wavenumber.py  wavenumber-dependent pole layer (project A), 1D and 2D, FOH time axis (D1 decision)

FNO and TFNO baselines come from the `neuraloperator` package; do not re-implement them.
"""
from .hybrid import ALNO1d, LFNO1d, PoleResidueTS1d, SpectralConv1d
from .lno import LNO1d, LNO2d, LNO2dV2, PoleResidue1d, PoleResidue2d, Steady2d, Transient2d
from .wavenumber import (WavenumberPoleConv1d, WavenumberPoleConv2d, WavenumberPoleLayer1d, WavenumberPoleLayer2d,
                         WavenumberPoleOperator1d, WavenumberPoleOperator2d)

__all__ = ["ALNO1d", "LFNO1d", "LNO1d", "LNO2d", "LNO2dV2", "PoleResidue1d", "PoleResidue2d",
           "PoleResidueTS1d", "SpectralConv1d", "Steady2d", "Transient2d", "WavenumberPoleConv1d",
           "WavenumberPoleConv2d", "WavenumberPoleLayer1d", "WavenumberPoleLayer2d",
           "WavenumberPoleOperator1d", "WavenumberPoleOperator2d"]
