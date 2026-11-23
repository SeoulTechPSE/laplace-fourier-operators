"""Operator implementations.

  lno.py     LNO v1 (pole-residue) and v2 (pole-residue transient + FRF steady),
             ported from github.com/qianyingcao/Laplace-Neural-Operator
  hybrid.py  ALNO (arXiv:2602.12706) and LFNO (arXiv:2606.07601), 1D and 2D (x, t),
             re-implemented from the papers (no public code)
  s4d.py     S4D layer, vendored from github.com/state-spaces/s4 (Apache-2.0, LICENSE.state-spaces-s4)
  delay.py   pole-residue-delay layer (project A, process dynamics), exact delayed first-order hold
  wavenumber.py  wavenumber-dependent pole layer (project A), 1D and 2D, FOH time axis (D1 decision)

FNO and TFNO baselines come from the `neuraloperator` package; do not re-implement them.
"""
from .delay import PoleResidueDelay
from .hybrid import ALNO1d, ALNO2d, LFNO1d, LFNO2d, PoleResidueTS1d, PoleResidueTS2d, SpectralConv1d, SpectralConv2d
from .lno import LNO1d, LNO2d, LNO2dV2, PoleResidue1d, PoleResidue2d, Steady2d, Transient2d
from .wavenumber import (WavenumberPoleConv1d, WavenumberPoleConv2d, WavenumberPoleLayer1d, WavenumberPoleLayer2d,
                         WavenumberPoleOperator1d, WavenumberPoleOperator2d, set_domain)

__all__ = ["PoleResidueDelay", "ALNO1d", "ALNO2d", "LFNO1d", "LFNO2d", "PoleResidueTS2d", "SpectralConv2d", "LNO1d", "LNO2d", "LNO2dV2", "PoleResidue1d", "PoleResidue2d",
           "PoleResidueTS1d", "SpectralConv1d", "Steady2d", "Transient2d", "WavenumberPoleConv1d",
           "WavenumberPoleConv2d", "WavenumberPoleLayer1d", "WavenumberPoleLayer2d",
           "WavenumberPoleOperator1d", "WavenumberPoleOperator2d", "set_domain"]
