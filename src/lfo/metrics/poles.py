"""Shared accuracy and pole metrics.

These definitions are used by all three projects (A, C, D) so that results are
comparable. Change them only together with a decision entry in STATUS.md.

All functions accept NumPy arrays or PyTorch tensors and return NumPy values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def _np(x) -> np.ndarray:
    """Convert a tensor/array-like to a NumPy array (detaching torch tensors)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def relative_l2(pred, true, axis=None) -> np.ndarray:
    """Relative L2 error ||pred - true|| / ||true||.

    With ``axis=None`` the norm is over all elements. Pass the non-batch axes
    (e.g. ``axis=(1, 2)``) to get one error per sample, then average yourself.
    """
    p, t = _np(pred), _np(true)
    num = np.sqrt(np.sum(np.abs(p - t) ** 2, axis=axis))
    den = np.sqrt(np.sum(np.abs(t) ** 2, axis=axis))
    return num / den


@dataclass
class PoleMatch:
    """Result of matching predicted poles to reference poles."""

    pred_idx: np.ndarray       # indices into the predicted poles
    true_idx: np.ndarray       # indices into the reference poles, same length
    unmatched_pred: np.ndarray  # predicted poles left over (N > N*)
    unmatched_true: np.ndarray  # reference poles not recovered (N < N*)


def match_poles(mu_pred, mu_true) -> PoleMatch:
    """Match predicted to reference poles by minimum total relative distance.

    Uses the Hungarian algorithm on cost |mu_p - mu_t| / |mu_t|. Handles
    N != N*: the smaller set is matched completely and the rest is reported
    as unmatched.
    """
    p = _np(mu_pred).ravel().astype(complex)
    t = _np(mu_true).ravel().astype(complex)
    scale = np.where(np.abs(t) > 0, np.abs(t), 1.0)
    cost = np.abs(p[:, None] - t[None, :]) / scale[None, :]
    rows, cols = linear_sum_assignment(cost)
    return PoleMatch(
        pred_idx=rows,
        true_idx=cols,
        unmatched_pred=np.setdiff1d(np.arange(p.size), rows),
        unmatched_true=np.setdiff1d(np.arange(t.size), cols),
    )


def pole_error(mu_pred, mu_true, reduce: str = "mean") -> float | np.ndarray:
    """Relative pole error |mu_n - mu_n*| / |mu_n*| after Hungarian matching.

    ``reduce`` is "mean", "max", or "none" (per matched pole, ordered as the
    reference poles). Unrecovered reference poles count as error 1.0.
    """
    p = _np(mu_pred).ravel().astype(complex)
    t = _np(mu_true).ravel().astype(complex)
    m = match_poles(p, t)
    err = np.ones(t.size)
    scale = np.where(np.abs(t[m.true_idx]) > 0, np.abs(t[m.true_idx]), 1.0)
    err[m.true_idx] = np.abs(p[m.pred_idx] - t[m.true_idx]) / scale
    return _reduce(err, reduce)


def residue_error(mu_pred, beta_pred, mu_true, beta_true, reduce: str = "mean"):
    """Relative residue error |beta_n - beta_n*| / |beta_n*| for poles matched by location."""
    p, t = _np(mu_pred).ravel(), _np(mu_true).ravel()
    bp, bt = _np(beta_pred).ravel().astype(complex), _np(beta_true).ravel().astype(complex)
    m = match_poles(p, t)
    err = np.ones(t.size)
    scale = np.where(np.abs(bt[m.true_idx]) > 0, np.abs(bt[m.true_idx]), 1.0)
    err[m.true_idx] = np.abs(bp[m.pred_idx] - bt[m.true_idx]) / scale
    return _reduce(err, reduce)


def dispersion_error(mu_k_pred, lam_k_true, reduce: str = "mean"):
    """Dispersion-relation error over wavenumbers.

    ``mu_k_pred``: (K, N) learned poles per wavenumber.
    ``lam_k_true``: (K, N*) true dispersion branches per wavenumber.
    Poles are matched separately at each wavenumber; returns the per-mode mean
    relative pole error, reduced over modes.
    """
    P, T = _np(mu_k_pred), _np(lam_k_true)
    if P.ndim == 1:
        P = P[:, None]
    if T.ndim == 1:
        T = T[:, None]
    per_mode = np.array([pole_error(P[k], T[k], reduce="mean") for k in range(T.shape[0])])
    return _reduce(per_mode, reduce)


def seed_spread(pole_sets, mu_ref) -> np.ndarray:
    """Standard deviation of matched poles across seeds.

    ``pole_sets``: list of pole arrays, one per seed. Each is matched to
    ``mu_ref`` (the truth, or a reference seed). Returns the std of |mu_matched|
    distance from the reference for each reference pole.
    """
    t = _np(mu_ref).ravel().astype(complex)
    devs = []
    for p in pole_sets:
        p = _np(p).ravel().astype(complex)
        m = match_poles(p, t)
        d = np.full(t.size, np.nan, dtype=complex)
        d[m.true_idx] = p[m.pred_idx]
        devs.append(d)
    devs = np.stack(devs)
    return np.nanstd(np.abs(devs - t[None, :]), axis=0)


def _reduce(x: np.ndarray, how: str):
    if how == "mean":
        return float(np.mean(x))
    if how == "max":
        return float(np.max(x))
    if how == "none":
        return x
    raise ValueError(f"unknown reduce: {how}")
