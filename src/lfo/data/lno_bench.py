"""Loader for the original LNO benchmark data, with two train/test splits.

The original LNO datasets (Cao et al., arXiv:2303.10528) are an
out-of-distribution benchmark: training inputs come from one forcing family
(e.g. A sin 5t) and validation/test inputs from another (A e^{-0.05t} sin 5t).
We report every result on two splits (STATUS.md decision, 2026-11-12):

``"ood"``  the original split (train 200 / vali 50 / test 130), unchanged.
``"iid"``  the same 380 samples pooled and re-split at random with the same
           sizes, stratified so each split has the same share of both forcing
           families. Train and test then come from the same distribution.
           The split is fixed by ``split_seed`` (default 0), independent of the
           model seed.

Only the split differs between the two; samples and grids are identical.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import torch

__all__ = ["DEFAULT_ROOTS", "find_case", "load_case", "SPLITS"]

SPLITS = ("ood", "iid")
DEFAULT_ROOTS = (Path("/srv/data/lno_drive"),
                 Path(__file__).resolve().parents[3] / "runs" / "data" / "lno_drive")
_PARTS = ("train", "vali", "test")


def find_case(folder: str, root: str | Path | None = None) -> Path:
    """Path to ``<root>/<folder>/Data/data.mat``; searches DEFAULT_ROOTS if root is None."""
    roots = [Path(root)] if root else list(DEFAULT_ROOTS)
    for r in roots:
        p = r / folder / "Data" / "data.mat"
        if p.exists():
            return p
    raise FileNotFoundError(f"{folder}/Data/data.mat not found under {[str(r) for r in roots]}")


def _iid_indices(n_train: int, n_vali: int, n_test: int, seed: int):
    """Stratified random split of the pooled samples.

    Pooled order: original train (family 0), then vali + test (family 1).
    Returns index arrays (train, vali, test) into the pooled arrays.
    """
    rng = np.random.default_rng(seed)
    sizes = np.array([n_train, n_vali, n_test])
    fam0 = rng.permutation(n_train)                        # original training family
    fam1 = n_train + rng.permutation(n_vali + n_test)      # original vali/test family
    # family-0 count per split: proportional, largest-remainder rounding
    want = sizes * n_train / sizes.sum()
    take0 = np.floor(want).astype(int)
    take0[np.argsort(take0 - want)[: n_train - take0.sum()]] += 1
    take1 = sizes - take0                                  # family 1 fills the rest
    cuts0, cuts1 = np.cumsum(take0)[:-1], np.cumsum(take1)[:-1]
    return tuple(np.sort(np.concatenate([a, b]))
                 for a, b in zip(np.split(fam0, cuts0), np.split(fam1, cuts1)))


def load_case(folder: str, split: str = "ood", root: str | Path | None = None, split_seed: int = 0):
    """Load an original LNO case.

    Returns ``(data, grids)``: ``data[part] = (f, u)`` for part in train/vali/test,
    with ``f`` of shape (n, *grid, 1) and ``u`` of shape (n, *grid), float32;
    ``grids`` is ``{"t": ...}`` for 1D cases and ``{"x": ..., "t": ...}`` for 2D.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    m = scipy.io.loadmat(find_case(folder, root))
    f = {p: np.asarray(m[f"f_{p}"], dtype=np.float32) for p in _PARTS}
    u = {p: np.asarray(m[f"u_{p}"], dtype=np.float32) for p in _PARTS}
    if "x_train" in m:   # 1D ODE cases store the time grid as x_train
        grids = {"t": torch.tensor(m["x_train"], dtype=torch.float32)}
    else:
        grids = {"x": torch.tensor(m["x"], dtype=torch.float32), "t": torch.tensor(m["t"], dtype=torch.float32)}

    if split == "iid":
        F = np.concatenate([f[p] for p in _PARTS])
        U = np.concatenate([u[p] for p in _PARTS])
        idx = _iid_indices(*(len(f[p]) for p in _PARTS), seed=split_seed)
        f = {p: F[i] for p, i in zip(_PARTS, idx)}
        u = {p: U[i] for p, i in zip(_PARTS, idx)}

    data = {p: (torch.from_numpy(f[p])[..., None], torch.from_numpy(u[p])) for p in _PARTS}
    return data, grids
