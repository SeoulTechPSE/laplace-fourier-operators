"""Tests for lfo.data.lno_bench (original LNO data, ood and iid splits)."""
import numpy as np
import pytest
import scipy.io

from lfo.data import lno_bench
from lfo.data.lno_bench import _iid_indices


def test_iid_indices_partition_and_stratify():
    tr, va, te = _iid_indices(200, 50, 130, seed=0)
    assert (len(tr), len(va), len(te)) == (200, 50, 130)
    allidx = np.concatenate([tr, va, te])
    assert np.array_equal(np.sort(allidx), np.arange(380))       # a partition
    share = [np.mean(s < 200) for s in (tr, va, te)]              # family-0 share
    assert np.allclose(share, 200 / 380, atol=0.01)


def test_iid_indices_depend_only_on_seed():
    a, b = _iid_indices(200, 50, 130, 0), _iid_indices(200, 50, 130, 0)
    c = _iid_indices(200, 50, 130, 1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])


def _have(folder):
    try:
        lno_bench.find_case(folder)
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _have("1D_Duffing_c0"), reason="LNO data not available")
def test_ood_is_original_and_iid_is_a_resplit():
    m = scipy.io.loadmat(lno_bench.find_case("1D_Duffing_c0"))
    ood, grids = lno_bench.load_case("1D_Duffing_c0", "ood")
    iid, _ = lno_bench.load_case("1D_Duffing_c0", "iid")
    assert grids["t"].numel() == 2048
    assert np.array_equal(ood["test"][1].numpy(), m["u_test"].astype(np.float32))
    assert ood["train"][0].shape == (200, 2048, 1)
    pool = lambda d: np.sort(np.concatenate([d[p][1].numpy()[:, 100] for p in ("train", "vali", "test")]))
    assert np.array_equal(pool(ood), pool(iid))                   # same samples
    assert {p: len(iid[p][0]) for p in iid} == {"train": 200, "vali": 50, "test": 130}


@pytest.mark.skipif(not _have("2D_Diffusion"), reason="LNO data not available")
def test_2d_case_shapes():
    data, grids = lno_bench.load_case("2D_Diffusion", "iid")
    assert data["train"][0].shape == (200, 50, 50, 1) and data["train"][1].shape == (200, 50, 50)
    assert set(grids) == {"x", "t"}
