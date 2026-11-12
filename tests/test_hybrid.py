"""Tests for lfo.models.hybrid (ALNO1d, LFNO1d re-implementations)."""
import numpy as np
import torch

from lfo.data import lti
from lfo.models import ALNO1d, LFNO1d, PoleResidue1d, PoleResidueTS1d
from lfo.training import count_params

T = torch.linspace(0, 20.47, 2048)


def _count_complex_once(model):
    """Parameter count as reported in the LFNO / NMI LNO papers (complex entry = 1)."""
    return sum(p.numel() for p in model.parameters())


def test_shapes_and_poles():
    t = torch.linspace(0, 1, 64)
    x = torch.randn(3, 64, 1)
    for m in (ALNO1d(4, 6, 8, t, n_layers=3), LFNO1d(4, 8, t, n_layers=3, n_steady=1, use_grid=True)):
        assert m(x).shape == (3, 64, 1)
        mu, beta = m.poles()[0]
        assert mu.shape == beta.shape and mu.dtype == torch.cfloat


def test_lfno_matches_published_parameter_count():
    # LFNO paper, Table 6 (ODEs: width 4, 16 modes) and Table 7 (3,417 parameters)
    m = LFNO1d(width=4, modes=16, t=T, n_layers=4, n_steady=2)
    assert _count_complex_once(m) == 3417
    assert count_params(m) == 3417 + 4 * 512 + 2 * 256          # complex counted twice (our convention)


def test_single_channel_layer_equals_original_lno_layer():
    # With one channel the original's second channel mixing is trivial, so the
    # textbook layer must reproduce the original LNO v1 layer (transient_sign=-1).
    torch.manual_seed(0)
    t = torch.linspace(0, 2, 50)
    ref = PoleResidue1d(1, 1, 5, t, transient_sign=-1.0)
    new = PoleResidueTS1d(1, 1, 5, t)
    with torch.no_grad():
        new.mu.copy_(ref.weights_pole - 0.5)                         # stable-ish, not conjugate-symmetric
        new.beta.copy_(ref.weights_residue)
        ref.weights_pole.copy_(new.mu)
    x = torch.randn(4, 1, 50)
    with torch.no_grad():
        torch.testing.assert_close(new(x), ref(x), rtol=1e-4, atol=1e-5)


def test_layer_is_exact_for_periodic_input_and_known_poles():
    # For a periodic input, transient + steady with the true poles is the exact LTI
    # response from rest (residue theorem), up to the DFT's periodic-extension error.
    t = np.linspace(0, 20, 2001)[:-1]                                # periodic grid, T = 20
    mu = np.array([-0.8 + 3j, -0.8 - 3j, -1.5])
    beta = np.array([0.5 - 0.2j, 0.5 + 0.2j, 1.0])
    u = np.sin(2 * np.pi * 3 * t / 20) + 0.5 * np.cos(2 * np.pi * 7 * t / 20)
    y = lti.response(t, u, mu, beta)
    layer = PoleResidueTS1d(1, 1, 3, torch.tensor(t, dtype=torch.float32))
    with torch.no_grad():
        layer.mu.copy_(torch.tensor(mu, dtype=torch.cfloat).reshape(1, 1, 3))
        layer.beta.copy_(torch.tensor(beta, dtype=torch.cfloat).reshape(1, 1, 3))
        pred = layer(torch.tensor(u, dtype=torch.float32).reshape(1, 1, -1))[0, 0].numpy()
    assert np.linalg.norm(pred - y) / np.linalg.norm(y) < 1e-2


def test_steady_fold_equals_real_part_of_two_sided_sum():
    torch.manual_seed(1)
    t = torch.linspace(0, 1, 33)                                     # odd length
    layer = PoleResidueTS1d(2, 3, 4, t)
    x = torch.randn(2, 2, 33)
    with torch.no_grad():
        alpha = torch.fft.fft(x)
        iw = (torch.fft.fftfreq(33, (t[1] - t[0]).item()) * 2 * np.pi * 1j)
        H = (layer.beta / (iw[:, None, None, None] - layer.mu)).sum(-1)
        ref = torch.real(torch.fft.ifft(torch.einsum("bil,lio->bol", alpha, H)))
        torch.testing.assert_close(torch.fft.irfft(layer.steady_coeffs(x), n=33), ref, rtol=1e-4, atol=1e-5)


def test_models_train_one_step():
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 64)
    x, y = torch.randn(4, 64, 1), torch.randn(4, 64, 1)
    for m in (ALNO1d(4, 4, 8, t), LFNO1d(4, 8, t)):
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)


def test_stable_pole_init_and_norm_option():
    t = torch.linspace(0, 20.47, 2048)
    m = ALNO1d(4, 16, 16, t)
    mu, _ = m.poles()[0]
    assert (mu.real < 0).all() and (mu.imag >= 0).all()
    assert (LFNO1d(4, 16, t, pole_init="lno").poles()[0][0].real >= 0).all()
    x = torch.randn(2, 2048, 1)
    assert torch.isfinite(LFNO1d(4, 16, t, norm=True)(x)).all()
