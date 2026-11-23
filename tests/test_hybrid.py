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


# ---------------------------------------------------------------- 2D (x, t)
def _grids(nx=16, nt=24):
    return torch.linspace(0, 2 * np.pi * (1 - 1 / nx), nx), torch.linspace(0, 2.3, nt)


def test_2d_shapes_and_poles():
    from lfo.models import ALNO2d, LFNO2d
    x, t = _grids()
    f = torch.randn(2, 16, 24, 1)
    for m in (ALNO2d(4, 2, 3, 3, 4, x, t, n_layers=2), LFNO2d(4, 2, 3, 3, 4, x, t, n_layers=3, n_steady=1)):
        assert m(f).shape == (2, 16, 24, 1)
        mu1, mu2, beta = m.poles()[0]
        assert mu1.shape == (4, 4, 2) and mu2.shape == (4, 4, 3) and beta.shape == (4, 4, 2, 3)
        assert (mu2.real < 0).all()


def test_2d_single_channel_matches_original_lno_layer():
    """One channel: no channel quirk, so transient and steady equal the original PoleResidue2d's (sign +1)."""
    from lfo.models import PoleResidue2d, PoleResidueTS2d
    x, t = _grids()
    torch.manual_seed(0)
    ours = PoleResidueTS2d(1, 1, 2, 3, x, t)
    ref = PoleResidue2d(1, 1, 2, 3, x, t, transient_sign=1.0)
    with torch.no_grad():
        ref.weights_pole1.copy_(ours.mu1)
        ref.weights_pole2.copy_(ours.mu2)
        ref.weights_residue.copy_(ours.beta)
    v = torch.randn(3, 1, 16, 24)
    full = ref(v)
    alpha = torch.fft.fft2(v)
    lam1 = 2j * np.pi * torch.fft.fftfreq(16, (x[1] - x[0]).item())[:, None, None, None]
    lam2 = 2j * np.pi * torch.fft.fftfreq(24, (t[1] - t[0]).item())[:, None, None, None]
    steady_ref, _ = ref.output_PR(lam1, lam2, alpha)                         # (b, out, n_x, n_t), fft2 order
    steady_field = torch.real(torch.fft.ifft2(steady_ref, s=(16, 24)))
    assert torch.allclose(ours.transient(v), full - steady_field, atol=1e-4)
    lam = ours.steady_coeffs(v, 3, 4)                                         # rows 0..2 and 13..15, cols 0..3
    rows = [0, 1, 2, 13, 14, 15]
    assert torch.allclose(lam, steady_ref[:, :, rows, :4], rtol=1e-4, atol=1e-4)


def test_spectral_conv2d_identity_on_band_limited_field():
    from lfo.models import SpectralConv2d
    conv = SpectralConv2d(1, 1, 3, 4)
    with torch.no_grad():
        conv.weight.fill_(1.0)
    a = torch.zeros(1, 1, 16, 24 // 2 + 1, dtype=torch.cfloat)
    a[0, 0, 1, 2], a[0, 0, 15, 3], a[0, 0, 2, 0] = 1 + 2j, 0.5j, 0.7
    v = torch.fft.irfft2(a, s=(16, 24))
    assert torch.allclose(conv(v), v, atol=1e-6)
