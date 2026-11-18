"""Tests for the wavenumber-dependent pole layer (project A, `lfo.models.wavenumber`)."""
import math

import numpy as np
import torch

from lfo.data import lti
from lfo.models.wavenumber import WavenumberPoleLayer1d, WavenumberPoleLayer2d, foh_kernel, foh_recursion, foh_response


def test_foh_coefficients_match_the_data_generator():
    """Same exact FOH step as `lfo.data.lti.response`, including the small-|mu dt| series branch."""
    t = np.linspace(0, 2, 101)
    a = np.random.default_rng(3).normal(size=101)
    for mu in (-1.2, -1e-3, -0.4 + 3.0j):
        ref = lti.response(t, a, np.array([complex(mu)]), np.array([1 + 0j]))
        y = foh_recursion(torch.tensor(complex(mu), dtype=torch.complex128), torch.tensor(a), 0.02).real.numpy()
        assert np.abs(y - ref).max() < 1e-8 * max(np.abs(ref).max(), 1e-3)   # lti cancels at |mu dt| < 1e-3


def test_foh_convolution_matches_recursion():
    torch.manual_seed(0)
    mu = torch.complex(-torch.rand(5, dtype=torch.float64) * 3, torch.randn(5, dtype=torch.float64) * 4)
    u = torch.randn(5, 200, dtype=torch.complex128)
    dt = 0.05
    y_conv = foh_response(*foh_kernel(mu, 200, dt), u)
    y_rec = foh_recursion(mu, u, dt)
    assert torch.allclose(y_conv, y_rec, atol=1e-10, rtol=1e-10)


def _diffusion_layer(t, D):
    layer = WavenumberPoleLayer1d(1, t, mu_degree=(2, 1), beta_degree=1)
    with torch.no_grad():
        layer.mu_re.copy_(torch.tensor([[0.0, -D]], dtype=torch.float64))
        layer.mu_im.zero_()
        layer.beta.zero_()
        layer.beta[0, 0, 0] = 1.0
    return layer


def test_true_diffusion_poles_give_the_exact_solution():
    """u_t = D u_xx + f, f = sin(2x) a(t) + cos(5x) b(t), a, b piecewise linear: exact per mode."""
    n_x, n_t, T, D = 32, 101, 2.0, 0.3
    t = np.linspace(0, T, n_t)
    x = np.arange(n_x) * 2 * math.pi / n_x
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=n_t), rng.normal(size=n_t)
    f = np.sin(2 * x)[:, None] * a[None] + np.cos(5 * x)[:, None] * b[None]
    ua = lti.response(t, a, np.array([-D * 4 + 0j]), np.array([1 + 0j]))
    ub = lti.response(t, b, np.array([-D * 25 + 0j]), np.array([1 + 0j]))
    u = np.sin(2 * x)[:, None] * ua[None] + np.cos(5 * x)[:, None] * ub[None]
    with torch.no_grad():
        out = _diffusion_layer(t, D)(torch.tensor(f[None]))[0].numpy()
    assert np.abs(out - u).max() / np.abs(u).max() < 1e-10


def test_layer_is_causal():
    torch.manual_seed(1)
    t = np.linspace(0, 1, 64)
    layer = WavenumberPoleLayer1d(3, t, stable=True)
    f = torch.randn(2, 16, 64, dtype=torch.float64)
    g = f.clone()
    g[..., 40:] += torch.randn(2, 16, 24, dtype=torch.float64)
    with torch.no_grad():
        yf, yg = layer(f), layer(g)
    assert (yf - yg)[..., :40].abs().max() < 1e-12 * yf.abs().max()


def test_layer_transfers_to_a_finer_spatial_grid():
    """Same parameters on 32 and 64 points: a band-limited forcing gives the same field on the shared points."""
    torch.manual_seed(2)
    t = np.linspace(0, 1, 50)
    layer = WavenumberPoleLayer1d(2, t, stable=True)
    a = torch.randn(50, dtype=torch.float64)
    field = lambda n: torch.sin(3 * torch.arange(n, dtype=torch.float64) * 2 * math.pi / n)[:, None] * a[None]
    with torch.no_grad():
        y32, y64 = layer(field(32)[None])[0], layer(field(64)[None])[0]
    assert torch.allclose(y32, y64[::2], atol=1e-12)


def test_mode_truncation_and_growth_cap():
    t = np.linspace(0, 5, 101)
    layer = WavenumberPoleLayer1d(1, t, modes=5, re_max=0.5)
    with torch.no_grad():
        layer.mu_re.fill_(1.0)                                   # Re mu = 1 + k^2 + k^4 without the cap
        k = layer.wavenumbers(32)[:5]
        re = layer.poles(k)[0].real
        f = torch.zeros(1, 32, 101, dtype=torch.float64)
        f[0, :, 10] = torch.cos(9 * torch.arange(32, dtype=torch.float64) * 2 * math.pi / 32)   # only k = 9 (dropped)
        out = layer(f)
    assert re.max() <= 0.5 and torch.isfinite(re).all()
    assert out.abs().max() < 1e-12


def test_even_imaginary_powers_give_the_beam_velocity():
    """Beam u_tt = -c^2 u_xxxx + f, velocity output: poles +-i c k^2, residues 1/2 (im_powers with k^2)."""
    n_x, n_t, c = 32, 201, 0.1
    t = np.linspace(0, 4, n_t)
    x = np.arange(n_x) * 2 * math.pi / n_x
    a = np.random.default_rng(4).normal(size=n_t)
    f = np.sin(3 * x)[:, None] * a[None]
    w = c * 9
    v = lti.response(t, a, np.array([1j * w, -1j * w]), np.array([0.5 + 0j, 0.5 + 0j]))
    layer = WavenumberPoleLayer1d(2, t, mu_degree=(1, 3), beta_degree=1, im_powers=(1, 2, 3))
    with torch.no_grad():
        layer.mu_re.zero_(), layer.mu_im.zero_(), layer.beta.zero_()
        layer.mu_im[0, 1], layer.mu_im[1, 1] = c, -c
        layer.beta[:, 0, 0] = 0.5
        out = layer(torch.tensor(f[None]))[0].numpy()
    ref = np.sin(3 * x)[:, None] * v[None]
    assert np.abs(out - ref).max() < 1e-10 * np.abs(ref).max()


def _adr2d_layer(t, D, v, kr):
    layer = WavenumberPoleLayer2d(1, t, re_degree=2, beta_degree=1)
    with torch.no_grad():
        layer.mu_re.copy_(torch.tensor([[-kr, -D]], dtype=torch.float64))
        layer.adv.copy_(-torch.tensor([v], dtype=torch.float64))
        layer.beta.zero_()
        layer.beta[0, 0, 0] = 1.0
    return layer


def test_2d_true_poles_give_the_exact_adr_solution():
    """u_t = D lap u - v . grad u - k_r u + f: each Fourier mode e^{i k.x} has lambda = -D |k|^2 - i v.k - k_r.
    Forcing with modes (2, 1) and (-3, 0) (cos, so both signs of k), piecewise linear in time."""
    n, n_t, T, D, v, kr = 16, 81, 2.0, 0.2, (0.7, -0.4), 0.3
    t = np.linspace(0, T, n_t)
    x = np.arange(n) * 2 * math.pi / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=n_t), rng.normal(size=n_t)
    lam = lambda kx, ky: -D * (kx ** 2 + ky ** 2) - 1j * (v[0] * kx + v[1] * ky) - kr
    # complex mode response from the real-valued generator: Re(x) with beta = 1, Im(x) = Re(-i x) with beta = -i
    resp = lambda kx, ky, p: sum(w * lti.response(t, p, np.array([lam(kx, ky)]), np.array([bb]))
                                 for w, bb in ((1, 1 + 0j), (1j, -1j)))
    f = np.cos(2 * X + Y)[..., None] * a + np.cos(-3 * X)[..., None] * b
    u = (0.5 * (np.exp(1j * (2 * X + Y))[..., None] * resp(2, 1, a) + np.exp(-1j * (2 * X + Y))[..., None] * resp(-2, -1, a))
         + 0.5 * (np.exp(-3j * X)[..., None] * resp(-3, 0, b) + np.exp(3j * X)[..., None] * resp(3, 0, b))).real
    with torch.no_grad():
        out = _adr2d_layer(t, D, v, kr)(torch.tensor(f[None]))[0].numpy()
    assert np.abs(out - u).max() / np.abs(u).max() < 1e-10


def test_2d_layer_reduces_to_1d_for_fields_constant_in_y():
    torch.manual_seed(6)
    t = np.linspace(0, 1, 40)
    l1 = WavenumberPoleLayer1d(2, t, mu_degree=(3, 1), beta_degree=2, stable=True)
    l2 = WavenumberPoleLayer2d(2, t, re_degree=3, beta_degree=2, stable=True)
    with torch.no_grad():
        l2.mu_re.copy_(l1.mu_re), l2.beta.copy_(l1.beta + 0.1 * torch.randn_like(l1.beta))
        l1.beta.copy_(l2.beta)
        l2.adv[:, 0] = l1.mu_im[:, 0]
        l2.adv[:, 1] = 0.7
        fk = torch.fft.rfft(torch.randn(3, 16, 40, dtype=torch.float64), dim=1)
        fk[:, 8] = 0                                    # no Nyquist mode: its k_x sign differs (rfft +8, fft -8)
        f = torch.fft.irfft(fk, n=16, dim=1)
        y1 = l1(f)
        y2 = l2(f[:, :, None, :].expand(3, 16, 8, 40).contiguous())
    assert torch.allclose(y2, y1[:, :, None, :].expand_as(y2), atol=1e-11)


def test_2d_layer_is_causal_and_truncates():
    torch.manual_seed(7)
    t = np.linspace(0, 1, 50)
    layer = WavenumberPoleLayer2d(2, t, stable=True, modes=3)
    f = torch.randn(2, 12, 10, 50, dtype=torch.float64)
    g = f.clone()
    g[..., 30:] += torch.randn(2, 12, 10, 20, dtype=torch.float64)
    with torch.no_grad():
        yf, yg = layer(f), layer(g)
        x = torch.arange(12, dtype=torch.float64) * 2 * math.pi / 12
        h = torch.cos(4 * x)[:, None, None].expand(12, 10, 50)[None].contiguous()   # k_x = 4 >= modes
        yh = layer(h)
    assert (yf - yg)[..., :30].abs().max() < 1e-12 * yf.abs().max()
    assert yh.abs().max() < 1e-12


def test_2d_isotropic_imaginary_terms_keep_the_output_real_consistent():
    """With Im beta and an isotropic Im mu power, the k_y = 0 row stays Hermitian: the layer's output equals the
    real part of the same operator applied mode by mode to the full (complex) 2D spectrum."""
    torch.manual_seed(8)
    n, n_t = 8, 30
    t = np.linspace(0, 1, n_t)
    layer = WavenumberPoleLayer2d(2, t, re_degree=2, im_powers=(2,), beta_degree=2, stable=True)
    with torch.no_grad():
        layer.beta.normal_()
        f = torch.randn(2, n, n, n_t, dtype=torch.float64)
        out = layer(f)
        kx = 2 * math.pi * torch.fft.fftfreq(n, 2 * math.pi / n, dtype=torch.float64)
        KX, KY = torch.meshgrid(kx, kx, indexing="ij")
        # full spectrum: evaluate on the half plane s(k) = +1 and conjugate the other half
        pos = (KY > 0) | ((KY == 0) & (KX >= 0))
        mu, beta = layer.poles(torch.where(pos, KX, -KX), torch.where(pos, KY, -KY))
        mu, beta = torch.where(pos, mu, mu.conj()), torch.where(pos, beta, beta.conj())
        g, c = foh_kernel(mu, n_t, float(t[1] - t[0]))
        g, c = (beta[..., None] * g).sum(0), (beta[..., None] * c).sum(0)
        ref = torch.fft.ifft2(foh_response(g[None], c[None], torch.fft.fft2(f, dim=(1, 2))), dim=(1, 2))
    # drop k = 0 (beta(0) may be complex; it is its own partner) and the Nyquist lines (no partner on the grid)
    drop = (KX.abs() == n / 2) | (KY.abs() == n / 2) | ((KX == 0) & (KY == 0))
    fk = torch.fft.fft2(f, dim=(1, 2))
    fk[:, drop] = 0
    f2 = torch.fft.ifft2(fk, dim=(1, 2)).real
    with torch.no_grad():
        out = layer(f2)
        ref = torch.fft.ifft2(foh_response(g[None], c[None], torch.fft.fft2(f2, dim=(1, 2))), dim=(1, 2))
    assert ref.imag.abs().max() < 1e-10 * ref.real.abs().max()
    assert torch.allclose(out, ref.real, atol=1e-10)
