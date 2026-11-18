"""Tests for the wavenumber-dependent pole layer (project A, `lfo.models.wavenumber`)."""
import math

import numpy as np
import torch

from lfo.data import lti
from lfo.models.wavenumber import WavenumberPoleLayer1d, foh_kernel, foh_recursion, foh_response


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
