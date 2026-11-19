"""Tests for the pole-residue-delay layer (`lfo.models.delay`)."""
import numpy as np
import torch

from lfo.models.delay import PoleResidueDelay, delayed_foh
from lfo.models.wavenumber import foh_recursion


def test_zero_and_integer_delays():
    torch.manual_seed(0)
    mu = torch.tensor([-0.7 + 0j, -0.2 + 2.0j], dtype=torch.complex128)
    u = torch.randn(3, 120, dtype=torch.float64)
    dt = 0.05
    ref = torch.stack([foh_recursion(m, u, dt) for m in mu], 1)
    y0 = delayed_foh(mu, torch.zeros(2, dtype=torch.float64), u, dt)
    assert torch.allclose(y0, ref, atol=1e-12)
    y7 = delayed_foh(mu, torch.full((2,), 7 * dt, dtype=torch.float64), u, dt)
    assert torch.allclose(y7[..., 7:], ref[..., :-7], atol=1e-12) and y7[..., :7].abs().max() < 1e-14


def test_fopdt_step_response_with_fractional_delay():
    K, tau, theta, dt = 2.0, 1.3, 0.737, 0.05
    t = np.arange(201) * dt
    u = torch.ones(1, 201, dtype=torch.float64)
    m = PoleResidueDelay(1, dt)
    with torch.no_grad():
        m.mu_re.fill_(float(np.log(np.expm1(1 / tau))))          # -softplus(raw) = -1 / tau
        m.beta_re.fill_(K / tau)
        m.theta_raw.fill_(float(np.log(np.expm1(theta))))
        y = m(u)[0].numpy()
    ref = np.where(t > theta, K * (1 - np.exp(-(t - theta) / tau)), 0.0)
    assert np.abs(y - ref).max() < 1e-10


def test_delay_gradient_matches_finite_difference():
    torch.manual_seed(1)
    mu = torch.tensor([-0.5 + 1.0j], dtype=torch.complex128)
    u = torch.randn(2, 80, dtype=torch.float64)
    f = lambda th: delayed_foh(mu, th, u, 0.05).real.pow(2).sum()
    th = torch.tensor([0.4321], dtype=torch.float64, requires_grad=True)
    f(th).backward()
    h = 1e-6
    fd = (f(torch.tensor([0.4321 + h], dtype=torch.float64)) - f(torch.tensor([0.4321 - h], dtype=torch.float64))) / (2 * h)
    assert abs(float(th.grad) - float(fd)) < 1e-5 * max(1.0, abs(float(fd)))


def test_delay_shift_matches_delayed_foh():
    from lfo.models.delay import delay_shift
    from lfo.models.wavenumber import foh_kernel, foh_response
    torch.manual_seed(2)
    mu = torch.tensor([-0.4 + 0.8j, -1.1 + 0j], dtype=torch.complex128)
    u = torch.randn(3, 90, dtype=torch.float64)
    th = torch.tensor(0.6181, dtype=torch.float64)
    g, c = foh_kernel(mu, 90, 0.05)
    x = foh_response(g[None], c[None], u.to(torch.complex128)[:, None])
    y = delay_shift(x, u[:, None], mu[None], th, 0.05)
    assert torch.allclose(y, delayed_foh(mu, th.expand(2), u, 0.05), atol=1e-12)


def test_wavenumber_layer_with_input_delay_shifts_the_diffusion_response():
    import math
    from lfo.models.wavenumber import WavenumberPoleLayer1d
    t = np.linspace(0, 2, 201)
    x = np.arange(16) * 2 * math.pi / 16
    a = np.sin(3 * t) + (t > 0.5)
    f = torch.tensor(np.cos(2 * x)[:, None] * a[None])[None]
    lay = lambda d: WavenumberPoleLayer1d(1, t, mu_degree=(2, 1), beta_degree=1, delay=d)
    l0, l1 = lay(None), lay(0.37)
    for l in (l0, l1):
        with torch.no_grad():
            l.mu_re.copy_(torch.tensor([[0.0, -0.3]], dtype=torch.float64)), l.mu_im.zero_(), l.beta.zero_()
            l.beta[0, 0, 0] = 1.0
    with torch.no_grad():
        y0, y1 = l0(f)[0], l1(f)[0]
    # mode k = 2 response, delayed by 0.37 = 37 samples exactly here (dt = 0.01)
    assert torch.allclose(y1[:, 37:], y0[:, :-37], atol=1e-6) and y1[:, :37].abs().max() < 1e-6
