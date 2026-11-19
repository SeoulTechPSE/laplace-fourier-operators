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
