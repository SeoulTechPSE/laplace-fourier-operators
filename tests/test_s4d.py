"""Tests for the vendored S4D layer (lfo.models.s4d): the convolution is the diagonal SSM's recursion, causal."""
import torch

from lfo.models.s4d import S4D, S4DKernel


def test_kernel_is_the_zoh_recursion_of_the_diagonal_ssm():
    torch.manual_seed(0)
    k = S4DKernel(3, N=8)
    L = 40
    K = k(L)                                                          # (H, L)
    dt = torch.exp(k.log_dt)[:, None]
    A = -torch.exp(k.log_A_real) + 1j * k.A_imag
    C = torch.view_as_complex(k.C)
    u = torch.randn(3, L)
    x = torch.zeros(3, 4, dtype=torch.cfloat)
    y = []
    for l in range(L):                                                # x_{l+1} = e^{dt A} x_l + (e^{dt A} - 1) / A u_l
        x = torch.exp(dt * A) * x + (torch.exp(dt * A) - 1) / A * u[:, l, None]
        y.append(2 * (C * x).sum(-1).real)
    y = torch.stack(y, -1)
    conv = torch.stack([sum(K[:, j] * u[:, l - j] for j in range(l + 1)) for l in range(L)], -1)
    assert torch.allclose(conv, y, atol=1e-5)


def test_layer_is_causal():
    torch.manual_seed(0)
    m = S4D(4, d_state=8)
    u = torch.randn(2, 4, 30)
    v = u.clone()
    v[..., 20:] += 1.0
    assert torch.allclose(m(u)[0][..., :20], m(v)[0][..., :20], atol=1e-6)
