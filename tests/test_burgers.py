"""Tests for the forced Burgers solver (`lfo.data.burgers`)."""
import math

import numpy as np

from lfo.data import burgers, lti


def _forcing(n_x, n, amp, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.01
    fk = np.zeros((2, n_x // 2 + 1, n), complex)
    for k in (1, 3, 5):
        fk[:, k] = amp * n_x / 2 * (rng.normal(size=(2, 1)) + 1j * rng.normal(size=(2, 1))) * np.sin(2 * t + k)
    return t, fk


def test_small_forcing_gives_the_linear_diffusion_response():
    n_x, n, nu = 32, 301, 0.2
    t, fk = _forcing(n_x, n, 1e-6)
    uk = burgers.solve(fk, 0.01, nu, n_x)
    for k in (1, 3, 5):
        for b in range(2):
            lin = sum(w * lti.response(t, fk[b, k].real if w == 1 else fk[b, k].imag, np.array([-nu * k ** 2 + 0j]),
                                       np.array([1 + 0j])) for w in (1, 1j))
            assert np.abs(uk[b, k] - lin).max() < 1e-6 * np.abs(lin).max()


def test_mean_is_driven_by_the_mean_forcing_only_and_converges_in_dt():
    n_x, nu = 32, 0.1
    t, fk = _forcing(n_x, 201, 0.5)
    fk[:, 0] = n_x * 0.3                                          # constant mean forcing 0.3
    u1 = burgers.solve(fk, 0.01, nu, n_x)
    assert np.allclose(u1[:, 0].real / n_x, 0.3 * t, atol=1e-10)
    # half the step (forcing linearly interpolated), same sample times
    fk2 = np.zeros(fk.shape[:2] + (401,), complex)
    fk2[..., ::2] = fk
    fk2[..., 1::2] = 0.5 * (fk[..., :-1] + fk[..., 1:])
    u2 = burgers.solve(fk2, 0.005, nu, n_x, stride=2)
    assert np.abs(u1 - u2).max() < 1e-6 * np.abs(u2).max()
    assert np.abs(u1[:, 2:]).max() > 0 and math.isfinite(np.abs(u1).max())


def test_initial_field_decays_as_diffusion_at_small_amplitude():
    n_x, nu = 32, 0.1
    fk = np.zeros((1, n_x // 2 + 1, 201), complex)
    u0k = np.zeros((1, n_x // 2 + 1), complex)
    u0k[0, 3] = 1e-6 * n_x / 2
    uk = burgers.solve(fk, 0.01, nu, n_x, u0k=u0k)
    t = np.arange(201) * 0.01
    assert np.allclose(uk[0, 3], u0k[0, 3] * np.exp(-nu * 9 * t), rtol=1e-6, atol=0)


def test_dispersion_rotates_each_mode_at_small_amplitude():
    n_x, nu, delta = 32, 0.1, 0.01
    fk = np.zeros((1, n_x // 2 + 1, 201), complex)
    u0k = np.zeros((1, n_x // 2 + 1), complex)
    u0k[0, [2, 5, 8]] = 1e-6 * n_x / 2
    uk = burgers.solve(fk, 0.01, nu, n_x, u0k=u0k, delta=delta)
    t = np.arange(201) * 0.01
    for k in (2, 5, 8):
        lam = -nu * k ** 2 + 1j * delta * k ** 3                  # -delta u_xxx on e^{ikx}: +i delta k^3
        assert np.allclose(uk[0, k], u0k[0, k] * np.exp(lam * t), rtol=1e-6, atol=0)


def test_kdv_without_viscosity_conserves_energy():
    n_x = 64
    rng = np.random.default_rng(1)
    u0k = np.zeros((1, n_x // 2 + 1), complex)
    u0k[0, 1:6] = (rng.normal(size=5) + 1j * rng.normal(size=5)) * n_x / 2 * 0.3
    fk = np.zeros((1, n_x // 2 + 1, 2001), complex)
    uk = burgers.solve(fk, 0.001, 0.0, n_x, u0k=u0k, delta=0.01)
    u = np.fft.irfft(uk, n=n_x, axis=1)
    e = (u ** 2).sum(axis=1)[0]
    assert np.abs(u[0, :, -1] - u[0, :, 0]).max() > 0.1 * np.abs(u[0, :, 0]).max()   # the field has evolved
    assert np.abs(e / e[0] - 1).max() < 1e-6
