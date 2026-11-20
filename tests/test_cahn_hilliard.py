"""Tests for the forced 1D Cahn-Hilliard solver (`lfo.data.cahn_hilliard`)."""
import numpy as np

from lfo.data import cahn_hilliard as ch, lti


def test_small_forcing_gives_the_linear_response_including_growing_modes():
    n_x, n, M, kappa = 32, 301, 0.1, 0.04
    t = np.arange(n) * 0.01
    rng = np.random.default_rng(0)
    fk = np.zeros((2, n_x // 2 + 1, n), complex)
    for k in (1, 3, 4, 6):                                    # k = 1, 3, 4 grow (k_c = 5), k = 6 decays
        fk[:, k] = 1e-6 * n_x / 2 * (rng.normal(size=(2, 1)) + 1j * rng.normal(size=(2, 1))) * np.cos(3 * t + k)
    uk = ch.solve(fk, 0.01, M, kappa, n_x)
    for k in (1, 3, 4, 6):
        lam = ch.growth_rate(k, M, kappa)
        assert (lam > 0) == (k < 5)
        for b in range(2):
            lin = lti.response(t, fk[b, k].real, np.array([lam + 0j]), np.array([1 + 0j])) \
                + 1j * lti.response(t, fk[b, k].imag, np.array([lam + 0j]), np.array([1 + 0j]))
            assert np.abs(uk[b, k] - lin).max() < 1e-6 * np.abs(lin).max()


def test_growth_saturates_and_mean_is_conserved():
    n_x, M, kappa = 64, 0.1, 0.04
    n = 2001
    fk = np.zeros((1, n_x // 2 + 1, n), complex)
    fk[0, 3, :200] = 0.5 * n_x / 2                            # a short push on k = 3, then free evolution
    uk = ch.solve(fk, 0.01, M, kappa, n_x)
    u = np.fft.irfft(uk[0], n=n_x, axis=0)
    assert np.abs(u[:, -1]).max() < 1.5                      # saturates near +-1 instead of growing as e^{0.58 t}
    assert np.abs(uk[0, 0]).max() < 1e-9                     # zero mean forcing keeps the mean at zero
    assert np.isfinite(u).all()


def test_2d_small_forcing_gives_the_linear_response():
    n, nt, M, kappa = 16, 201, 0.1, 0.04
    t = np.arange(nt) * 0.01
    fk = np.zeros((1, n, n // 2 + 1, nt), complex)
    a = 1e-6 * n * n / 2 * np.cos(2 * t)
    fk[0, 2, 3] = a                                           # k = (2, 3): |k|^2 = 13, decays
    fk[0, n - 1, 2] = a                                       # k = (-1, 2): |k|^2 = 5, grows
    uk = ch.solve2d(fk, 0.01, M, kappa, n)
    for (i, j), k2 in (((2, 3), 13), ((n - 1, 2), 5)):
        lam = M * k2 * (1 - kappa * k2)
        lin = lti.response(t, a, np.array([lam + 0j]), np.array([1 + 0j]))
        assert np.abs(uk[0, i, j] - lin).max() < 1e-6 * np.abs(lin).max()


def test_initial_field_at_a_mean_composition_grows_with_the_shifted_rate():
    n_x, n, M, kappa, phi0 = 32, 301, 0.1, 0.04, 0.3
    t = np.arange(n) * 0.01
    fk = np.zeros((1, n_x // 2 + 1, n), complex)
    u0k = np.zeros((1, n_x // 2 + 1), complex)
    u0k[0, 0] = phi0 * n_x
    for k in (2, 4):                                          # k_c(0.3) = sqrt(0.73 / 0.04) = 4.27: 2, 4 grow
        u0k[0, k] = 1e-7 * n_x / 2
    u0k[0, 5] = 1e-7 * n_x / 2                               # k = 5 decays
    uk = ch.solve(fk, 0.01, M, kappa, n_x, u0k=u0k)
    assert np.allclose(uk[0, 0], phi0 * n_x, rtol=0, atol=1e-9)
    for k in (2, 4, 5):
        lam = ch.growth_rate(k, M, kappa, phi0)
        assert (lam > 0) == (k < 4.27)
        assert np.allclose(uk[0, k], u0k[0, k] * np.exp(lam * t), rtol=1e-5, atol=0)


def test_default_rest_is_unchanged():
    n_x, n = 32, 101
    rng = np.random.default_rng(1)
    fk = np.zeros((1, n_x // 2 + 1, n), complex)
    fk[0, 3] = 0.1 * n_x / 2 * rng.normal(size=n)
    assert np.array_equal(ch.solve(fk, 0.01, 0.1, 0.04, n_x),
                          ch.solve(fk, 0.01, 0.1, 0.04, n_x, u0k=np.zeros((1, n_x // 2 + 1), complex)))
