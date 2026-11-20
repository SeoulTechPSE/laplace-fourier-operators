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


def test_2d_unforced_initial_field_matches_forced_zero_and_linear_rate():
    n, steps, M, kappa, phi0 = 16, 100, 0.1, 0.04, 0.3
    u0k = np.zeros((2, n, n // 2 + 1), complex)
    u0k[:, 0, 0] = phi0 * n * n
    u0k[0, 2, 1] = 1e-7 * n * n / 2                          # |k|^2 = 5: grows
    u0k[1, 0, 4] = 1e-7 * n * n / 2                          # |k|^2 = 16: decays
    free = ch.solve2d(None, 0.01, M, kappa, n, u0k=u0k, n_steps=steps)
    zero = ch.solve2d(np.zeros((2, n, n // 2 + 1, steps + 1), complex), 0.01, M, kappa, n, u0k=u0k)
    assert np.allclose(free, zero, rtol=0, atol=1e-12 * n * n)
    t = np.arange(steps + 1) * 0.01
    for b, (i, j), k2 in ((0, (2, 1), 5), (1, (0, 4), 16)):
        lam = M * k2 * (1 - 3 * phi0 ** 2 - kappa * k2)
        assert np.allclose(free[b, i, j], u0k[b, i, j] * np.exp(lam * t), rtol=1e-5, atol=0)
    assert np.allclose(free[:, 0, 0], phi0 * n * n)          # mean conserved


def test_2d_free_torch_matches_numpy_in_the_nonlinear_regime():
    n, steps, M, kappa = 16, 200, 0.1, 0.04
    rng = np.random.default_rng(0)
    u0k = np.fft.rfft2(0.5 * rng.normal(size=(2, n, n)), axes=(1, 2))
    ref = ch.solve2d(None, 0.01, M, kappa, n, stride=50, u0k=u0k, n_steps=steps)
    got = ch.solve2d_free_torch(u0k, 0.01, M, kappa, n, steps, stride=50).numpy()
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-10 * np.abs(ref).max()


def test_2d_long_unstable_run_stays_a_real_field():
    n, M, kappa, dt, steps = 16, 1.0, 0.1, 0.01, 2500           # k_c = 3.2 < n / 4; lambda_max t = 62
    rng = np.random.default_rng(1)
    u0k = np.fft.rfft2(0.05 * rng.normal(size=(1, n, n)), axes=(1, 2))
    uk = ch.solve2d_free_torch(u0k, dt, M, kappa, n, steps, stride=steps).numpy()[..., -1]
    proj = np.fft.rfft2(np.fft.irfft2(uk, s=(n, n), axes=(1, 2)), axes=(1, 2))
    assert np.isfinite(uk).all()
    assert np.abs(uk - proj).max() < 1e-8 * np.abs(uk).max()
    assert np.abs(np.fft.irfft2(uk, s=(n, n), axes=(1, 2))).max() < 1.2      # phase separated, bounded
