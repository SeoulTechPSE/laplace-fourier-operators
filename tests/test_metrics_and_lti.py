import numpy as np
import pytest

from lfo.data import lti
from lfo.metrics import dispersion_error, match_poles, pole_error, relative_l2, residue_error, seed_spread


def test_first_order_step_response_is_exact():
    a = 2.0
    t = np.linspace(0, 5, 501)
    y = lti.response(t, lti.step(t), mu=[-a], beta=[a])
    assert np.max(np.abs(y - (1 - np.exp(-a * t)))) < 1e-12


def test_ramp_response_matches_analytic():
    a = 1.5
    t = np.linspace(0, 4, 401)
    y = lti.response(t, lti.ramp(t), mu=[-a], beta=[1.0])
    exact = t / a - (1 - np.exp(-a * t)) / a**2
    assert np.max(np.abs(y - exact)) < 1e-12


def test_conjugate_pair_gives_real_damped_oscillation():
    mu = np.array([-0.5 + 3j, -0.5 - 3j])
    beta = np.array([0.5j, -0.5j])
    t = np.linspace(0, 6, 1201)
    y = lti.response(t, np.zeros_like(t), mu, beta, x0=beta)  # homogeneous only
    assert np.max(np.abs(y - np.real(beta @ np.exp(np.outer(mu, t))))) < 1e-12


def test_random_poles_are_stable_and_conjugate_closed():
    mu, beta = lti.random_poles(5, np.random.default_rng(0))
    assert np.all(mu.real < 0)
    assert np.allclose(np.sort_complex(mu), np.sort_complex(np.conj(mu)))
    t = np.linspace(0, 3, 301)
    y = lti.response(t, lti.prbs(t, np.random.default_rng(1)), mu, beta)
    assert np.isrealobj(y)


def test_pole_matching_is_permutation_invariant():
    true = np.array([-1.0, -2 + 3j, -2 - 3j])
    pred = true[[2, 0, 1]] * (1 + 1e-3)
    assert pole_error(pred, true) == pytest.approx(1e-3, rel=1e-6)
    m = match_poles(pred, true)
    assert m.unmatched_pred.size == 0 and m.unmatched_true.size == 0


def test_missing_pole_counts_as_full_error():
    true = np.array([-1.0, -5.0])
    assert pole_error([-1.0], true) == pytest.approx(0.5)


def test_residue_error_uses_location_matching():
    mu_t, b_t = np.array([-1.0, -3.0]), np.array([2.0, 4.0])
    assert residue_error(mu_t[::-1], b_t[::-1], mu_t, b_t) == pytest.approx(0.0)


def test_dispersion_error_for_diffusion():
    k = np.arange(1, 17)
    lam = -0.1 * k**2
    assert dispersion_error(lam * 1.01, lam) == pytest.approx(0.01)


def test_relative_l2_and_seed_spread():
    x = np.ones((4, 10))
    assert relative_l2(1.1 * x, x) == pytest.approx(0.1)
    ref = np.array([-1.0, -2.0])
    spread = seed_spread([ref, ref[::-1], ref], ref)
    assert np.allclose(spread, 0.0)
