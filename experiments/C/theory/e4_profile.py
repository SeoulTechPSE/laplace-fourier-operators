"""(C) E4: minimal-form refit, order test and per-pole profile intervals.

The chi-square certificate (e2_floor.py, rule ``noise``) under-certifies at low noise: the
spread of S between runs is set by the optimiser, so often only the best run is certified and
the consensus (>= 2 certified runs) is empty. A union with the ratio rule does not help: the
1.5 x ratio certifies every run whenever the loss is at the noise level. Here, per group of 5
runs after the noise-aware restarts (stage ``rescue_na``), truth-free:

  certified   S - S_min <= chi2_p(0.99) sigma^2 (as e2_floor.py), noise level from the outputs
  report      consensus of the certified runs if >= 2, else the effective poles of the best run
  refit       the report as a minimal tied layer (one slot per reported pole, real ones real):
              residues first with the poles frozen, then all parameters (L-BFGS). The 8-slot runs
              split a decay over several real slots under noise; the refit removes that freedom
  order test  S_minimal - S_best <= chi2_dp(0.99) sigma^2 (dp = parameters removed): the report
              explains the data as well as the best 8-slot run. If the consensus fails, the best
              run's effective poles are tried
  profile     for each fitted pole, Re mu fixed at c x its value (c = exp(+-u), u log-spaced in
              [2e-3, ln 3]), all else re-fitted (warm start from the neighbouring c); interval = the
              c range around 1 with S(c) - min S <= chi2_1(0.95) sigma^2 (pointwise 95 % profile
              likelihood interval)
  identified  a fitted pole whose interval closes on both sides

``--hold foh``: the refits use the exact sampled response to a first-order-hold (piecewise-linear)
input, the input model of the data generator (lfo.data.lti.response), instead of the layer's
band-limited one. Per pole, the discrete filter x_{j+1} = E x_j + b0 u_j + b1 u_{j+1}
(E = e^{mu dt}, b0 = I0 - I1 / dt, b1 = I1 / dt) replaces 1 / (i omega - mu) by
(b0 + b1 z_l) / (z_l - E), z_l = e^{i omega_l dt}; the transient term keeps its form. At the truth
this leaves S ~ 1e-13 (float32 data) instead of 8e-7 (p6), and no pole bias: the layer's
"discretisation floor" is the hold mismatch. With foh the certificate and the order test also
use FOH refits. Each run's effective poles are refitted and then reduced by backward elimination
(drop a pole or make a complex pole real, the move that raises S least, while the rise stays below
chi2_dp(0.99) sigma^2); the
reduced refit replaces the run (poles and S) in the certificate, the consensus and the best-run
fallback, and the best full refit is the reference of the order test (dp = its free parameters -
the report's). Earlier versions used the full refits' poles (spare poles drift, unpruned) or the
runs' band-model poles with refitted S (wrong runs certified); both broke the consensus.
Outputs then go to ``results/e4_profile_foh``.
Noise-free conditions (``CLEAN``, stage ``rescue``): sigma^2 is the float32 rounding variance of the
stored outputs (``round_var``), so the chi-square certificate, the order test and the intervals
work at the precision of the data; reported in their own section (no CRB). There the fits minimise
S / (n sigma^2) and end with damped Newton steps (L-BFGS on S ~ 1e-13 stalled), and each reported
pole also gets a bootstrap interval (``bootstrap``: dithers of the stored inputs and fitted outputs),
because the rounding of the inputs, missing from sigma^2, widens the pole spread (e4_bias.py).
``--hold band_red``: the same per-run refit and
reduction with the band-limited model (control: the reduction without the right hold); outputs in
``results/e4_profile_band_red``. Conditions ending in ``_foh`` (runs of the FOH-trained layer,
e2_train.py) are reported in their own section; refit them with ``--hold foh``.

The truth is used only to score: per true pole, whether a fitted pole is within 0.1 with a
closed interval, whether the interval covers the true Re mu, and fitted poles with closed
intervals that match no true pole (false).

    python experiments/C/theory/e4_profile.py --cond p6_ms16_T80_N4096_n1e-1 --group 10
    python experiments/C/theory/e4_profile.py --hold foh --cond p6_ms16_T80_N4096_n1e-3 --group 10
    python experiments/C/theory/e4_profile.py --summarize    # writes results/e4_profile{,_foh}.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_floor as F  # noqa: E402

P, RS, T, E = F.P, F.RS, F.T, F.E
OUT = T.R.RESULTS / "e4_profile"
HOLDS = ("band", "foh", "band_red", "foh_eiv", "foh_ml", "foh_eivb", "foh_eivc")
# foh_eiv: EIV-whitened refits (eiv_weights); foh_ml: EIVML refits; foh_eivb: foh_eiv + calibrated_reduce of the report;
# foh_eivc: foh_eivb + calibrated_reduce of every run before the certificate (calibrated certificate)
N_CAL = 16                                          # bootstrap draws per calibrated elimination step
STAGE, VARIANT = "rescue_na", "damp8_ms"
CLEAN_STAGE = "rescue"                              # noise-free data: ratio-rule restarts
CLEAN = ("p4_ms16_T20", "p6_ms16_T80", "p6_ms16_T80_N4096", "p4_e1_T20", "p6_e1_T20")   # + "_foh" variants
CONDS = [c for c, v in F.CASES if v == VARIANT and F.NOISY(c)]
BAD = 1e30                                          # objective returned for non-finite S
STEPS, U_MIN, C_MAX, VP_ITERS = 12, 2e-3, 3.0, 500  # per side: c = exp(+-u), u log-spaced in [U_MIN, ln C_MAX]
LEVEL = chi2.ppf(0.95, 1)
E1_NOISY = {s: (f"{s}_e1_T20", {float(e): f"{s}_e1_T20_n{e}" for e in es})        # E1 data under noise
            for s, es in (("p4", ("1e-1", "3e-1")), ("p6", ("1e-2", "3e-2", "1e-1")))}
N_BOOT = 32                                         # bootstrap draws per group (noise-free: dithers; noisy: EIV)
EIV = {s: {u: f"{base}_u{u}" for u in us} | {"0": base}                           # input noise, eta_y = 1e-2
       for s, base, us in (("p4", "p4_ms16_T20_n1e-2", ("1e-2", "3e-2", "1e-1")), ("p6", "p6_e1_T20_n1e-2", ("1e-2", "3e-2")))}
EIV_CONDS = {c for d in EIV.values() for c in d.values()}


def round_var(y):
    """Noise-free data: the float32 rounding variance per element of y_i / ||y_i|| (ulp^2 / 12),
    from the stored data alone. At the true poles the FOH refit has S = (1.0-1.3) x n sigma^2 here,
    so it takes the place of the noise estimate (whose 6th differences see the signal and the
    steps of the E1 inputs on clean data)."""
    y = y.numpy()
    u = np.spacing(np.abs(y).astype(np.float32)).astype(np.float64)
    return float(np.mean((u ** 2 / 12).mean(1) / (y.astype(np.float64) ** 2).sum(1)))


def dither(a, rng):
    """a + uniform(-ulp/2, ulp/2) with the float32 ulp of each element (a: float64 tensor)."""
    ulp = np.spacing(np.abs(a.numpy()).astype(np.float32)).astype(np.float64)
    return a + torch.tensor(rng.uniform(-0.5, 0.5, a.shape) * ulp)


def bootstrap(vp, data, t, theta, real, device, n_boot=N_BOOT, seed=0):
    """Noise-free data, truth-free: the fitted outputs are taken as exact; the stored inputs and the
    fitted outputs are dithered by +-ulp/2 (float32) and refitted from theta. Returns the std of each
    Re mu over the refits. The input rounding is missing from sigma^2 (outputs only) and doubles
    the pole spread on some data (e4_bias.py)."""
    x, y = (a.double().cpu() for a in data["train"])
    with torch.no_grad():
        y_hat = (vp.pred(torch.tensor(theta, device=device), real).reshape(y.shape) / vp.scale).cpu()
    rng = np.random.default_rng(seed)
    idx = np.cumsum([0] + [1 if r else 2 for r in real])[:-1]
    out = []
    for _ in range(n_boot):
        vb = VarPro({"train": (dither(x, rng), dither(y_hat, rng))}, t, device, vp.hold)
        vb.s_unit = vp.s_unit
        out.append(vb.fit(theta, real)[0][idx])
    return np.std(out, axis=0, ddof=1)


def noise_var_robust(v):
    """Per-element noise variance of v_i / ||v_i|| from the median absolute 6th difference (robust to
    the few large differences at the steps / switches of the E1 inputs: there the mean-square
    version, e2_rescue.noise_var, reads eta ~ 0.05 on clean inputs, this one 4e-13)."""
    d = torch.diff(v.double(), n=RS.DIFF, dim=1)
    s = 1.4826 * d.abs().median(1).values / math.sqrt(math.comb(2 * RS.DIFF, RS.DIFF))
    return float((s ** 2 / v.double().pow(2).sum(1)).mean())


def eiv_bootstrap(vp, data, t, theta, real, sig2_y, sig2_u, device, n_boot=N_BOOT, seed=0):
    """Noisy data, truth-free errors-in-variables bootstrap (e4_eiv.py): y* = G_fit x~ + e_y*,
    x~* = x~ + e_u*, white Gaussian with the per-element variances sig2_y (outputs, e2_rescue.noise_var)
    and sig2_u (inputs, noise_var_robust) of the normalised samples; refit from theta. Returns the
    mean and std of each Re mu over the refits (bias-corrected estimate 2 theta - mean)."""
    x, y = (a.double().cpu() for a in data["train"])
    with torch.no_grad():
        y_hat = (vp.pred(torch.tensor(theta, device=device), real).reshape(y.shape) / vp.scale).cpu()
    gen = torch.Generator().manual_seed(seed)
    nz = lambda a, s2: a + math.sqrt(s2) * a.norm(dim=1, keepdim=True) * torch.randn(a.shape, generator=gen,
                                                                                    dtype=a.dtype)
    idx = np.cumsum([0] + [1 if r else 2 for r in real])[:-1]
    out = []
    for _ in range(n_boot):
        vb = vp.like({"train": (nz(x, sig2_u), nz(y_hat, sig2_y))})
        out.append(vb.fit(theta, real)[0][idx])
    return np.mean(out, axis=0), np.std(out, axis=0, ddof=1)


def add_eiv(res, profs, vp, data, t, theta, real, sig2, device):
    """Store the EIV bootstrap per reported pole: boot_mean, boot_std, interval_boot = bias-corrected
    Re mu +- 1.96 std."""
    sig2_u = noise_var_robust(data["train"][0])
    bm, bs = eiv_bootstrap(vp, data, t, theta, real, sig2, sig2_u, device)
    res["eta_u_hat"] = float(np.sqrt(data["train"][0].shape[1] * sig2_u))
    for pr, m, sk in zip(profs, bm, bs):
        c = 2 * pr["mu"][0] - float(m)
        pr.update(boot_mean=float(m), boot_std=float(sk), interval_boot=[c - 1.96 * float(sk), c + 1.96 * float(sk)])


def eiv_weights(vp, theta, real, sig2_y, sig2_u):
    """Frequency weights that whiten the errors-in-variables residual. With y = G x and the stored
    x~ = x + e_u, the residual of the true model is e_y - G e_u: per normalised sample i its spectrum
    is v_il = sig2_y + sig2_u (||x_i|| / ||y_i||)^2 |G(i omega_l)|^2 (stationary approximation; the
    transient of e_u is ignored). G from the fit theta: G = 1/2 sum_k (b_k H_k(omega) + conj(b_k H_k(-omega))),
    b_k the (complex) residue of pole k. Returns w = 1 / sqrt(v) (b, n_t), so that S is in units of the
    noise (sigma^2 = 1 in every test). With sig2_u = 0, w = 1 / sigma_y: the unweighted tests."""
    th = torch.tensor(theta, dtype=torch.float64, device=vp.device)
    with torch.no_grad():
        _, c = vp.coef(th, real)
        H = vp.transfer(vp.mu(th, real))                                     # (L, n)
        b, i = [], 0
        for r in real:
            b.append(complex(float(c[i]), 0.0) if r else complex(float(c[i]), float(c[i + 1])))
            i += 1 if r else 2
        bk = torch.tensor(b, dtype=torch.complex128, device=vp.device)
        neg = (-torch.arange(vp.n_t, device=vp.device)) % vp.n_t
        G = 0.5 * ((H * bk).sum(1) + torch.conj((H[neg] * bk).sum(1)))
        ratio = (vp.alpha.abs().pow(2).sum(1) / vp.n_t) * vp.scale[:, 0] ** 2     # ||x_i||^2 / ||y_i||^2
        v = sig2_y + sig2_u * ratio[:, None] * G.abs().pow(2)[None]
    return 1 / v.sqrt()


def load_group(cond, g0, device):
    """Runs of one group (seeds g0 .. g0 + 4): model, S, effective poles."""
    system, family, Tw, Kk, N = T.unpack(cond)
    data, t, mu_true, beta_true = T.setup(cond, g0)
    out, suffix = F.STAGES[STAGE if F.NOISY(cond) else CLEAN_STAGE]
    runs = []
    for s in range(g0, g0 + F.K):
        name = T.run_name(cond, VARIANT, s)
        ck = T.RUNS / f"{name}{suffix}.pt"
        if not ck.exists():
            continue
        model, _ = T.build_for(cond, VARIANT, s, t, mu_true, beta_true)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.to(device)
        mu, beta = (a.detach().cpu().numpy() for a in model.pole_sets())
        runs.append(dict(seed=s, model=model, S=RS.sse(model, data, device), eff=F.effective(mu, beta, Tw)))
    return data, t, mu_true, beta_true, runs


class VarPro:
    """Minimal tied layer in float64 by variable projection. For poles mu_k the layer output is
    Re sum_k beta_k phi_k(x), linear in the residues, with (as PoleResidueTS1d, beta = 1)

        phi_k = -(1 / n_t) (sum_l alpha_l H_kl) e^{mu_k t} + ifft_l(alpha_l H_kl),  H_kl = 1 / (i omega_l - mu_k),

    alpha = fft(x). Residues are solved exactly (weighted least squares, rows scaled by 1 / ||y_i||,
    so the objective is S); only the poles are iterated (L-BFGS on S with autograd gradients).
    Real poles have a real residue (1 basis column), complex poles a complex one (2 columns)."""

    def __init__(self, data, t, device, hold="band", w=None):
        self.hold = hold
        self.w = w                      # (b, n_t) frequency weights 1 / sqrt(residual spectrum), or None
        x, y = (a.to(device).double() for a in data["train"])
        self.n_t = x.shape[1]
        self.alpha = torch.fft.fft(x)
        self.t_np = np.asarray(t)
        self.t = torch.tensor(np.asarray(t), dtype=torch.float64, device=device)
        dt = float(self.t[1] - self.t[0])                                          # i omega_l as _omega, float64
        self.iw = (2j * np.pi * torch.fft.fftfreq(self.n_t, dt, dtype=torch.float64, device=device))
        self.scale = 1 / y.norm(dim=1, keepdim=True)
        self.y = (y * self.scale).reshape(-1)
        self.device = device
        self.b = x.shape[0]
        self.yw = None if w is None else self.whiten(self.y[:, None])[:, 0]
        self.s_unit = 1.0               # fit() minimises S / s_unit (noise-free data: n sigma^2, see run())

    def like(self, data):
        """The same refit model on other data (bootstrap draws)."""
        return VarPro(data, self.t_np, self.device, self.hold, self.w)

    def mu(self, theta, real):
        out, i = [], 0
        for r in real:
            out.append(torch.complex(theta[i], torch.zeros((), dtype=theta.dtype, device=theta.device)) if r
                       else torch.complex(theta[i], theta[i + 1]))
            i += 1 if r else 2
        return torch.stack(out)

    def transfer(self, mu):
        """(L, n) frequency response per pole: band-limited input (the layer) or first-order hold."""
        if self.hold.startswith("band"):
            return 1 / (self.iw[:, None] - mu[None, :])
        dt = self.t[1] - self.t[0]
        z = mu * dt
        Ek = torch.exp(z)
        I0, I1 = (Ek - 1) / mu, (Ek - 1 - z) / mu ** 2       # int_0^dt e^{mu(dt-s)} (1, s) ds
        zl = torch.exp(self.iw * dt)[:, None]
        return ((I0 - I1 / dt)[None] + (I1 / dt)[None] * zl) / (zl - Ek[None])

    def whiten(self, M):
        """(b n_t, m) columns -> ifft(w * fft(.)) per sample (EIV weighting; w even in omega, so real)."""
        m = M.shape[1]
        F_ = torch.fft.fft(M.reshape(self.b, self.n_t, m), dim=1) * self.w[:, :, None]
        return torch.fft.ifft(F_, dim=1).real.reshape(-1, m)

    def S(self, theta, real):
        if self.w is None:
            r = self.y - self.pred(theta, real)
        else:
            A = self.design(theta, real)
            Aw = self.whiten(A)
            r = self.yw - Aw @ self.solve(Aw, self.yw)
        return (r @ r)

    def solve(self, A, y):
        G = A.T @ A
        return torch.linalg.solve(G + 1e-12 * torch.eye(A.shape[1], dtype=G.dtype, device=G.device) * G.diag().max(),
                                  A.T @ y)                                     # small ridge: coinciding poles

    def coef(self, theta, real):
        """Design matrix and residues (weighted least squares when self.w is set)."""
        A = self.design(theta, real)
        if self.w is None:
            return A, self.solve(A, self.y)
        Aw = self.whiten(A)
        return A, self.solve(Aw, self.yw)

    def pred(self, theta, real):
        """Fitted outputs (flattened, rows scaled by 1 / ||y_i||) with the exact residues."""
        A, c = self.coef(theta, real)
        return A @ c

    def design(self, theta, real):
        mu = self.mu(theta, real)
        H = self.transfer(mu)                                                         # (L, n)
        aH = self.alpha[:, :, None] * H[None]                                         # (b, L, n)
        trans = -(aH.sum(1) / self.n_t)[:, :, None] * torch.exp(mu[:, None] * self.t[None])[None]   # (b, n, n_t)
        phi = trans + torch.fft.ifft(aH, dim=1).transpose(1, 2)
        phi = phi * self.scale[:, :, None]
        cols = []
        for k, r in enumerate(real):
            cols.append(phi[:, k].real.reshape(-1))
            if not r:
                cols.append(-phi[:, k].imag.reshape(-1))
        return torch.stack(cols, 1)

    def fit(self, theta0, real, fixed=None, iters=VP_ITERS):
        """argmin_theta S; fixed = index of theta held at its value. Returns (theta, S)."""
        theta0 = np.asarray(theta0, float)
        free = np.ones(theta0.size, bool)
        if fixed is not None:
            free[fixed] = False

        def f(z):
            th = theta0.copy()
            th[free] = z
            tt = torch.tensor(th, dtype=torch.float64, device=self.device, requires_grad=True)
            try:
                val = self.S(tt, real) / self.s_unit
                val.backward()
                g = tt.grad.cpu().numpy()[free]
                if np.isfinite(float(val)) and np.isfinite(g).all():
                    return float(val), g
            except torch.linalg.LinAlgError:
                pass
            return BAD, np.zeros(free.sum())                  # unstable / degenerate poles: line search backs off

        res = minimize(f, theta0[free], jac=True, method="L-BFGS-B",
                       options=dict(maxiter=iters, ftol=0.0, gtol=1e-14, maxcor=20))
        th = theta0.copy()
        th[free] = res.x
        if self.s_unit != 1.0:                  # noise-free data: L-BFGS stalls in flat valleys (see run())
            return self.newton(th, real, free)
        return th, float(res.fun) * self.s_unit

    def newton(self, theta, real, free, steps=30):
        """Damped Newton on the free parameters (autograd Hessian, |eigenvalues|, step halving) after
        L-BFGS. On p6 fine the L-BFGS fit started at the truth stalled 3.5 sigma^2 above the minimum
        that the start from the report reached (Re mu of -1 / -2 off by 3e-9 / 1e-8, ~2 std); a few
        Newton steps bring both starts to the same point."""
        th = torch.tensor(theta, dtype=torch.float64, device=self.device)
        idx = torch.tensor(np.flatnonzero(free), device=self.device)

        def f(z):
            return self.S(th.index_put((idx,), z), real) / self.s_unit

        z = th[idx].clone()
        try:
            S = float(f(z))
            for _ in range(steps):
                zz = z.clone().requires_grad_(True)
                g = torch.autograd.grad(f(zz), zz)[0]
                w, V = torch.linalg.eigh(torch.autograd.functional.hessian(f, z))
                step = -(V @ ((V.T @ g) / w.abs().clamp_min(1e-12 * float(w.abs().max()))))
                lam = 1.0
                while lam > 1e-9 and not float(f(z + lam * step)) < S:
                    lam /= 2
                if lam <= 1e-9:
                    break
                z = z + lam * step
                S_new = float(f(z))
                done, S = S - S_new < 1e-12, S_new
                if done:
                    break
        except torch.linalg.LinAlgError:
            S = float(f(z))
        out = theta.copy()
        out[free] = z.cpu().numpy()
        return out, S * self.s_unit


class EIVML(VarPro):
    """Frequency-domain errors-in-variables ML refit (`--hold foh_ml`). With known noise variances the
    ML cost of y = G x observed as (x~, y~), the true inputs eliminated, is

        V = sum_il |R_il|^2 / v_il(theta),  v_il = sig2_y + sig2_u (||x_i|| / ||y_i||)^2 |G_theta(i omega_l)|^2,

    R_i = fft of the normalised time-domain residual of sample i (the transient of e_u ignored, as in
    eiv_weights). The weights move with theta, so lowering |G| where the input is pure noise no
    longer pays (the plug-in weights of foh_eiv keep that bias). The residues enter v, so they are
    no longer projected out: theta = (pole parameters, residues), fitted jointly; a pole-only theta
    (from pack) gets its residues from the unweighted least squares first. V is in noise units
    (sigma^2 = 1 in every test); V / n ~ 1 at the truth."""

    def __init__(self, data, t, device, hold, sig2_y, sig2_u):
        super().__init__(data, t, device, hold)
        self.sig2_y, self.sig2_u = sig2_y, sig2_u
        self.ratio = (self.alpha.abs().pow(2).sum(1) / self.n_t) * self.scale[:, 0] ** 2
        self.neg = (-torch.arange(self.n_t, device=device)) % self.n_t

    def like(self, data):
        return EIVML(data, self.t_np, self.device, self.hold, self.sig2_y, self.sig2_u)

    @staticmethod
    def n_pole(real):
        return int(sum(1 if r else 2 for r in real))

    def full(self, theta, real):
        """Append the unweighted LS residues to a pole-only theta (numpy)."""
        theta = np.asarray(theta, float)
        if theta.size > self.n_pole(real):
            return theta
        with torch.no_grad():
            A = self.design(torch.tensor(theta, dtype=torch.float64, device=self.device), real)
            c = self.solve(A, self.y)
        return np.concatenate([theta, c.cpu().numpy()])

    def split(self, theta, real):
        k = self.n_pole(real)
        return theta[:k], theta[k:]

    def coef(self, theta, real):
        thp, c = self.split(theta, real)
        return self.design(thp, real), c

    def S(self, theta, real):
        thp, c = self.split(theta, real)
        A = self.design(thp, real)
        R = torch.fft.fft((self.y - A @ c).reshape(self.b, self.n_t), dim=1)
        H = self.transfer(self.mu(thp, real))
        b, i = [], 0
        for r in real:
            b.append(torch.complex(c[i], torch.zeros((), dtype=c.dtype, device=c.device)) if r
                     else torch.complex(c[i], c[i + 1]))
            i += 1 if r else 2
        bk = torch.stack(b)
        G = 0.5 * ((H * bk).sum(1) + torch.conj((H[self.neg] * bk).sum(1)))
        v = self.sig2_y + self.sig2_u * self.ratio[:, None] * G.abs().pow(2)[None]
        return (R.abs().pow(2) / v).sum() / self.n_t                     # Parseval: time-domain units

    def fit(self, theta0, real, fixed=None, iters=VP_ITERS):
        return super().fit(self.full(theta0, real), real, fixed, max(iters, 2000))


def denoise_input(x, sig2_u):
    """Wiener-type shrinkage of the stored inputs, truth-free: per frequency line the mean normalised power
    P_l over the samples, noise power n_t sigma_u^2; gain max(0, 1 - n_t sigma_u^2 / P_l)."""
    X = torch.fft.fft(x, dim=1)
    P = (X.abs().pow(2) / x.pow(2).sum(1, keepdim=True)).mean(0)
    gain = torch.clamp(1 - x.shape[1] * sig2_u / P, min=0)
    return torch.fft.ifft(X * gain[None], dim=1).real


def calibrated_reduce(vp, theta, real, data, sig2_y, sig2_u, device, n_cal=N_CAL, seed=0):
    """Backward elimination of a report with bootstrap-calibrated thresholds (errors-in-variables data).
    Under input noise the drop in S from an extra pole is far above chi2 sigma^2 even when the data
    have no such pole (p4 multisine, eta_u = 10 eta_y, a real pole at -28 added to the true model:
    dS ~ 100-130 against chi2_2(0.99) = 9.2, on simulated truth data). Per step: the cheapest move
    (drop pole k / make complex pole k real) gives dS = S_reduced - S_current; n_cal EIV draws from the
    reduced fit (denoised stored inputs as clean inputs, y* = G_reduced x^ + e_y*, x~* = x^ + e_u*) give
    dS* = S*_reduced - S*_current; the move is taken while dS <= kappa chi2_dp(0.99), kappa = max(1,
    mean dS* / dp). S*_current is the better of two starts (the current fit, and the reduced bootstrap fit
    with the removed pole put back): from the current fit alone it landed above S*_reduced in one group
    (mean dS* < 0). Returns (theta, real, steps)."""
    x, y = (a.double().cpu() for a in data["train"])
    xh = denoise_input(x, sig2_u)
    vpx = vp.like({"train": (xh, y)})
    gen = torch.Generator().manual_seed(seed)
    nz = lambda a, s2: a + math.sqrt(s2) * a.norm(dim=1, keepdim=True) * torch.randn(a.shape, generator=gen,
                                                                                    dtype=a.dtype)
    theta, S_cur = vp.fit(theta, real)
    steps = []
    while len(real) > 1:
        cur = unpack_theta(theta, real)
        moves = [(cur[:k] + cur[k + 1:], ("drop", k)) for k in range(len(cur))]
        moves += [(cur[:k] + [complex(z.real, 0.0)] + cur[k + 1:], ("real", k)) for k, z in enumerate(cur) if z.imag]
        trials = []
        for m, mv in moves:
            a, b = pack(m)
            a, S = vp.fit(a, b)
            trials.append((S, a, b, mv))
        S, a, b, (kind, k) = min(trials, key=lambda q: q[0])

        def nested(th_red):
            """The current model started from a reduced fit (the removed pole put back)."""
            z = unpack_theta(th_red, b)
            z = z[:k] + [cur[k]] + z[k:] if kind == "drop" else z[:k] + [complex(z[k].real, cur[k].imag)] + z[k + 1:]
            return pack(z)[0]
        dp = n_eff(real) - n_eff(b)
        with torch.no_grad():
            yc = (vpx.pred(torch.tensor(a, device=device), b).reshape(y.shape) / vpx.scale).cpu()
        d = []
        for _ in range(n_cal):
            vb = vp.like({"train": (nz(xh, sig2_u), nz(yc, sig2_y))})
            th_r, S_r = vb.fit(a, b)
            S_c = min(vb.fit(theta, real)[1], vb.fit(nested(th_r), real)[1])   # two starts: local minima
            d.append(max(S_r - S_c, 0.0))
        kappa = max(1.0, float(np.mean(d)) / dp)
        thr = kappa * chi2.ppf(0.99, dp)
        ok = bool(S - S_cur <= thr)
        steps.append(dict(poles=[[float(z.real), float(z.imag)] for z in unpack_theta(a, b)], dS=float(S - S_cur),
                          dp=int(dp), dS_boot_mean=float(np.mean(d)), kappa=kappa, threshold=float(thr), removed=ok))
        if not ok:
            break
        theta, real, S_cur = a, b, S
    return theta, real, steps


def pack(poles):
    real = [z.imag == 0 for z in poles]
    theta = [v for z in poles for v in ((z.real,) if z.imag == 0 else (z.real, abs(z.imag)))]
    return np.array(theta), real


def unpack_theta(theta, real):
    out, i = [], 0
    for r in real:
        out.append(complex(theta[i], 0.0) if r else complex(theta[i], theta[i + 1]))
        i += 1 if r else 2
    return out


def n_eff(real):
    """Free real parameters: real pole 2 (decay, residue), complex pole 4."""
    return int(sum(2 if r else 4 for r in real))


def profile(vp, theta, real, k):
    """S(c) with theta[k] (a Re mu) fixed at c x its fitted value, all other poles re-fitted
    (residues exact); continuation outward from c = 1 on each side."""
    pts = {1.0: float(vp.S(torch.tensor(theta, device=vp.device), real))}
    for side in (1, -1):
        th = theta.copy()
        for c in np.exp(side * np.geomspace(U_MIN, np.log(C_MAX), STEPS)):
            th[k] = theta[k] * c
            th, Sc = vp.fit(th, real, fixed=k)
            pts[float(c)] = Sc
    cs = np.array(sorted(pts))
    return cs, np.array([pts[c] for c in cs])


def interval(cs, S, thr):
    """c range around c = 1 with S - min S <= thr; None side = open. The crossing is interpolated
    linearly in sqrt(S - min S) vs log c, which is exact for a quadratic profile. (A first version
    interpolated S itself: when the interval is narrower than the first grid step, U_MIN = 2e-3,
    that shrinks the half-width by about thr / (S(c_1) - min S), i.e. ~ eta^2 instead of eta.)"""
    i0 = int(np.argmin(np.abs(cs - 1.0)))
    ok = S - S.min() <= thr
    out = []
    for step in (-1, 1):
        i = i0
        while 0 <= i + step < len(cs) and ok[i + step]:
            i += step
        j = i + step
        if not (0 <= j < len(cs)):
            out.append(None)
            continue
        a, b = np.log(cs[i]), np.log(cs[j])
        fa, fb = (np.sqrt(max(S[k] - S.min(), 0.0)) - np.sqrt(thr) for k in (i, j))
        out.append(float(np.exp(a + (b - a) * fa / (fa - fb))) if fb != fa else float(cs[j]))
    return out


def reduce(vp, poles, sig2):
    """Refit a run's effective poles, then backward elimination: of the moves "drop pole k" and "make
    complex pole k real", take the one that raises S least, while S_reduced - S_full <= chi2_dp(0.99)
    sigma^2 (dp = parameters removed). Returns
    (reduced poles, S_reduced, S_full, free parameters of the full refit)."""
    th, re_ = pack(list(poles))
    th, S_full = vp.fit(th, re_)
    p_full = n_eff(re_)
    cur, S_cur = unpack_theta(th, re_), S_full
    cur = list(cur)
    while len(cur) > 1:
        trials = []
        moves = [cur[:k] + cur[k + 1:] for k in range(len(cur))]                    # drop pole k
        moves += [cur[:k] + [complex(z.real, 0.0)] + cur[k + 1:] for k, z in enumerate(cur) if z.imag]  # make it real
        for m in moves:
            th, re_ = pack(m)
            th, S = vp.fit(th, re_)
            trials.append((S, unpack_theta(th, re_), n_eff(re_)))
        S, cand, pc = min(trials, key=lambda x: x[0])
        if not S - S_full <= chi2.ppf(0.99, p_full - pc) * sig2:
            break
        cur, S_cur = cand, S
    return np.array(cur), S_cur, S_full, p_full


def out_dir(hold):
    return OUT if hold == "band" else OUT.with_name(f"{OUT.name}_{hold}")


def run(cond, g0, device, hold="band", reuse=None):
    out = out_dir(hold)
    if (out / f"{cond}_g{g0}.json").exists():
        return
    system, family, Tw, Kk, N = T.unpack(cond)
    t0 = time.time()
    data, t, mu_true, beta_true, runs = load_group(cond, g0, device)
    if len(runs) < F.K:
        print(f"skip {cond} {g0}: {len(runs)} runs", flush=True)
        return
    sig2 = RS.noise_var(data["train"][1]) if F.NOISY(cond) else round_var(data["train"][1])
    mask = runs[0]["model"].pr.parametrizations.mu[0].mask.flatten().cpu().numpy()
    p = n_eff(list(mask == 0))                               # effective free parameters of the 8-slot layer
    vp = VarPro(data, t, device, hold)
    sig2_y = sig2
    if hold in ("foh_eiv", "foh_eivb", "foh_eivc"):
        # weights from the best run's refit (unweighted FOH); then every S is in noise units, sigma^2 = 1
        best0 = min(runs, key=lambda r: r["S"])
        th0, re0 = pack(list(best0["eff"]))
        th0, _ = vp.fit(th0, re0)
        sig2_u = noise_var_robust(data["train"][0])
        vp = VarPro(data, t, device, hold, eiv_weights(vp, th0, re0, sig2_y, sig2_u))
        sig2 = 1.0
    if hold == "foh_ml":
        vp = EIVML(data, t, device, hold, sig2_y, noise_var_robust(data["train"][0]))
        sig2 = 1.0
    if not F.NOISY(cond):
        # S ~ 1e-13 here, so gtol = 1e-14 on S itself stops L-BFGS early: a fit started at the truth
        # ended above the fit started at the report (p6 fine: S 1.29352322e-13 vs 1.29352183e-13), and
        # pole errors of 1e-9 were off by up to 1e-8. Minimise S / (n sigma^2) = O(1) instead.
        vp.s_unit = data["train"][1].numel() * sig2
    ref = None
    if hold != "band":                                       # runs replaced by their reduced refits
        for r in runs:
            r["eff"], r["S"], r["S_full"], r["p_full"] = reduce(vp, r["eff"], sig2)
        p = max(r["p_full"] for r in runs)
        ref = min(runs, key=lambda r: r["S_full"])
    if hold == "foh_eivc":
        # calibrated certificate: every run is reduced with bootstrap-calibrated thresholds first, so runs that
        # differ only by spurious poles (whose S drop is far above chi2 sigma^2 under input noise) agree in
        # their poles and S; identical reduced runs (within MERGE) share one calibration
        same = lambda a, b: len(a) == len(b) and all(np.min(np.abs(np.asarray(a) - z)) < F.MERGE * abs(z) for z in b)
        old = {}
        if reuse is not None and (reuse / f"{cond}_g{g0}.json").exists():   # per-run calibrations of an earlier output
            old = {x["seed"]: x for x in json.loads((reuse / f"{cond}_g{g0}.json").read_text())["runs"]}
        done = []
        for i, r in enumerate(runs):
            prev = next((d for d in done if same(d[0], r["eff"])), None)
            o = old.get(r["seed"])
            if prev is None and o is not None and same([complex(*z) for z in o["poles_before"]], r["eff"]):
                th, re_ = pack([complex(*z) for z in o["poles"]])
                steps = o.get("steps") or [dict(kappa=k, removed=j < len(o["kappa"]) - 1 or len(o["poles"]) == 1)
                                           for j, k in enumerate(o["kappa"])]
                th, S = vp.fit(th, re_)
                prev = (np.array(r["eff"]), np.array(unpack_theta(th, re_)), S, steps)
                done.append(prev)
            if prev is None:
                th, re_ = pack(list(r["eff"]))
                th, re_, steps = calibrated_reduce(vp, th, re_, data, sig2_y, sig2_u, device, seed=100 + i)
                th, S = vp.fit(th, re_)
                prev = (np.array(r["eff"]), np.array(unpack_theta(th, re_)), S, steps)
                done.append(prev)
            r["eff_before"], r["eff"], r["S"], r["steps"] = r["eff"], prev[1], prev[2], prev[3]
        # pooled calibration of the certificate and the order test: what a spurious pole buys on these data,
        # the median kappa of the elimination steps that were taken in the group's runs (1 if none). A run
        # whose own bootstrap does not reproduce its extra poles (kappa ~ 1 at its first step) keeps them and
        # the lowest S; with chi2 sigma^2 alone it alone is certified and the order test fails for every report.
        acc = [st["kappa"] for r in runs for st in r["steps"] if st["removed"]]
        kappa_g = max(1.0, float(np.median(acc))) if acc else 1.0
    if hold != "foh_eivc":
        kappa_g = 1.0
    s_min = min(r["S"] for r in runs)
    cert = [r for r in runs if r["S"] - s_min <= kappa_g * chi2.ppf(0.99, p) * sig2]
    best = min(runs, key=lambda r: r["S"])
    S_ref, p_ref = (best["S"], p) if ref is None else (ref["S_full"], ref["p_full"])   # order-test reference
    # candidate reports: the consensus (>= 2 certified runs), then the best run's effective poles
    cands = ([("consensus", F.consensus([r["eff"] for r in cert]))] if len(cert) >= 2 else []) + [("best run", best["eff"])]
    tried = []
    for src, rep in cands:
        if rep.size == 0:
            continue
        theta, real = pack(list(rep))
        theta, S_fit = vp.fit(theta, real)
        dS, dp = S_fit - S_ref, max(p_ref - n_eff(real), 1)
        tried.append(dict(source=src, theta=theta, real=real, S=S_fit, dS=dS / sig2, dp=dp,
                          order_ok=bool(dS <= kappa_g * chi2.ppf(0.99, dp) * sig2),
                          ratio=float(dS / (kappa_g * chi2.ppf(0.99, dp) * sig2))))
        if tried[-1]["order_ok"]:
            break
    # every candidate failed: foh_eivc takes the least violating one (smallest dS / threshold); the
    # last tried (the best run) can be the run whose own bootstrap missed its spurious poles
    fallback = min(tried, key=lambda x: x["ratio"]) if hold == "foh_eivc" else tried[-1]
    chosen = next((x for x in tried if x["order_ok"]), fallback)
    theta, real = chosen["theta"], chosen["real"]
    calib = {}
    if hold in ("foh_eivb", "foh_eivc"):
        theta, real, steps = calibrated_reduce(vp, theta, real, data, sig2_y, sig2_u, device)
        calib = dict(calibrated_steps=steps, report_before=[[float(z.real), float(z.imag)]
                                                            for z in unpack_theta(chosen["theta"], chosen["real"])])
    mu_fit = unpack_theta(theta, real)
    profs, k = [], 0
    for z, r in zip(mu_fit, real):
        cs, S = profile(vp, theta, real, k)
        lo, hi = interval(cs, S, LEVEL * sig2)
        profs.append(dict(mu=[float(z.real), float(z.imag)], c=cs.tolist(), dS=((S - S.min()) / sig2).tolist(),
                          interval=[None if lo is None else z.real * lo, None if hi is None else z.real * hi]))
        k += 1 if r else 2
    if not F.NOISY(cond) and hold == "foh":
        sd = bootstrap(vp, data, t, theta, real, device)
        for pr, sk in zip(profs, sd):
            pr["boot_std"] = float(sk)
            pr["interval_boot"] = [pr["mu"][0] - 1.96 * float(sk), pr["mu"][0] + 1.96 * float(sk)]
    eiv = {}
    if F.NOISY(cond) and hold in ("foh", "foh_eiv", "foh_ml", "foh_eivb", "foh_eivc") and cond in EIV_CONDS:
        add_eiv(eiv, profs, vp, data, t, theta, real, sig2_y, device)
    true = E.true_folded(mu_true, beta_true)[0]
    res = dict(cond=cond, group=g0, system=system, eta_hat=float(np.sqrt(data["train"][1].shape[1] * sig2_y)),
               sigma2=sig2, p=p, n_cert=len(cert), best_seed=best["seed"], best_S=best["S"], S_ref=S_ref, p_ref=p_ref,
               runs=[dict(seed=r["seed"], S=r["S"], S_full=r.get("S_full"), certified=any(r is c for c in cert),
                          poles=[[float(z.real), float(z.imag)] for z in r["eff"]],
                          **({"poles_before": [[float(z.real), float(z.imag)] for z in r["eff_before"]],
                              "kappa": [s["kappa"] for s in r["steps"]], "steps": r["steps"]} if "steps" in r else {}))
                     for r in runs], kappa_group=kappa_g,
               tried=[dict(source=x["source"], S=x["S"], dS=x["dS"], dp=x["dp"], order_ok=x["order_ok"], ratio=x["ratio"],
                           poles=[[float(z.real), float(z.imag)] for z in unpack_theta(x["theta"], x["real"])])
                      for x in tried],
               source=chosen["source"], order_ok=chosen["order_ok"],
               report=[[float(z.real), float(z.imag)] for z in mu_fit], profiles=profs,
               true_poles=[[float(z.real), float(z.imag)] for z in true], hold=hold,
               seconds=round(time.time() - t0, 1), **eiv, **calib)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cond}_g{g0}.json").write_text(json.dumps(res, indent=1, default=float))
    fmt = lambda iv: "[" + ", ".join("open" if x is None else f"{x:.3f}" for x in iv) + "]"
    print(f"{cond} g{g0}: {len(cert)} cert., best s{best['seed']}, "
          + ", ".join(f"{x['source']} dS {x['dS']:.1f} (dp {x['dp']}, ok {x['order_ok']})" for x in tried)
          + "; Re mu intervals " + "; ".join(f"{complex(*pr['mu']):.3f} {fmt(pr['interval'])}" for pr in profs)
          + f" ({res['seconds']:.0f}s)", flush=True)


def add_boot_existing(cond, g0, device):
    """EIV bootstrap for an existing FOH group JSON that lacks it (the eta_u = 0 references)."""
    f = out_dir("foh") / f"{cond}_g{g0}.json"
    r = json.loads(f.read_text())
    if "eta_u_hat" in r:
        return
    data, t, _, _ = T.setup(cond, g0)
    vp = VarPro(data, t, device, "foh")
    theta, real = pack([complex(*z) for z in r["report"]])
    theta, _ = vp.fit(theta, real)
    add_eiv(r, r["profiles"], vp, data, t, theta, real, r["sigma2"], device)
    f.write_text(json.dumps(r, indent=1, default=float))
    print(f"{cond} g{g0}: EIV bootstrap added (eta_u_hat {r['eta_u_hat']:.2g})", flush=True)


def score(r):
    """Truth-scored view of one group, per true pole: (fitted pole, interval closed, interval covers
    the true Re mu, relative error of the fit, relative half-width); and the false identified poles."""
    true = [complex(*z) for z in r["true_poles"]]
    fit = []
    for pr in r["profiles"]:                         # interval recomputed from the stored profile
        z = complex(*pr["mu"])
        lo, hi = interval(np.array(pr["c"]), np.array(pr["dS"]), LEVEL)
        fit.append((z, [None if lo is None else z.real * lo, None if hi is None else z.real * hi], pr.get("interval_boot")))
    per = {}
    used = set()
    for z in true:
        k = min(range(len(fit)), key=lambda i: abs(fit[i][0] - z)) if fit else None
        if k is None or abs(fit[k][0] - z) >= T.RECOVERED * abs(z):
            per[z] = None
            continue
        used.add(k)
        f, (lo, hi), ib = fit[k]
        closed = lo is not None and hi is not None
        per[z] = dict(fit=f, closed=closed, err=abs(f - z) / abs(z),
                      cover=(hi is None or hi <= z.real) and (lo is None or z.real <= lo),
                      half=abs(lo - hi) / 2 / abs(z) if closed else None,
                      boot_cover=None if ib is None else bool(ib[0] <= z.real <= ib[1]))
    false = sum(1 for i, (f, (lo, hi), _) in enumerate(fit) if i not in used and lo is not None and hi is not None)
    return per, false


def summarize():
    for hold in HOLDS:
        summarize_hold(hold)


def summarize_hold(hold):
    import e2_noise as NZ
    rs = [json.loads(f.read_text()) for f in sorted(out_dir(hold).glob("*.json"))]
    if not rs:
        return
    lab = lambda z: f"{z.real:+g}{z.imag:+g}i" if z.imag else f"{z.real:+g}"
    lines = ["# E4: minimal-form refit, order test and profile intervals"
             + {"band": "", "foh": ", first-order-hold refits", "band_red": ", band-limited refits with reduction",
                 "foh_eiv": ", first-order-hold refits, errors-in-variables weighting",
                 "foh_ml": ", first-order-hold refits, errors-in-variables ML",
                 "foh_eivb": ", first-order-hold refits, EIV weighting and bootstrap-calibrated reduction",
                 "foh_eivc": ", first-order-hold refits, EIV weighting and a bootstrap-calibrated certificate"}[hold], "",
             *(["Refits, certificate and order-test reference use the exact first-order-hold response "
                "(the data generator's input model) instead of the layer's band-limited one. Each run is refitted "
                "and reduced (backward elimination) before the certificate.", ""]
               if hold == "foh" else []),
             *(["As `--hold foh`, but every refit is a weighted least-squares fit whose frequency weights whiten the "
                "errors-in-variables residual e_y − G e_u (`eiv_weights`: residual spectrum σ_y² + σ_u² (‖x_i‖ / ‖y_i‖)² "
                "|G(iω)|², σ_y² from the outputs, σ_u² from the inputs by a robust estimate, G from the best run's "
                "refit). S is then in noise units and every test uses σ² = 1. With σ_u² = 0 this is `--hold foh`.", ""]
               if hold == "foh_eiv" else []),
             *(["As `--hold foh`, but every refit minimises the frequency-domain errors-in-variables ML cost "
                "Σ |R|² / (σ_y² + σ_u² (‖x_i‖ / ‖y_i‖)² |G_θ(iω)|²) (`EIVML`; poles and residues jointly, the "
                "weights move with θ). S is in noise units and every test uses σ² = 1.", ""]
               if hold == "foh_ml" else []),
             *(["As `--hold foh_eiv`; then the report is reduced once more by backward elimination with "
                f"bootstrap-calibrated thresholds (`calibrated_reduce`, {N_CAL} EIV draws per step from the reduced "
                "fit with a denoised input): drop the cheapest pole / make it real while ΔS ≤ κ χ²_Δp(0.99), "
                "κ = max(1, mean ΔS* / Δp).", ""]
               if hold == "foh_eivb" else []),
             *(["As `--hold foh_eivb`, but every run is reduced by `calibrated_reduce` too (after its FOH reduction, "
                "before the certificate; runs with the same poles share one calibration). The certificate and the "
                "order test use κ_g χ²_p(0.99), κ_g = max(1, median κ of the elimination steps taken in the group's "
                "runs), the pooled bootstrap estimate of what a spurious pole buys on these data. If every candidate "
                "fails the order test, the least violating one (smallest ΔS / threshold) is reported (flagged). "
                "The report is reduced once more as with `foh_eivb`.", ""]
               if hold == "foh_eivc" else []),
             *(["Control: refits with the band-limited (layer) model, each run refitted and reduced (backward "
                "elimination) before the certificate, as with `--hold foh`.", ""]
               if hold == "band_red" else []),
             "Stage `rescue_na` (noise-aware restarts), groups of 5 seeds (one noise draw each), truth-free up "
             "to the scores. Certified: S − S_min ≤ χ²_p(0.99) σ̂². Report: the consensus of the certified runs "
             "(≥ 2), else the best run's effective poles. The report is refitted as a minimal tied layer (one "
             "slot per reported pole); order test: S_minimal − S_best ≤ χ²_Δp(0.99) σ̂²; if the consensus "
             "fails it, the best run's poles are tried. Each fitted pole: 95 % profile interval of Re μ "
             f"(others re-fitted, c ∈ [1/{C_MAX:g}, {C_MAX:g}]); identified = within 0.1 of a true pole with the "
             "interval closed. Cell per true pole: identified groups / intervals covering the true Re μ; "
             "RMS relative error of the minimal fit; mean relative half-width of the interval; relative CRB "
             "(e2_noise.py).", ""]
    sets = [(s, s, base, noisy, suf) for suf in ("", "_foh") for s, (base, noisy) in NZ.SYSTEMS.items()]
    sets += [(f"{s}, E1 data", s, base, noisy, "") for s, (base, noisy) in E1_NOISY.items()]
    for label, system, base, noisy, suf in sets:
        g = [r for r in rs if r["system"] == system and r["cond"] in {c + suf for c in noisy.values()}]
        if not g:
            continue
        cr = NZ.crb_unit(base)
        true = [complex(*z) for z in cr["poles"]]
        lines += [f"## {label}" + (", runs of the FOH-trained layer (`*_foh`)" if suf else ""), "",
                  "| η | groups | certified runs (mean) | report from consensus | order test passed | "
                  + " | ".join(f"{lab(z)}: identified / covered, error, half-width, CRB" for z in true)
                  + " | false identified |", "|---|---|---|---|---|" + "---|" * len(true) + "---|"]
        for eta, cond in noisy.items():
            gr = sorted((r for r in g if r["cond"] == cond + suf), key=lambda r: r["group"])
            if not gr:
                continue
            sc = [score(r) for r in gr]
            cells = []
            for j, z in enumerate(true):
                xs = [s[0][z] for s in sc]
                idf = sum(bool(x and x["closed"]) for x in xs)
                cov = sum(bool(x and x["cover"]) for x in xs)
                errs = [x["err"] for x in xs if x]
                halves = [x["half"] for x in xs if x and x["half"] is not None]
                crb = cr["unit"][j] * eta / np.sqrt(cr["n_t"])
                cells.append(f"{idf}/{cov}, " + (f"{np.sqrt(np.mean(np.square(errs))):.1e}" if errs else "–")
                             + (f", {np.mean(halves):.1e}" if halves else ", –") + f", {crb:.1e}")
            lines.append(f"| {eta:g} | {len(gr)} | {np.mean([r['n_cert'] for r in gr]):.1f} | "
                         f"{sum(r['source'] == 'consensus' for r in gr)}/{len(gr)} | {sum(r['order_ok'] for r in gr)}/{len(gr)} | "
                         + " | ".join(cells) + f" | {sum(s[1] for s in sc)} |")
        lines.append("")
    for system, conds in (EIV.items() if hold in ("foh", "foh_eiv", "foh_ml", "foh_eivb", "foh_eivc") else ()):
        g = [r for r in rs if r["cond"] in conds.values()]
        if not g:
            continue
        true = [complex(*z) for z in g[0]["true_poles"]]
        base = conds["0"]
        lines += [f"## {system}, input noise (errors-in-variables), `{base}` + η_u", "",
                  "Noise on the stored train / validation inputs as well (outputs from the clean inputs), output "
                  "noise 1e-2. The pipeline above unchanged (σ̂² from the outputs). Covered: by the profile interval / "
                  f"by the truth-free EIV bootstrap ({N_BOOT} draws; σ_u² from the inputs by a robust estimate, "
                  "bias-corrected Re μ ± 1.96 std). Half-width: profile / bootstrap.", "",
                  "| η_u | η̂_u | groups | certified runs (mean) | report from consensus | order test passed | "
                  + " | ".join(f"{lab(z)}: identified / covered (profile / bootstrap), error, half-width (profile / bootstrap)"
                               for z in true) + " | false identified |",
                  "|---|---|---|---|---|---|" + "---|" * len(true) + "---|"]
        for u, cond in sorted(conds.items(), key=lambda kv: float(kv[0])):
            gr = sorted((r for r in g if r["cond"] == cond), key=lambda r: r["group"])
            if not gr:
                continue
            sc = [score(r) for r in gr]
            cells = []
            for z in true:
                xs = [x[0][z] for x in sc]
                errs = [x["err"] for x in xs if x]
                halves = [x["half"] for x in xs if x and x["half"] is not None]
                bh = [1.96 * pr["boot_std"] / abs(z) for r in gr for pr in r["profiles"]
                      if "boot_std" in pr and abs(complex(*pr["mu"]) - z) < T.RECOVERED * abs(z)]
                cells.append(f"{sum(bool(x and x['closed']) for x in xs)} / {sum(bool(x and x['cover']) for x in xs)} / "
                             f"{sum(bool(x and x['boot_cover']) for x in xs)}, "
                             + (f"{np.sqrt(np.mean(np.square(errs))):.1e}" if errs else "–")
                             + (f", {np.mean(halves):.1e}" if halves else ", –")
                             + (f" / {np.mean(bh):.1e}" if bh else " / –"))
            eu = [r["eta_u_hat"] for r in gr if "eta_u_hat" in r]
            lines.append(f"| {float(u):g} | {np.mean(eu):.2g} | {len(gr)} | {np.mean([r['n_cert'] for r in gr]):.1f} | "
                         f"{sum(r['source'] == 'consensus' for r in gr)}/{len(gr)} | {sum(r['order_ok'] for r in gr)}/{len(gr)} | "
                         + " | ".join(cells) + f" | {sum(x[1] for x in sc)} |")
        lines.append("")
    clean = [c + suf for c in CLEAN for suf in ("", "_foh")]
    for system in ("p4", "p6"):
        g = [r for r in rs if r["system"] == system and r["cond"] in clean]
        if not g:
            continue
        true = [complex(*z) for z in g[0]["true_poles"]]
        lines += [f"## {system}, noise-free data (σ² = float32 rounding of the outputs)", "",
                  "Fits minimise S / (n σ²) and end with damped Newton steps. Covered: by the 95 % profile "
                  f"interval (σ² from the outputs only) / by the bootstrap interval Re μ ± 1.96 std ({N_BOOT} "
                  "dithers of the stored inputs and fitted outputs by ± ulp/2). All groups of a condition share "
                  "one data set, so coverage is one realisation per condition.", "",
                  "| condition | groups | certified runs (mean) | report from consensus | order test passed | "
                  + " | ".join(f"{lab(z)}: identified / covered (profile / bootstrap), error, half-width" for z in true)
                  + " | false identified |", "|---|---|---|---|---|" + "---|" * len(true) + "---|"]
        for cond in clean:
            gr = sorted((r for r in g if r["cond"] == cond), key=lambda r: r["group"])
            if not gr:
                continue
            sc = [score(r) for r in gr]
            cells = []
            for z in true:
                xs = [x[0][z] for x in sc]
                errs = [x["err"] for x in xs if x]
                halves = [x["half"] for x in xs if x and x["half"] is not None]
                cells.append(f"{sum(bool(x and x['closed']) for x in xs)} / {sum(bool(x and x['cover']) for x in xs)} / "
                             f"{sum(bool(x and x['boot_cover']) for x in xs)}, "
                             + (f"{np.sqrt(np.mean(np.square(errs))):.1e}" if errs else "–")
                             + (f", {np.mean(halves):.1e}" if halves else ", –"))
            lines.append(f"| {cond} | {len(gr)} | {np.mean([r['n_cert'] for r in gr]):.1f} | "
                         f"{sum(r['source'] == 'consensus' for r in gr)}/{len(gr)} | {sum(r['order_ok'] for r in gr)}/{len(gr)} | "
                         + " | ".join(cells) + f" | {sum(x[1] for x in sc)} |")
        lines.append("")
    (T.R.RESULTS / f"{out_dir(hold).name}.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=CONDS)
    ap.add_argument("--group", nargs="+", type=int, default=[10, 15, 20, 25])
    ap.add_argument("--hold", choices=HOLDS, default="band")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--reuse", type=Path, help="foh_eivc: reuse the per-run calibrations of an earlier output dir")
    ap.add_argument("--add-boot", action="store_true", help="EIV bootstrap for existing FOH groups")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.add_boot:
        for c in a.cond:
            for g in a.group:
                add_boot_existing(c, g, device)
        return
    for c in a.cond:
        for g in a.group:
            run(c, g, device, a.hold, a.reuse)


if __name__ == "__main__":
    main()
