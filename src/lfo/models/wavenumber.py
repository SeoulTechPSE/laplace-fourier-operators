"""Wavenumber-dependent pole layer (project A), 1D space x time.

Plan: `docs/proposal_A_wavenumber_pole_operator.qmd`, Section 4. For a linear PDE u_t = L u + f with a periodic
spatial axis, every spatial Fourier mode k obeys d u_k / dt = lambda(k) u_k + f_k. The layer learns the temporal
kernel of each mode as a pole-residue sum whose poles and residues are functions of k:

    K(k, t) = sum_n beta_n(k) e^{mu_n(k) t},    v_k(t) = int_0^t K(k, t - s) f_k(s) ds   (zero initial state).

**Time axis** (D1 decision, `experiments/D/d1_inputs/README.md`): the first-order-hold exponential-integrator
recursion. The response is exact for an input that is linear between samples; it is causal by construction and makes
no periodicity assumption in time. It is computed as a causal convolution with the discrete FOH impulse response
(FFT with zero padding, O(n log n)) plus a correction for the first sample; `foh_recursion` is the O(n) loop
form, used to check it.

**Pole functions** (Section 4.3, polynomial form). Over the wavenumber k of the spatial grid:

    mu_n(k)   = sum_j a_nj k^{2j}  +  i sum_j c_nj k^{p_j}         (even real part; Im powers p_j, default odd)
    beta_n(k) = sum_j (b_nj + i e_nj) k^{2j}

Only k >= 0 is evaluated (rfft), and irfft makes the field real, so any powers p_j are admissible. Odd powers
(advection, k^3 dispersion) are the default; `im_powers=(1, 2, 3)` adds even ones, needed for a frequency c k^2
(Euler-Bernoulli beam, where the two poles per k are +-i c k^2). Residues are polynomials in k^2: a transfer function
whose residues are not (for example the beam's displacement, with residues 1 / (2 i c k^2)) needs another output
(the beam's velocity: residues 1/2) or a rational residue. `stable=True` maps the real part through
-softplus(.) so Re mu_n(k) <= 0. Parameters are continuous in k, so the layer can be evaluated on a finer spatial
grid than it was trained on.

Options: `modes` keeps only the lowest `modes` rfft wavenumbers (like FNO truncation; the polynomial is not
evaluated where the data carry no information, so a k^4 term cannot overflow there). `re_max` caps Re mu smoothly at
re_max (soft minimum with temperature `re_tau`), which bounds the growth e^{Re mu T} while leaving the polynomial
unchanged well below the cap. Above the cap the soft minimum has almost no gradient, so a start whose Re mu lies
there stays trapped (`experiments/A/layer`, a7); `re_leak` > 0 keeps a slope re_leak above the cap. `k_ref`
evaluates the polynomials in k / k_ref, so that all coefficients have comparable scale for a first-order optimizer
(with k up to 16, a k^4 coefficient otherwise moves Re mu by 6.5e4 per unit step).

`ic=True` adds the initial-condition path of project (D2): `forward(f, u0)` adds the free response of the
initial field, which shares the poles mu_n(k) with the forced response.
`delay` adds one shared input delay theta >= 0, applied exactly (`lfo.models.delay.delay_shift`).
`mu_mlp` (RQ4) replaces the pole polynomials by an MLP of k / k_ref (the residues stay polynomial; the
polynomial coefficients are then unused).

Input / output: (batch, n_x, n_t) real fields on a uniform periodic x grid of length `length` and a uniform t grid.

**2D** (`WavenumberPoleLayer2d`, input (batch, n_x, n_y, n_t)). The real part is isotropic, a polynomial in |k|^2;
the imaginary part is a linear advection term c . k plus optional isotropic powers |k|^p:

    mu_n(k)   = sum_j a_nj |k|^{2j}  +  i (c_nx k_x + c_ny k_y + sum_j e_nj |k|^{p_j})
    beta_n(k) = sum_j (b_nj + i e'_nj) |k|^{2j}

A real field needs mu(-k) = conj mu(k) and beta(-k) = conj beta(k). Re mu is even and c . k is odd, so they comply.
The isotropic imaginary terms (Im beta and the |k|^p terms of Im mu) are even, so they are multiplied by the
half-plane sign s(k) = +1 for k_y > 0 or (k_y = 0, k_x >= 0), and -1 otherwise. This is the 1D convention (evaluate
on k >= 0, extend by conjugation): the rfft2 layout holds negative k_x only in the row k_y = 0. `modes` keeps
|k_x index| < modes and k_y index < modes (the FNO corner block); outside it the output is zero.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["foh_coefficients", "foh_kernel", "foh_recursion", "foh_response", "WavenumberPoleLayer1d",
           "WavenumberPoleLayer2d", "WavenumberPoleConv1d", "WavenumberPoleOperator1d", "WavenumberPoleConv2d",
           "WavenumberPoleOperator2d"]


def foh_coefficients(mu: torch.Tensor, dt: float):
    """(E, b0, b1) of the exact FOH step x_{j+1} = E x_j + b0 u_j + b1 u_{j+1} for the mode 1/(s - mu).
    Computed in complex128; I1 cancels for small mu dt, so a series is used there."""
    mu = mu.to(torch.complex128)
    z = mu * dt
    E = torch.exp(z)
    small = z.abs() < 1e-3
    zs = torch.where(small, torch.ones_like(z), z)
    # I0 = (E - 1) / mu, I1 = (E - 1 - z) / mu^2, written in z to avoid 0 / 0
    I0 = torch.where(small, dt * (1 + z / 2 + z ** 2 / 6 + z ** 3 / 24), (E - 1) / zs * dt)
    I1 = torch.where(small, dt ** 2 * (0.5 + z / 6 + z ** 2 / 24 + z ** 3 / 120), (E - 1 - zs) / zs ** 2 * dt ** 2)
    return E, I0 - I1 / dt, I1 / dt


def foh_kernel(mu: torch.Tensor, n_t: int, dt: float):
    """Discrete FOH impulse responses of 1/(s - mu), each of shape mu.shape + (n_t,):

    - g: response to a unit sample u_k at an interior index k >= 1, shifted to start at k (g_0 = b1,
      g_j = E^{j-1} (E b1 + b0));
    - c: the correction for the first sample. From x_0 = 0 the ramp before t_0 does not exist, so the response
      to u_0 is g_j - b1 E^j."""
    E, b0, b1 = foh_coefficients(mu, dt)
    j = torch.arange(n_t, device=mu.device, dtype=torch.float64)
    Ej = torch.exp(mu.to(torch.complex128)[..., None] * dt * j)                  # E^j
    # E^{j-1}; at j = 0 the value is unused (masked below), so evaluate it at j = 0 instead of j = -1: exp(-mu dt)
    # overflows for Re(mu) dt < -709 and its inf turns the masked branch's gradient into nan
    Ejm1 = torch.exp(mu.to(torch.complex128)[..., None] * dt * (j - 1).clamp(min=0))
    g = torch.where(j == 0, b1[..., None], (E * b1 + b0)[..., None] * Ejm1)
    return g, b1[..., None] * Ej


def foh_recursion(mu: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """O(n) loop form, for tests: x_{j+1} = E x_j + b0 u_j + b1 u_{j+1}, x_0 = 0. mu: (...), u: (..., n_t)."""
    E, b0, b1 = foh_coefficients(mu, dt)
    u = u.to(torch.complex128)
    x = torch.zeros(u.shape[:-1], dtype=torch.complex128, device=u.device)
    out = [x]
    for j in range(u.shape[-1] - 1):
        x = E * x + b0 * u[..., j] + b1 * u[..., j + 1]
        out.append(x)
    return torch.stack(out, dim=-1)


def causal_conv(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """y_j = sum_{k <= j} g_{j-k} u_k along the last axis (complex), by FFT with zero padding."""
    n = u.shape[-1]
    return torch.fft.ifft(torch.fft.fft(g, 2 * n) * torch.fft.fft(u, 2 * n))[..., :n]


def foh_response(g: torch.Tensor, c: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Exact FOH response from rest: causal convolution with g, minus the first-sample correction u_0 c."""
    return causal_conv(g, u) - u[..., :1] * c


def _half_plane_sign(kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """s(k) = +1 for k_y > 0 or (k_y = 0, k_x >= 0), else -1 (2D layers, module docstring)."""
    return torch.where((ky > 0) | ((ky == 0) & (kx >= 0)), 1.0, -1.0).to(kx.dtype)


def _cap(re: torch.Tensor, cap: float, tau: float, leak: float) -> torch.Tensor:
    """Soft minimum of re and cap (temperature tau), plus slope `leak` above the cap."""
    soft = cap - tau * F.softplus((cap - re) / tau)
    return soft + leak * (re - soft) if leak else soft


class WavenumberPoleLayer1d(nn.Module):
    """K(k, s) = sum_n beta_n(k) / (s - mu_n(k)), polynomial in k, FOH time axis. See the module docstring.

    n_poles     N
    mu_degree   number of even (real) and odd (imaginary) coefficients of mu_n(k): (n_re, n_im)
    beta_degree number of coefficients of beta_n(k) in k^2
    """

    def __init__(self, n_poles: int, t, length: float = 2 * math.pi, mu_degree=(3, 2), beta_degree: int = 2,
                 stable: bool = False, init_scale: float = 0.1, modes: int | None = None, re_max: float | None = None,
                 re_tau: float = 0.05, k_ref: float = 1.0, im_powers: tuple | None = None, re_leak: float = 0.0,
                 mu_mlp: int | None = None, delay: float | None = None, ic: bool = False):
        super().__init__()
        # initial-condition path (project D2): homogeneous term u0_hat(k) sum_n gamma_n(k) e^{mu_n(k) t}, gamma_n a
        # polynomial in k^2 like beta_n; it shares the poles with the forced response
        self.gamma = None
        if ic:
            self.gamma = nn.Parameter(torch.zeros(n_poles, beta_degree, 2, dtype=torch.float64))
            with torch.no_grad():
                self.gamma[:, 0, 0] = 1.0 / n_poles
        self.modes, self.re_max, self.re_tau, self.k_ref = modes, re_max, re_tau, float(k_ref)
        # optional shared input delay theta = softplus(raw) >= 0 (process dynamics; `delay` is its initial value)
        self.theta_raw = None if delay is None else nn.Parameter(
            torch.log(torch.expm1(torch.tensor(float(delay), dtype=torch.float64))))
        self.re_leak = float(re_leak)
        # RQ4 option: mu_n(k) from an MLP of k / k_ref (hidden width mu_mlp) instead of the polynomial
        self.mlp = None if mu_mlp is None else nn.Sequential(
            nn.Linear(1, mu_mlp, dtype=torch.float64), nn.Tanh(), nn.Linear(mu_mlp, mu_mlp, dtype=torch.float64),
            nn.Tanh(), nn.Linear(mu_mlp, 2 * n_poles, dtype=torch.float64))
        t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
        self.register_buffer("t", t)
        self.length, self.stable = float(length), stable
        n_re, n_im = mu_degree
        # powers of k in Im mu: default odd (1, 3, ...); any powers are valid because only k >= 0 is evaluated
        self.im_powers = tuple(im_powers) if im_powers is not None else tuple(2 * j + 1 for j in range(n_im))
        n_im = len(self.im_powers)
        self.mu_re = nn.Parameter(init_scale * torch.randn(n_poles, n_re, dtype=torch.float64))
        self.mu_im = nn.Parameter(init_scale * torch.randn(n_poles, n_im, dtype=torch.float64))
        self.beta = nn.Parameter(torch.zeros(n_poles, beta_degree, 2, dtype=torch.float64))
        with torch.no_grad():
            self.beta[:, 0, 0] = 1.0 / n_poles

    def wavenumbers(self, n_x: int) -> torch.Tensor:
        """Angular wavenumbers of the rfft of an n_x-point periodic grid of length `length`."""
        return 2 * math.pi * torch.fft.rfftfreq(n_x, self.length / n_x, device=self.t.device, dtype=torch.float64)

    def mu(self, k: torch.Tensor) -> torch.Tensor:
        """Poles mu_n(k), (N, len(k)) complex128. The polynomials (or the MLP) are in k / k_ref."""
        k = k / self.k_ref
        k2 = k ** 2
        if self.mlp is not None:
            o = self.mlp(k[:, None].to(torch.float64)).T                        # (2N, K)
            n = o.shape[0] // 2
            re, im = o[:n], o[n:]
            if self.stable:
                re = -F.softplus(re)
            elif self.re_max is not None:
                re = _cap(re, self.re_max, self.re_tau, self.re_leak)
            return torch.complex(re, im)
        re = sum(self.mu_re[:, j, None] * k2 ** j for j in range(self.mu_re.shape[1]))
        if self.stable:
            re = -F.softplus(re)
        elif self.re_max is not None:          # smooth cap Re mu <= re_max; identity to < 1e-6 below re_max - 15 tau
            re = _cap(re, self.re_max, self.re_tau, self.re_leak)
        im = sum(self.mu_im[:, j, None] * k ** p for j, p in enumerate(self.im_powers))
        return torch.complex(re, im)

    def poles(self, k: torch.Tensor):
        """(mu, beta), each (N, len(k)) complex128. The polynomials are in k / k_ref."""
        k2 = (k / self.k_ref) ** 2
        b = sum(torch.complex(self.beta[:, j, 0, None], self.beta[:, j, 1, None]) * k2 ** j
                for j in range(self.beta.shape[1]))
        return self.mu(k), b

    def homogeneous(self, u0: torch.Tensor, n_t: int) -> torch.Tensor:
        """Free response from the initial field u0 (b, n_x): sum_n gamma_n(k) u0_hat(k) e^{mu_n(k) t_j}, (b, K, n_t)."""
        n_x = u0.shape[1]
        u0k = torch.fft.rfft(u0.to(torch.float64), dim=1)
        K = u0k.shape[1] if self.modes is None else min(self.modes, u0k.shape[1])
        k = self.wavenumbers(n_x)[:K]
        mu = self.mu(k)                                                       # (N, K)
        k2 = (k / self.k_ref) ** 2
        gam = sum(torch.complex(self.gamma[:, j, 0, None], self.gamma[:, j, 1, None]) * k2 ** j
                  for j in range(self.gamma.shape[1]))
        dt = float(self.t[1] - self.t[0])
        tj = dt * torch.arange(n_t, device=u0.device, dtype=torch.float64)
        return (gam[..., None] * torch.exp(mu[..., None] * tj)).sum(0)[None] * u0k[:, :K, None]

    def forward(self, f: torch.Tensor, u0: torch.Tensor | None = None) -> torch.Tensor:
        b_, n_x, n_t = f.shape
        dt = float(self.t[1] - self.t[0])
        fk = torch.fft.rfft(f.to(torch.float64), dim=1)                       # (b, K, n_t)
        K = fk.shape[1] if self.modes is None else min(self.modes, fk.shape[1])
        fk = fk[:, :K]                                                        # modes beyond K are set to zero
        mu, beta = self.poles(self.wavenumbers(n_x)[:K])                      # (N, K)
        g, c = foh_kernel(mu, n_t, dt)                                         # (N, K, n_t) each
        if self.theta_raw is not None:                                        # delayed: per pole, then shift exactly
            from .delay import delay_shift
            x = foh_response(g[:, None], c[:, None], fk[None])                # (N, b, K, n_t)
            x = delay_shift(x, fk[None], mu[:, None, :], F.softplus(self.theta_raw), dt)
            vk = (beta[:, None, :, None] * x).sum(0)
        else:
            g, c = (beta[..., None] * g).sum(0), (beta[..., None] * c).sum(0) # (K, n_t): sum_n beta_n (.)
            vk = foh_response(g[None], c[None], fk)                           # (b, K, n_t)
        if u0 is not None:
            if self.gamma is None:
                raise ValueError("pass ic=True to use an initial condition")
            vk = vk + self.homogeneous(u0, n_t)
        return torch.fft.irfft(vk, n=n_x, dim=1).to(f.dtype)


class WavenumberPoleLayer2d(nn.Module):
    """2D version of `WavenumberPoleLayer1d` on a periodic (x, y) grid; see the module docstring.

    n_poles     N
    re_degree   number of coefficients of Re mu_n in |k|^2
    advection   learn c_n . k in Im mu_n
    im_powers   isotropic powers |k|^p in Im mu_n (default none)
    beta_degree number of coefficients of beta_n in |k|^2
    """

    def __init__(self, n_poles: int, t, length=(2 * math.pi, 2 * math.pi), re_degree: int = 3, advection: bool = True,
                 im_powers: tuple = (), beta_degree: int = 2, stable: bool = False, init_scale: float = 0.1,
                 modes: int | None = None, re_max: float | None = None, re_tau: float = 0.05, k_ref: float = 1.0,
                 re_leak: float = 0.0):
        super().__init__()
        self.modes, self.re_max, self.re_tau, self.k_ref, self.stable = modes, re_max, re_tau, float(k_ref), stable
        self.re_leak = float(re_leak)
        self.length = tuple(float(a) for a in length)
        self.register_buffer("t", torch.as_tensor(t, dtype=torch.float64).reshape(-1))
        self.im_powers = tuple(im_powers)
        self.mu_re = nn.Parameter(init_scale * torch.randn(n_poles, re_degree, dtype=torch.float64))
        self.adv = nn.Parameter(init_scale * torch.randn(n_poles, 2, dtype=torch.float64)) if advection else None
        self.mu_im = nn.Parameter(init_scale * torch.randn(n_poles, len(self.im_powers), dtype=torch.float64))
        self.beta = nn.Parameter(torch.zeros(n_poles, beta_degree, 2, dtype=torch.float64))
        with torch.no_grad():
            self.beta[:, 0, 0] = 1.0 / n_poles

    def wavenumbers(self, n_x: int, n_y: int):
        """(k_x, k_y), each (n_x, n_y // 2 + 1): angular wavenumbers of the rfft2 layout."""
        dev = self.t.device
        kx = 2 * math.pi * torch.fft.fftfreq(n_x, self.length[0] / n_x, device=dev, dtype=torch.float64)
        ky = 2 * math.pi * torch.fft.rfftfreq(n_y, self.length[1] / n_y, device=dev, dtype=torch.float64)
        return torch.meshgrid(kx, ky, indexing="ij")

    def mask(self, n_x: int, n_y: int):
        """Boolean (n_x, n_y // 2 + 1): the kept modes."""
        ix = torch.fft.fftfreq(n_x, 1 / n_x, device=self.t.device).abs()
        iy = torch.fft.rfftfreq(n_y, 1 / n_y, device=self.t.device)
        if self.modes is None:
            return torch.ones(ix.numel(), iy.numel(), dtype=torch.bool, device=self.t.device)
        return (ix[:, None] < self.modes) & (iy[None, :] < self.modes)

    def mu(self, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        """Poles mu_n(k), (N,) + kx.shape complex128. Even imaginary terms carry the half-plane sign s(k)."""
        kx, ky = kx / self.k_ref, ky / self.k_ref
        k2 = kx ** 2 + ky ** 2
        e = lambda w: w.reshape(w.shape + (1,) * kx.dim())
        re = sum(e(self.mu_re[:, j]) * k2 ** j for j in range(self.mu_re.shape[1]))
        if self.stable:
            re = -F.softplus(re)
        elif self.re_max is not None:
            re = _cap(re, self.re_max, self.re_tau, self.re_leak)
        im = torch.zeros_like(re)
        if self.adv is not None:
            im = im + e(self.adv[:, 0]) * kx + e(self.adv[:, 1]) * ky
        kabs = k2.sqrt()
        for j, p in enumerate(self.im_powers):
            im = im + _half_plane_sign(kx, ky) * e(self.mu_im[:, j]) * kabs ** p
        return torch.complex(re, im)

    def poles(self, kx: torch.Tensor, ky: torch.Tensor):
        """(mu, beta), each (N,) + kx.shape complex128. The polynomials are in k / k_ref. Even imaginary terms carry
        the half-plane sign s(k) (module docstring)."""
        k2 = (kx / self.k_ref) ** 2 + (ky / self.k_ref) ** 2
        e = lambda w: w.reshape(w.shape + (1,) * kx.dim())
        sgn = _half_plane_sign(kx, ky)
        b = sum(torch.complex(e(self.beta[:, j, 0]), sgn * e(self.beta[:, j, 1])) * k2 ** j
                for j in range(self.beta.shape[1]))
        return self.mu(kx, ky), b

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b_, n_x, n_y, n_t = f.shape
        dt = float(self.t[1] - self.t[0])
        keep = self.mask(n_x, n_y)
        kx, ky = self.wavenumbers(n_x, n_y)
        kx, ky = kx[keep], ky[keep]                                           # (M,) kept modes only
        fk = torch.fft.rfft2(f.to(torch.float64), dim=(1, 2))[:, keep]        # (b, M, n_t)
        mu, beta = self.poles(kx, ky)                                         # (N, M)
        g, c = foh_kernel(mu, n_t, dt)
        g, c = (beta[..., None] * g).sum(0), (beta[..., None] * c).sum(0)     # (M, n_t)
        vk = torch.zeros(b_, n_x, n_y // 2 + 1, n_t, dtype=torch.complex128, device=f.device)
        vk[:, keep] = foh_response(g[None], c[None], fk)
        return torch.fft.irfft2(vk, s=(n_x, n_y), dim=(1, 2)).to(f.dtype)


class WavenumberPoleConv1d(WavenumberPoleLayer1d):
    """Channel-mixing version of `WavenumberPoleLayer1d` for deep models (proposal A, Section 4.4):

        v_o(k, s) = sum_n sum_i beta_{n,o,i}(k) / (s - mu_n(k)) f_i(k, s).

    The N poles mu_n(k) are shared by all channel pairs (the modal form: one dispersion relation per pole); each pole
    has a residue matrix beta_n(k), polynomial in k^2. Pole options as in `WavenumberPoleLayer1d`.
    `beta_powers` (optional) gives the powers of k / k_ref in beta_n(k) instead of (0, 2, ..., 2 beta_degree - 2).
    Odd powers are valid (only k >= 0 is evaluated, the field stays real) and let a layer apply odd symbols such as
    the derivative i k, which even polynomials cannot.
    Input (batch, c_in, n_x, n_t), output (batch, c_out, n_x, n_t).
    """

    def __init__(self, c_in: int, c_out: int, n_poles: int, t, beta_degree: int = 2, ic: bool = False,
                 beta_powers: tuple | None = None, **kw):
        super().__init__(n_poles, t, beta_degree=1, **kw)
        del self.beta
        self.c_in, self.c_out = c_in, c_out
        self.beta_powers = tuple(beta_powers) if beta_powers is not None else tuple(2 * j for j in range(beta_degree))
        b = torch.zeros(n_poles, len(self.beta_powers), c_out, c_in, 2, dtype=torch.float64)
        b[:, 0, :, :, 0] = torch.randn(n_poles, c_out, c_in, dtype=torch.float64) / math.sqrt(c_in * n_poles)
        self.beta = nn.Parameter(b)
        # initial-state path (D2): free response of an initial channel field, gamma matrices share the poles
        self.gamma = nn.Parameter(b[:, :1].clone()) if ic else None          # constant in k (budget)

    def poles(self, k: torch.Tensor):
        """(mu (N, K), beta (N, c_out, c_in, K)) complex128."""
        kr = k / self.k_ref
        b = sum(torch.complex(self.beta[:, j, ..., 0, None], self.beta[:, j, ..., 1, None]) * kr ** p
                for j, p in enumerate(self.beta_powers))
        return self.mu(k), b

    def forward(self, f: torch.Tensor, v0: torch.Tensor | None = None) -> torch.Tensor:
        b_, c_in, n_x, n_t = f.shape
        dt = float(self.t[1] - self.t[0])
        fk = torch.fft.rfft(f.to(torch.float64), dim=2)
        K = fk.shape[2] if self.modes is None else min(self.modes, fk.shape[2])
        fk = fk[:, :, :K]                                                     # (b, c_in, K, n_t)
        mu, beta = self.poles(self.wavenumbers(n_x)[:K])                      # (N, K), (N, c_out, c_in, K)
        free = None
        if v0 is not None:                                                    # free response of the initial field
            k2 = (self.wavenumbers(n_x)[:K] / self.k_ref) ** 2
            gam = sum(torch.complex(self.gamma[:, j, ..., 0, None], self.gamma[:, j, ..., 1, None]) * k2 ** j
                      for j in range(self.gamma.shape[1]))                   # (N, c_out, c_in, K)
            v0k = torch.fft.rfft(v0.to(torch.float64), dim=2)[:, :, :K]       # (b, c_in, K)
            ej = torch.exp(mu[..., None] * dt * torch.arange(n_t, device=f.device, dtype=torch.float64))  # (N, K, n_t)
            free = torch.einsum("noik,bik,nkt->bokt", gam, v0k, ej)
        g, c = foh_kernel(mu, n_t, dt)                                         # (N, K, n_t)
        # sum over poles first (the FOH response is linear in the kernel): one causal convolution per channel pair
        # in the frequency domain, H_oi(k, w) = sum_n beta_noi(k) G_n(k, w), instead of N x c_in convolutions
        H = torch.einsum("noik,nkw->oikw", beta, torch.fft.fft(g, 2 * n_t))   # (c_out, c_in, K, 2 n_t)
        C = torch.einsum("noik,nkt->oikt", beta, c)                            # first-sample correction kernels
        Fw = torch.fft.fft(fk, 2 * n_t)                                        # (b, c_in, K, 2 n_t)
        vk = torch.fft.ifft(torch.einsum("oikw,bikw->bokw", H, Fw))[..., :n_t]
        vk = vk - torch.einsum("oikt,bik->bokt", C, fk[..., 0])
        if free is not None:
            vk = vk + free
        return torch.fft.irfft(vk, n=n_x, dim=2).to(f.dtype)


def set_domain(model: nn.Module, length) -> nn.Module:
    """Move every wavenumber pole layer in `model` to a periodic domain of another size, at the same grid spacing.

    The poles and residues are functions of the physical wavenumber, so a model trained on one domain is evaluated
    on another by changing the layers' `length` and scaling their mode count by the same factor (the same physical
    band). The caller gives inputs on the new grid (n_x scaled by the same factor). `length` is a float (1D) or a
    pair (2D, equal factors on both axes, because a 2D layer keeps one mode count). Returns `model`, changed in
    place. Pointwise parts are unchanged; FNO-type spectral weights, indexed by mode number, have no counterpart
    (`experiments/A/deep`, a34)."""
    for m in model.modules():
        if isinstance(m, WavenumberPoleLayer2d):
            new = tuple(float(a) for a in length)
            r = {n / o for n, o in zip(new, m.length)}
            if len(r) != 1:
                raise ValueError("a 2D layer needs the same scale factor on both axes")
            r = r.pop()
            m.length = new
        elif isinstance(m, WavenumberPoleLayer1d):
            r = float(length) / m.length
            m.length = float(length)
        else:
            continue
        if m.modes is not None:
            m.modes = int(round(m.modes * r))
    return model


class WavenumberPoleOperator1d(nn.Module):
    """Deep wavenumber pole operator (proposal A, Section 4.4), 1D space x time.

    lift (pointwise, 1 -> width) -> n_layers x [ v <- act(K_l v + W_l v) ] -> projection (pointwise, width -> hidden
    -> 1). K_l is a `WavenumberPoleConv1d`, W_l a pointwise channel mix; no activation after the last layer. Every
    part is causal in time, so the operator is causal. Input / output (batch, n_x, n_t) real.
    """

    def __init__(self, t, width: int = 16, n_layers: int = 4, n_poles: int = 4, hidden: int = 64, ic: bool = False,
                 **layer_kw):
        super().__init__()
        f64 = dict(dtype=torch.float64)
        self.lift = nn.Linear(1, width, **f64)
        # initial field (D2): lifted pointwise and fed as the initial state of the first pole layer (shared poles)
        self.lift0 = nn.Linear(1, width, bias=False, **f64) if ic else None
        self.K = nn.ModuleList(WavenumberPoleConv1d(width, width, n_poles, t, ic=ic and i == 0, **layer_kw)
                               for i in range(n_layers))
        self.W = nn.ModuleList(nn.Linear(width, width, **f64) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Linear(width, hidden, **f64), nn.GELU(), nn.Linear(hidden, 1, **f64))

    def forward(self, f: torch.Tensor, u0: torch.Tensor | None = None) -> torch.Tensor:
        v = self.lift(f.to(torch.float64)[..., None]).permute(0, 3, 1, 2)    # (b, width, n_x, n_t)
        v0 = None if u0 is None else self.lift0(u0.to(torch.float64)[..., None]).permute(0, 2, 1)   # (b, width, n_x)
        for i, (K, W) in enumerate(zip(self.K, self.W)):
            v = (K(v, v0) if i == 0 and v0 is not None else K(v)) + W(v.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            if i < len(self.K) - 1:
                v = F.gelu(v)
        return self.proj(v.permute(0, 2, 3, 1))[..., 0].to(f.dtype)


class WavenumberPoleConv2d(WavenumberPoleLayer2d):
    """Channel-mixing version of `WavenumberPoleLayer2d` (as `WavenumberPoleConv1d` in 1D): N poles mu_n(k) shared by
    all channel pairs, a residue matrix beta_n(k) per pole (polynomial in |k|^2, imaginary part with the half-plane
    sign). Input (batch, c_in, n_x, n_y, n_t), output (batch, c_out, n_x, n_y, n_t)."""

    def __init__(self, c_in: int, c_out: int, n_poles: int, t, beta_degree: int = 2, ic: bool = False, **kw):
        super().__init__(n_poles, t, beta_degree=1, **kw)
        del self.beta
        self.c_in, self.c_out = c_in, c_out
        b = torch.zeros(n_poles, beta_degree, c_out, c_in, 2, dtype=torch.float64)
        b[:, 0, :, :, 0] = torch.randn(n_poles, c_out, c_in, dtype=torch.float64) / math.sqrt(c_in * n_poles)
        self.beta = nn.Parameter(b)
        # initial-state path (D2): free response of an initial channel field, gamma matrices share the poles
        self.gamma = nn.Parameter(b[:, :1].clone()) if ic else None          # constant in k (budget)

    def poles(self, kx: torch.Tensor, ky: torch.Tensor):
        """(mu (N, M), beta (N, c_out, c_in, M)) complex128 for 1D tensors kx, ky of length M."""
        k2 = (kx / self.k_ref) ** 2 + (ky / self.k_ref) ** 2
        sgn = _half_plane_sign(kx, ky)
        b = sum(torch.complex(self.beta[:, j, ..., 0, None], sgn * self.beta[:, j, ..., 1, None]) * k2 ** j
                for j in range(self.beta.shape[1]))
        return self.mu(kx, ky), b

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b_, c_in, n_x, n_y, n_t = f.shape
        dt = float(self.t[1] - self.t[0])
        keep = self.mask(n_x, n_y)
        kx, ky = self.wavenumbers(n_x, n_y)
        kx, ky = kx[keep], ky[keep]
        fk = torch.fft.rfft2(f.to(torch.float64), dim=(2, 3))[:, :, keep]   # (b, c_in, M, n_t)
        mu, beta = self.poles(kx, ky)
        g, c = foh_kernel(mu, n_t, dt)                                         # (N, M, n_t)
        H = torch.einsum("noik,nkw->oikw", beta, torch.fft.fft(g, 2 * n_t))
        C = torch.einsum("noik,nkt->oikt", beta, c)
        vk_m = torch.fft.ifft(torch.einsum("oikw,bikw->bokw", H, torch.fft.fft(fk, 2 * n_t)))[..., :n_t]
        vk_m = vk_m - torch.einsum("oikt,bik->bokt", C, fk[..., 0])
        vk = torch.zeros(b_, self.c_out, n_x, n_y // 2 + 1, n_t, dtype=torch.complex128, device=f.device)
        vk[:, :, keep] = vk_m
        return torch.fft.irfft2(vk, s=(n_x, n_y), dim=(2, 3)).to(f.dtype)


class WavenumberPoleOperator2d(nn.Module):
    """2D version of `WavenumberPoleOperator1d`: input / output (batch, n_x, n_y, n_t) real."""

    def __init__(self, t, width: int = 16, n_layers: int = 4, n_poles: int = 4, hidden: int = 64, **layer_kw):
        super().__init__()
        f64 = dict(dtype=torch.float64)
        self.lift = nn.Linear(1, width, **f64)
        self.K = nn.ModuleList(WavenumberPoleConv2d(width, width, n_poles, t, **layer_kw) for _ in range(n_layers))
        self.W = nn.ModuleList(nn.Linear(width, width, **f64) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Linear(width, hidden, **f64), nn.GELU(), nn.Linear(hidden, 1, **f64))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        v = self.lift(f.to(torch.float64)[..., None]).permute(0, 4, 1, 2, 3)   # (b, width, n_x, n_y, n_t)
        for i, (K, W) in enumerate(zip(self.K, self.W)):
            v = K(v) + W(v.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            if i < len(self.K) - 1:
                v = F.gelu(v)
        return self.proj(v.permute(0, 2, 3, 4, 1))[..., 0].to(f.dtype)
