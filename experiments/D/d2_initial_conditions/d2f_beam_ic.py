"""(D2) A damped beam with initial displacement and velocity, observed through its displacement.

The plan's problem table lists the Euler–Bernoulli beam with non-zero initial displacement and velocity. So far the
IC path has had one initial field and a first-order mode per pole (gamma_n(k) a polynomial law). A second-order
system has two initial fields per mode, and through its displacement u the coefficients are not polynomial in k.

**System.** u_tt = -c^2 u_xxxx + eta u_xxt + f on periodic x in [0, 2 pi): per mode, u'' + eta k^2 u' + c^2 k^4 u = f,
poles mu_{1,2}(k) with mu1 + mu2 = -eta k^2, mu1 mu2 = c^2 k^4. c = 0.04, eta = 0.002 (damping ratio 0.025). With
x_n the modal states and r = 1 / (mu1 - mu2):

    u = x1 + x2,   x_n(t) = a_n e^{mu_n t} + (+-r) [e^{mu_n .} * f](t),
    a1 = (v0 - mu2 u0) r,   a2 = (mu1 u0 - v0) r,   u_t = mu1 x1 + mu2 x2.

The forcing residues and the coefficient of v0 are +-1 / (mu1 - mu2), proportional to 1 / k^2: not a polynomial in
k (`a4_beam.py` used the velocity output to avoid this). They are fixed by the poles.

**Data.** n_x = 64, T = 5, n_t = 201, forcing as in a4 (4 modes in k = 1..16, E1 profiles). Initial fields u0, v0
on modes 1..16: u0 amplitude ~ s / k, v0 amplitude ~ s c k (so c k^2 u0 and v0 are comparable), with s log-uniform
in [0.3, 3] times a fixed calibration that makes the free and forced parts equal in mean norm (set from the data
alone). Truth: the exact modal solution with forcing on a 32x finer time grid. Splits 200 / 50 / 100, data seeds
0-4. Rollout: 50 trajectories over [0, 20], forcing drawn per window and concatenated.

**Models** (mode 0 is excluded; it is not excited). Poles from two laws in k^2 (k / k_ref, k_ref = 16), each on
{1, k^2, k^4}: the sum s(k) = mu1 + mu2 and the product p(k) = mu1 mu2 (6 reals).
- `struct`: residues +-g(k) / (mu1 - mu2) with g on {1, k^2} (true 1, 0), and a_n from the poles as above. 8 reals.
- `poly`: the same poles; forcing residue and the two IC coefficient laws per pole are free complex polynomials on
  {1, k^2}, as in the shared layer's gamma: 6 + 24 = 30 reals.
- `struct_u0`: `struct` with v0 withheld (the IC path of the earlier passes: one initial field).
- `oracle`: `struct` at the true parameters (the FOH representation floor).
Fit: L-BFGS from 8 random starts (c0 log-uniform in [0.01, 0.1], eta0 in [1e-4, 1e-2]), pick by validation.

**Rollout.** Each model passes (u, u_t) at the window end, u_t = mu1 x1 + mu2 x2 from its own modal states (only
u is observed, so the velocity must come from the model).

**Checks before any fit** (seed 0). IC scale 1.24. Oracle test error 2.8e-3; oracle 4-window rollout 6.2e-3. The
oracle's end-of-window velocity agrees with a central difference of the truth to 3e-3 (the difference's own error).

**Predictions** (written before running).

1. `struct`, in every seed: test error <= 1.2x the oracle, c and eta within 1 %, and 4-window rollout error
   <= 1.05x the oracle's rollout error.
2. `poly`: >= 3x the oracle in every seed.
3. `struct_u0`: >= 5x the oracle in every seed.

    python experiments/D/d2_initial_conditions/d2f_beam_ic.py      # results/d2f_beam_ic.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "A" / "layer"))
import a1_dispersion as A  # noqa: E402
from lfo.models.wavenumber import foh_kernel, foh_response  # noqa: E402

C_B, ETA, K_REF, KMAX, N_WIN = 0.04, 0.002, 16.0, 16, 4
KS = np.arange(1, KMAX + 1)
OUT = HERE / "results" / "d2f_beam_ic.json"


def true_poles(k):
    s, p = -ETA * k ** 2, C_B ** 2 * k ** 4
    d = np.sqrt((s ** 2 / 4 - p).astype(complex))
    return s / 2 + d, s / 2 - d


def forcing(rng, n, tf):
    K = A.N_X // 2 + 1
    fk = np.zeros((n, K, tf.size), complex)
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        km = rng.integers(1, KMAX + 1, size=n)
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * A.N_X / 2
        fk[np.arange(n), km] += amp[:, None] * prof
    return fk


def ic_fields(rng, n, scale):
    K = A.N_X // 2 + 1
    s = np.exp(rng.uniform(np.log(0.3), np.log(3.0), size=n)) * scale
    z = lambda: rng.normal(size=(n, KMAX)) + 1j * rng.normal(size=(n, KMAX))
    u0k, v0k = np.zeros((n, K), complex), np.zeros((n, K), complex)
    u0k[:, 1:KMAX + 1] = z() / KS * A.N_X / 2 * s[:, None]
    v0k[:, 1:KMAX + 1] = z() * C_B * KS * A.N_X / 2 * s[:, None]
    return u0k, v0k


def solve(fk, dtf, u0k, v0k, stride):
    """Exact modal solution: forcing FOH on the fine grid, then subsampled. Returns u_hat (n, K, n_t)."""
    K = fk.shape[1]
    kk = np.arange(K, dtype=float)
    m1, m2 = true_poles(kk)
    out = np.zeros(fk.shape[:2] + ((fk.shape[2] - 1) // stride + 1,), complex)
    tf = np.arange(fk.shape[2]) * dtf
    for k in KS:
        r = 1.0 / (m1[k] - m2[k])
        forced = r * (A.foh_np(np.array([m1[k]]), fk[:, k:k + 1], dtf) - A.foh_np(np.array([m2[k]]), fk[:, k:k + 1], dtf))[:, 0]
        a1, a2 = (v0k[:, k] - m2[k] * u0k[:, k]) * r, (m1[k] * u0k[:, k] - v0k[:, k]) * r
        free = a1[:, None] * np.exp(m1[k] * tf) + a2[:, None] * np.exp(m2[k] * tf)
        out[:, k] = (forced + free)[:, ::stride]
    return out


def to_x(zk):
    return np.fft.irfft(zk, n=A.N_X, axis=1)


def sample(rng, n, n_win, scale):
    tf1 = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    fks = [forcing(rng, n, tf1) for _ in range(n_win)]
    fk = np.concatenate([fks[0]] + [f[..., 1:] for f in fks[1:]], -1)
    u0k, v0k = ic_fields(rng, n, scale)
    uk = solve(fk, tf1[1] - tf1[0], u0k, v0k, A.FINE)
    t = lambda a: torch.tensor(a)
    return t(to_x(fk[..., ::A.FINE])), t(to_x(u0k)), t(to_x(v0k)), t(to_x(uk))


def calibrate():
    """IC scale so that free and forced parts have equal mean norm (data only, seed 99)."""
    rng = np.random.default_rng(99)
    tf1 = np.linspace(0, A.T_END, (A.N_T - 1) * A.FINE + 1)
    fk = forcing(rng, 100, tf1)
    u0k, v0k = ic_fields(rng, 100, 1.0)
    z = np.zeros_like(u0k)
    forced = solve(fk, tf1[1] - tf1[0], z, z, A.FINE)
    free = solve(np.zeros_like(fk), tf1[1] - tf1[0], u0k, v0k, A.FINE)
    return float(np.linalg.norm(forced) / np.linalg.norm(free))


def make_data(seed, scale):
    rng = np.random.default_rng(700 + seed)
    f, u0, v0, u = sample(rng, sum(A.SPLIT), 1, scale)
    c = np.cumsum([0, *A.SPLIT])
    d = {k: (f[a:b], u0[a:b], v0[a:b], u[a:b]) for k, a, b in zip(("train", "vali", "test"), c[:-1], c[1:])}
    d["roll"] = sample(np.random.default_rng(800 + seed), 50, N_WIN, scale)
    return d


class Beam(nn.Module):
    """Two poles per mode from sum / product laws; residues and IC coefficients by `kind` (module docstring)."""

    def __init__(self, t, kind):
        super().__init__()
        self.kind, self.n_t, self.dt = kind, len(t), float(t[1] - t[0])
        self.register_buffer("q", torch.tensor(KS / K_REF, dtype=torch.float64) ** 2)
        self.s = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.p = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        if kind == "poly":
            self.res = nn.Parameter(torch.zeros(3, 2, 2, 2, dtype=torch.float64))   # (law, pole, degree, re/im)
        else:
            self.g = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))

    def law(self, c):
        return c[0] + c[1] * self.q + (c[2] * self.q ** 2 if len(c) > 2 else 0)

    def poles(self):
        s, p = self.law(self.s), self.law(self.p)
        d = torch.sqrt((s ** 2 / 4 - p).to(torch.complex128))
        return s / 2 + d, s / 2 - d

    def modal(self, f, u0, v0):
        """Modal states x_1, x_2: (b, K_used, n_t) each."""
        m1, m2 = self.poles()
        fk = torch.fft.rfft(f, dim=1)[:, 1:KMAX + 1]
        u0k = torch.fft.rfft(u0, dim=1)[:, 1:KMAX + 1]
        v0k = torch.fft.rfft(v0, dim=1)[:, 1:KMAX + 1] if self.kind != "struct_u0" else torch.zeros_like(u0k)
        j = torch.arange(self.n_t, device=f.device, dtype=torch.float64) * self.dt
        if self.kind == "poly":
            r = torch.complex(self.res[..., 0], self.res[..., 1])                     # (3, 2, 2)
            lw = r[:, :, 0, None] + r[:, :, 1, None] * self.q                          # (3, 2, K)
            rf, gu, gv = lw[0], lw[1], lw[2]
            a = (gu[0] * u0k + gv[0] * v0k, gu[1] * u0k + gv[1] * v0k)
            rr = (rf[0], rf[1])
        else:
            r = self.law(self.g) / (m1 - m2)
            a = ((v0k - m2 * u0k) / (m1 - m2), (m1 * u0k - v0k) / (m1 - m2))
            rr = (r, -r)
        xs = []
        for mu, an, rn in zip((m1, m2), a, rr):
            g, c = foh_kernel(mu, self.n_t, self.dt)
            xs.append(rn[None, :, None] * foh_response(g, c, fk.to(torch.complex128))
                      + an[..., None] * torch.exp(mu[:, None] * j))
        return xs, (m1, m2)

    def full(self, xk):
        K = A.N_X // 2 + 1
        z = torch.zeros(xk.shape[0], K, xk.shape[-1], dtype=torch.complex128, device=xk.device)
        z[:, 1:KMAX + 1] = xk
        return torch.fft.irfft(z, n=A.N_X, dim=1)

    def forward(self, f, u0, v0):
        xs, _ = self.modal(f, u0, v0)
        return self.full(xs[0] + xs[1])

    def end_velocity(self, f, u0, v0):
        xs, (m1, m2) = self.modal(f, u0, v0)
        return self.full(xs[0] + xs[1]), self.full((m1[:, None] * xs[0] + m2[:, None] * xs[1]))[..., -1]

    def set_true(self):
        with torch.no_grad():
            self.s.zero_(), self.p.zero_()
            self.s[1] = -ETA * K_REF ** 2
            self.p[2] = C_B ** 2 * K_REF ** 4
            if self.kind != "poly":
                self.g.copy_(torch.tensor([1.0, 0.0]))

    def constants(self):
        return math.sqrt(max(float(self.p[2]), 0.0)) / K_REF ** 2, -float(self.s[1]) / K_REF ** 2


def lbfgs(m, d, iters=400):
    opt = torch.optim.LBFGS(m.parameters(), max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50)

    def closure():
        opt.zero_grad()
        loss = A.rel(m(*d[:3]), d[3])
        if torch.isfinite(loss):
            loss.backward()
        return loss

    try:
        opt.step(closure)
    except (RuntimeError, IndexError):
        pass


def val(m, d):
    with torch.no_grad():
        v = float(A.rel(m(*d[:3]), d[3]))
    return v if math.isfinite(v) else float("inf")


def fit(kind, data, t, device):
    tr, va = ([z.to(device) for z in data[s]] for s in ("train", "vali"))
    best = None
    for st in range(8):
        g = np.random.default_rng(st)
        m = Beam(t, kind).to(device)
        c0, e0 = math.exp(g.uniform(math.log(0.01), math.log(0.1))), math.exp(g.uniform(math.log(1e-4), math.log(1e-2)))
        with torch.no_grad():
            m.s[1], m.p[2] = -e0 * K_REF ** 2, c0 ** 2 * K_REF ** 4
            if kind == "poly":
                m.res.normal_(0, 0.3, generator=torch.Generator(device=device).manual_seed(st))
        lbfgs(m, tr)
        v = val(m, va)
        if best is None or v < best[0]:
            best = (v, m)
    return best[1]


@torch.no_grad()
def rollout(m, f, u0, v0, n_win):
    w = A.N_T - 1
    out, su, sv = [u0[..., None]], u0, v0
    for i in range(n_win):
        seg, vend = m.end_velocity(f[..., i * w:(i + 1) * w + 1], su, sv)
        out.append(seg[..., 1:])
        su, sv = seg[..., -1], vend
    return torch.cat(out, -1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, A.T_END, A.N_T)
    scale = calibrate()
    print("IC scale", scale, flush=True)
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    for seed in range(5):
        data = make_data(seed, scale)
        te = [z.to(device) for z in data["test"]]
        ro = [z.to(device) for z in data["roll"]]
        orc = Beam(t, "struct").to(device)
        orc.set_true()
        floor, roll_floor = val(orc, te), float(A.rel(rollout(orc, *ro[:3], N_WIN), ro[3]))
        for kind in ("struct", "poly", "struct_u0"):
            if any(r["seed"] == seed and r["model"] == kind for r in out):
                continue
            t0 = time.time()
            m = fit(kind, data, t, device)
            test = val(m, te)
            roll = float(A.rel(rollout(m, *ro[:3], N_WIN), ro[3]))
            c, eta = m.constants()
            r = dict(model=kind, seed=seed, oracle=floor, ratio=test / floor, roll=roll, roll_oracle=roll_floor,
                     roll_ratio=roll / roll_floor, c=c, eta=eta, seconds=round(time.time() - t0, 1))
            out.append(r)
            print({q: (round(v, 6) if isinstance(v, float) else v) for q, v in r.items()}, flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D2) Damped beam with initial displacement and velocity, displacement output", "",
             f"c = {C_B}, eta = {ETA}; IC scale {scale:.3g}. Range over data seeds 0–4.", "",
             "| model | test / oracle | ĉ | η̂ | rollout / oracle rollout (4 windows) |", "|---|---|---|---|---|"]
    for kind in ("struct", "poly", "struct_u0"):
        rs = [r for r in out if r["model"] == kind]
        rg = lambda q, f: f"{min(r[q] for r in rs):{f}}–{max(r[q] for r in rs):{f}}"
        lines.append(f"| {kind} | {rg('ratio', '.2f')} | {rg('c', '.5f')} | {rg('eta', '.6f')} | {rg('roll_ratio', '.3f')} |")
    lines += ["", f"Oracle test error {min(r['oracle'] for r in out):.2e}–{max(r['oracle'] for r in out):.2e}; "
              f"oracle rollout error {min(r['roll_oracle'] for r in out):.2e}–{max(r['roll_oracle'] for r in out):.2e}."]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
