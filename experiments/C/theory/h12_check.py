"""(C) Numerical checks of the H1 / H2 statements in `H1_H2.md` (single-channel pole-residue layer, float64).

1. Lemma 1 (the tied layer is the zero-initial-state response of the trigonometric interpolant): the layer
   output against an independent first-order-hold solve of the interpolant on a 32x finer grid.
2. Theorem 2 (sparse grid-line input, exact recovery): output of the tied layer for a multisine on DFT lines,
   poles recovered by a matrix pencil of order |S| + N_tot, the |S| nodes on the unit circle discarded.
3. Proposition 3 (generic local identifiability from one input): singular values of the Jacobian of the tied
   layer (minimal form) for one input of each kind.
4. Theorem 4 / Proposition 5 (free steady branch): information on each Re mu, profiled over the free steady
   symbol R (all lines) and the other parameters, against the tied layer; one input, circular shifts of one
   input, two inputs, the E1 training set.

    python experiments/C/theory/h12_check.py            # writes results/h12_check.md
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e1_audit"))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import identifiability as ID  # noqa: E402
from lfo.data import lti  # noqa: E402

torch.set_default_dtype(torch.float64)
RESULTS = HERE / "results"


class Model:
    """Minimal-form tied layer: one slot per conjugate pair (residue 2 beta) and one per real pole.
    Real parameter vector v: complex slot (Re mu, Im mu, Re b, Im b), real slot (Re mu, Re b)."""

    def __init__(self, mu, beta, t):
        self.cplx = np.asarray(mu).imag != 0
        self.v0 = torch.tensor(np.concatenate([[z.real, z.imag, b.real, b.imag] if c else [z.real, b.real]
                                               for z, b, c in zip(mu, beta, self.cplx)]))
        self.t = t
        self.n = len(t)
        self.iw = 2j * math.pi * torch.fft.fftfreq(self.n, float(t[1] - t[0]))

    def unpack(self, v):
        mu, b, i = [], [], 0
        for c in self.cplx:
            if c:
                mu.append(torch.complex(v[i], v[i + 1]))
                b.append(torch.complex(v[i + 2], v[i + 3]))
                i += 4
            else:
                mu.append(torch.complex(v[i], torch.zeros_like(v[i])))
                b.append(torch.complex(v[i + 1], torch.zeros_like(v[i])))
                i += 2
        return torch.stack(mu), torch.stack(b)

    def branches(self, v, x):
        """(transient, steady) of the tied layer, each (J, n); same arithmetic as identifiability.layer_out."""
        mu, b = self.unpack(v)
        alpha = torch.fft.fft(x.to(torch.complex128))
        H = b[None, :] / (self.iw[:, None] - mu[None, :])
        gamma = -(alpha @ H) / self.n
        tr = torch.real(gamma @ torch.exp(mu[:, None] * self.t[None, :].to(torch.complex128)))
        lam = alpha * H.sum(1)[None, :]
        pos = torch.arange(self.n // 2 + 1)
        c = (lam[:, pos] + torch.conj(lam[:, (-pos) % self.n])) / 2
        return tr, torch.fft.irfft(c, n=self.n)

    def out(self, v, x):
        tr, st = self.branches(v, x)
        return tr + st

    def jac(self, f, x):
        """(J, n, p) Jacobian of f(v, x) at v0 (forward mode)."""
        return torch.func.jacfwd(lambda v: f(v, x))(self.v0)

    def re_index(self):
        out, i = [], 0
        for c in self.cplx:
            out.append(i)
            i += 4 if c else 2
        return out


# ---- 1. Lemma 1 --------------------------------------------------------------------------------------------

def check_lemma1(system="p6", n_in=4, refine=32):
    t, x, _, m, b = ID.load(system)
    mdl = Model(m, b, t)
    x = x[:n_in]
    y_layer = mdl.out(mdl.v0, x)
    n, dt = len(t), float(t[1] - t[0])
    tf = np.arange(n * refine) * dt / refine
    alpha = np.fft.fft(x.numpy(), axis=1)
    w = 2 * np.pi * np.fft.fftfreq(n, dt)
    xt = np.real(np.exp(1j * np.outer(tf, w)) @ alpha.T / n).T            # trigonometric interpolant (Re)
    mu_full = np.concatenate([[z, np.conj(z)] if z.imag else [z] for z in m])
    b_full = np.concatenate([[bb / 2, np.conj(bb) / 2] if z.imag else [bb] for z, bb in zip(m, b)])
    y_ode = lti.response(tf, xt, mu_full, b_full)[:, ::refine]
    return float(np.linalg.norm(y_ode - y_layer.numpy()) / np.linalg.norm(y_layer.numpy()))


# ---- 2. Theorem 2: exact recovery from a sparse grid-line input ------------------------------------------------

def pencil(y, M):
    """Matrix pencil: M nodes z of y_k = sum c_m z_m^k (noise-free)."""
    n = len(y)
    L = n // 2
    Y = np.array([y[i:i + L + 1] for i in range(n - L)])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh[:M].conj().T
    V1, V2 = V[:-1], V[1:]
    return np.linalg.eigvals(np.linalg.pinv(V1) @ V2), s


def check_theorem2(system="p6", lines=(3, 11)):
    t, _, _, m, b = ID.load(system)
    mdl = Model(m, b, t)
    n, dt = len(t), float(t[1] - t[0])
    x = sum(torch.cos(2 * math.pi * l / (n * dt) * t + 0.3 * l) for l in lines)[None]
    y = mdl.out(mdl.v0, x)[0].numpy()
    n_tot = int(sum(2 if c else 1 for c in mdl.cplx))
    S = 2 * len(lines)
    z, s = pencil(y, S + n_tot)
    mu_hat = np.log(z) / dt
    poles = sorted([u for u in mu_hat if abs(abs(np.exp(u * dt)) - 1) > 1e-6], key=lambda u: (u.real, u.imag))
    true = sorted(np.concatenate([[u, np.conj(u)] if u.imag else [u] for u in m]), key=lambda u: (u.real, u.imag))
    err = max(min(abs(p - q) / abs(q) for p in poles) for q in true) if len(poles) == len(true) else float("nan")
    return dict(lines=list(lines), order=S + n_tot, found=len(poles), n_tot=n_tot, max_rel_err=float(err),
                sv_gap=float(s[S + n_tot - 1] / s[S + n_tot]))


# ---- 3. Proposition 3: local identifiability from one input --------------------------------------------------

def check_prop3(system):
    t, x, _, m, b = ID.load(system)
    mdl = Model(m, b, t)
    n, dt = len(t), float(t[1] - t[0])
    gen = torch.Generator().manual_seed(0)
    cases = {"grid cosine (line 5)": torch.cos(2 * math.pi * 5 / (n * dt) * t)[None],
             "E1 input 0": x[:1], "E1 input 1": x[1:2],
             "white noise": torch.randn(1, n, generator=gen)}
    out = {}
    for k, xi in cases.items():
        Jm = mdl.jac(mdl.out, xi)[0]                                    # (n, p)
        Jm = Jm / Jm.norm(dim=0, keepdim=True)                          # column scaling
        s = torch.linalg.svdvals(Jm)
        out[k] = (float(s[-1]), float(s[0] / s[-1]))
    return out, int(mdl.v0.numel())


# ---- 4. Theorem 4 / Proposition 5: free steady branch ----------------------------------------------------------

def profiled_info(mdl, x, w, untied):
    """1 / (F^-1)_kk for each Re mu: tied layer (F of the output), or transient branch profiled over a free
    steady symbol R_l on every DFT line (per-line weighted projection onto the inputs' spectra)."""
    f = (lambda v, xx: mdl.branches(v, xx)[0]) if untied else mdl.out
    J = mdl.jac(f, x)                                                    # (J, n, p)
    D = torch.fft.fft(J.to(torch.complex128), dim=1)                     # (J, n, p)
    if untied:
        A = torch.fft.fft(x.to(torch.complex128), dim=1)                 # (J, n)
        den = (w[:, None] * A.abs() ** 2).sum(0)                         # (n,)
        num = torch.einsum("j,jl,jlp->lp", w.to(torch.complex128), A.conj(), D)
        coef = torch.where(den[:, None] > 0, num / den.clamp_min(1e-300)[:, None], 0)
        D = D - A[:, :, None] * coef[None]
    F = torch.einsum("j,jlp,jlq->pq", w.to(torch.complex128), D.conj(), D).real / mdl.n
    Fi = torch.linalg.pinv(F, rtol=1e-13, hermitian=True)
    idx = mdl.re_index()
    return np.array([1 / float(Fi[k, k]) if Fi[k, k] > 0 else 0.0 for k in idx]), float(torch.linalg.eigvalsh(F)[0] / torch.linalg.eigvalsh(F)[-1])


def check_theorem4(system):
    t, x, y, m, b = ID.load(system)
    mdl = Model(m, b, t)
    gen = torch.Generator().manual_seed(1)
    x0 = x[:1]
    sets = {"1 E1 input": x0,
            "5 circular shifts of it": torch.cat([torch.roll(x0, 97 * k, dims=1) for k in range(5)]),
            "2 E1 inputs": x[:2],
            "1 white-noise input": torch.randn(1, len(t), generator=gen),
            "2 white-noise inputs": torch.randn(2, len(t), generator=gen),
            f"E1 training set ({x.shape[0]})": x}
    rows = []
    for k, xs in sets.items():
        ys = mdl.out(mdl.v0, xs)
        w = 1 / ys.pow(2).sum(1)
        it, _ = profiled_info(mdl, xs, w, untied=False)
        iu, _ = profiled_info(mdl, xs, w, untied=True)
        rows.append((k, it, iu))
    return rows, m


def main():
    lines = ["# H1 / H2: numerical checks (`h12_check.py`)", "",
             "Single-channel pole-residue layer in float64, true parameters of the E1 systems (minimal form), "
             "E1 time grid (T = 20, n = 1024). See `H1_H2.md` for the statements.", ""]
    e = {s: check_lemma1(s) for s in ("p4", "p6")}
    lines += ["## Lemma 1: tied layer = zero-initial-state response of the trigonometric interpolant", "",
              "Relative difference between the tied layer and an independent first-order-hold solve "
              "(`lfo.data.lti.response`) of the trigonometric interpolant on a 32× finer grid, 4 E1 inputs: "
              + ", ".join(f"{s} {v:.1e}" for s, v in e.items()) + " (the residual is the hold's O(h²) error).", ""]
    lines += ["## Theorem 2: exact recovery from a sparse grid-line input", "",
              "Tied-layer output for a multisine on DFT lines S, matrix pencil of order |S| + N_tot; nodes on the "
              "unit circle (the input lines) discarded, the rest are the poles.", "",
              "| system | lines | order | poles found / N_tot | max relative pole error | singular-value gap at the order |",
              "|---|---|---|---|---|---|"]
    for s, ln in (("p4", (3, 11)), ("p6", (3, 11)), ("p6", (7,))):
        r = check_theorem2(s, ln)
        lines.append(f"| {s} | {r['lines']} | {r['order']} | {r['found']} / {r['n_tot']} | {r['max_rel_err']:.1e} | "
                     f"{r['sv_gap']:.1e} |")
    lines += ["", "## Proposition 3: local identifiability of the tied layer from one input", "",
              "Jacobian of the output for one input, columns scaled to unit norm; smallest singular value and "
              "condition number. Full rank = locally identifiable.", "",
              "| system | parameters | input | smallest singular value | condition number |", "|---|---|---|---|---|"]
    for s in ("p4", "p6"):
        res, p = check_prop3(s)
        for k, (smin, cond) in res.items():
            lines.append(f"| {s} | {p} | {k} | {smin:.1e} | {cond:.1e} |")
    lines += ["", "## Theorem 4 / Proposition 5: a free steady branch", "",
              "Information on each Re μ (1 / (F⁻¹)_kk, other parameters profiled; per-sample weights 1 / ‖y‖², "
              "as the E1 loss). Tied: the tied layer. Free R: transient branch tied to (μ, β), steady branch a free "
              "symbol R_l on every DFT line, profiled out. Ratio = free R / tied.", ""]
    for s in ("p4", "p6"):
        rows, m = check_theorem4(s)
        lab = [f"{z.real:g}{z.imag:+g}i" if z.imag else f"{z.real:g}" for z in m]
        lines += [f"### {s}", "", "| inputs | " + " | ".join(f"{l}: tied, ratio" for l in lab) + " |",
                  "|---|" + "---|" * len(lab)]
        for k, it, iu in rows:
            lines.append(f"| {k} | " + " | ".join(f"{a:.2e}, {c / a:.1e}" for a, c in zip(it, iu)) + " |")
        lines.append("")
    (RESULTS / "h12_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
