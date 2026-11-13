"""(C) E1: practical identifiability of each true pole for the tied (LNO v1) layer.

Question: the E1 runs never identify the fast real poles (-1, -2) although the fit is
good. Is the loss insensitive to them?

Model. One tied pole-residue layer in its minimal form: one slot per conjugate pair
(residue 2*beta) and one per real pole, computed in float64 with the same arithmetic as
``lfo.models.hybrid.PoleResidueTS1d`` (checked against it in ``check_against_layer``).
Data: the E1 training inputs of the system (``e1_audit.make_data``), exact outputs.
Error measure: per-sample relative residual r_i = (yhat_i - y_i) / ||y_i||, and
e = sqrt(mean_i ||r_i||^2) (RMS relative L2; close to the E1 test metric at small error).

1. Local (Gauss-Newton) at the truth. F = mean_i J_i^T J_i / ||y_i||^2 over real
   parameters (Re mu, Im mu, Re beta, Im beta for complex slots; Re mu, Re beta for real
   ones: for a real pole Im mu and Im beta have zero first-order effect, so a real pole
   sits at a flat point in those directions). For a relative change of 10 % in one
   parameter theta_k:
       fixed     e = 0.1 |theta_k| sqrt(F_kk)               (others held at the truth)
       profiled  e = 0.1 |theta_k| / sqrt((F^-1)_kk)        (others re-fitted, linearised)
2. Profile (nonlinear). Fix Re mu of one pole at c * Re mu* (c on a grid), re-fit all other
   parameters with L-BFGS from the truth (including the fixed pole's Im mu and residue,
   which are free in the real model), report the minimum e.

    python experiments/C/e1_audit/identifiability.py       # writes results/identifiability.md (+ .json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_audit as E  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
SYSTEMS = ("p2", "p4", "p6")
PROFILE_FACTORS = (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)


def minimal_poles(mu, beta):
    """One slot per conjugate pair (Im > 0, residue 2 beta) plus the real poles."""
    keep = mu.imag >= 0
    m, b = mu[keep], beta[keep]
    return m, np.where(m.imag > 0, 2 * b, b)


def layer_out(mu: torch.Tensor, beta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Tied layer output (transient + steady), float64; x (b, n_t) real -> (b, n_t)."""
    n = x.shape[-1]
    dt = float(t[1] - t[0])
    alpha = torch.fft.fft(x)                                              # (b, L)
    iw = 2j * math.pi * torch.fft.fftfreq(n, dt, dtype=torch.float64)    # (L,)
    H = beta[None, :] / (iw[:, None] - mu[None, :])                       # (L, N)
    gamma = -(alpha @ H) / n                                              # (b, N)
    tr = torch.real(gamma @ torch.exp(mu[:, None] * t[None, :].to(torch.complex128)))
    lam = alpha * H.sum(1)[None, :]
    pos = torch.arange(n // 2 + 1)
    c = (lam[:, pos] + torch.conj_physical(lam[:, (-pos) % n])) / 2
    return tr + torch.fft.irfft(c, n=n)


class Params:
    """Real parameter vector <-> (mu, beta). Real poles: Re mu, Re beta only unless ``full``."""

    def __init__(self, mu, beta, full: bool = False):
        self.mu0, self.beta0 = np.asarray(mu, complex), np.asarray(beta, complex)
        self.cplx = (self.mu0.imag != 0) | full
        self.names = []
        for j, z in enumerate(self.mu0):
            parts = ("Re mu", "Im mu", "Re beta", "Im beta") if self.cplx[j] else ("Re mu", "Re beta")
            self.names += [(j, p) for p in parts]

    def vector(self, mu=None, beta=None) -> np.ndarray:
        mu = self.mu0 if mu is None else mu
        beta = self.beta0 if beta is None else beta
        v = []
        for j, p in self.names:
            z = mu[j] if "mu" in p else beta[j]
            v.append(z.real if p.startswith("Re") else z.imag)
        return np.array(v)

    def unpack(self, v: torch.Tensor):
        n = self.mu0.size
        re_mu, im_mu = [torch.zeros((), dtype=torch.float64) for _ in range(n)], [torch.zeros((), dtype=torch.float64) for _ in range(n)]
        re_b, im_b = [torch.zeros((), dtype=torch.float64) for _ in range(n)], [torch.zeros((), dtype=torch.float64) for _ in range(n)]
        tgt = {"Re mu": re_mu, "Im mu": im_mu, "Re beta": re_b, "Im beta": im_b}
        for k, (j, p) in enumerate(self.names):
            tgt[p][j] = v[k]
        mu = torch.complex(torch.stack(re_mu), torch.stack(im_mu))
        beta = torch.complex(torch.stack(re_b), torch.stack(im_b))
        return mu, beta


def load(system: str):
    data, t, mu, beta = E.make_data(system)
    x, y = (a.double() for a in data["train"])
    m, b = minimal_poles(mu, beta)
    return torch.tensor(t), x, y, m, b


def check_against_layer(system: str = "p6") -> float:
    """Max relative difference between ``layer_out`` and the float32 ``PoleResidueTS1d``."""
    t, x, y, m, b = load(system)
    layer = E.PoleResidueTS1d(1, 1, m.size, t.float())
    with torch.no_grad():
        layer.mu.copy_(torch.tensor(m, dtype=torch.cfloat).reshape(1, 1, -1))
        layer.beta.copy_(torch.tensor(b, dtype=torch.cfloat).reshape(1, 1, -1))
        ref = layer(x.float()[:, None])[:, 0].double()
    ours = layer_out(torch.tensor(m), torch.tensor(b), x, t)
    return float((ours - ref).norm() / ref.norm())


def residual(P: Params, v, x, y, t):
    mu, beta = P.unpack(v)
    return (layer_out(mu, beta, x, t) - y) / y.norm(dim=1, keepdim=True)


def rms_error(P, v, x, y, t) -> float:
    r = residual(P, v, x, y, t)
    return float(torch.sqrt((r ** 2).sum(1).mean()))


def local_analysis(system: str) -> list[dict]:
    t, x, y, m, b = load(system)
    P = Params(m, b)
    v0 = torch.tensor(P.vector())
    J = torch.autograd.functional.jacobian(lambda v: residual(P, v, x, y, t).reshape(-1), v0)  # (N*T, p)
    F = (J.T @ J) / x.shape[0]
    Finv = torch.linalg.inv(F)
    ev = torch.linalg.eigvalsh(F)
    rows = []
    for k, (j, p) in enumerate(P.names):
        th = abs(float(v0[k]))
        rows.append(dict(system=system, pole=[m[j].real, m[j].imag], param=p,
                         value=float(v0[k]), fixed_e=0.1 * th * float(F[k, k].sqrt()),
                         profiled_e=0.1 * th / float(Finv[k, k].sqrt()),
                         cond=float(ev.max() / ev.min()), e_truth=rms_error(P, v0, x, y, t)))
    return rows


def profile(system: str, j: int, factors=PROFILE_FACTORS, iters: int = 200) -> list[dict]:
    """Profile of Re mu_j: fix it at c * Re mu_j*, re-fit everything else (L-BFGS)."""
    t, x, y, m, b = load(system)
    P = Params(m, b, full=True)                       # all parts free, as in the model
    k_fix = P.names.index((j, "Re mu"))
    v_true = torch.tensor(P.vector())
    out = []
    for c in factors:
        v = v_true.clone()
        v[k_fix] = c * v_true[k_fix]
        free = torch.ones_like(v, dtype=torch.bool)
        free[k_fix] = False
        w = v[free].clone().requires_grad_(True)
        opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-15,
                                history_size=50, line_search_fn="strong_wolfe")

        def full(wv):
            vv = v.clone()
            vv[free] = wv
            return vv

        def closure():
            opt.zero_grad()
            loss = (residual(P, full(w), x, y, t) ** 2).sum(1).mean()
            loss.backward()
            return loss

        opt.step(closure)
        vv = full(w.detach())
        mu, _ = P.unpack(vv)
        out.append(dict(system=system, pole=[m[j].real, m[j].imag], factor=c,
                        re_mu=float(vv[k_fix]), e=rms_error(P, vv, x, y, t),
                        im_mu_fitted=float(mu[j].imag)))
    return out


def main():
    diff = check_against_layer()
    print(f"float64 layer vs PoleResidueTS1d (float32): rel. diff {diff:.1e}")
    local = [r for s in SYSTEMS for r in local_analysis(s)]
    prof = []
    for s in SYSTEMS:
        _, _, _, m, _ = load(s)
        for j in range(m.size):
            prof += profile(s, j)
    (RESULTS / "identifiability.json").write_text(json.dumps(dict(local=local, profile=prof), indent=1))

    fmt = lambda z: f"{z[0]:+.1f}{z[1]:+.1f}i" if z[1] else f"{z[0]:+.1f}"
    lines = ["# E1 practical identifiability (tied layer, at the truth)", "",
             f"Float64 re-implementation vs `PoleResidueTS1d`: rel. diff {diff:.1e}. "
             "e = RMS relative L2 error over the 200 training inputs. "
             "Compare with the oracle floor (≈ 1e-3, discretisation) and the E1 test errors.", "",
             "## Local: error caused by a 10 % change of one parameter", "",
             "fixed: all other parameters at the truth; profiled: the others re-fitted (linearised). "
             "Small profiled e = the other poles/residues compensate the change.", "",
             "| system | pole | parameter | fixed e | profiled e | cond(F) |", "|---|---|---|---|---|---|"]
    for r in local:
        lines.append(f"| {r['system']} | {fmt(r['pole'])} | {r['param']} | {r['fixed_e']:.1e} | "
                     f"{r['profiled_e']:.1e} | {r['cond']:.1e} |")
    lines += ["", "## Profile of Re mu (nonlinear): minimum e with Re mu fixed at c × truth", "",
              "All other parameters re-fitted with L-BFGS from the truth (the fixed pole's Im mu and "
              "residue included). Im mu column: fitted Im of the profiled pole.", ""]
    head = "| system | pole | " + " | ".join(f"c={c:g}" for c in PROFILE_FACTORS) + " | Im mu at c=0.25 |"
    lines += [head, "|" + "---|" * (len(PROFILE_FACTORS) + 3)]
    keys = sorted({(r["system"], tuple(r["pole"])) for r in prof}, key=lambda k: (k[0], -k[1][1], -k[1][0]))
    for s, z in keys:
        rs = [r for r in prof if r["system"] == s and tuple(r["pole"]) == z]
        lines.append(f"| {s} | {fmt(z)} | " + " | ".join(f"{r['e']:.1e}" for r in rs) +
                     f" | {rs[0]['im_mu_fitted']:+.2f} |")
    lines.append("")
    (RESULTS / "identifiability.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
