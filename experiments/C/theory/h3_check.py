"""(C) H3: the partial tie R = K + R_res, |R_res| <= delta. Pole deviation from the tied estimate vs delta, against
the first-order bound of Theorem 6 in `H1_H2.md`.

Model (ALNO partial tie, `e2_train.PartialTie` with kind alno_pt): the tied layer plus R_res,l alpha_l on the
lowest `modes` rfft lines, R_res,l = delta z_l / (1 + |z_l|). Data: E2 conditions, training split (generated with
the first-order hold, so the band layer has a mismatch floor and the tied estimate theta_0 is biased). Loss:
mean_j ||r_j||^2 / ||y_j||^2. Per delta (continuation from the previous delta): fit error, pole errors against the
truth, |Re mu_k(delta) - Re mu_k(0)| and the bound C_k delta, C_k = sum_l |(H^-1 G)_{k, l}| (H = Hessian of the
loss in theta at theta_0, G = cross Hessian theta x R_res; complex R_res,l with |R_res,l| <= 1 gives the modulus).

    python experiments/C/theory/h3_check.py --cond p4_ms16_T20 p6_e1_T20      # writes results/h3_check.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_train as T  # noqa: E402
import h12_check as HC  # noqa: E402

torch.set_default_dtype(torch.float64)
DELTAS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)
OUT = HERE / "results" / "h3_check.json"


class PT(HC.Model):
    def __init__(self, mu, beta, t, modes, device):
        super().__init__(mu, beta, t)
        self.modes = modes
        self.iw = self.iw.to(device)
        self.t = self.t.to(device)
        self.v0 = self.v0.to(device)
        self.p = int(self.v0.numel())

    def out_pt(self, v, z, x, delta):
        """Tied layer + R_res on the lowest `modes` rfft lines; z: (modes, 2) real."""
        tr, st = self.branches(v, x)
        zc = torch.complex(z[:, 0], z[:, 1])
        R = delta * zc / (1 + zc.abs())
        a = torch.fft.rfft(x)[:, :self.modes]
        add = torch.zeros(x.shape[0], self.n // 2 + 1, dtype=torch.complex128, device=x.device)
        add[:, :self.modes] = R[None] * a
        return tr + st + torch.fft.irfft(add, n=self.n)

    def out_lin(self, v, R, x):
        """Tied layer + linear R_res (R: (modes, 2) real = Re, Im), for the derivatives at R = 0."""
        tr, st = self.branches(v, x)
        a = torch.fft.rfft(x)[:, :self.modes]
        add = torch.zeros(x.shape[0], self.n // 2 + 1, dtype=torch.complex128, device=x.device)
        add[:, :self.modes] = torch.complex(R[:, 0], R[:, 1])[None] * a
        return tr + st + torch.fft.irfft(add, n=self.n)


def loss_fn(mdl, x, y, w):
    return lambda out: (w[:, None] * (out - y) ** 2).sum() / x.shape[0]


def fit(mdl, x, y, w, v, z, delta, iters=3000):
    L = loss_fn(mdl, x, y, w)
    p = mdl.p
    dev = x.device

    def f(u):
        u = torch.tensor(u, device=dev, requires_grad=True)
        val = L(mdl.out_pt(u[:p], u[p:].reshape(-1, 2), x, delta) if delta > 0 else mdl.out(u[:p], x))
        val.backward()
        return float(val), u.grad.cpu().numpy()

    u0 = np.concatenate([v.cpu().numpy(), z.cpu().numpy().ravel()]) if delta > 0 else v.cpu().numpy()
    r = minimize(f, u0, jac=True, method="L-BFGS-B", options=dict(maxiter=iters, ftol=0, gtol=1e-16, maxcor=30))
    u = torch.tensor(r.x, device=dev)
    return u[:p], (u[p:].reshape(-1, 2) if delta > 0 else z), float(r.fun)


def bound(mdl, x, y, w, v0):
    """C_k per Re mu (first-order bound) and the derivative map M = -H^-1 G (p x 2 modes)."""
    L = loss_fn(mdl, x, y, w)
    R0 = torch.zeros(mdl.modes, 2, device=x.device)
    g = lambda v, R: torch.func.grad(lambda vv: L(mdl.out_lin(vv, R, x)))(v)
    H = torch.func.jacfwd(lambda v: g(v, R0))(v0)                                  # (p, p)
    G = torch.func.jacfwd(lambda R: g(v0, R))(R0).reshape(mdl.p, -1)              # (p, 2 modes)
    M = -torch.linalg.solve(H, G)
    Mc = M.reshape(mdl.p, mdl.modes, 2)
    C = Mc.pow(2).sum(-1).sqrt().sum(-1)                                            # sup over |R_l| <= 1
    return C, M, float(torch.linalg.eigvalsh(H)[0])


def run(cond, modes, device, n_train=200):
    data, t, mu, beta = T.setup(cond, 0)
    x, y = (a.double().to(device) for a in data["train"])
    x, y = x[:n_train], y[:n_train]
    m, b = HC.ID.minimal_poles(mu, beta)
    mdl = PT(m, b, torch.tensor(t), modes, device)
    w = 1 / y.pow(2).sum(1)
    v0, _, L0 = fit(mdl, x, y, w, mdl.v0, None, 0.0)                               # tied estimate theta_0
    C, M, hmin = bound(mdl, x, y, w, v0)
    idx = mdl.re_index()
    true = list(m)
    z = torch.full((modes, 2), 1e-8, device=device)          # |z| is not differentiable at 0
    v = v0
    rows = []
    d_prev = 0.0
    for d in DELTAS:
        if d > 0:
            # two starts: continuation (z rescaled so that R_res is the previous fit's) and the tied fit (R_res = 0,
            # always feasible, so the loss cannot exceed the tied loss); keep the lower
            zc = torch.complex(z[:, 0], z[:, 1])
            r = (d_prev * zc / (1 + zc.abs())) / d if d_prev > 0 else torch.zeros_like(zc)
            r = r * torch.clamp(0.999 / r.abs().clamp_min(1e-300), max=1.0)       # |R_prev| / d < 1
            zs = r / (1 - r.abs())
            zs = torch.where(zs.abs() > 1e-8, zs, torch.full_like(zs, 1e-8))
            z_cont = torch.stack([zs.real, zs.imag], -1)
            a = fit(mdl, x, y, w, v, z_cont, d)
            b2 = fit(mdl, x, y, w, v0, torch.full_like(z, 1e-8), d)
            v, z, L = min(a, b2, key=lambda q: q[2])
        else:
            L = L0
        d_prev = d
        mu_d, _ = mdl.unpack(v)
        R = (d * torch.complex(z[:, 0], z[:, 1]) / (1 + torch.complex(z[:, 0], z[:, 1]).abs())) if d > 0 else None
        pred = (M @ torch.stack([R.real, R.imag], -1).reshape(-1)) if d > 0 else torch.zeros(mdl.p, device=device)
        rows.append(dict(delta=d, fit=math.sqrt(L), max_R=float(R.abs().max()) if d > 0 else 0.0,
                         pole_err=[float(abs(complex(mu_d[k]) - true[k]) / abs(true[k])) for k in range(len(true))],
                         dev=[float(abs(v[i] - v0[i])) for i in idx],
                         bound=[float(C[i] * d) for i in idx],
                         first_order=[float(abs(pred[i])) for i in idx]))
        print(cond, f"delta {d:g}: fit {rows[-1]['fit']:.2e}, pole err "
              + ", ".join(f"{e:.1e}" for e in rows[-1]["pole_err"])
              + "; dev / bound " + ", ".join(f"{a:.1e}/{c:.1e}" for a, c in zip(rows[-1]["dev"], rows[-1]["bound"])),
              flush=True)
    return dict(cond=cond, modes=modes, poles=[[z_.real, z_.imag] for z_ in true], C=[float(C[i]) for i in idx],
                h_min=hmin, rows=rows)


def summarize(res):
    lines = ["# H3: partial tie, pole deviation vs δ (`h3_check.py`)", "",
             "ALNO partial tie R = K + R_res on the lowest rfft modes, |R_res| ≤ δ, fitted directly (L-BFGS, "
             "continuation in δ) to the E2 training data (first-order-hold data, so the band layer has a mismatch "
             "floor and the tied estimate θ₀ is biased). Deviation = |Re μ(δ) − Re μ(0)|; bound = C δ, C from the "
             "Hessian and cross Hessian at θ₀ (Theorem 6); first order = |(−H⁻¹G R̂_δ)_k| with the fitted R̂_δ.", ""]
    for r in res:
        lab = [f"{p[0]:g}{p[1]:+g}i" if p[1] else f"{p[0]:g}" for p in r["poles"]]
        lines += [f"## {r['cond']} ({r['modes']} modes)", "",
                  "C per Re μ: " + ", ".join(f"{l} {c:.2e}" for l, c in zip(lab, r["C"])), "",
                  "| δ | fit (RMS rel.) | max |R̂| | " + " | ".join(f"{l}: pole error; deviation / first order / bound"
                                                          for l in lab) + " |",
                  "|---|---|---|" + "---|" * len(lab)]
        for x in r["rows"]:
            lines.append(f"| {x['delta']:g} | {x['fit']:.2e} | {x['max_R']:.1e} | "
                         + " | ".join(f"{e:.1e}; {a:.1e} / {f:.1e} / {c:.1e}" for e, a, f, c in
                                      zip(x["pole_err"], x["dev"], x["first_order"], x["bound"])) + " |")
        lines.append("")
    (HERE / "results" / "h3_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=["p4_ms16_T20", "p6_e1_T20"])
    ap.add_argument("--modes", type=int, default=64)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = [run(c, a.modes, device) for c in a.cond]
    OUT.write_text(json.dumps(res, indent=1))
    summarize(res)


if __name__ == "__main__":
    main()
