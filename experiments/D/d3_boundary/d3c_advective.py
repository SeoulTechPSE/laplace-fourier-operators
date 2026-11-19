"""(D) Advective boundaries: an advection-diffusion problem with Dirichlet values, made diagonal by one learned weight.

With advection, u_t = -v u_x + D u_xx + f on [0, 1] is not diagonal in the sine basis: -v u_x couples sine and
cosine modes. The substitution w = e^{-alpha x} u with alpha = v / (2 D) removes the advection:
w_t = D w_xx - (v^2 / 4D) w + e^{-alpha x} f, with Dirichlet values w(0) = a, w(1) = b e^{-alpha}. So the sine
modes of the weighted field diagonalise the problem, with lambda(k) = -D k^2 - v^2 / (4D).

**Model** (`adv`): the d3b basis layer (sine modes, a free cubic law per boundary channel) applied to the weighted
fields. A single learned alpha weights the input by e^{-alpha x} before the transform and the output by e^{alpha x}
after it. The right channel's law should then come out as 2 D k e^{-alpha}. Recovered: v_hat = 2 D_hat alpha_hat; and
the offset of Re mu, -a0, should equal v_hat^2 / (4 D_hat) (a consistency check the model is not told about).

**Control** (`noadv`): alpha fixed at 0, i.e. the d3b DD model.

**System.** v = 0.2, D = 0.05 (alpha = 2, offset 0.2), N = 32 modes and grid points x_i = i / 33. Forcing on the
weighted modes 1..12 (E1 profiles); boundary values a, b from E1 profiles starting at rest. Truth: the 32-mode
Galerkin solution of the weighted problem (exact FOH, 32x finer time grid), mapped back by e^{alpha x}. T = 5,
n_t = 201, splits 200 / 50 / 100, data seeds 0-4. Fit: the a7 joint fit with 8 starts; alpha starts at 0.

**Oracle, measured before any fit** (seed 0): 9.0e-3.

**Predictions** (written before running).

1. `adv`: test error <= 1.2x the oracle, with alpha and D within 2 % and v_hat within 3 %, in every seed. The
   offset -a0 is within 5 % of v_hat^2 / (4 D_hat).
2. `noadv`: test error >= 3x the oracle in every seed.

    python experiments/D/d3_boundary/d3c_advective.py      # results/d3c_advective.md / .json
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
sys.path.insert(0, str(HERE))
import d3b_bc_types as B  # noqa: E402

A = B.A
V, DIF, N, T, NT, FINE, SPLIT = 0.2, 0.05, 32, 5.0, 201, 32, (200, 50, 100)
ALPHA, OFFSET = V / (2 * DIF), V ** 2 / (4 * DIF)
OUT = HERE / "results" / "d3c_advective.json"
B.N, B.DIF = N, DIF


def make_data(seed):
    k, phi, sgn, w, _ = B.case("DD")
    x = np.arange(1, N + 1) / (N + 1)
    law = np.stack([2 * DIF * k, 2 * DIF * k * math.exp(-ALPHA)])
    rng = np.random.default_rng(700 + seed)
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    fm = np.zeros((n, N, tf.size))
    for _ in range(4):
        prof = A.profiles(tf, rng, n)
        mm = rng.integers(1, 13, size=n)
        fm[np.arange(n), mm - 1] += (rng.normal(size=n) / mm)[:, None] * prof
    bc = np.stack([A.profiles(tf, rng, n), A.profiles(tf, rng, n)], 1)
    bc = bc - bc[..., :1]
    src = fm + np.einsum("jm,njt->nmt", law * sgn, bc)
    wm = A.foh_np((-DIF * k ** 2 - OFFSET).astype(complex), src.astype(complex), tf[1] - tf[0]).real
    ex = np.exp(ALPHA * x)[None, :, None]
    f = ex * np.einsum("xm,nmt->nxt", phi, fm)
    u = ex * np.einsum("xm,nmt->nxt", phi, wm)
    to = lambda a: torch.tensor(a[..., ::FINE])
    f, bc, u = to(f), to(bc), to(u)
    c = np.cumsum([0, *SPLIT])
    return {s: (f[p:q], bc[p:q], u[p:q]) for s, p, q in zip(("train", "vali", "test"), c[:-1], c[1:])}, x


class AdvLayer(nn.Module):
    def __init__(self, t, x, learn_alpha=True):
        super().__init__()
        self.inner = B.BasisLayer("DD", t, True)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=learn_alpha)
        self.register_buffer("x", torch.tensor(x, dtype=torch.float64))

    def forward(self, f, bc):
        e = torch.exp(self.alpha * self.x)[None, :, None]
        return e * self.inner(f / e, bc)


def fit(learn_alpha, data, t, x, device):
    (ftr, btr, ytr), (fva, bva, yva) = ((z.to(device) for z in data[s]) for s in ("train", "vali"))
    best = None
    for s in range(8):
        torch.manual_seed(s)
        m = AdvLayer(t, x, learn_alpha).to(device)
        with torch.no_grad():
            m.inner.core.mu_re.normal_(0, 3.0), m.inner.core.mu_im.normal_(0, 3.0)
        opt = torch.optim.LBFGS([p for p in m.parameters() if p.requires_grad], max_iter=300, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = A.rel(m(ftr, btr), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            v = float(A.rel(m(fva, bva), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    return None if best is None else best[1]


@torch.no_grad()
def oracle(data, t, x, device):
    m = AdvLayer(t, x).to(device)
    m.alpha.fill_(ALPHA)
    core = m.inner.core
    core.mu_re.zero_(), core.mu_im.zero_(), core.beta.zero_()
    core.mu_re[0, 0], core.mu_re[0, 1] = -OFFSET, -DIF * m.inner.kref ** 2
    core.beta[0, 0, 0] = 1.0
    m.inner.c.zero_()
    m.inner.c[0, 1] = 2 * DIF * m.inner.kref
    m.inner.c[1, 1] = 2 * DIF * m.inner.kref * math.exp(-ALPHA)
    fte, bte, yte = (z.to(device) for z in data["test"])
    return float(A.rel(m(fte, bte), yte))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = np.linspace(0, T, NT)
    out = []
    for seed in range(5):
        data, x = make_data(seed)
        floor = oracle(data, t, x, device)
        fte, bte, yte = (z.to(device) for z in data["test"])
        for learn in (True, False):
            t0 = time.time()
            m = fit(learn, data, t, x, device)
            if m is None:
                out.append(dict(seed=seed, model="adv" if learn else "noadv", ratio=float("inf"), failed=True))
                continue
            with torch.no_grad():
                test = float(A.rel(m(fte, bte), yte))
            r = dict(seed=seed, model="adv" if learn else "noadv", oracle=floor, ratio=test / floor, seconds=round(time.time() - t0, 1))
            if learn:
                ar = m.inner.core.mu_re[0].detach().cpu().numpy()
                D_hat, a_hat = float(-ar[1] / m.inner.kref ** 2), float(m.alpha)
                r.update(alpha=a_hat, D=D_hat, v=2 * D_hat * a_hat, offset=float(-ar[0]), offset_pred=(2 * D_hat * a_hat) ** 2 / (4 * D_hat))
            out.append(r)
            print(seed, r["model"], f"ratio {r['ratio']:.2f} (oracle {floor:.1e})",
                  {k: round(r[k], 5) for k in ("alpha", "D", "v", "offset", "offset_pred") if k in r}, f"{r['seconds']} s", flush=True)
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (D) Advection–diffusion with Dirichlet boundaries: one learned weight e^{αx}", "",
             f"v = {V}, D = {DIF}: α = {ALPHA}, offset v²/4D = {OFFSET}. Range over data seeds 0–4.", "",
             "| model | test / oracle | α̂ | D̂ | v̂ = 2D̂α̂ | offset −a₀ | v̂²/(4D̂) |", "|---|---|---|---|---|---|---|"]
    for model in ("adv", "noadv"):
        rs = [r for r in out if r["model"] == model and not r.get("failed")]
        g = lambda q: f"{min(r[q] for r in rs):.4f}–{max(r[q] for r in rs):.4f}" if rs and q in rs[0] else "–"
        lines.append(f"| {model} | {g('ratio')} | {g('alpha')} | {g('D')} | {g('v')} | {g('offset')} | {g('offset_pred')} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
