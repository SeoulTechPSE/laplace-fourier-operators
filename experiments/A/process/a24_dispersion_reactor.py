"""(A) Application 1: lumped delay against distributed dispersion (plan Section 5.2).

In a tubular reactor with axial dispersion, the delay seen at the outlet is not a separate factor: it is the
transport term -i v k of the dispersion relation. A lumped FOPDT / SOPDT fit at a sensor reads it as theta. Here
both readings are compared.

**System** (periodic analogue of a dispersion reactor). c_t = -v c_x + D c_xx - k_r c + g(x - x0) u(t) on
[0, 2 pi), with v = 0.5, D = 0.02, k_r = 0.1, so lambda(k) = -i v k - D k^2 - k_r. The source g is a narrow
Gaussian (sigma 0.15) at x0 = 1. Sensors y_L(t) = integral of g(x - x0 - L) c(x, t) dx sit at distances
L in {1, 2, 3}, with transport times L / v = 2, 4, 6. T = 12, dt = 0.05: the field does not wrap round the domain
(2 pi / v = 12.6) within the window. Field truth per mode on a 32x finer time grid (exact FOH, as a1); n_x = 64.
Inputs: E1 profiles; splits 200 / 50 / 100; data seed 0.

**Readings.**
- Lumped: `PoleResidueDelay` FOPDT (N = 1) and SOPDT (N = 2, real) fitted to (u, y_L) for each L, using the a20
  protocol. For each: theta, the time constants, and the mean delay theta + sum tau (the first moment of a
  lumped model).
- Truth: the mean residence time t_bar(L) of the exact sensor impulse response, its first moment on [0, 12].
- Distributed: the wavenumber pole layer (N = 1, the a7 / a8 joint fit with leak) on the field data f -> c,
  giving v, D and k_r.

**Predictions** (written before running).

1. For every L, the SOPDT mean delay theta + tau1 + tau2 is within 5 % of t_bar(L), and the FOPDT's within 10 %.
2. A line through the SOPDT mean delays against L has a slope within 5 % of the slope of t_bar(L). Measured
   before any fit: t_bar = 2.230, 4.077, 6.044 for L = 1, 2, 3, so that slope is 1.907, itself 4.7 % below 1 / v
   because the window truncates the tail. The wavenumber layer gives v within 1 %.
3. The lumped theta alone is below L / v for every L and both models: dispersion makes the response arrive early.

    python experiments/A/process/a24_dispersion_reactor.py      # results/a24_dispersion_reactor.md / .json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "layer"))
sys.path.insert(0, str(HERE))
import a1_dispersion as A  # noqa: E402
import a20_fopdt as P  # noqa: E402
from lfo.models import PoleResidueDelay  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

V, DIF, KR, SIG, X0 = 0.5, 0.02, 0.1, 0.15, 1.0
LS = (1.0, 2.0, 3.0)
T, NT, FINE, NX, SPLIT = 12.0, 241, 32, 64, (200, 50, 100)
DT = T / (NT - 1)
OUT = HERE / "results" / "a24_dispersion_reactor.json"


def lam(k):
    return -1j * V * k - DIF * k ** 2 - KR


def gauss(x, c):
    d = (x - c + np.pi) % (2 * np.pi) - np.pi
    return np.exp(-0.5 * (d / SIG) ** 2) / (SIG * math.sqrt(2 * math.pi))


def make_data():
    rng = np.random.default_rng(0)
    n = sum(SPLIT)
    tf = np.linspace(0, T, (NT - 1) * FINE + 1)
    x = np.arange(NX) * 2 * np.pi / NX
    u = A.profiles(tf, rng, n)                                           # (n, n_tf)
    gk = np.fft.rfft(gauss(x, X0))                                       # (K,)
    K = NX // 2 + 1
    fk = gk[None, :, None] * u[:, None, :]                               # (n, K, n_tf)
    ck = A.foh_np(lam(np.arange(K)), fk, tf[1] - tf[0])
    f = np.fft.irfft(fk[..., ::FINE], n=NX, axis=1)
    c = np.fft.irfft(ck[..., ::FINE], n=NX, axis=1)                      # (n, NX, NT)
    dx = 2 * np.pi / NX
    ys = {L: np.einsum("x,nxt->nt", gauss(x, X0 + L) * dx, c) for L in LS}
    cut = np.cumsum([0, *SPLIT])
    split = lambda a: {k: torch.tensor(a[p:q]) for k, p, q in zip(("train", "vali", "test"), cut[:-1], cut[1:])}
    uu = split(u[:, ::FINE])
    sensors = {L: {k: (uu[k], v) for k, v in split(ys[L]).items()} for L in LS}
    fs, cs = split(f), split(c)
    field = {k: (fs[k], cs[k]) for k in fs}
    return sensors, field


def mean_residence():
    """First moment of the exact sensor impulse response on [0, T], per L."""
    t = np.linspace(0, T, 24001)
    k = np.arange(-NX // 2 + 1, NX // 2)
    x = np.arange(NX) * 2 * np.pi / NX
    gk_full = np.fft.fft(gauss(x, 0.0)) * (2 * np.pi / NX)               # Fourier coefficients (scaled)
    out = {}
    for L in LS:
        coef = np.abs(gk_full[k % NX]) ** 2 * np.exp(1j * k * L) / (2 * np.pi)
        h = np.real(np.exp(np.outer(t, lam(k.astype(float)))) @ coef)
        out[L] = float(np.trapezoid(t * h, t) / np.trapezoid(h, t))
    return out


def fit_lumped(data, order, device):
    best = None
    for i, th0 in enumerate((0.5, 1.0, 2.0, 4.0, 6.0)):
        for draw in range(2):
            torch.manual_seed(10 * i + draw)
            m = PoleResidueDelay(order, DT, theta0=th0).to(device)
            P.DT = DT
            v = P.lbfgs(m, data, device)
            if math.isfinite(v) and (best is None or v < best[0]):
                best = (v, m)
    m = best[1]
    with torch.no_grad():
        mu, beta, theta = m.poles()
        test = float(P.rel(m(data["test"][0].to(device)), data["test"][1].to(device)))
    taus = sorted((-1 / mu.real.cpu().numpy()).tolist(), reverse=True)
    return dict(theta=float(theta[0]), taus=taus, mean=float(theta[0]) + sum(taus), test=test, vali=best[0])


def fit_field(field, device):
    t = np.linspace(0, T, NT)
    (xtr, ytr), (xva, yva) = ((x.to(device) for x in field[s]) for s in ("train", "vali"))
    best = None
    for s in range(8):
        torch.manual_seed(s)
        m = WavenumberPoleLayer1d(1, t, modes=33, re_max=0.5, re_leak=0.05, k_ref=16.0).to(device)
        with torch.no_grad():
            m.mu_re.normal_(0, 3.0), m.mu_im.normal_(0, 3.0)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=300, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                                tolerance_change=1e-15, history_size=50)

        def closure():
            opt.zero_grad()
            loss = A.rel(m(xtr), ytr)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, IndexError):
            pass
        with torch.no_grad():
            v = float(A.rel(m(xva), yva))
        if math.isfinite(v) and (best is None or v < best[0]):
            best = (v, m)
    m = best[1]
    ar, ci = m.mu_re[0].detach().cpu().numpy(), m.mu_im[0].detach().cpu().numpy()
    with torch.no_grad():
        test = float(A.rel(m(field["test"][0].to(device)), field["test"][1].to(device)))
    return dict(v=float(-ci[0] / 16), D=float(-ar[1] / 256), kr=float(-ar[0]), test=test)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensors, field = make_data()
    tbar = mean_residence()
    out = dict(tbar=tbar, lumped={}, field=None)
    for L in LS:
        out["lumped"][L] = {name: fit_lumped(sensors[L], order, device) for name, order in (("fopdt", 1), ("sopdt", 2))}
        print(L, f"L/v {L / V:.3f} t_bar {tbar[L]:.4f}", {n: {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                                                              for k, v in r.items()} for n, r in out["lumped"][L].items()}, flush=True)
    out["field"] = fit_field(field, device)
    print("field", out["field"], flush=True)
    s = np.polyfit(LS, [out["lumped"][L]["sopdt"]["mean"] for L in LS], 1)
    sf = np.polyfit(LS, [out["lumped"][L]["fopdt"]["mean"] for L in LS], 1)
    out["slope"] = dict(sopdt=float(s[0]), fopdt=float(sf[0]), inv_v=1 / V)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    lines = ["# (A) Lumped delay against distributed dispersion (periodic dispersion reactor)", "",
             f"v = {V}, D = {DIF}, k_r = {KR}; source at x0, sensors at distance L.", "",
             "| L | L / v | t̄ (truth) | FOPDT θ | FOPDT θ + τ | SOPDT θ | SOPDT θ + τ₁ + τ₂ | FOPDT / SOPDT test |",
             "|---|---|---|---|---|---|---|---|"]
    for L in LS:
        f_, s_ = out["lumped"][L]["fopdt"], out["lumped"][L]["sopdt"]
        lines.append(f"| {L:g} | {L / V:.2f} | {tbar[L]:.3f} | {f_['theta']:.3f} | {f_['mean']:.3f} | {s_['theta']:.3f} | {s_['mean']:.3f} | "
                     f"{f_['test']:.2e} / {s_['test']:.2e} |")
    fd = out["field"]
    lines += ["", f"Slope of the mean delay against L: SOPDT {s[0]:.4f}, FOPDT {sf[0]:.4f} (1 / v = {1 / V:.4f}).",
              f"Wavenumber layer on the field: v = {fd['v']:.5f}, D = {fd['D']:.5f}, k_r = {fd['kr']:.5f} (test {fd['test']:.2e})."]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
