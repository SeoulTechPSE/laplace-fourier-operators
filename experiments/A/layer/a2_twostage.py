"""(A) Pole layer, second validation: two-stage identification of the dispersion relation.

The first validation (`a1_dispersion.py`) fitted the polynomial mu(k) jointly and failed by optimization:
- wrong basins appeared through k^4 / k^3 terms;
- phase wrapping in v k added local minima.
Here the dispersion relation is identified in two stages, and the layer is then fine-tuned from it:

1. **Per mode.** For each excited wavenumber k = 1..16 there is one complex pole mu_k with a complex residue b_k.
   For fixed mu the residue is linear (least squares over all samples, using the layer's FOH convolution).
   mu is found by a prior-free grid, then refined:
   - grid: Re mu in -[1e-3, 1e2] (log, 60 points) and Im mu in [-20, 20] (161 points);
   - refinement: Nelder-Mead on the residual.
2. **Polynomial.** Least-squares fit of Re mu_k on {1, k^2, k^4} and of Im mu_k on {k, k^3}, over k = 1..16 with
   equal weights. The degree is chosen by BIC: each of k^4 and k^3 is kept only if it lowers the BIC.
3. **Fine-tune.** `WavenumberPoleLayer1d` (N = 1) is initialized with the stage-2 polynomial and its beta(k) fitted
   to the b_k (b0 + b1 k^2), then trained with Adam, lr 1e-3, 1000 steps. Settings otherwise as in
   `a1_dispersion.py`: modes = 17, Re cap 0.5.

Data: `a1_dispersion.make_data` (truth on a 32x finer time grid), systems `diff` and `adr`, data seeds 0-4. There
are no model seeds, because stages 1 and 2 are deterministic.

Predictions (written before running):

1. Stage 1: every per-mode pole within 1 % of lambda(k), k = 1..16, in every data seed and both systems.
2. Stage 2: BIC picks the true degree (no k^4 term; no k^3 term) in at least 4 of 5 data seeds per system. The
   recovered D, v, k_r are within 1 % (k_r: within 0.002 absolute, since it is 0 for `diff`).
3. Stage 3: test error <= 1.2x the oracle floor in every data seed and both systems, and the dispersion error over
   k = 1..16 stays <= 1 %.

    python experiments/A/layer/a2_twostage.py      # results/a2_twostage.md / .json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a1_dispersion as A  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d, foh_kernel, foh_response  # noqa: E402

KS = np.arange(1, 17)
GRID_RE = -np.logspace(-3, 2, 60)
GRID_IM = np.linspace(-20, 20, 161)
OUT = HERE / "results" / "a2_twostage.json"


def mode_series(data, k, split="train"):
    """Complex time series of mode k for the samples that excite it: (f_k, u_k), each (B_k, n_t)."""
    f, u = data[split]
    fk, uk = torch.fft.rfft(f, dim=1)[:, k], torch.fft.rfft(u, dim=1)[:, k]
    keep = fk.abs().amax(1) > 1e-8 * fk.abs().max()
    return fk[keep], uk[keep]


def residual(mu, fk, uk, dt):
    """Relative residual of the best residue for pole(s) mu (tensor of shape (G,)); returns (res (G,), b (G,))."""
    g, c = foh_kernel(mu, fk.shape[-1], dt)                       # (G, n_t)
    x = foh_response(g[:, None], c[:, None], fk[None])           # (G, B, n_t) response with unit residue
    b = (x.conj() * uk[None]).sum((1, 2)) / (x.abs() ** 2).sum((1, 2))
    r = (uk[None] - b[:, None, None] * x).abs().pow(2).sum((1, 2)).sqrt() / uk.abs().pow(2).sum().sqrt()
    return r, b


def per_mode(data, dt, device):
    out = []
    for k in KS:
        fk, uk = (a.to(device) for a in mode_series(data, int(k)))
        best = (np.inf, None)
        for re in GRID_RE:                                          # grid row by row (161 poles at a time)
            mu = torch.tensor(re + 1j * GRID_IM, dtype=torch.complex128, device=device)
            r, _ = residual(mu, fk, uk, dt)
            i = int(r.argmin())
            if float(r[i]) < best[0]:
                best = (float(r[i]), complex(mu[i]))
        f = lambda p: float(residual(torch.tensor([complex(p[0], p[1])], dtype=torch.complex128, device=device),
                                     fk, uk, dt)[0][0])
        opt = minimize(f, [best[1].real, best[1].imag], method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
        mu = complex(opt.x[0], opt.x[1])
        r, b = residual(torch.tensor([mu], dtype=torch.complex128, device=device), fk, uk, dt)
        out.append(dict(k=int(k), mu=[mu.real, mu.imag], b=[float(b[0].real), float(b[0].imag)], resid=float(r[0]),
                        n_samples=int(fk.shape[0])))
    return out


def poly_fit(ks, y, powers):
    V = np.stack([ks ** p for p in powers], 1)
    c, *_ = np.linalg.lstsq(V, y, rcond=None)
    rss = float(((V @ c - y) ** 2).sum())
    n = ks.size
    return c, n * np.log(max(rss / n, 1e-300)) + len(powers) * np.log(n)


def stage2(modes):
    ks = np.array([m["k"] for m in modes], float)
    mu = np.array([complex(*m["mu"]) for m in modes])
    re_opts = {"a0,a1": [0, 2], "a0,a1,a2": [0, 2, 4]}
    im_opts = {"c1": [1], "c1,c3": [1, 3]}
    re_fit = {k: poly_fit(ks, mu.real, p) for k, p in re_opts.items()}
    im_fit = {k: poly_fit(ks, mu.imag, p) for k, p in im_opts.items()}
    re_key = min(re_fit, key=lambda k: re_fit[k][1])
    im_key = min(im_fit, key=lambda k: im_fit[k][1])
    a = list(re_fit[re_key][0]) + [0.0] * (3 - len(re_fit[re_key][0]))
    c = list(im_fit[im_key][0]) + [0.0] * (2 - len(im_fit[im_key][0]))
    b = np.array([complex(*m["b"]) for m in modes])
    bc, _ = poly_fit(ks, b.real, [0, 2])
    bs, _ = poly_fit(ks, b.imag, [0, 2])
    return dict(re_degree=re_key, im_degree=im_key, a=a, c=c, beta=[[bc[0], bs[0]], [bc[1], bs[1]]],
                D=-a[1], v=-c[0], kr=-a[0])


def finetune(data, t, s2, device):
    model = WavenumberPoleLayer1d(1, t, modes=A.MODES, re_max=A.RE_MAX).to(device)
    with torch.no_grad():
        model.mu_re.copy_(torch.tensor([s2["a"]], dtype=torch.float64))
        model.mu_im.copy_(torch.tensor([s2["c"]], dtype=torch.float64))
        model.beta.copy_(torch.tensor([s2["beta"]], dtype=torch.float64))
    (xtr, ytr), (xte, yte) = ((a.to(device) for a in data[k]) for k in ("train", "test"))
    with torch.no_grad():
        test0 = float(A.rel(model(xte), yte))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(1000):
        opt.zero_grad()
        loss = A.rel(model(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        test = float(A.rel(model(xte), yte))
        mu = model.poles(torch.tensor(KS, dtype=torch.float64, device=device))[0][0].cpu().numpy()
    return test0, test, mu, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for system, p in A.SYSTEMS.items():
        for ds in range(5):
            t0 = time.time()
            data, t = A.make_data(system, seed=ds)
            dt = float(t[1] - t[0])
            modes = per_mode(data, dt, device)
            true = A.lam(KS, **p)
            err1 = [abs(complex(*m["mu"]) - true[i]) / abs(true[i]) for i, m in enumerate(modes)]
            s2 = stage2(modes)
            test0, test, mu_ft, _ = finetune(data, t, s2, device)
            r = dict(system=system, data_seed=ds, modes=modes, mode_err=err1, stage2=s2, test_init=test0, test=test,
                     disp_err=float(np.linalg.norm(mu_ft - true) / np.linalg.norm(true)), seconds=round(time.time() - t0, 1))
            out.append(r)
            print(system, ds, f"mode err max {max(err1):.2e}", s2["re_degree"], s2["im_degree"],
                  f"D {s2['D']:.5f} v {s2['v']:.5f} kr {s2['kr']:.5f}", f"test {test0:.2e} -> {test:.2e}",
                  f"disp {r['disp_err']:.2e}", flush=True)
    floors = {s: A.oracle(s, device) for s in A.SYSTEMS}
    OUT.write_text(json.dumps(dict(floors=floors, runs=out), indent=1))
    summarize(floors, out)


def summarize(floors, out):
    lines = ["# (A) Two-stage dispersion identification, then fine-tuning", "",
             "Stage 1: one pole per excited k (grid + Nelder–Mead, residue by least squares). Stage 2: polynomial LS, "
             "degree by BIC. Stage 3: layer initialized from stage 2, Adam 1000 steps. Data seeds 0–4.", "",
             "| system | data seed | stage 1: max pole error (k = 1..16) | degree (Re; Im) | D | v | k_r | test: init → tuned | oracle | dispersion error |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        s2 = r["stage2"]
        lines.append(f"| {r['system']} | {r['data_seed']} | {max(r['mode_err']):.1e} | {s2['re_degree']}; {s2['im_degree']} | "
                     f"{s2['D']:.5f} | {s2['v']:.5f} | {s2['kr']:.5f} | {r['test_init']:.2e} → {r['test']:.2e} | "
                     f"{floors[r['system']]:.2e} | {r['disp_err']:.1e} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
