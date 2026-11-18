"""(A) Pole layer, seventh validation: is stage 1 needed? A joint fit from random starts.

a3-a6 recover every linear system with a two-stage recipe: per-mode identification (stage 1), cuts and weighted
polynomials (stage 2), then an L-BFGS fine-tune with k_ref scaling. a1 showed that a joint fit with Adam and no
scaling fails. This validation keeps everything of the recipe except stages 1 and 2: the layer starts from random
coefficients and is fitted jointly with the same L-BFGS and k_ref. Several starts per data seed; the start with the
lowest validation error is selected (no truth used).

**Systems and data** (unchanged generators, data seeds 0-4):

- 1D `diff`, `adr` (`a1_dispersion.make_data`), N = 1, layer as in a3 (Re: 1, k^2, k^4; Im: k, k^3), k_ref = 16;
- 1D `beam` (`a4_beam.make_data`), N = 2, Im powers (1, 2, 3), k_ref = 16;
- 2D `diff2d`, `adr2d` (`a6_diffusion2d.make_data`), N = 1, layer as in a6, k_ref = 8.

**Starts.** All mu coefficients (in k / k_ref units) ~ N(0, 3^2): |mu(k_ref)| of order 5, inside the resolvable
range |mu| dt <= 0.25. Residues start at the layer default (beta = 1 / N, constant). Re mu is capped at 0.5 as in
a3. 16 starts per data seed in 1D, 8 in 2D. L-BFGS (strong Wolfe), 300 iterations, full batch, float64, relative L2
loss. A start whose loss becomes non-finite counts as failed.

**Metrics.** Per start: test error / oracle floor (true lambda in the same layer, same data seed); success means
<= 1.2. Per data seed: the start with the lowest validation error, and its recovered constants. Wall time against
the two-stage recipe.

**Predictions** (written before running).

1. diff and diff2d: at least half the starts succeed, and the selected start succeeds in 5 of 5 seeds.
2. adr and adr2d: the selected start succeeds in at least 4 of 5 seeds per system. The success rate per start is
   lower than for diffusion (the phase v k wraps, which adds local minima, a1).
3. beam: at most a quarter of the starts succeed, and the selected start succeeds in at most 2 of 5 seeds; that is,
   stage 1 is needed for undamped, oscillatory modes.

    python experiments/A/layer/a7_joint.py [system ...]      # results/a7_joint.md / .json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a1_dispersion as A  # noqa: E402
import a2_twostage as B  # noqa: E402
import a4_beam as Bm  # noqa: E402
import a6_diffusion2d as S  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d, WavenumberPoleLayer2d  # noqa: E402

SIGMA, ITERS = 3.0, 300
STARTS = {"diff": 16, "adr": 16, "beam": 16, "diff2d": 8, "adr2d": 8}
OUT = HERE / "results" / "a7_joint.json"


def build(system, t, device, leak=0.0):
    if system in ("diff", "adr"):
        return WavenumberPoleLayer1d(1, t, modes=A.MODES, re_max=A.RE_MAX, k_ref=16.0, re_leak=leak).to(device)
    if system == "beam":
        return WavenumberPoleLayer1d(2, t, mu_degree=(3, 3), beta_degree=2, im_powers=(1, 2, 3), modes=A.MODES,
                                     re_max=A.RE_MAX, k_ref=16.0, re_leak=leak).to(device)
    m = S.layer(t, device)
    m.re_leak = float(leak)
    return m


def data_for(system, ds):
    if system in ("diff", "adr"):
        return A.make_data(system, seed=ds)
    if system == "beam":
        return Bm.make_data(ds)
    return S.make_data(system, seed=ds)


@torch.no_grad()
def oracle(system, data, t, device):
    m = build(system, t, device)
    for q in m.parameters():
        q.zero_()
    if system in ("diff", "adr"):
        p = A.SYSTEMS[system]
        m.mu_re[0, 0], m.mu_re[0, 1], m.mu_im[0, 0] = -p["kr"], -p["D"] * 256, -p["v"] * 16
        m.beta[0, 0, 0] = 1.0
    elif system == "beam":
        m.mu_im[0, 1], m.mu_im[1, 1] = Bm.C_BEAM * 256, -Bm.C_BEAM * 256
        m.beta[:, 0, 0] = 0.5
    else:
        return S.oracle(system, data, t, device)
    xte, yte = (a.to(device) for a in data["test"])
    return float(A.rel(m(xte), yte))


@torch.no_grad()
def readout(system, m, device):
    """Recovered constants and dispersion error."""
    if system in ("diff", "adr"):
        ar, ci = m.mu_re[0].cpu().numpy(), m.mu_im[0].cpu().numpy()
        mu = m.poles(torch.tensor(B.KS, dtype=torch.float64, device=device))[0][0].cpu().numpy()
        true = A.lam(B.KS, **A.SYSTEMS[system])
        const = dict(D=float(-ar[1] / 256), v=float(-ci[0] / 16), kr=float(-ar[0]))
    elif system == "beam":
        mu = m.poles(torch.tensor(Bm.KS, dtype=torch.float64, device=device))[0].cpu().numpy()
        lab = 0 if mu[0, -1].imag > 0 else 1
        mu = np.stack([mu[lab], mu[1 - lab]])
        true = np.stack(Bm.true_poles(Bm.KS))
        const = dict(c=float(m.mu_im[lab, 1]) / 256)
    else:
        kk = torch.tensor(S.H, dtype=torch.float64, device=device)
        mu = m.poles(kk[:, 0], kk[:, 1])[0][0].cpu().numpy()
        true = S.lam(np.array(S.H)[:, 0], np.array(S.H)[:, 1], **S.SYSTEMS[system])
        ar, av = m.mu_re[0].cpu().numpy(), m.adv[0].cpu().numpy()
        const = dict(D=float(-ar[1] / 64), v=[float(-av[0] / 8), float(-av[1] / 8)], kr=float(-ar[0]))
    return const, float(np.linalg.norm(mu - true) / np.linalg.norm(true))


def fit(system, data, t, start, device, leak=0.0, stable_start=False):
    """One start. `leak`: slope above the Re cap (a8). `stable_start`: Re coefficients -|N(0, SIGMA^2)|, so that the
    start has Re mu <= 0 at every k (a8); the draws are otherwise those of a7."""
    torch.manual_seed(start)
    m = build(system, t, device, leak)
    with torch.no_grad():
        for name, q in m.named_parameters():
            if name != "beta":
                q.normal_(0.0, SIGMA)
        if stable_start:
            m.mu_re.abs_().neg_()
    (xtr, ytr), (xva, yva), (xte, yte) = ((x.to(device) for x in data[k]) for k in ("train", "vali", "test"))
    opt = torch.optim.LBFGS(m.parameters(), max_iter=ITERS, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50)

    def closure():
        opt.zero_grad()
        loss = A.rel(m(xtr), ytr)
        if torch.isfinite(loss):
            loss.backward()
        return loss

    try:
        opt.step(closure)
    except (RuntimeError, IndexError):   # IndexError: torch's strong-Wolfe search can fail on a non-finite loss (a8)
        pass
    with torch.no_grad():
        vali, test = float(A.rel(m(xva), yva)), float(A.rel(m(xte), yte))
    if not (math.isfinite(vali) and math.isfinite(test)):
        return dict(start=start, vali=float("inf"), test=float("inf"), const=None, disp_err=float("inf"))
    const, disp = readout(system, m, device)
    return dict(start=start, vali=vali, test=test, const=const, disp_err=disp)


def main(systems):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = json.loads(OUT.read_text()) if OUT.exists() else []
    out = [r for r in out if r["system"] not in systems]
    for system in systems:
        for ds in range(5):
            t0 = time.time()
            data, t = data_for(system, ds)
            floor = oracle(system, data, t, device)
            starts = [fit(system, data, t, s, device) for s in range(STARTS[system])]
            for s in starts:
                s["ratio"] = s["test"] / floor
            best = min(starts, key=lambda s: s["vali"])
            r = dict(system=system, data_seed=ds, oracle=floor, starts=starts, best=best,
                     n_success=sum(s["ratio"] <= 1.2 for s in starts), seconds=round(time.time() - t0, 1))
            out.append(r)
            print(system, ds, f"success {r['n_success']}/{len(starts)}", f"best ratio {best['ratio']:.3f}",
                  best["const"], f"disp {best['disp_err']:.1e}", f"{r['seconds']} s", flush=True)
            del data
            torch.cuda.empty_cache()
            OUT.write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out):
    fmt = lambda c: ("–" if c is None else ", ".join(
        f"{k} {v:.5f}" if not isinstance(v, list) else f"{k} ({v[0]:.4f}, {v[1]:.4f})" for k, v in c.items()))
    lines = ["# (A) Joint fit from random starts (no stage 1)", "",
             f"Coefficients (k / k_ref units) ~ N(0, {SIGMA:g}²); L-BFGS {ITERS} iterations; selection by validation "
             "error. Success: test ≤ 1.2× oracle. Data seeds 0–4.", "",
             "| system | seed | starts succeeded | selected: test / oracle | selected: dispersion error | selected: constants | "
             "time (s) |", "|---|---|---|---|---|---|---|"]
    for r in out:
        b = r["best"]
        lines.append(f"| {r['system']} | {r['data_seed']} | {r['n_success']} / {len(r['starts'])} | {b['ratio']:.3f} | "
                     f"{b['disp_err']:.1e} | {fmt(b['const'])} | {r['seconds']} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or list(STARTS))
