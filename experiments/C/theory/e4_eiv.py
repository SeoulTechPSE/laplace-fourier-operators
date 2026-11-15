"""(C) E4, errors-in-variables: input noise under output noise.

The E4 certificate and its intervals take the inputs as exact; sigma^2 holds the output noise
only. Noise-free, the float32 rounding of the inputs already widened the pole spread up to 2 x
(e4_bias.py). Here the inputs carry measurement noise as well: the stored input is
x~ = x + e_u, the outputs y = G x + e_y come from the clean x, and the refit uses x~.

Per data set (training split, 200 samples) and input-noise level eta_u (output noise eta_y fixed),
N_MC independent draws of (e_u, e_y), white Gaussian with relative L2 norm eta per sample (as
e2_train.make_splits). For each draw, the minimal first-order-hold refit (VarPro, started at the
truth; a local diagnosis) and three 95 % intervals per Re mu:

  naive     the E4 interval: sigma^2 = output-noise estimate (e2_rescue.noise_var of y),
            Wald form sqrt(2 sigma^2 [H^-1]_kk), H the Hessian of S. For a quadratic profile this
            is the profile interval of e4_profile.
  resid     the same with sigma^2 = S_min / (n - p) (the residual variance, which contains the
            input noise passed through the system); widens, no bias correction.
  eiv boot  truth-free parametric bootstrap: sigma_u^2 from the stored inputs (robust MAD of
            6th differences, which ignores the steps of the E1 inputs), sigma_y^2 as naive;
            y* = G_fit x~ + e_y*, x~* = x~ + e_u*, refit from the fit (N_BOOT draws).
            Bias-corrected estimate 2 theta_fit - mean theta*, interval +- 1.96 std(theta*).

Coverage is counted over the N_MC draws (independent data, unlike the groups of e4_profile).

    python experiments/C/theory/e4_eiv.py --cond p4_ms16_T20 --eta-u 0 1e-2 3e-2 1e-1
    python experiments/C/theory/e4_eiv.py --summarize      # writes results/e4_eiv.md
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_noise as NZ  # noqa: E402
import e2_rescue as RS  # noqa: E402
import e4_profile as EP  # noqa: E402

T, E = EP.T, EP.E
R = T.R
OUT = R.RESULTS / "e4_eiv"
ETA_Y = 1e-2
N_MC, N_BOOT = 64, 32
Z = 1.96


def clean_split(cond):
    """Clean float64 training inputs and outputs (as e2_train.make_splits, no cast, no noise)."""
    system, family, Tw, K, N = T.unpack(cond)
    R.DT = 20.0 / N
    t, u, y, _, _ = R.make(system, family, Tw, K, seed=0)
    return t.numpy(), u[:200].double(), y[:200].double()


def add_noise(a, eta, gen):
    if eta == 0:
        return a
    return a + eta * a.norm(dim=1, keepdim=True) / math.sqrt(a.shape[1]) * torch.randn(a.shape, generator=gen,
                                                                                        dtype=a.dtype)


def noise_var_robust(v):
    """Per-element noise variance of v_i / ||v_i||: MAD of the 6th differences (robust to the few
    large differences at the steps / switches of the E1 inputs, where the mean-square version of
    e2_rescue.noise_var reads eta ~ 0.05 on clean inputs; with the MAD 4e-13)."""
    d = torch.diff(v.double(), n=RS.DIFF, dim=1)
    s = 1.4826 * d.abs().median(1).values / math.sqrt(math.comb(2 * RS.DIFF, RS.DIFF))
    return float((s ** 2 / v.double().pow(2).sum(1)).mean())


def re_index(real):
    return np.cumsum([0] + [1 if r else 2 for r in real])[:-1]


def wald_std(vp, theta, real, sig2):
    """sqrt(2 sigma^2 [H^-1]_kk) per Re mu, H = Hessian of S at theta."""
    th = torch.tensor(theta, dtype=torch.float64, device=vp.device)
    H = torch.autograd.functional.hessian(lambda z: vp.S(z, real), th)
    C = torch.linalg.inv(H).diagonal().cpu().numpy()
    return np.sqrt(np.maximum(2 * sig2 * C[re_index(real)], 0))


def fit(x, y, t, theta0, real, device):
    vp = EP.VarPro({"train": (x, y)}, t, device, "foh")
    th, S = vp.fit(theta0, real)
    return vp, th, S


def boot(vp, x, y, t, theta, real, sig2_u, sig2_y, device, gen):
    """EIV parametric bootstrap from the fit: returns the refitted Re mu (N_BOOT, n_poles)."""
    with torch.no_grad():
        y_hat = (vp.pred(torch.tensor(theta, device=device), real).reshape(y.shape) / vp.scale).cpu()
    out = []
    for _ in range(N_BOOT):
        xb = add_noise(x, math.sqrt(x.shape[1] * sig2_u), gen)
        yb = add_noise(y_hat, math.sqrt(y.shape[1] * sig2_y), gen)
        out.append(fit(xb, yb, t, theta, real, device)[1][re_index(real)])
    return np.array(out)


def run(cond, eta_u, device):
    out = OUT / f"{cond}_u{eta_u:g}.json"
    if out.exists():
        return
    t0 = time.time()
    t, x0, y0 = clean_split(cond)
    system = T.unpack(cond)[0]
    mu_true, beta_true = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    true = E.true_folded(mu_true, beta_true)[0]
    theta0, real = EP.pack(list(true))
    re0 = theta0[re_index(real)]
    gen = torch.Generator().manual_seed(int(1e6 * eta_u) + 7)
    draws = []
    for m in range(N_MC):
        x = add_noise(x0, eta_u, gen)
        y = add_noise(y0, ETA_Y, gen)
        vp, th, S = fit(x, y, t, theta0, real, device)
        sig2_y, sig2_u = RS.noise_var(y), noise_var_robust(x)
        n, p = y.numel(), EP.n_eff(real)
        re = th[re_index(real)]
        tb = boot(vp, x, y, t, th, real, sig2_u, sig2_y, device, gen)
        draws.append(dict(re=re.tolist(), S=S, eta_y_hat=math.sqrt(y.shape[1] * sig2_y),
                          eta_u_hat=math.sqrt(x.shape[1] * sig2_u),
                          std_naive=wald_std(vp, th, real, sig2_y).tolist(),
                          std_resid=wald_std(vp, th, real, S / (n - p)).tolist(),
                          boot_mean=tb.mean(0).tolist(), boot_std=tb.std(0, ddof=1).tolist()))
    res = dict(cond=cond, eta_u=eta_u, eta_y=ETA_Y, n_mc=N_MC, n_boot=N_BOOT, re_true=re0.tolist(),
               poles=[[float(z.real), float(z.imag)] for z in true], crb_unit=NZ.crb_unit(cond)["unit"],
               draws=draws, seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"{cond} eta_u {eta_u:g} done ({res['seconds']:.0f}s)", flush=True)


def stats(r):
    """Per pole: relative bias and spread of the fit, mean relative std of each interval and its coverage."""
    re0 = np.array(r["re_true"])
    d = r["draws"]
    re = np.array([x["re"] for x in d])
    err = (re - re0) / np.abs(re0)
    bm, bs = (np.array([x[k] for x in d]) for k in ("boot_mean", "boot_std"))
    bc = 2 * re - bm                                           # bias-corrected
    s = {}
    for name, centre, sd in (("naive", re, np.array([x["std_naive"] for x in d])),
                             ("resid", re, np.array([x["std_resid"] for x in d])),
                             ("boot", bc, bs)):
        s[name] = dict(std=(sd / np.abs(re0)).mean(0), cover=(np.abs(centre - re0) <= Z * sd).mean(0))
    return dict(bias=err.mean(0), spread=err.std(0, ddof=1), bias_bc=((bc - re0) / np.abs(re0)).mean(0),
                eta_u_hat=np.mean([x["eta_u_hat"] for x in d]), eta_y_hat=np.mean([x["eta_y_hat"] for x in d]),
                **s)


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    lines = ["# E4: errors-in-variables (input noise under output noise)", "",
             f"Output noise η_y = {ETA_Y:g}; input noise η_u on the stored inputs (outputs from the clean inputs). "
             f"{N_MC} independent draws per row; minimal first-order-hold refit started at the truth. Per Re μ: "
             "relative bias (mean error) and spread (std) of the fit over the draws; for each 95 % interval its "
             "mean relative std and its coverage over the draws. naive = the E4 interval (σ² = output-noise "
             "estimate, Wald form = profile for a quadratic profile); resid = σ² from the residual S / (n − p); "
             f"boot = truth-free EIV bootstrap ({N_BOOT} draws, σ_u² from the inputs by a robust estimate), "
             "bias-corrected. The η_u = 0 rows are the output-noise reference.", ""]
    for cond in dict.fromkeys(r["cond"] for r in rs):
        g = sorted((r for r in rs if r["cond"] == cond), key=lambda r: r["eta_u"])
        poles = g[0]["poles"]
        lines += [f"## {cond}", "",
                  "| η_u | η̂_u / η̂_y | pole | bias | spread | bias after correction | naive: std, cover | "
                  "resid: std, cover | boot: std, cover |", "|---|---|---|---|---|---|---|---|---|"]
        for r in g:
            s = stats(r)
            for j, z in enumerate(poles):
                head = (f"| {r['eta_u']:g} | {s['eta_u_hat']:.2g} / {s['eta_y_hat']:.2g} " if j == 0 else "| | ")
                lines.append(head + f"| {lab(z)} | {s['bias'][j]:+.1e} | {s['spread'][j]:.1e} | "
                             f"{s['bias_bc'][j]:+.1e} | "
                             + " | ".join(f"{s[k]['std'][j]:.1e}, {s[k]['cover'][j]:.0%}" for k in ("naive", "resid", "boot"))
                             + " |")
        lines.append("")
    (R.RESULTS / "e4_eiv.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=["p4_ms16_T20", "p4_e1_T20", "p6_e1_T20", "p6_ms16_T80_N4096"])
    ap.add_argument("--eta-u", nargs="+", type=float, default=[0.0, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for e in a.eta_u:
            run(c, e, device)


if __name__ == "__main__":
    main()
