"""(C) E4: minimal-form refit, order test and per-pole profile intervals.

The chi-square certificate (e2_floor.py, rule ``noise``) under-certifies at low noise: the
spread of S between runs is set by the optimiser, so often only the best run is certified and
the consensus (>= 2 certified runs) is empty. A union with the ratio rule does not help: the
1.5 x ratio certifies every run whenever the loss is at the noise level. Here, per group of 5
runs after the noise-aware restarts (stage ``rescue_na``), truth-free:

  certified   S - S_min <= chi2_p(0.99) sigma^2 (as e2_floor.py), noise level from the outputs
  report      consensus of the certified runs if >= 2, else the effective poles of the best run
  refit       the report as a minimal tied layer (one slot per reported pole, real ones real):
              residues first with the poles frozen, then all parameters (L-BFGS). The 8-slot runs
              split a decay over several real slots under noise; the refit removes that freedom
  order test  S_minimal - S_best <= chi2_dp(0.99) sigma^2 (dp = parameters removed): the report
              explains the data as well as the best 8-slot run. If the consensus fails, the best
              run's effective poles are tried
  profile     for each fitted pole, Re mu fixed at c x its value (c = exp(+-u), u log-spaced in
              [2e-3, ln 3]), all else re-fitted (warm start from the neighbouring c); interval = the
              c range around 1 with S(c) - min S <= chi2_1(0.95) sigma^2 (pointwise 95 % profile
              likelihood interval)
  identified  a fitted pole whose interval closes on both sides

The truth is used only to score: per true pole, whether a fitted pole is within 0.1 with a
closed interval, whether the interval covers the true Re mu, and fitted poles with closed
intervals that match no true pole (false).

    python experiments/C/theory/e4_profile.py --cond p6_ms16_T80_N4096_n1e-1 --group 10
    python experiments/C/theory/e4_profile.py --summarize    # writes results/e4_profile.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_floor as F  # noqa: E402

P, RS, T, E = F.P, F.RS, F.T, F.E
OUT = T.R.RESULTS / "e4_profile"
STAGE, VARIANT = "rescue_na", "damp8_ms"
CONDS = [c for c, v in F.CASES if v == VARIANT and F.NOISY(c)]
BAD = 1e30                                          # objective returned for non-finite S
STEPS, U_MIN, C_MAX, VP_ITERS = 12, 2e-3, 3.0, 500  # per side: c = exp(+-u), u log-spaced in [U_MIN, ln C_MAX]
LEVEL = chi2.ppf(0.95, 1)


def load_group(cond, g0, device):
    """Runs of one group (seeds g0 .. g0 + 4): model, S, effective poles."""
    system, family, Tw, Kk, N = T.unpack(cond)
    data, t, mu_true, beta_true = T.setup(cond, g0)
    out, suffix = F.STAGES[STAGE]
    runs = []
    for s in range(g0, g0 + F.K):
        name = T.run_name(cond, VARIANT, s)
        ck = T.RUNS / f"{name}{suffix}.pt"
        if not ck.exists():
            continue
        model, _ = T.build_for(cond, VARIANT, s, t, mu_true, beta_true)
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.to(device)
        mu, beta = (a.detach().cpu().numpy() for a in model.pole_sets())
        runs.append(dict(seed=s, model=model, S=RS.sse(model, data, device), eff=F.effective(mu, beta, Tw)))
    return data, t, mu_true, beta_true, runs


class VarPro:
    """Minimal tied layer in float64 by variable projection. For poles mu_k the layer output is
    Re sum_k beta_k phi_k(x), linear in the residues, with (as PoleResidueTS1d, beta = 1)

        phi_k = -(1 / n_t) (sum_l alpha_l H_kl) e^{mu_k t} + ifft_l(alpha_l H_kl),  H_kl = 1 / (i omega_l - mu_k),

    alpha = fft(x). Residues are solved exactly (weighted least squares, rows scaled by 1 / ||y_i||,
    so the objective is S); only the poles are iterated (L-BFGS on S with autograd gradients).
    Real poles have a real residue (1 basis column), complex poles a complex one (2 columns)."""

    def __init__(self, data, t, device):
        x, y = (a.to(device).double() for a in data["train"])
        self.n_t = x.shape[1]
        self.alpha = torch.fft.fft(x)
        self.t = torch.tensor(np.asarray(t), dtype=torch.float64, device=device)
        dt = float(self.t[1] - self.t[0])                                          # i omega_l as _omega, float64
        self.iw = (2j * np.pi * torch.fft.fftfreq(self.n_t, dt, dtype=torch.float64, device=device))
        self.scale = 1 / y.norm(dim=1, keepdim=True)
        self.y = (y * self.scale).reshape(-1)
        self.device = device

    def mu(self, theta, real):
        out, i = [], 0
        for r in real:
            out.append(torch.complex(theta[i], torch.zeros((), dtype=theta.dtype, device=theta.device)) if r
                       else torch.complex(theta[i], theta[i + 1]))
            i += 1 if r else 2
        return torch.stack(out)

    def S(self, theta, real):
        mu = self.mu(theta, real)
        H = 1 / (self.iw[:, None] - mu[None, :])                                      # (L, n)
        aH = self.alpha[:, :, None] * H[None]                                         # (b, L, n)
        trans = -(aH.sum(1) / self.n_t)[:, :, None] * torch.exp(mu[:, None] * self.t[None])[None]   # (b, n, n_t)
        phi = trans + torch.fft.ifft(aH, dim=1).transpose(1, 2)
        phi = phi * self.scale[:, :, None]
        cols = []
        for k, r in enumerate(real):
            cols.append(phi[:, k].real.reshape(-1))
            if not r:
                cols.append(-phi[:, k].imag.reshape(-1))
        A = torch.stack(cols, 1)
        G = A.T @ A
        c = torch.linalg.solve(G + 1e-12 * torch.eye(len(cols), dtype=G.dtype, device=G.device) * G.diag().max(),
                               A.T @ self.y)                                  # small ridge: coinciding poles
        r = self.y - A @ c
        return (r @ r)

    def fit(self, theta0, real, fixed=None, iters=VP_ITERS):
        """argmin_theta S; fixed = index of theta held at its value. Returns (theta, S)."""
        theta0 = np.asarray(theta0, float)
        free = np.ones(theta0.size, bool)
        if fixed is not None:
            free[fixed] = False

        def f(z):
            th = theta0.copy()
            th[free] = z
            tt = torch.tensor(th, dtype=torch.float64, device=self.device, requires_grad=True)
            try:
                val = self.S(tt, real)
                val.backward()
                g = tt.grad.cpu().numpy()[free]
                if np.isfinite(float(val)) and np.isfinite(g).all():
                    return float(val), g
            except torch.linalg.LinAlgError:
                pass
            return BAD, np.zeros(free.sum())                  # unstable / degenerate poles: line search backs off

        res = minimize(f, theta0[free], jac=True, method="L-BFGS-B",
                       options=dict(maxiter=iters, ftol=0.0, gtol=1e-14, maxcor=20))
        th = theta0.copy()
        th[free] = res.x
        return th, float(res.fun)


def pack(poles):
    real = [z.imag == 0 for z in poles]
    theta = [v for z in poles for v in ((z.real,) if z.imag == 0 else (z.real, abs(z.imag)))]
    return np.array(theta), real


def unpack_theta(theta, real):
    out, i = [], 0
    for r in real:
        out.append(complex(theta[i], 0.0) if r else complex(theta[i], theta[i + 1]))
        i += 1 if r else 2
    return out


def n_eff(real):
    """Free real parameters: real pole 2 (decay, residue), complex pole 4."""
    return int(sum(2 if r else 4 for r in real))


def profile(vp, theta, real, k):
    """S(c) with theta[k] (a Re mu) fixed at c x its fitted value, all other poles re-fitted
    (residues exact); continuation outward from c = 1 on each side."""
    pts = {1.0: float(vp.S(torch.tensor(theta, device=vp.device), real))}
    for side in (1, -1):
        th = theta.copy()
        for c in np.exp(side * np.geomspace(U_MIN, np.log(C_MAX), STEPS)):
            th[k] = theta[k] * c
            th, Sc = vp.fit(th, real, fixed=k)
            pts[float(c)] = Sc
    cs = np.array(sorted(pts))
    return cs, np.array([pts[c] for c in cs])


def interval(cs, S, thr):
    """c range around c = 1 with S - min S <= thr (linear crossing in log c); None side = open."""
    i0 = int(np.argmin(np.abs(cs - 1.0)))
    ok = S - S.min() <= thr
    out = []
    for step in (-1, 1):
        i = i0
        while 0 <= i + step < len(cs) and ok[i + step]:
            i += step
        j = i + step
        if not (0 <= j < len(cs)):
            out.append(None)
            continue
        a, b = np.log(cs[i]), np.log(cs[j])
        fa, fb = S[i] - S.min() - thr, S[j] - S.min() - thr
        out.append(float(np.exp(a + (b - a) * fa / (fa - fb))) if fb != fa else float(cs[j]))
    return out


def run(cond, g0, device):
    if (OUT / f"{cond}_g{g0}.json").exists():
        return
    system, family, Tw, Kk, N = T.unpack(cond)
    t0 = time.time()
    data, t, mu_true, beta_true, runs = load_group(cond, g0, device)
    if len(runs) < F.K:
        print(f"skip {cond} {g0}: {len(runs)} runs", flush=True)
        return
    sig2 = RS.noise_var(data["train"][1])
    mask = runs[0]["model"].pr.parametrizations.mu[0].mask.flatten().cpu().numpy()
    p = n_eff(list(mask == 0))                               # effective free parameters of the 8-slot layer
    s_min = min(r["S"] for r in runs)
    cert = [r for r in runs if r["S"] - s_min <= chi2.ppf(0.99, p) * sig2]
    best = min(runs, key=lambda r: r["S"])
    # candidate reports: the consensus (>= 2 certified runs), then the best run's effective poles
    cands = ([("consensus", F.consensus([r["eff"] for r in cert]))] if len(cert) >= 2 else []) + [("best run", best["eff"])]
    vp = VarPro(data, t, device)
    tried = []
    for src, rep in cands:
        if rep.size == 0:
            continue
        theta, real = pack(list(rep))
        theta, S_fit = vp.fit(theta, real)
        dS, dp = S_fit - best["S"], max(p - n_eff(real), 1)
        tried.append(dict(source=src, theta=theta, real=real, S=S_fit, dS=dS / sig2, dp=dp,
                          order_ok=bool(dS <= chi2.ppf(0.99, dp) * sig2)))
        if tried[-1]["order_ok"]:
            break
    chosen = next((x for x in tried if x["order_ok"]), tried[-1])
    theta, real = chosen["theta"], chosen["real"]
    mu_fit = unpack_theta(theta, real)
    profs, k = [], 0
    for z, r in zip(mu_fit, real):
        cs, S = profile(vp, theta, real, k)
        lo, hi = interval(cs, S, LEVEL * sig2)
        profs.append(dict(mu=[float(z.real), float(z.imag)], c=cs.tolist(), dS=((S - S.min()) / sig2).tolist(),
                          interval=[None if lo is None else z.real * lo, None if hi is None else z.real * hi]))
        k += 1 if r else 2
    true = E.true_folded(mu_true, beta_true)[0]
    res = dict(cond=cond, group=g0, system=system, eta_hat=float(np.sqrt(data["train"][1].shape[1] * sig2)),
               sigma2=sig2, p=p, n_cert=len(cert), best_seed=best["seed"], best_S=best["S"],
               tried=[dict(source=x["source"], S=x["S"], dS=x["dS"], dp=x["dp"], order_ok=x["order_ok"],
                           poles=[[float(z.real), float(z.imag)] for z in unpack_theta(x["theta"], x["real"])])
                      for x in tried],
               source=chosen["source"], order_ok=chosen["order_ok"],
               report=[[float(z.real), float(z.imag)] for z in mu_fit], profiles=profs,
               true_poles=[[float(z.real), float(z.imag)] for z in true], seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cond}_g{g0}.json").write_text(json.dumps(res, indent=1, default=float))
    fmt = lambda iv: "[" + ", ".join("open" if x is None else f"{x:.3f}" for x in iv) + "]"
    print(f"{cond} g{g0}: {len(cert)} cert., best s{best['seed']}, "
          + ", ".join(f"{x['source']} dS {x['dS']:.1f} (dp {x['dp']}, ok {x['order_ok']})" for x in tried)
          + "; Re mu intervals " + "; ".join(f"{complex(*pr['mu']):.3f} {fmt(pr['interval'])}" for pr in profs)
          + f" ({res['seconds']:.0f}s)", flush=True)


def score(r):
    """Truth-scored view of one group, per true pole: (fitted pole, interval closed, interval covers
    the true Re mu, relative error of the fit, relative half-width); and the false identified poles."""
    true = [complex(*z) for z in r["true_poles"]]
    fit = [(complex(*pr["mu"]), pr["interval"]) for pr in r["profiles"]]
    per = {}
    used = set()
    for z in true:
        k = min(range(len(fit)), key=lambda i: abs(fit[i][0] - z)) if fit else None
        if k is None or abs(fit[k][0] - z) >= T.RECOVERED * abs(z):
            per[z] = None
            continue
        used.add(k)
        f, (lo, hi) = fit[k]
        closed = lo is not None and hi is not None
        per[z] = dict(fit=f, closed=closed, err=abs(f - z) / abs(z),
                      cover=(hi is None or hi <= z.real) and (lo is None or z.real <= lo),
                      half=abs(lo - hi) / 2 / abs(z) if closed else None)
    false = sum(1 for i, (f, (lo, hi)) in enumerate(fit) if i not in used and lo is not None and hi is not None)
    return per, false


def summarize():
    import e2_noise as NZ
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lab = lambda z: f"{z.real:+g}{z.imag:+g}i" if z.imag else f"{z.real:+g}"
    lines = ["# E4: minimal-form refit, order test and profile intervals", "",
             "Stage `rescue_na` (noise-aware restarts), groups of 5 seeds (one noise draw each), truth-free up "
             "to the scores. Certified: S − S_min ≤ χ²_p(0.99) σ̂². Report: the consensus of the certified runs "
             "(≥ 2), else the best run's effective poles. The report is refitted as a minimal tied layer (one "
             "slot per reported pole); order test: S_minimal − S_best ≤ χ²_Δp(0.99) σ̂²; if the consensus "
             "fails it, the best run's poles are tried. Each fitted pole: 95 % profile interval of Re μ "
             f"(others re-fitted, c ∈ [1/{C_MAX:g}, {C_MAX:g}]); identified = within 0.1 of a true pole with the "
             "interval closed. Cell per true pole: identified groups / intervals covering the true Re μ; "
             "RMS relative error of the minimal fit; mean relative half-width of the interval; relative CRB "
             "(e2_noise.py).", ""]
    for system, (base, noisy) in NZ.SYSTEMS.items():
        g = [r for r in rs if r["system"] == system]
        if not g:
            continue
        cr = NZ.crb_unit(base)
        true = [complex(*z) for z in cr["poles"]]
        lines += [f"## {system}", "",
                  "| η | groups | certified runs (mean) | report from consensus | order test passed | "
                  + " | ".join(f"{lab(z)}: identified / covered, error, half-width, CRB" for z in true)
                  + " | false identified |", "|---|---|---|---|---|" + "---|" * len(true) + "---|"]
        for eta, cond in noisy.items():
            gr = sorted((r for r in g if r["cond"] == cond), key=lambda r: r["group"])
            if not gr:
                continue
            sc = [score(r) for r in gr]
            cells = []
            for j, z in enumerate(true):
                xs = [s[0][z] for s in sc]
                idf = sum(bool(x and x["closed"]) for x in xs)
                cov = sum(bool(x and x["cover"]) for x in xs)
                errs = [x["err"] for x in xs if x]
                halves = [x["half"] for x in xs if x and x["half"] is not None]
                crb = cr["unit"][j] * eta / np.sqrt(cr["n_t"])
                cells.append(f"{idf}/{cov}, " + (f"{np.sqrt(np.mean(np.square(errs))):.1e}" if errs else "–")
                             + (f", {np.mean(halves):.1e}" if halves else ", –") + f", {crb:.1e}")
            lines.append(f"| {eta:g} | {len(gr)} | {np.mean([r['n_cert'] for r in gr]):.1f} | "
                         f"{sum(r['source'] == 'consensus' for r in gr)}/{len(gr)} | {sum(r['order_ok'] for r in gr)}/{len(gr)} | "
                         + " | ".join(cells) + f" | {sum(s[1] for s in sc)} |")
        lines.append("")
    (T.R.RESULTS / "e4_profile.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=CONDS)
    ap.add_argument("--group", nargs="+", type=int, default=[10, 15, 20, 25])
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for g in a.group:
            run(c, g, device)


if __name__ == "__main__":
    main()
