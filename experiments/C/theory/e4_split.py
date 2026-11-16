"""(C) E4, input noise: a location-calibrated split test for reported real poles.

The calibrated reduction (`e4_profile.py --hold foh_eivo`) merges the p6 E1 pair -1 / -2 at
eta_u = 10 eta_y, and rules read from its reduction path cannot tell that merge from the EIV bias
of an isolated pole (sixteenth pass, `e4_absorb.py`). Its threshold is calibrated on the move it
takes, usually a spurious fast pole, whose gain under input noise is large. This test calibrates
an added pole *at a given location* instead.

Per group and reported real pole z, truth-free:

  G(w)   = S_report - S_split(w), the split model = report + a real pole held at w = f z,
           f in FACTORS (skipped within MERGE of a reported pole), everything else refitted;
  G*(w)  the same gain on N_CAL EIV draws (denoised stored inputs as clean inputs, as
           calibrated_order);
  kappa_w = max(1, mean G*(w) / DP), ratio(w) = G(w) / (kappa_w chi2_DP(0.99)), DP = 2.

z is flagged (data hold more structure near z than the report) if max_w ratio > 1. For a
flagged pole the split model is also refitted with w free (start: the best w), to see what the
extra pole resolves into.

Null (`--null`): `denoise` draws from the report's fit; `bc` from its bias-corrected fit
2 theta - mean theta* (a first pass of N_CAL draws from the report's fit, each refitted from
theta). Under strong input noise the report is EIV-biased, and draws from it lack the structure
by which the real data differ from the biased fit.

Two further post-report checks, no refits: a reported pole within MERGE of another (duplicate)
and a complex pole with |Im mu| > pi / dt (above the Nyquist frequency of the sampled data). The
summary scores the `foh_eivo` reports with every flagged pole read as unresolved (open interval).

    python experiments/C/theory/e4_split.py --cond p6_e1_T20_n1e-2_u1e-1 --group 10 15 20 25 [--null bc]
    python experiments/C/theory/e4_split.py --summarize      # writes results/e4_split.md
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
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e4_profile as EP  # noqa: E402

F, RS, T, E = EP.F, EP.RS, EP.T, EP.E
SRC = T.R.RESULTS / "e4_profile_foh_eivo"
FACTORS = (0.25, 0.4, 0.63, 1.6, 2.5, 4.0)
DP = 2
N_CAL = EP.N_CAL
NULLS = ("denoise", "bc")


def out_dir(null):
    return T.R.RESULTS / ("e4_split" if null == "denoise" else f"e4_split_{null}")


def with_pole(theta, real, w):
    """Report + a real pole at w (appended last); returns (theta, real, index of w in theta)."""
    return np.append(theta, w), list(real) + [True], len(theta)


def run(cond, g0, device, null="denoise"):
    out = out_dir(null) / f"{cond}_g{g0}.json"
    if out.exists():
        return
    t0 = time.time()
    rep = json.loads((SRC / f"{cond}_g{g0}.json").read_text())
    data, t, _, _, runs = EP.load_group(cond, g0, device)
    sig2_y = RS.noise_var(data["train"][1])
    sig2_u = EP.noise_var_robust(data["train"][0])
    vp = EP.VarPro(data, t, device, "foh")
    best0 = min(runs, key=lambda r: r["S"])                 # EIV weights as e4_profile.run
    th0, re0 = EP.pack(list(best0["eff"]))
    th0, _ = vp.fit(th0, re0)
    vp = EP.VarPro(data, t, device, "foh", EP.eiv_weights(vp, th0, re0, sig2_y, sig2_u))
    full = [complex(*z) for z in rep["report"]]
    theta, real = EP.pack(full)
    theta, S_rep = vp.fit(theta, real)
    poles = EP.unpack_theta(theta, real)
    x, y = (a.double().cpu() for a in data["train"])
    xh = EP.denoise_input(x, sig2_u)
    vpx = vp.like({"train": (xh, y)})
    gen = torch.Generator().manual_seed(500 + g0)
    nz = lambda a, s2: a + math.sqrt(s2) * a.norm(dim=1, keepdim=True) * torch.randn(a.shape, generator=gen,
                                                                                    dtype=a.dtype)

    def pred(th):
        with torch.no_grad():
            return (vpx.pred(torch.tensor(th, device=device), real).reshape(y.shape) / vpx.scale).cpu()
    theta_gen = theta
    if null == "bc":
        yc = pred(theta)
        tb = [vp.like({"train": (nz(xh, sig2_u), nz(yc, sig2_y))}).fit(theta, real)[0] for _ in range(N_CAL)]
        theta_gen = 2 * theta - np.mean(tb, axis=0)
        is_re = np.zeros(theta.size, bool)
        is_re[np.cumsum([0] + [1 if r else 2 for r in real])[:-1]] = True
        theta_gen = np.where(is_re, np.minimum(theta_gen, -1e-3), np.abs(theta_gen))   # decays < 0, Im >= 0
    yc = pred(theta_gen)
    sites = []                                              # (pole index, w)
    for k, z in enumerate(poles):
        if z.imag:
            continue
        for f in FACTORS:
            w = f * z.real
            if all(abs(w - p) >= F.MERGE * abs(p) for p in poles):
                sites.append((k, w))
    G = []
    for k, w in sites:
        a, b, i = with_pole(theta, real, w)
        G.append(S_rep - vp.fit(a, b, fixed=i)[1])
    Gb = np.zeros((N_CAL, len(sites)))
    for j in range(N_CAL):
        vb = vp.like({"train": (nz(xh, sig2_u), nz(yc, sig2_y))})
        th_b, S_b = vb.fit(theta, real)
        for s, (k, w) in enumerate(sites):
            a, b, i = with_pole(th_b, real, w)
            Gb[j, s] = S_b - vb.fit(a, b, fixed=i)[1]
    q = chi2.ppf(0.99, DP)
    rows = []
    for k, z in enumerate(poles):
        if z.imag:
            continue
        idx = [s for s, (kk, _) in enumerate(sites) if kk == k]
        sc = []
        for s in idx:
            kap = max(1.0, float(Gb[:, s].mean()) / DP)
            sc.append(dict(w=float(sites[s][1]), G=float(G[s]), G_boot_mean=float(Gb[:, s].mean()),
                           G_boot_max=float(Gb[:, s].max()), kappa=kap, ratio=float(G[s] / (kap * q))))
        top = max(sc, key=lambda d: d["ratio"]) if sc else None
        row = dict(pole=[z.real, z.imag], sites=sc, max_ratio=top["ratio"] if top else None,
                   flagged=bool(top and top["ratio"] > 1))
        if row["flagged"]:
            a, b, _ = with_pole(theta, real, top["w"])
            a, S = vp.fit(a, b)
            row["split_fit"] = [[float(p.real), float(p.imag)] for p in EP.unpack_theta(a, b)]
            row["split_gain"] = float(S_rep - S)
        rows.append(row)
        print(f"{cond} g{g0} {z.real:+.3f}: max ratio {row['max_ratio']}, flagged {row['flagged']}", flush=True)
    res = dict(cond=cond, group=g0, null=null, report=[[p.real, p.imag] for p in poles], S_report=S_rep,
               theta_gen=[float(v) for v in theta_gen], dt=float(t[1] - t[0]), rows=rows,
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def checks(r):
    """Reported poles flagged without refits: duplicates (within MERGE of another) and aliased
    (complex, |Im mu| > pi / dt)."""
    P = [complex(*z) for z in r["report"]]
    dup = [z for i, z in enumerate(P) if any(abs(z - w) < F.MERGE * abs(z) for j, w in enumerate(P) if j != i)]
    dt = r.get("dt") or 20.0 / T.unpack(r["cond"])[4]        # outputs before `dt` was stored: T = 20 only
    alias = [z for z in P if abs(z.imag) > math.pi / dt]
    return dup, alias


def rescore(r, flagged, bc=False):
    """EP.score of the foh_eivo group with the flagged poles read as unresolved; bc: every pole moved to
    its bias-corrected value (theta_gen of the `bc` null; the relative profile interval moves with it), where
    the pole's EIV bootstrap 95 % half-width (stored by foh_eivo) is within the identification tolerance
    T.RECOVERED |z|: wider, the bootstrap is too non-linear for a bias correction (p6 E1 -1 / -2 at
    eta_u = 0.1, spread 9-18 %, are pushed apart to -0.7 / -2.5)."""
    g = json.loads((SRC / f"{r['cond']}_g{r['group']}.json").read_text())
    mus = [complex(*pr["mu"]) for pr in g["profiles"]]
    if bc:
        rep = [complex(*z) for z in r["report"]]
        new = EP.unpack_theta(r["theta_gen"], [not z.imag for z in rep])
        for k, m in enumerate(mus):
            j = min(range(len(rep)), key=lambda i: abs(rep[i] - m))
            sd = g["profiles"][k].get("boot_std")
            if abs(rep[j] - m) < 0.05 * abs(m) and sd is not None and 1.96 * sd <= T.RECOVERED * abs(m):
                g["profiles"][k]["mu"] = [new[j].real, new[j].imag]
    for f in flagged:                                  # the nearest stored pole (refits differ slightly)
        k = min(range(len(mus)), key=lambda i: abs(mus[i] - f))
        if abs(mus[k] - f) < 0.05 * abs(f):
            g["profiles"][k]["unresolved"] = True
    return EP.score(g)


def summarize():
    lab = lambda z: f"{z[0]:+.3g}{z[1]:+.3g}i" if z[1] else f"{z[0]:+.3g}"
    labc = lambda z: lab([z.real, z.imag])
    lines = ["# E4: location-calibrated split test of reported real poles", "",
             f"Per reported real pole z of the `foh_eivo` report: a real pole added at w = f z, f ∈ {FACTORS}, "
             f"everything else refitted; gain G(w) against {N_CAL} EIV draws, ratio = G / (κ_w χ²_{DP}(0.99)), "
             f"κ_w = max(1, mean G* / {DP}). Flagged if max ratio > 1. Split fit: the flagged pole's split model "
             "refitted with w free. Null `denoise`: draws from the report's fit; `bc`: from its bias-corrected "
             "fit. Checks without refits: dup = a reported pole within 10 % of another, alias = a complex pole "
             "above the Nyquist frequency.", ""]
    for null in NULLS:
        rs = [json.loads(f.read_text()) for f in sorted(out_dir(null).glob("*.json"))]
        if not rs:
            continue
        lines += [f"## null `{null}`", "",
                  "| condition | group | reported real poles: max ratio (flag) | max G / mean G* | split fit of flagged "
                  "poles (real, |z| < 30) | dup / alias |", "|---|---|---|---|---|---|"]
        tally = {}
        for r in rs:
            cells = ", ".join(f"{lab(x['pole'])}: {x['max_ratio']:.3g}{' **F**' if x['flagged'] else ''}"
                              for x in r["rows"] if x["max_ratio"] is not None)
            gm = max((s["G"] / s["G_boot_mean"] for x in r["rows"] for s in x["sites"] if s["G_boot_mean"] > 0),
                     default=float("nan"))
            sp = "; ".join(f"{lab(x['pole'])} → " + ", ".join(lab(p) for p in x["split_fit"]
                                                               if not p[1] and abs(p[0]) < 30)
                           for x in r["rows"] if x["flagged"])
            dup, alias = checks(r)
            dl = ", ".join(map(labc, dup)) or "–"
            al = ", ".join(map(labc, alias)) or "–"
            lines.append(f"| {r['cond']} | {r['group']} | {cells} | {gm:.3g} | {sp or '–'} | {dl} / {al} |")
            split = [complex(*x["pole"]) for x in r["rows"] if x["flagged"]]
            for name, fl, bc in (("report", [], False), ("split", split, False),
                                 ("split + checks", split + dup + alias, False),
                                 ("bc values", split + dup + alias, True)):
                if bc and null != "bc":
                    continue
                per, false = rescore(r, fl, bc)
                ident = sum(v is not None and v["closed"] for v in per.values())
                t_ = tally.setdefault((r["cond"], name), [0, 0, 0, 0])
                t_[0] += 1
                t_[1] += ident
                t_[2] += len(per)
                t_[3] += false
        lines += ["", "Scored (truth used only here): identified true poles / false identified, over the groups.", "",
                  "| condition | groups | report | report, split flags unresolved | + dup / alias unresolved |"
                  + (" + bias-corrected values |" if null == "bc" else ""),
                  "|---|---|---|---|---|" + ("---|" if null == "bc" else "")]
        for cond in dict.fromkeys(c for c, _ in tally):
            v = [tally[(cond, n)] for n in ("report", "split", "split + checks", "bc values") if (cond, n) in tally]
            lines.append(f"| {cond} | {v[0][0]} | " + " | ".join(f"{a[1]}/{a[2]}, {a[3]} false" for a in v) + " |")
        lines.append("")
    (T.R.RESULTS / "e4_split.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=[])
    ap.add_argument("--group", nargs="+", type=int, default=[10, 15, 20, 25])
    ap.add_argument("--null", choices=NULLS, default="denoise")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for g in a.group:
            run(c, g, device, a.null)


if __name__ == "__main__":
    main()
