"""(C) E4, input noise: backward elimination under EIVML, started from the calibrated `foh_eivo` report.

EIVML removes the EIV bias when it refits a finished report (twentieth pass), but as the search cost of the whole
pipeline it admits undamped poles on the DFT grid (twenty-first pass). Here it only refits and reduces: the
`foh_eivo` report of each group is refitted under EIVML, then the cheapest move (drop pole k / make complex pole
k real) is taken while dS = S_reduced - S_current <= chi2_dp(0.99) (noise units, no calibration). The final model
gets 95 % EIVML profile intervals of Re mu. A reported pole that is a duplicate (within 10 % of another), aliased
(|Im mu| > pi / dt) or undamped (|Re mu| T_rec < 1) is unresolved. The rule was fixed before any group was run.

    python experiments/C/theory/e4_mlelim.py --cond p6_e1_T20_n1e-2_u1e-1 --group 10 15 20 25
    python experiments/C/theory/e4_mlelim.py --summarize      # writes results/e4_mlelim.md

Finding: 0 false poles up to p4 multisine at 30 eta_y and p6 E1 at 5 eta_y on fresh groups, but the EIVML profile
intervals undercover once input noise dominates (17 / 36 at p4 0.3). Hence `--boot`: a parametric EIV bootstrap
around the final EIVML fit, fixed before it was run. Clean inputs: the stored inputs denoised (EP.denoise_input);
per draw x* = x^ + e_u*, y* = G_fit x^ + e_y* (the stored-input noise is not added twice, as EP.eiv_bootstrap
does), refitted under EIVML from the fit; interval Re mu +- 1.96 std over N_BOOT draws.

    python experiments/C/theory/e4_mlelim.py --boot --cond p4_ms16_T20_n1e-2_u3e-1 --group 10 15 20 25
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
import e2_rescue as RS  # noqa: E402
import e4_mlpipe as MP  # noqa: E402
import e4_profile as EP  # noqa: E402

T = EP.T
SRC = EP.out_dir("foh_eivo")
N_BOOT = 32
OUT = T.R.RESULTS / "e4_mlelim"


def unresolved(poles, dt, t_rec):
    return [z for i, z in enumerate(poles) if abs(z.imag) > math.pi / dt or EP.undamped([z], t_rec)
            or any(abs(z - w) < EP.DUP * abs(z) for j, w in enumerate(poles) if j != i)]


def eliminate(ml, poles):
    """Backward elimination under ml from poles; returns (theta, real, S, steps)."""
    th, re_ = EP.pack(poles)
    th, S_cur = ml.fit(th, re_)
    steps = []
    while len(re_) > 1:
        cur = EP.unpack_theta(th, re_)
        moves = [(cur[:k] + cur[k + 1:], ("drop", k)) for k in range(len(cur))]
        moves += [(cur[:k] + [complex(z.real, 0.0)] + cur[k + 1:], ("real", k)) for k, z in enumerate(cur) if z.imag]
        trials = []
        for m, mv in moves:
            a, b = EP.pack(m)
            a, S = ml.fit(a, b)
            trials.append((S, a, b, mv))
        S, a, b, (kind, k) = min(trials, key=lambda q: q[0])
        dp = EP.n_eff(re_) - EP.n_eff(b)
        thr = float(chi2.ppf(0.99, dp))
        ok = bool(S - S_cur <= thr)
        steps.append(dict(move=kind, moved=[cur[k].real, cur[k].imag], dS=float(S - S_cur), dp=int(dp),
                          threshold=thr, removed=ok))
        if not ok:
            break
        th, re_, S_cur = a, b, S
    return th, re_, S_cur, steps


def run(cond, g0, device):
    out = OUT / f"{cond}_g{g0}.json"
    if out.exists():
        return
    t0 = time.time()
    rep = json.loads((SRC / f"{cond}_g{g0}.json").read_text())
    data, t, _, _, _ = EP.load_group(cond, g0, device)
    dt = float(t[1] - t[0])
    t_rec = dt * len(t)
    ml = EP.EIVML(data, t, device, "foh", RS.noise_var(data["train"][1]), EP.noise_var_robust(data["train"][0]))
    th, re_, S, steps = eliminate(ml, [complex(*z) for z in rep["report"]])
    th, S = ml.fit(th, re_)
    mu = EP.unpack_theta(th, re_)
    flagged = unresolved(mu, dt, t_rec)
    profs, k = [], 0
    for z, r in zip(mu, re_):
        cs, Sp = EP.profile(ml, th, re_, k)
        lo, hi = EP.interval(cs, Sp, EP.LEVEL)
        profs.append(dict(mu=[z.real, z.imag], c=cs.tolist(), dS=(Sp - Sp.min()).tolist(),
                          interval=[None if lo is None else z.real * lo, None if hi is None else z.real * hi],
                          **({"unresolved": True} if z in flagged else {})))
        k += 1 if r else 2
    res = dict(cond=cond, group=g0, start=rep["report"], S=S, n=int(data["train"][1].numel()), steps=steps,
               report=[[z.real, z.imag] for z in mu], unresolved=[[z.real, z.imag] for z in flagged],
               profiles=profs, true_poles=rep["true_poles"], seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"{cond} g{g0}: {len(rep['report'])} -> {len(mu)} poles, S/n {S / res['n']:.3f}; "
          + "; ".join(f"{z:.3g} {pr['interval']}" + (" (u)" if pr.get("unresolved") else "")
                      for z, pr in zip(mu, profs)) + f" ({res['seconds']:.0f}s)", flush=True)


def boot(cond, g0, device, n_boot=N_BOOT, seed=0):
    f = OUT / f"{cond}_g{g0}.json"
    r = json.loads(f.read_text())
    if "boot_seconds" in r:
        return
    t0 = time.time()
    data, t, _, _, _ = EP.load_group(cond, g0, device)
    s2y, s2u = RS.noise_var(data["train"][1]), EP.noise_var_robust(data["train"][0])
    ml = EP.EIVML(data, t, device, "foh", s2y, s2u)
    th, re_ = EP.pack([complex(*z) for z in r["report"]])
    th, _ = ml.fit(th, re_)
    x, y = (a.double().cpu() for a in data["train"])
    xh = EP.denoise_input(x, s2u)
    with torch.no_grad():
        mlx = ml.like({"train": (xh, y)})
        yc = (mlx.pred(torch.tensor(th, device=device), re_).reshape(y.shape) / mlx.scale).cpu()
    gen = torch.Generator().manual_seed(seed)
    nz = lambda a, s2: a + math.sqrt(s2) * a.norm(dim=1, keepdim=True) * torch.randn(a.shape, generator=gen,
                                                                                    dtype=a.dtype)
    idx = np.cumsum([0] + [1 if q else 2 for q in re_])[:-1]
    draws = []
    for _ in range(n_boot):
        vb = ml.like({"train": (nz(xh, s2u), nz(yc, s2y))})
        draws.append(vb.fit(th, re_)[0][idx])
    sd = np.std(draws, axis=0, ddof=1)
    for pr, s_ in zip(r["profiles"], sd):
        pr["boot_std"] = float(s_)
        pr["interval_boot"] = [pr["mu"][0] - 1.96 * float(s_), pr["mu"][0] + 1.96 * float(s_)]
    r["boot_seconds"] = round(time.time() - t0, 1)
    f.write_text(json.dumps(r, indent=1))
    print(f"{cond} g{g0}: boot std " + ", ".join(f"{p['mu'][0]:.3g}: {p['boot_std']:.2e}" for p in r["profiles"])
          + f" ({r['boot_seconds']:.0f}s)", flush=True)


def coverage(rs):
    """{cond: [profile covered, boot covered, scored, profile half-widths, boot half-widths]} over identified poles."""
    out = {}
    for r in rs:
        per, _ = EP.score(r)
        c = out.setdefault(r["cond"], [0, 0, 0, [], []])
        for z, x in per.items():
            if not (x and x["closed"]):
                continue
            c[0] += bool(x["cover"])
            c[2] += 1
            c[3].append(x["half"])
            if x["boot_cover"] is not None:
                c[1] += bool(x["boot_cover"])
                pr = min(r["profiles"], key=lambda p: abs(complex(*p["mu"]) - z))
                c[4].append(1.96 * pr["boot_std"] / abs(z))
    return out


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    tl, ref = {}, MP.tally("foh_eivo")
    rows = []
    for r in rs:
        per, false = EP.score(r)
        key = (r["cond"], (r["cond"], r["group"]) in MP.FRESH)
        t_ = tl.setdefault(key, [0, 0, 0, 0, 0])
        t_[0] += 1
        t_[1] += sum(bool(x and x["closed"]) for x in per.values())
        t_[2] += len(per)
        t_[3] += false
        t_[4] += len(r["unresolved"])
        un = [complex(*z) for z in r["unresolved"]]
        lab = lambda z: (f"{z.real:.3g}{z.imag:+.3g}i" if z.imag else f"{z.real:.3g}") + (" (u)" if z in un else "")
        rows.append(f"| {r['cond']} | {r['group']}{' (fresh)' if key[1] else ''} | {len(r['start'])} → "
                    f"{len(r['report'])} | {r['S'] / r['n']:.3f} | {', '.join(lab(complex(*z)) for z in r['report'])} | "
                    f"{sum(bool(x and x['closed']) for x in per.values())}/{len(per)} | {false} |")
    lines = ["# E4: backward elimination under EIVML from the `foh_eivo` report", "",
             "Each group's `foh_eivo` report refitted under EIVML, then the cheapest move (drop a pole / make a complex "
             "pole real) taken while ΔS ≤ χ²_Δp(0.99) (noise units, no calibration); 95 % EIVML profile intervals. "
             "Unresolved: duplicates (10 %), aliased or undamped (|Re μ| T_rec < 1) poles. Identified = within 10 % "
             "of a true pole with a closed interval; false = other poles with a closed interval. Fresh groups: "
             "trained after the twentieth pass.", "",
             "| condition | groups | EIVML elimination: identified, false, unresolved | `foh_eivo`: identified, false |",
             "|---|---|---|---|"]
    for k in sorted(tl):
        a, b = tl[k], ref.get(k)
        lines.append(f"| {k[0]}{' (fresh)' if k[1] else ''} | {a[0]} | {a[1]}/{a[2]}, {a[3]}, {a[4]} | "
                     + (f"{b[1]}/{b[2]}, {b[3]}" if b else "–") + " |")
    cv = coverage(rs)
    lines += ["", "Coverage of the true Re μ by the identified poles' intervals: 95 % EIVML profile / EIV bootstrap "
              f"around the EIVML fit (`--boot`, {N_BOOT} draws); mean relative half-width.", "",
              "| condition | identified poles | covered: profile / bootstrap | half-width: profile / bootstrap |",
              "|---|---|---|---|"]
    for c in sorted(cv):
        v = cv[c]
        lines.append(f"| {c} | {v[2]} | {v[0]} / " + (f"{v[1]} (of {len(v[4])})" if v[4] else "–") + " | "
                     + f"{np.mean(v[3]):.1e} / " + (f"{np.mean(v[4]):.1e}" if v[4] else "–") + " |")
    lines += ["", "| condition | group | poles | S / n | report ((u) = unresolved) | identified | false |",
              "|---|---|---|---|---|---|---|"] + rows
    (T.R.RESULTS / "e4_mlelim.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+")
    ap.add_argument("--group", nargs="+", type=int, default=[10, 15, 20, 25])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--boot", action="store_true", help="EIV bootstrap intervals around the final EIVML fit")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for g in a.group:
            boot(c, g, device) if a.boot else run(c, g, device)


if __name__ == "__main__":
    main()
