"""(C) E4, input noise: backward elimination under EIVML, started from the calibrated `foh_eivo` report.

EIVML removes the EIV bias when it refits a finished report (twentieth pass), but as the search cost of the whole
pipeline it admits undamped poles on the DFT grid (twenty-first pass). Here it only refits and reduces: the
`foh_eivo` report of each group is refitted under EIVML, then the cheapest move (drop pole k / make complex pole
k real) is taken while dS = S_reduced - S_current <= chi2_dp(0.99) (noise units, no calibration). The final model
gets 95 % EIVML profile intervals of Re mu. A reported pole that is a duplicate (within 10 % of another), aliased
(|Im mu| > pi / dt) or undamped (|Re mu| T_rec < 1) is unresolved. The rule was fixed before any group was run.

    python experiments/C/theory/e4_mlelim.py --cond p6_e1_T20_n1e-2_u1e-1 --group 10 15 20 25
    python experiments/C/theory/e4_mlelim.py --summarize      # writes results/e4_mlelim.md
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
    lines += ["", "| condition | group | poles | S / n | report ((u) = unresolved) | identified | false |",
              "|---|---|---|---|---|---|---|"] + rows
    (T.R.RESULTS / "e4_mlelim.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+")
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
