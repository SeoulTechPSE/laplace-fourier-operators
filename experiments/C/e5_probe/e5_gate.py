"""(C) E5, seventh pass: data-side gates of the RQ4 trust rule, compared over all simulated / original sets.

A stored pole that passes rule v1 (valid Jacobian probe, agreement with a stored pole, fit) is the data's equivalent
linear pole over the training regime (fifth pass). The gate decides whether that pole may be read as the
small-signal pole.

- **v2 gate** (fifth pass): the best linear pair of all training samples leaves < 0.1 of the output.
- **v3 gate** (seventh pass, a pole-shift gate): with mu_low the equivalent pole of the lowest-amplitude fifth of the
  training samples (by peak |f|) and mu_all that of all samples, the low band must be linear (residual < 0.1)
  and the shift |mu_all - mu_low| / |mu_low| < 0.1 (the 10 % accuracy the rule claims; first set to 0.05).

For each set the gate values come from the data alone; the trusted runs and their errors come from
`results/e5_probe_<set>.json`.

    python experiments/C/e5_probe/e5_gate.py [set ...]      # writes results/e5_gate.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e5_probe as P  # noqa: E402
import e5_train as TR  # noqa: E402

EP = P.EP
LOW_FRAC, LOW_RESID, SHIFT = 0.2, 0.1, 0.1     # SHIFT was 0.05 in the first run (see README, seventh pass)
SETS = ("small", "mixed", "medium", "moderate", "pend_orig", "pend_medium", "pend_large", "wide", "pend_wide",
        "c0_orig", "c0_small", "c0_medium", "lz_orig", "lz_small", "lz_wide",
        "lz_r10_orig", "lz_r10_small", "lz_r10_wide")


STARTS = ([complex(-0.3, 1.0)], [complex(-0.25, 0.97)], [complex(-0.5, 1.5)])
STARTS_LZ = ([complex(-0.9, 4.0)], [complex(-0.5, 5.0)], [complex(-1.5, 3.0)], [complex(-0.6, 6.2)])   # last: rho = 10


def eq_pole(f, u, t, device, center=False):
    """center (Lorenz, ninth pass): fit u - mean(u) to f - mean(f), which removes the common zero-input response."""
    if center:
        f, u = f - f.mean(0), u - u.mean(0)
    vp = EP.VarPro({"train": (torch.tensor(f), torch.tensor(u))}, t, device, "foh")
    best = None
    for z0 in (STARTS_LZ if center else STARTS):
        th, re_ = EP.pack(z0)
        th, S = vp.fit(th, re_)
        if best is None or S < best[1]:
            best = (th, S, re_)
    th, _, re_ = best
    with torch.no_grad():
        pred = vp.pred(torch.tensor(th, device=device), re_)
    return EP.unpack_theta(th, re_)[0], float((vp.y - pred).norm() / vp.y.norm())


def gates(tag, device):
    data, grids = TR.data_and_grid(tag)
    f, u = (a.double().numpy() for a in data["train"])
    t = np.asarray(grids["t"], float).ravel()
    amp = np.abs(f).max(1)
    low = np.argsort(amp)[: int(LOW_FRAC * len(amp))]
    c = tag.startswith("lz_")
    mu_all, r_all = eq_pole(f, u, t, device, c)
    mu_low, r_low = eq_pole(f[low], u[low], t, device, c)
    shift = abs(mu_all - mu_low) / abs(mu_low)
    return dict(mu_all=[mu_all.real, mu_all.imag], r_all=r_all, mu_low=[mu_low.real, mu_low.imag], r_low=r_low,
                low_amp=float(amp[low].max()), shift=shift, v2=r_all < P.GATE,
                v3=bool(r_low < LOW_RESID and shift < SHIFT))


def main(tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_p = P.RESULTS / "e5_gate.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for tag in tags:
        g = gates(tag, device)
        pr = json.loads((P.RESULTS / f"e5_probe_{tag}.json").read_text())
        ref = complex(*pr["mu_lin"])
        tr = [dict(variant=v, seed=r["seed"], err=abs(complex(*r["jac"]["h_poles"][0]) - ref) / abs(ref))
              for v, rows in pr["variants"].items() for r in rows if P.trusted(r)]
        g["v1_trusted"] = tr
        res[tag] = g
        print(tag, {k: v for k, v in g.items() if k != "v1_trusted"}, "v1 errors", [round(x["err"], 3) for x in tr],
              flush=True)
    out_p.write_text(json.dumps(res, indent=1))
    summarize(res)


def summarize(res):
    c = lambda z: f"{z[0]:.3f}{z[1]:+.3f}i"
    lines = ["# E5: data-side gates of the trust rule", "",
             f"v2 gate: all-sample linear residual < {P.GATE:g}. v3 gate: lowest {LOW_FRAC:.0%} of samples by peak |f| "
             f"linear (residual < {LOW_RESID:g}) and equivalent-pole shift all vs low < {SHIFT:g}. A trusted run is "
             "correct if its Jacobian pole is within 10 % of the small-signal pole: −0.25 ± 0.9682i (damped Duffing, "
             "pendulum), ± i (`c0_*`, undamped Duffing), −0.9287 ± 4.1476i (`lz_*`, Lorenz ρ = 5 at C+) or −0.5955 ± 6.1742i (`lz_r10_*`, ρ = 10 at C+); Lorenz from centred data.", "",
             "| set | all: pole, resid | low band (peak |f| ≤): pole, resid | shift | v2 | v3 | v1 trusted: correct / all "
             "(errors) | v2 false / missed | v3 false / missed |", "|---|---|---|---|---|---|---|---|---|"]
    tot = dict(v1f=0, v2f=0, v2m=0, v3f=0, v3m=0)
    for tag, g in res.items():
        errs = [x["err"] for x in g["v1_trusted"]]
        ok = sum(e < 0.1 for e in errs)
        bad = len(errs) - ok
        v2f, v2m = (bad, 0) if g["v2"] else (0, ok)
        v3f, v3m = (bad, 0) if g["v3"] else (0, ok)
        for k, v in dict(v1f=bad, v2f=v2f, v2m=v2m, v3f=v3f, v3m=v3m).items():
            tot[k] += v
        lines.append(f"| `{tag}` | {c(g['mu_all'])}, {g['r_all']:.3f} | ({g['low_amp']:.2f}) {c(g['mu_low'])}, "
                     f"{g['r_low']:.3f} | {g['shift']:.3f} | {'pass' if g['v2'] else 'fail'} | "
                     f"{'pass' if g['v3'] else 'fail'} | {ok} / {len(errs)} ("
                     + (f"{min(errs):.2f}–{max(errs):.2f}" if errs else "–") + f") | {v2f} / {v2m} | {v3f} / {v3m} |")
    lines += ["", f"Totals: v1 alone {tot['v1f']} false trusts; v2 {tot['v2f']} false, {tot['v2m']} correct runs not "
              f"trusted; v3 {tot['v3f']} false, {tot['v3m']} correct runs not trusted."]
    (P.RESULTS / "e5_gate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:] or SETS)
