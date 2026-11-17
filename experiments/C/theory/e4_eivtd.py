"""(C) E4, input noise: exact time-domain errors-in-variables cost, and what a spurious fast pole gains.

Under input noise the E1 reports keep spurious fast poles (-48 ... -121) with narrow intervals, which
no post-report rule removes (nineteenth pass). The eleventh pass found that on truth data a fast
pole still lowers the frequency-domain EIV ML cost (`EIVML`) by 66 sigma^2 on E1 inputs, and named a
candidate cause: its stationary noise model ignores the start-up of the propagated input noise
(zero initial state). This script replaces it by the exact time-domain covariance.

The model is linear in the input: y_i = M_theta x_i, with M_theta (n_t x n_t) the first-order-hold
response matrix with zero initial state (the VarPro design applied to unit impulses). For the
stored x~_i = x_i + e_u,i, the residual r_i = y_i - M x~_i = e_y,i - M e_u,i has, per normalised
sample, the covariance

    C_i = sig2_y I + sig2_u (||x_i|| / ||y_i||)^2 M M^T,

and the cost (true inputs eliminated, no log det, as EIVML) is V = sum_i r_i^T C_i^{-1} r_i, with
poles and residues fitted jointly (batched Cholesky of the C_i, autograd through it).

Diagnosis on truth data (`e4_eiv.clean_split`, independent noise draws): the true model and the
true model plus fast real poles, fitted under three costs (EIV-weighted FOH = the pipeline's,
EIVML, EIVTD); gain dS = S_true - S_extra in noise units, against chi2_dp(0.99).

    python experiments/C/theory/e4_eivtd.py --cond p6_e1_T20 --eta-u 0.1 --draws 2
    python experiments/C/theory/e4_eivtd.py --summarize      # writes results/e4_eivtd.md

Finding (truth data, eta_u = 0.1): the EIVTD cost degenerates (S / n of the refitted true model 0.00 on
E1 inputs, 0.25 on multisine): without a log det term it can be driven towards zero by an ill-conditioned
M. The frequency-domain EIVML, in contrast, gives a spurious -50 on E1 inputs a gain of 0.1-0.7, far
below chi2. Hence `--drop`: the `foh_eivo` reports (group data) refitted under EIVML, each pole dropped
in turn, gain dS = S_dropped - S_report against chi2_dp(0.99) (> chi2: the pole is supported).

    python experiments/C/theory/e4_eivtd.py --drop --cond p6_e1_T20_n1e-2_u1e-1 --group 10 15 20 25
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
import e4_eiv as V  # noqa: E402
import e4_profile as EP  # noqa: E402
import e4_split as SP  # noqa: E402

T, E = EP.T, EP.E
OUT = T.R.RESULTS / "e4_eivtd"
OUT_DROP = T.R.RESULTS / "e4_mldrop"
SRC = T.R.RESULTS / "e4_profile_foh_eivo"
TD_ITERS = 400
EXTRA = {"p6": ([complex(-50, 0)], [complex(-50, 0), complex(-120, 0)]),
         "p4": ([complex(-28, 0)], [complex(-28, 0), complex(-244, 0)])}


class EIVTD(EP.EIVML):
    """Exact time-domain EIV cost (see the module docstring); interface as EIVML."""

    def __init__(self, data, t, device, hold, sig2_y, sig2_u):
        super().__init__(data, t, device, hold, sig2_y, sig2_u)
        eye = torch.eye(self.n_t, dtype=torch.float64)
        self.imp = EP.VarPro({"train": (eye, torch.ones_like(eye))}, t, device, hold)   # unit impulses
        x, _ = (a.to(device).double() for a in data["train"])
        self.x = x * self.scale                                       # inputs in normalised-output units

    def like(self, data):
        return EIVTD(data, self.t_np, self.device, self.hold, self.sig2_y, self.sig2_u)

    def matrix(self, thp, c, real):
        """M_theta: column j = response to the unit impulse e_j (zero initial state)."""
        A = self.imp.design(thp, real) / self.imp.scale[0, 0]            # (n_t * n_t, m), undo the row scale
        return (A @ c).reshape(self.n_t, self.n_t).T                     # [time, input index]

    def S(self, theta, real):
        thp, c = self.split(theta, real)
        M = self.matrix(thp, c, real)
        r = self.y.reshape(self.b, self.n_t) - self.x @ M.T              # (b, n_t)
        K = M @ M.T
        eye = torch.eye(self.n_t, dtype=K.dtype, device=K.device)
        s = self.sig2_u * self.ratio                                     # (b,)
        C = self.sig2_y * eye[None] + s[:, None, None] * K[None]
        L = torch.linalg.cholesky(C)
        z = torch.linalg.solve_triangular(L, r[:, :, None], upper=False)
        return z.pow(2).sum()

    def fit(self, theta0, real, fixed=None, iters=TD_ITERS):
        """L-BFGS from theta0 (with residues: start from the EIVML fit); about 0.85 s per evaluation."""
        return EP.VarPro.fit(self, self.full(theta0, real), real, fixed, iters)


def costs(x, y, t, device, poles):
    """Fits of `poles` under the three costs; returns {name: S} in noise units."""
    sig2_y, sig2_u = RS.noise_var(y), EP.noise_var_robust(x)
    d = {"train": (x, y)}
    vp0 = EP.VarPro(d, t, device, "foh")
    th, re_ = EP.pack(poles)
    th0, _ = vp0.fit(th, re_)
    out = {}
    vp = EP.VarPro(d, t, device, "foh", EP.eiv_weights(vp0, th0, re_, sig2_y, sig2_u))
    out["foh_eiv"] = vp.fit(th0, re_)[1]
    ml = EP.EIVML(d, t, device, "foh", sig2_y, sig2_u)
    th_ml, out["eivml"] = ml.fit(th0, re_)
    out["eivtd"] = EIVTD(d, t, device, "foh", sig2_y, sig2_u).fit(th_ml, re_)[1]   # from the ML fit
    return out


def run(cond, eta_u, draws, device):
    out = OUT / f"{cond}_u{eta_u:g}.json"
    if out.exists():
        return
    t0 = time.time()
    t, x0, y0 = V.clean_split(cond)
    system = T.unpack(cond)[0]
    mu, beta = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    true = list(E.true_folded(mu, beta)[0])
    gen = torch.Generator().manual_seed(int(1e6 * eta_u) + 11)
    rows = []
    for k in range(draws):
        x, y = V.add_noise(x0, eta_u, gen), V.add_noise(y0, V.ETA_Y, gen)
        S0 = costs(x, y, t, device, true)
        for extra in EXTRA[system]:
            S1 = costs(x, y, t, device, true + extra)
            dp = 2 * len(extra)
            rows.append(dict(draw=k, extra=[[z.real, z.imag] for z in extra], dp=dp,
                             chi2=float(chi2.ppf(0.99, dp)), S_true=S0, S_extra=S1,
                             gain={n: S0[n] - S1[n] for n in S0}))
            print(f"{cond} eta_u {eta_u:g} draw {k} +{[z.real for z in extra]}: gain "
                  + ", ".join(f"{n} {v:.1f}" for n, v in rows[-1]["gain"].items())
                  + f" (chi2 {rows[-1]['chi2']:.1f}); S_true " + ", ".join(f"{n} {v:.0f}" for n, v in S0.items()),
                  flush=True)
    res = dict(cond=cond, eta_u=eta_u, eta_y=V.ETA_Y, n=int(y0.numel()), rows=rows, seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def drop(cond, g0, device):
    out = OUT_DROP / f"{cond}_g{g0}.json"
    if out.exists():
        return
    t0 = time.time()
    rep = json.loads((SRC / f"{cond}_g{g0}.json").read_text())
    data, t, _, _, _ = EP.load_group(cond, g0, device)
    sig2_y, sig2_u = RS.noise_var(data["train"][1]), EP.noise_var_robust(data["train"][0])
    ml = EP.EIVML(data, t, device, "foh", sig2_y, sig2_u)
    full = [complex(*z) for z in rep["report"]]
    th, re_ = EP.pack(full)
    th, S_rep = ml.fit(th, re_)
    poles = EP.unpack_theta(ml.split(th, re_)[0], re_)
    rows = []
    for k, z in enumerate(poles):
        a, b = EP.pack(poles[:k] + poles[k + 1:])
        a, S = ml.fit(a, b)
        dp = EP.n_eff(re_) - EP.n_eff(b)
        rows.append(dict(pole=[z.real, z.imag], dS=float(S - S_rep), dp=int(dp), chi2=float(chi2.ppf(0.99, dp)),
                         supported=bool(S - S_rep > chi2.ppf(0.99, dp))))
        print(f"{cond} g{g0} drop {z:.3g}: dS {S - S_rep:.1f} (chi2 {rows[-1]['chi2']:.1f})", flush=True)
    res = dict(cond=cond, group=g0, S_report=S_rep, n=int(data["train"][1].numel()),
               true_poles=rep["true_poles"], rows=rows, seconds=round(time.time() - t0, 1))
    OUT_DROP.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def summarize_drop():
    rs = [json.loads(f.read_text()) for f in sorted(OUT_DROP.glob("*.json"))]
    if not rs:
        return []
    lines = ["", "## EIVML drop test of the `foh_eivo` reports", "",
             "Each reported pole dropped in turn, EIVML refit (group training data); dS = S_dropped − S_report in "
             "noise units against χ²_Δp(0.99). Truth (only for the last two columns): a reported pole is true if "
             "within 10 % of a true pole.", "",
             "| condition | group | S_report / n | reported poles: dS (✗ = not supported) | true poles not supported "
             "| other poles supported |", "|---|---|---|---|---|---|"]
    lab = lambda z: f"{z.real:+.3g}{z.imag:+.3g}i" if z.imag else f"{z.real:+.3g}"
    for r in rs:
        tr = [complex(*z) for z in r["true_poles"]]
        is_true = lambda z: any(abs(z - w) < T.RECOVERED * abs(w) or abs(z - w.conjugate()) < T.RECOVERED * abs(w)
                                for w in tr)
        cells, miss, extra = [], 0, 0
        for x in r["rows"]:
            z = complex(*x["pole"])
            cells.append(f"{lab(z)}: {x['dS']:.3g}{'' if x['supported'] else ' ✗'}")
            miss += is_true(z) and not x["supported"]
            extra += (not is_true(z)) and x["supported"]
        lines.append(f"| {r['cond']} | {r['group']} | {r['S_report'] / r['n']:.3f} | {', '.join(cells)} | {miss} | {extra} |")
    # report: the EIVML-refitted poles that are supported, duplicates (within 10 % of another) and aliased
    # complex poles (|Im mu| > pi / dt) read as unresolved; scored against the truth (within 10 %)
    tally = {}
    for r in rs:
        tr = [complex(*z) for z in r["true_poles"]]
        keep = [complex(*x["pole"]) for x in r["rows"] if x["supported"]]
        dt = 20.0 / T.unpack(r["cond"])[4]
        rep = [z for i, z in enumerate(keep) if abs(z.imag) <= math.pi / dt
               and not any(abs(z - w) < T.RECOVERED * abs(z) for j, w in enumerate(keep) if j != i)]
        used = set()
        for w in tr:
            c = [i for i, z in enumerate(rep) if i not in used and abs(z - w) < T.RECOVERED * abs(w)]
            if c:
                used.add(c[0])
        key = r["cond"] + (" (new groups)" if (r["cond"], r["group"]) in SP.OOS else "")
        t_ = tally.setdefault(key, [0, 0, 0, 0])
        t_[0] += 1
        t_[1] += len(used)
        t_[2] += len(tr)
        t_[3] += len(rep) - len(used)
    lines += ["", "Report = the supported EIVML-refitted poles, duplicates and aliased poles unresolved. Identified true "
              "poles (within 10 %) / false poles, over the groups.", "",
              "| condition | groups | identified | false |", "|---|---|---|---|"]
    lines += [f"| {k} | {v[0]} | {v[1]}/{v[2]} | {v[3]} |" for k, v in tally.items()]
    return lines


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lines = ["# E4: exact time-domain EIV cost — gain of spurious fast poles on truth data", "",
             f"Truth data (`e4_eiv.clean_split`, training split), output noise η_y = {V.ETA_Y:g}. Per draw: the "
             "true model and the true model plus fast real poles, fitted under the EIV-weighted FOH cost "
             "(the pipeline's), the frequency-domain EIV ML (`EIVML`) and the exact time-domain EIV cost "
             "(`EIVTD`). Gain = S_true − S_extra in noise units; χ² = χ²_Δp(0.99). S_true / n ≈ 1 for a "
             "consistent cost.", "",
             "| condition | η_u | added poles | χ² | gain FOH-EIV | gain EIVML | gain EIVTD | S_true / n (FOH-EIV / ML / TD) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rs:
        for x in r["rows"]:
            g = x["gain"]
            st = x["S_true"]
            lines.append(f"| {r['cond']} | {r['eta_u']:g} | {', '.join(f'{z[0]:g}' for z in x['extra'])} "
                         f"(draw {x['draw']}) | {x['chi2']:.1f} | {g['foh_eiv']:.1f} | {g['eivml']:.1f} | "
                         f"{g['eivtd']:.1f} | {st['foh_eiv'] / r['n']:.3f} / {st['eivml'] / r['n']:.3f} / "
                         f"{st['eivtd'] / r['n']:.3f} |")
    lines += summarize_drop()
    (T.R.RESULTS / "e4_eivtd.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=["p6_e1_T20"])
    ap.add_argument("--eta-u", nargs="+", type=float, default=[0.1])
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--drop", action="store_true", help="EIVML drop test of the foh_eivo reports")
    ap.add_argument("--group", nargs="+", type=int, default=[10, 15, 20, 25])
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.drop:
        for c in a.cond:
            for g in a.group:
                drop(c, g, device)
        return
    for c in a.cond:
        for e in a.eta_u:
            run(c, e, a.draws, device)


if __name__ == "__main__":
    main()
