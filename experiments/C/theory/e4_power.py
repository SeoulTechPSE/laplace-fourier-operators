"""(C) E4, input noise: power of the bootstrap-calibrated order test against a missed true pole.

`e4_profile.py --hold foh_eivo` passes its calibrated order test in every input-noise group, with
reports that hold exactly the true poles. A test that passes everything is only useful if it
also fails wrong reports. Here, per group, the reported model minus one pole (each pole in turn;
truth is used only to pick what to drop, the test itself is truth-free) is refitted and put
through the same test:

  dS = S_candidate - S_reference, dp = p_reference - p_candidate,
  pass if dS <= kappa chi2_dp(0.99), with
    chi2       kappa = 1
    pooled     kappa = kappa_g of foh_eivc (from the group JSON)
    calibrated kappa = kappa_o of calibrated_order, bootstrapped from the dropped-pole candidate

The setup (EIV weights from the best run, FOH reduction of every run, reference = the run with
the lowest full-model S) repeats e4_profile.run. A passed test on a dropped-pole candidate is a
miss (no power).

    python experiments/C/theory/e4_power.py --cond p4_ms16_T20_n1e-2_u1e-1 --group 10 15
    python experiments/C/theory/e4_power.py --summarize      # writes results/e4_power.md
"""
from __future__ import annotations

import argparse
import json
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
OUT = T.R.RESULTS / "e4_power"
SRC = T.R.RESULTS / "e4_profile_foh_eivo"


def run(cond, g0, device):
    out = OUT / f"{cond}_g{g0}.json"
    if out.exists():
        return
    t0 = time.time()
    rep = json.loads((SRC / f"{cond}_g{g0}.json").read_text())
    data, t, _, _, runs = EP.load_group(cond, g0, device)
    sig2_y = RS.noise_var(data["train"][1])
    sig2_u = EP.noise_var_robust(data["train"][0])
    vp = EP.VarPro(data, t, device, "foh")
    best0 = min(runs, key=lambda r: r["S"])
    th0, re0 = EP.pack(list(best0["eff"]))
    th0, _ = vp.fit(th0, re0)
    vp = EP.VarPro(data, t, device, "foh", EP.eiv_weights(vp, th0, re0, sig2_y, sig2_u))
    for r in runs:
        r["eff_full"] = r["eff"]
        r["eff"], r["S"], r["S_full"], r["p_full"] = EP.reduce(vp, r["eff"], 1.0)
    ref = min(runs, key=lambda r: r["S_full"])
    S_ref, p_ref = ref["S_full"], ref["p_full"]
    kappa_g = rep.get("kappa_group", 1.0)
    full = [complex(*z) for z in rep["report"]]
    th, re_ = EP.pack(full)
    th, S_rep = vp.fit(th, re_)
    rows = []
    for k, z in enumerate(full):
        a, b = EP.pack(full[:k] + full[k + 1:])
        a, S = vp.fit(a, b)
        dS, dp = S - S_ref, max(p_ref - EP.n_eff(b), 1)
        kappa_o, d = EP.calibrated_order(vp, a, b, ref["eff_full"], data, sig2_y, sig2_u, device, seed=300 + k)
        q = chi2.ppf(0.99, dp)
        rows.append(dict(dropped=[z.real, z.imag], dS=float(dS), dp=int(dp), kappa_g=kappa_g, kappa_o=kappa_o,
                         ratio_chi2=float(dS / q), ratio_pooled=float(dS / (kappa_g * q)),
                         ratio_cal=float(dS / (kappa_o * q)), dS_boot=d))
        print(f"{cond} g{g0} drop {z:.3f}: dS {dS:.1f} (dp {dp}), kappa_o {kappa_o:.1f}, ratio cal "
              f"{rows[-1]['ratio_cal']:.2f}", flush=True)
    res = dict(cond=cond, group=g0, S_ref=S_ref, p_ref=p_ref, S_report=S_rep,
               report_ratio_cal=next((x["ratio"] for x in rep["tried"] if x["source"] == rep["source"]), None),
               rows=rows, seconds=round(time.time() - t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lab = lambda z: f"{z[0]:+.3g}{z[1]:+.3g}i" if z[1] else f"{z[0]:+.3g}"
    lines = ["# E4: power of the calibrated order test against a missed true pole", "",
             "Per group: the `foh_eivo` report (exactly the true poles) minus one pole, refitted (EIV-weighted FOH), "
             "against the same reference as the order test (the run with the lowest full-model S). Ratio = "
             "ΔS / (κ χ²_Δp(0.99)); > 1 rejects the wrong report. κ = 1 (χ²), the pooled κ_g of `foh_eivc`, and "
             f"the calibrated κ_o ({EP.N_CAL} EIV draws from the dropped-pole candidate). Reported model: its own "
             "calibrated ratio.", "",
             "| condition | groups | reported model: ratio (max) | dropped pole | rejected χ² / pooled / calibrated "
             "| ratio calibrated (min) | κ_o (range) |", "|---|---|---|---|---|---|---|"]
    for cond in dict.fromkeys(r["cond"] for r in rs):
        g = [r for r in rs if r["cond"] == cond]
        poles = [r_["dropped"] for r_ in g[0]["rows"]]
        near = lambda rows, z: min(rows, key=lambda x: abs(complex(*x["dropped"]) - complex(*z)))
        for j, z in enumerate(poles):
            xs = [near(r["rows"], z) for r in g]                 # pole order differs between groups
            head = (f"| {cond} | {len(g)} | {max(r['report_ratio_cal'] for r in g):.2f} "
                    if j == 0 else "| | | ")
            lines.append(head + f"| {lab(z)} | {sum(x['ratio_chi2'] > 1 for x in xs)} / "
                         f"{sum(x['ratio_pooled'] > 1 for x in xs)} / {sum(x['ratio_cal'] > 1 for x in xs)} of {len(xs)} "
                         f"| {min(x['ratio_cal'] for x in xs):.3g} | {min(x['kappa_o'] for x in xs):.3g}–"
                         f"{max(x['kappa_o'] for x in xs):.3g} |")
    (T.R.RESULTS / "e4_power.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=[])
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
