"""(C) E4, noise-free undercoverage: where does the 1e-9 - 1e-8 pole bias come from?

On noise-free data the first-order-hold refit with sigma^2 = float32 rounding of the outputs
(e4_profile.py, CLEAN conditions) reports every pole, but some 95 % profile intervals miss the
truth by 1-1.7 half-widths. All groups of a condition share the same data, so the miss is one
realisation, not a repeated bias. Candidates: the float32 rounding of the inputs (not in sigma^2),
of the outputs, and the arithmetic of the generator / refit.

Per noise-free data set, the minimal FOH refit (VarPro, started at the truth) on four versions
of the training data:

  x32 y32   as stored (the E4 setting)
  x64 y32   exact inputs, rounded outputs
  x32 y64   rounded inputs, exact outputs
  x64 y64   both exact (generator + refit arithmetic only)

and a Monte-Carlo check: N_MC dithers of the exact data with independent uniform errors of
+-ulp/2 (float32) on the inputs and outputs (both / outputs only). The spread of the fitted Re mu
is compared with the profile half-width / 1.96 of e4_profile (sigma^2 from the outputs only).

Truth-free fix (parametric bootstrap): from the stored data alone, fit, take the fitted outputs
as exact, dither the stored inputs and the fitted outputs by +-ulp/2 (N_BOOT draws), refit;
the std of the refitted Re mu is the interval scale (mean +- 1.96 std). Coverage is checked
against the stored-data error. (The refits here start at the truth for the diagnosis; on the
noise-free data the reported poles equal this fit to the printed digits.)

    python experiments/C/theory/e4_bias.py              # writes results/e4_bias.{json,md}
    python experiments/C/theory/e4_bias.py --summarize  # table only, after e4_profile reruns
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e4_profile as EP  # noqa: E402

T, E = EP.T, EP.E
R = T.R
N_MC = N_BOOT = 32


def splits64(cond):
    """Training split as in e2_train.make_splits, without the float32 cast."""
    system, family, Tw, K, N = T.unpack(cond)
    R.DT = 20.0 / N
    t, u, y, _, _ = R.make(system, family, Tw, K, seed=0)
    return t.numpy(), u[:200].double(), y[:200].double()


dither = EP.dither


def fit(x, y, t, theta0, real, device):
    vp = EP.VarPro({"train": (x, y)}, t, device, "foh")
    vp.s_unit = y.numel() * EP.round_var(y.float())       # S / (n sigma^2) = O(1), as in e4_profile.run
    th, S = vp.fit(theta0, real)
    return th, S


def re_err(th, theta0, real):
    """Relative error of each Re mu (the profiled coordinate) against the truth."""
    idx = np.cumsum([0] + [1 if r else 2 for r in real])[:-1]
    return (th[idx] - theta0[idx]) / np.abs(theta0[idx])


def halfwidths(cond):
    """Relative profile half-width of each true pole (mean over groups) from e4_profile_foh."""
    out = {}
    for f in sorted((R.RESULTS / "e4_profile_foh").glob(f"{cond}_g*.json")):
        r = json.loads(f.read_text())
        for pr in r["profiles"]:
            lo, hi = pr["interval"]
            if lo is None or hi is None:
                continue
            z = complex(*pr["mu"])
            out.setdefault(round(z.real, 2), []).append(abs(hi - lo) / 2 / abs(z.real))
    return {k: float(np.mean(v)) for k, v in out.items()}


def run(cond, device):
    t, x64, y64 = splits64(cond)
    x32, y32 = (a.float().double() for a in (x64, y64))
    system = T.unpack(cond)[0]
    mu_true, beta_true = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    true = E.true_folded(mu_true, beta_true)[0]
    theta0, real = EP.pack(list(true))
    res = dict(cond=cond, poles=[[float(z.real), float(z.imag)] for z in true],
               sigma2_out=EP.round_var(y32.float()), n=int(y32.numel()))
    for name, (x, y) in dict(x32y32=(x32, y32), x64y32=(x64, y32), x32y64=(x32, y64), x64y64=(x64, y64)).items():
        th, S = fit(x, y, t, theta0, real, device)
        res[name] = dict(S=S, re_err=re_err(th, theta0, real).tolist())
    rng = np.random.default_rng(0)
    for name, both in (("mc_both", True), ("mc_out", False)):
        errs = []
        for _ in range(N_MC):
            x = dither(x64, rng) if both else x64
            th, _ = fit(x, dither(y64, rng), t, theta0, real, device)
            errs.append(re_err(th, theta0, real))
        errs = np.array(errs)
        res[name] = dict(std=errs.std(0, ddof=1).tolist(), mean=errs.mean(0).tolist())
    th_hat, _ = fit(x32, y32, t, theta0, real, device)          # the stored-data fit
    vp = EP.VarPro({"train": (x32, y32)}, t, device, "foh")
    with torch.no_grad():
        y_hat = (vp.pred(torch.tensor(th_hat, device=device), real).reshape(y32.shape) / vp.scale).cpu()
    for name, both in (("boot_both", True), ("boot_out", False)):
        errs = []
        for _ in range(N_BOOT):
            x = dither(x32, rng) if both else x32
            th, _ = fit(x, dither(y_hat, rng), t, th_hat, real, device)
            errs.append(re_err(th, th_hat, real))
        res[name] = dict(std=np.array(errs).std(0, ddof=1).tolist())
    res["halfwidth"] = halfwidths(cond)
    return res


def summarize(rs):
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    lines = ["# E4: the noise-free pole bias (float32 rounding of inputs vs outputs)", "",
             "Minimal first-order-hold refit started at the truth, training split (200 samples). "
             "Per pole: relative error of Re μ for the stored data (x32 y32), exact inputs (x64 y32), exact "
             "outputs (x32 y64) and both exact (x64 y64); Monte-Carlo std of the error over "
             f"{N_MC} uniform ±ulp/2 dithers of both / of the outputs only; profile half-width / 1.96 of "
             "`e4_profile_foh` (σ² = output rounding), the std the intervals assume; truth-free parametric "
             f"bootstrap std ({N_BOOT} dithers of the stored inputs and fitted outputs / of the fitted outputs "
             "only); z = |error of the stored data| / bootstrap std (both).", "",
             "| data | pole | x32 y32 | x64 y32 | x32 y64 | x64 y64 | MC std, both | MC std, outputs | half-width / 1.96 "
             "| boot std, both | boot std, outputs | z (profile) | z (boot) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    zp, zb = [], []
    for r in rs:
        for j, z in enumerate(r["poles"]):
            hw = r["halfwidth"].get(round(z[0], 2))
            lines.append(f"| {r['cond']} | {lab(z)} | " + " | ".join(f"{r[k]['re_err'][j]:+.1e}"
                         for k in ("x32y32", "x64y32", "x32y64", "x64y64"))
                         + f" | {r['mc_both']['std'][j]:.1e} | {r['mc_out']['std'][j]:.1e} | "
                         + (f"{hw / 1.96:.1e}" if hw else "–")
                         + f" | {r['boot_both']['std'][j]:.1e} | {r['boot_out']['std'][j]:.1e} | ")
            e = abs(r["x32y32"]["re_err"][j])
            zp.append(e / (hw / 1.96) if hw else np.nan)
            zb.append(e / r["boot_both"]["std"][j])
            lines[-1] += f"{zp[-1]:.1f} | {zb[-1]:.1f} |"
    zp, zb = np.array(zp), np.array(zb)
    lines += ["", f"Within ± 1.96 std (95 % interval): profile (σ² = output rounding) {np.sum(zp <= 1.96)}/{zp.size}, "
              f"bootstrap (inputs + outputs) {np.sum(zb <= 1.96)}/{zb.size}. Every group of a condition shares "
              "these data, so this is one realisation per data set.", ""]
    lines += ["", "S at the truth-started fit / n σ²_out: " + ", ".join(
        f"{r['cond']} {r['x32y32']['S'] / (r['n'] * r['sigma2_out']):.2f} (exact inputs "
        f"{r['x64y32']['S'] / (r['n'] * r['sigma2_out']):.2f})" for r in rs), ""]
    (R.RESULTS / "e4_bias.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    if "--summarize" in sys.argv:                  # rebuild the table (profile half-widths re-read)
        rs = json.loads((R.RESULTS / "e4_bias.json").read_text())
        for r in rs:
            r["halfwidth"] = halfwidths(r["cond"])
        return summarize(rs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = []
    for cond in EP.CLEAN:
        t0 = time.time()
        rs.append(run(cond, device))
        print(f"{cond} done ({time.time() - t0:.0f}s)", flush=True)
    (R.RESULTS / "e4_bias.json").write_text(json.dumps(rs, indent=1))
    summarize(rs)


if __name__ == "__main__":
    main()
