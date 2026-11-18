"""(A) Pole layer, third validation: two-stage identification with a resolution cut, scaled polynomials and an
L-BFGS fine-tune.

The second validation (`a2_twostage.py`) found three faults, all diagnosed from its own results:

- **Stage-1 bias at high k.** Per-mode poles are within 0.5 % for mid k, but 1-4 % off where |lambda| dt ≳ 0.4:
  jumps of steps and PRBS fall between samples, and fast modes see that. On `diff` the slowest mode (k = 1, decay
  0.1 over T = 5) is also 1-2 % off.
- **Degree selection follows the bias.** BIC adds k^4 / k^3 terms to fit the biased high-k poles. The true degree
  was chosen in only 2-3 of 5 seeds.
- **The Adam fine-tune made fits worse.** With k up to 16, one Adam step on a k^4 coefficient moves Re mu by up to
  65 lr.

Changes (everything else as in `a2_twostage.py`):

1. **Resolution cut**, without the truth: stage 2 uses only modes whose stage-1 pole has |mu_k| dt <= 0.25.
2. **Weighted stage 2.** Each kept mode is weighted by 1 / resid_k^2, its stage-1 relative residual, so the worse
   fitted modes count less.
3. **Fine-tune.** The layer uses k_ref = 16 (polynomials in k / 16), its coefficients are converted from stage 2,
   and it is trained with L-BFGS (strong Wolfe, 200 iterations, full batch, float64) instead of Adam.

Predictions (written before running):

1. On the kept modes, every stage-1 pole is within 2 % of lambda(k), in every data seed and both systems.
2. BIC picks the true degree in at least 4 of 5 data seeds per system. D and v are within 1 %, and k_r within 0.01
   absolute.
3. The fine-tuned test error is <= 1.2x the oracle floor in at least 4 of 5 seeds per system, with a dispersion
   error over k = 1..16 of <= 2 %.

    python experiments/A/layer/a3_twostage.py      # results/a3_twostage.md / .json
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
import a1_dispersion as A  # noqa: E402
import a2_twostage as B  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleLayer1d  # noqa: E402

CUT, K_REF = 0.25, 16.0
OUT = HERE / "results" / "a3_twostage.json"


def wls(ks, y, w, powers):
    V = np.stack([ks ** p for p in powers], 1)
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(V * sw[:, None], y * sw, rcond=None)
    rss = float((w * (V @ c - y) ** 2).sum() / w.mean())
    n = ks.size
    return c, n * np.log(max(rss / n, 1e-300)) + len(powers) * np.log(n)


def stage2(modes, dt):
    kept = [m for m in modes if abs(complex(*m["mu"])) * dt <= CUT]
    ks = np.array([m["k"] for m in kept], float)
    mu = np.array([complex(*m["mu"]) for m in kept])
    w = 1 / np.array([max(m["resid"], 1e-6) for m in kept]) ** 2
    re_fit = {k: wls(ks, mu.real, w, p) for k, p in {"a0,a1": [0, 2], "a0,a1,a2": [0, 2, 4]}.items()}
    im_fit = {k: wls(ks, mu.imag, w, p) for k, p in {"c1": [1], "c1,c3": [1, 3]}.items()}
    re_key, im_key = min(re_fit, key=lambda k: re_fit[k][1]), min(im_fit, key=lambda k: im_fit[k][1])
    a = list(re_fit[re_key][0]) + [0.0] * (3 - len(re_fit[re_key][0]))
    c = list(im_fit[im_key][0]) + [0.0] * (2 - len(im_fit[im_key][0]))
    b = np.array([complex(*m["b"]) for m in kept])
    bc, _ = wls(ks, b.real, w, [0, 2])
    bs, _ = wls(ks, b.imag, w, [0, 2])
    return dict(kept=[int(k) for k in ks], re_degree=re_key, im_degree=im_key, a=a, c=c,
                beta=[[bc[0], bs[0]], [bc[1], bs[1]]], D=-a[1], v=-c[0], kr=-a[0])


def finetune(data, t, s2, device):
    model = WavenumberPoleLayer1d(1, t, modes=A.MODES, re_max=A.RE_MAX, k_ref=K_REF).to(device)
    a, c, b = s2["a"], s2["c"], s2["beta"]
    with torch.no_grad():                                   # coefficients of (k / k_ref)^p: multiply by k_ref^p
        model.mu_re.copy_(torch.tensor([[a[0], a[1] * K_REF ** 2, a[2] * K_REF ** 4]], dtype=torch.float64))
        model.mu_im.copy_(torch.tensor([[c[0] * K_REF, c[1] * K_REF ** 3]], dtype=torch.float64))
        model.beta.copy_(torch.tensor([[[b[0][0], b[0][1]], [b[1][0] * K_REF ** 2, b[1][1] * K_REF ** 2]]],
                                      dtype=torch.float64))
    (xtr, ytr), (xte, yte) = ((x.to(device) for x in data[k]) for k in ("train", "test"))
    with torch.no_grad():
        test0 = float(A.rel(model(xte), yte))
    opt = torch.optim.LBFGS(model.parameters(), max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-15, history_size=50)

    def closure():
        opt.zero_grad()
        loss = A.rel(model(xtr), ytr)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        test = float(A.rel(model(xte), yte))
        mu = model.poles(torch.tensor(B.KS, dtype=torch.float64, device=device))[0][0].cpu().numpy()
        ar = model.mu_re.detach().cpu().numpy()[0]
        ci = model.mu_im.detach().cpu().numpy()[0]
    tuned = dict(D=float(-ar[1] / K_REF ** 2), v=float(-ci[0] / K_REF), kr=float(-ar[0]))
    return test0, test, mu, tuned


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    floors = {s: A.oracle(s, device) for s in A.SYSTEMS}
    out = []
    for system, p in A.SYSTEMS.items():
        for ds in range(5):
            t0 = time.time()
            data, t = A.make_data(system, seed=ds)
            dt = float(t[1] - t[0])
            modes = B.per_mode(data, dt, device)
            true = A.lam(B.KS, **p)
            err1 = {m["k"]: abs(complex(*m["mu"]) - true[i]) / abs(true[i]) for i, m in enumerate(modes)}
            s2 = stage2(modes, dt)
            test0, test, mu_ft, tuned = finetune(data, t, s2, device)
            r = dict(system=system, data_seed=ds, modes=modes, mode_err=err1,
                     kept_err_max=max(err1[k] for k in s2["kept"]), stage2=s2, test_init=test0, test=test,
                     tuned=tuned, disp_err=float(np.linalg.norm(mu_ft - true) / np.linalg.norm(true)),
                     seconds=round(time.time() - t0, 1))
            out.append(r)
            print(system, ds, f"kept {s2['kept'][0]}..{s2['kept'][-1]} err max {r['kept_err_max']:.2e}",
                  s2["re_degree"], s2["im_degree"], f"D {s2['D']:.5f} v {s2['v']:.5f} kr {s2['kr']:.5f}",
                  f"test {test0:.2e} -> {test:.2e}", f"disp {r['disp_err']:.2e}", tuned, flush=True)
    OUT.write_text(json.dumps(dict(floors=floors, runs=out), indent=1, default=float))
    summarize(floors, out)


def summarize(floors, out):
    lines = ["# (A) Two-stage identification with a resolution cut, scaled polynomials and an L-BFGS fine-tune", "",
             f"Stage 2 uses modes with |μ_k| Δt ≤ {CUT} (from stage 1, no truth), weights 1 / residual². Fine-tune: "
             f"k_ref = {K_REF:g}, L-BFGS 200 iterations. Data seeds 0–4.", "",
             "| system | seed | kept k | stage 1 max error (kept) | degree (Re; Im) | D | v | k_r | test: init → tuned (oracle) | "
             "dispersion error | tuned D, v, k_r |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        s2, tu = r["stage2"], r["tuned"]
        lines.append(f"| {r['system']} | {r['data_seed']} | {s2['kept'][0]}–{s2['kept'][-1]} | {r['kept_err_max']:.1e} | "
                     f"{s2['re_degree']}; {s2['im_degree']} | {s2['D']:.5f} | {s2['v']:.5f} | {s2['kr']:.5f} | "
                     f"{r['test_init']:.2e} → {r['test']:.2e} ({floors[r['system']]:.2e}) | {r['disp_err']:.1e} | "
                     f"{tu['D']:.5f}, {tu['v']:.5f}, {tu['kr']:.5f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
