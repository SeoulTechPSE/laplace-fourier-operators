"""(C) E5, third pass: equivalent-linear (secant) probes along the amplitude, model against data.

At an operating point x0 = A sin 5t the linearization of the Duffing system is time-varying (stiffness
k + 3 k3 x0(t)^2), so it has no poles. The comparable quantity is the equivalent linear pole: the best linear
pair (first-order-hold VarPro, one sample) for the pair (x_A, y_A).

- data: the stored response of the sample with amplitude A (training waveform A sin 5t; test waveform, a decaying
  envelope, as an out-of-sample family);
- model: the model's own response G(x_A) to the same input.

For each variant and seed of `e5_train.py` / the first pass, the equivalent pole of the model is compared with the
data's at the same A; the stored layer poles are fixed, so a model that reproduces the amplitude drift must do it
through its nonlinearity, not through its poles.

    python experiments/C/e5_probe/e5_oppoint.py          # writes results/e5_oppoint.md / .json
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

EP, RL = P.EP, P.RL
AMPS = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
STARTS = ([complex(-0.25, 0.97)], [complex(-0.3, 1.2)], [complex(-0.2, 0.8)])


def eq_pole(x, y, t, device):
    vp = EP.VarPro({"train": (torch.tensor(x[None]), torch.tensor(y[None]))}, t, device, "foh")
    best = None
    for s in STARTS:
        th, re_ = EP.pack(s)
        th, S = vp.fit(th, re_)
        if best is None or S < best[1]:
            best = (th, S, re_)
    th, S, re_ = best
    with torch.no_grad():
        pred = vp.pred(torch.tensor(th, device=device), re_)
    z = EP.unpack_theta(th, re_)[0]
    return z, float((vp.y - pred).norm() / vp.y.norm())


def pick(f, amps):
    a = np.abs(f).max(1)
    return [int(np.argmin(np.abs(a - A))) for A in amps], a


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case = RL.CASES[P.CASE]
    split, grids = RL.load(case, None, "ood")
    t = np.asarray(grids["t"], float).ravel()
    fam = {}
    for name, key in (("train", "train"), ("test", "test")):
        f, u = (a.double().reshape(a.shape[0], -1).numpy() for a in split[key])
        idx, a = pick(f, AMPS)
        fam[name] = (f[idx], u[idx], a[idx])
    res = dict(amps=list(AMPS), data={}, variants={})
    for name, (f, u, a) in fam.items():
        res["data"][name] = [dict(A=float(A), pole=[z.real, z.imag], rel=r)
                             for A, (z, r) in zip(a, (eq_pole(f[i], u[i], t, device) for i in range(len(a))))]
        print(name, "data:", ", ".join(f"A {x['A']:.2f}: {complex(*x['pole']):.3f}" for x in res["data"][name]), flush=True)
    for v in P.VARIANTS:
        rows = []
        for s in P.SEEDS:
            model, test, mu = P.load_variant(v, s, case, grids, device)
            row = dict(seed=s, test=test)
            for name, (f, u, a) in fam.items():
                with torch.no_grad():
                    g = model(torch.tensor(f, dtype=torch.float32, device=device)[..., None])[..., 0].double().cpu().numpy()
                row[name] = []
                for i in range(len(a)):
                    z, r = eq_pole(f[i], g[i], t, device)
                    fit = float(np.linalg.norm(g[i] - u[i]) / np.linalg.norm(u[i]))
                    row[name].append(dict(A=float(a[i]), pole=[z.real, z.imag], rel=r, fit=fit))
            rows.append(row)
            print(v, s, "train:", ", ".join(f"{complex(*x['pole']):.3f}" for x in row["train"]), flush=True)
        res["variants"][v] = rows
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    (P.RESULTS / "e5_oppoint.json").write_text(json.dumps(res, indent=1))
    summarize(res)


def summarize(res):
    lines = ["# E5: equivalent-linear (secant) poles along the amplitude", "",
             "Equivalent pole = best linear pair (first-order hold) for one input–response pair at amplitude A. Data: the "
             "stored response; model: its own response to the same input. Cells: Im of the equivalent pole (median "
             "over 5 seeds for the models), and in brackets the model's relative output error at that sample (median). "
             "Re of the poles in the JSON.", ""]
    for fam in ("train", "test"):
        d = res["data"][fam]
        lines += [f"## {fam} waveform", "",
                  "| source | " + " | ".join(f"A = {x['A']:.2g}" for x in d) + " |", "|---|" + "---|" * len(d),
                  "| data | " + " | ".join(f"{x['pole'][1]:.3f} (Re {x['pole'][0]:.3f}; resid {x['rel']:.2f})" for x in d) + " |"]
        for v, rows in res["variants"].items():
            im = np.median([[abs(x["pole"][1]) for x in r[fam]] for r in rows], axis=0)
            fit = np.median([[x["fit"] for x in r[fam]] for r in rows], axis=0)
            lines.append(f"| {v} | " + " | ".join(f"{a:.3f} ({b:.2f})" for a, b in zip(im, fit)) + " |")
        lines.append("")
    (P.RESULTS / "e5_oppoint.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
