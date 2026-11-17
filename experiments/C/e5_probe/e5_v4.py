"""(C) E5, tenth pass: rule v4 of the RQ4 trust rule (pre-registered in `results/e5_v4_prereg.md`).

Rule v3 = v1 (valid Jacobian probe, d < 0.1, test rel. L2 < 0.5) and the v3 data gate (`e5_gate.py`). In the ninth pass
(Lorenz rho = 5) it trusted one wrong pole: an original-init LNO whose Jacobian and an unmoved stored pole agreed on a
slow pole that is not the system's. Rule v4 changes the model-side checks:

- **probe agreement** a = |mu_LTI - mu_J| / |mu_J| < 0.1, with mu_LTI the one-pair pole of the odd-part LTI probe at
  eps = 0.01 and mu_J the Jacobian impulse-response pole (two probes of the same linearisation must agree);
- **centred fit** e_c < 0.5: test error relative to the test outputs minus the training-mean trajectory, per sample,
  averaged (the plain relative L2 is dominated by a large output mean on Lorenz);
- valid Jacobian, d < 0.1 and the v3 data gate unchanged.

Reads `results/e5_probe_<set>.json` and `results/e5_gate.json`; loads each model once for e_c.

    python experiments/C/e5_probe/e5_v4.py [set ...]      # writes results/e5_v4.md / .json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e5_gate as G  # noqa: E402
import e5_probe as P  # noqa: E402
import e5_train as TR  # noqa: E402

RL = P.RL
AGREE, FIT_C, CORRECT = 0.1, 0.5, 0.1
OUT = P.RESULTS / "e5_v4.json"


def fold(z):
    return z if z.imag >= 0 else z.conjugate()


@torch.no_grad()
def centred_error(model, data, device):
    x, y = data["test"]
    m = data["train"][1].mean(0)
    pred = model(x.to(device)[..., None])[..., 0].cpu()
    return float(((pred - y).norm(dim=1) / (y - m).norm(dim=1)).mean())


def evaluate(tag, device):
    pr = json.loads((P.RESULTS / f"e5_probe_{tag}.json").read_text())
    gate = json.loads((P.RESULTS / "e5_gate.json").read_text())[tag]
    ref = complex(*pr["mu_lin"])
    data, grids = TR.data_and_grid(tag)
    case = RL.CASES[TR.case_of(tag)]
    rows = []
    for v, rs in pr["variants"].items():
        for r in rs:
            j = r["jac"]
            mj = complex(*j["h_poles"][0]) if j["h_poles"] else None
            ml = fold(complex(*r["probes"][str(pr["eps"][0])]["pair"]["poles"][0]))
            model, _, _ = P.load_variant(v, r["seed"], case, grids, device, tag)
            ec = centred_error(model, data, device)
            a = abs(ml - mj) / abs(mj) if mj is not None else None
            v1 = P.trusted(r)
            v3 = v1 and gate["v3"]
            v4 = bool(j["valid"] and j["d_probe_layer"] is not None and j["d_probe_layer"] < P.TRUST_D
                      and a is not None and a < AGREE and ec < FIT_C and gate["v3"])
            err = abs(mj - ref) / abs(ref) if mj is not None else None
            rows.append(dict(variant=v, seed=r["seed"], test=r["test"], test_c=ec, agree=a, err=err, v1=v1, v3=v3,
                             v4=v4, valid=j["valid"], d=j["d_probe_layer"]))
    return dict(gate_v3=gate["v3"], rows=rows)


def counts(rows, key):
    t = [r for r in rows if r[key]]
    ok = [r for r in rows if r["v1"] and r["err"] < CORRECT]           # correct runs v1 would trust
    return dict(trusted=len(t), false=sum(r["err"] >= CORRECT for r in t),
                missed=sum(1 for r in ok if not r[key]))


def summarize(res):
    lines = ["# E5: rule v4 (probe agreement, centred fit) against v3", "",
             f"v4 = valid Jacobian ∧ d < {P.TRUST_D:g} ∧ probe agreement |μ_LTI − μ_J| / |μ_J| < {AGREE:g} ∧ centred "
             f"test error < {FIT_C:g} ∧ v3 data gate. v3 = v1 (valid ∧ d < {P.TRUST_D:g} ∧ test < {P.TRUST_TEST:g}) ∧ v3 "
             f"gate. Correct: J pole within {CORRECT:.0%} of the reference. Missed: correct runs that v1 trusts but the "
             "rule does not.", "",
             "| set | v3 gate | v3: trusted / false / missed | v4: trusted / false / missed | v1-trusted runs: agreement a, "
             "centred error e_c, pole error |", "|---|---|---|---|---|"]
    tot = {k: dict(trusted=0, false=0, missed=0) for k in ("v3", "v4")}
    for tag, g in res.items():
        c3, c4 = counts(g["rows"], "v3"), counts(g["rows"], "v4")
        for k, c in (("v3", c3), ("v4", c4)):
            for q in c:
                tot[k][q] += c[q]
        tr = [r for r in g["rows"] if r["v1"]]
        det = "; ".join(f"{r['variant']} s{r['seed']}: {r['agree']:.2f}, {r['test_c']:.2f}, {r['err']:.2f}" for r in tr)
        lines.append(f"| `{tag}` | {'pass' if g['gate_v3'] else 'fail'} | {c3['trusted']} / {c3['false']} / "
                     f"{c3['missed']} | {c4['trusted']} / {c4['false']} / {c4['missed']} | {det or '–'} |")
    lines += ["", "Totals: " + "; ".join(f"{k} {v['trusted']} trusted, {v['false']} false, {v['missed']} missed"
                                         for k, v in tot.items()) + "."]
    (P.RESULTS / "e5_v4.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(tags):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for tag in tags:
        res[tag] = evaluate(tag, device)
        print(tag, "done", flush=True)
    res = {t: res[t] for t in G.SETS if t in res}
    OUT.write_text(json.dumps(res, indent=1))
    summarize(res)


if __name__ == "__main__":
    main(sys.argv[1:] or G.SETS)
