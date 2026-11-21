"""(A) Deep operator: data efficiency on forced Burgers, WPO against the matched-budget TFNO.

a11 / a13 (200 training samples) left both models with a generalization gap (train 0.009-0.011, test 0.024-0.031).
Here the training-set size varies; the protocol is fixed (a11 WPO configuration: stable, lr 3e-3, pole ratio 30;
a13 TFNO: lr 3e-3; 12000 steps, batch 25, seeds 0-2; no new sweep).

**Training sets (nested).** N = 50 and 100 are the first samples of the a9 training set; N = 200 is the a9 training
set (results reused from a11 / a13, identical protocol); N = 400 adds 200 new samples (same distribution, new random
stream). Validation and test sets are a9's.

**Predictions** (written before running).

1. The WPO's mean test error is below the TFNO's at every N, and the ratio TFNO / WPO is largest at N = 50.
2. The WPO at N = 100 is at or below the TFNO at N = 200 (0.031).
3. Both models improve monotonically with N.

    python experiments/A/deep/a14_data_eff.py      # results/a14_data_eff.md / .json
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
import a9_burgers as M  # noqa: E402
import a11_fair as F  # noqa: E402
import a13_baselines as B13  # noqa: E402

SIZES = (50, 100, 400)
CFG = {"wpo": dict(lr=3e-3, plr=30.0), "tfno": dict(lr=3e-3)}
OUT = HERE / "results" / "a14_data_eff.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = M.make_data()
    extra = M.fields(*M.forcing(np.random.default_rng(4), 200, M.T1, M.NT1))
    pool = (torch.cat([data["train"][0], extra[0]]), torch.cat([data["train"][1], extra[1]]))
    t = np.linspace(0, M.T1, M.NT1)
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
    for n in SIZES:
        d = dict(data, train=(pool[0][:n], pool[1][:n]))
        for kind in CFG:
            for seed in M.SEEDS:
                if any(r["model"] == kind and r["n"] == n and r["seed"] == seed for r in out["runs"]):
                    continue
                t1 = time.time()
                m = F.build("wpo", seed, t, device) if kind == "wpo" else B13.build("tfno", seed, device)
                F.train(m, d, device, steps=F.FINAL_STEPS, **CFG[kind])
                m.eval()
                r = dict(model=kind, n=n, seed=seed, train=F.err(m, d, "train", device), test=F.err(m, d, "test", device),
                         seconds=round(time.time() - t1, 1))
                out["runs"].append(r)
                print(r, flush=True)
                OUT.write_text(json.dumps(out, indent=1))
                del m
                torch.cuda.empty_cache()
    prev = [dict(model=r["model"], n=200, seed=r["seed"], train=r["train"], test=r["test"])
            for f, k in ((F.OUT, "wpo"), (B13.OUT, "tfno")) for r in json.loads(f.read_text())["final"] if r["model"] == k]
    out["n200"] = prev
    OUT.write_text(json.dumps(out, indent=1))
    rows = out["runs"] + prev
    lines = ["# (A) Data efficiency on forced Burgers: WPO against matched-budget TFNO", "",
             "Nested training sets; 12000 steps; seeds 0–2; N = 200 from a11 / a13. Mean (min–max) over seeds.", "",
             "| N | WPO train | WPO test | TFNO train | TFNO test | TFNO / WPO (test) |", "|---|---|---|---|---|---|"]
    for n in sorted({r["n"] for r in rows}):
        g = lambda k, q: [r[q] for r in rows if r["model"] == k and r["n"] == n]
        f = lambda v: f"{np.mean(v):.4f} ({min(v):.4f}–{max(v):.4f})"
        lines.append(f"| {n} | {f(g('wpo', 'train'))} | {f(g('wpo', 'test'))} | {f(g('tfno', 'train'))} | {f(g('tfno', 'test'))} | "
                     f"{np.mean(g('tfno', 'test')) / np.mean(g('wpo', 'test')):.2f} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
