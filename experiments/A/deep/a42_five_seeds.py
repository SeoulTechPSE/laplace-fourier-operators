"""(A) Plan-review item: five seeds with intervals for the headline (A) claims.

The headline comparisons of the A paper were run with 3 seeds and reported as mean (range). This pass adds seeds
3 and 4, unchanged otherwise, to the two comparisons the abstract leads with, and reports 95 % t-intervals over 5
seeds (t_{0.975, 4} = 2.776):

**F, the form claim (a36, forced Burgers).** `wpo` (a11 configuration), `wpo_index` (free pole per mode index) and
`tfno_r` (TFNO at the WPO's real budget), each with its a36-selected configuration (wpo / wpo_index lr 3e-3,
pole lr x30; tfno_r lr 3e-3), 12000 steps, a36's evaluation. Seeds 0-2 are a36's runs (wpo: a36's re-evaluation of
the a11 checkpoints).

**P, the public benchmark (a37, PDEBench 1D Burgers).** `wpo` (a37's post hoc rerun configuration, after the
`foh_kernel` fix: lr 3e-3, pole lr x30) and `tfno_r` (lr 1e-2), 12000 steps, a37's metrics. Seeds 0-2 are a37's
rerun / final runs.

Training batches are drawn from a fixed generator in both protocols, so the seed sets the initialization only.

**Predictions** (written before running).

1. F: the unseen-mode ratio separates the forms: the WPO's 95 % interval lies below both `wpo_index`'s and
   `tfno_r`'s (no overlap).
2. F: the probe error outside the band separates the same way (no overlap with either).
3. F: in band the forms do not differ: the WPO's and `wpo_index`'s test-error intervals overlap.
4. P: the TFNO is more accurate: its nRMSE interval lies below the WPO's, and the ratio of means is in [1.6, 2.0].
5. P: the WPO's probe ν' is within 5 % of 0.1257 in at most 2 of 5 seeds (1 of 3 so far).

    python experiments/A/deep/a42_five_seeds.py [F:wpo F:wpo_index F:tfno_r P:wpo P:tfno_r]   # results/a42_five_seeds.md
"""
from __future__ import annotations

import fcntl
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
import a36_index_poles as X36  # noqa: E402
import a37_pdebench_burgers as X37  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402

OUT = HERE / "results" / "a42_five_seeds.json"
NEW_SEEDS = (3, 4)
T975_4 = 2.776
CFG_F = {"wpo": dict(lr=3e-3, plr=30.0), "wpo_index": dict(lr=3e-3, plr=30.0), "tfno_r": dict(lr=3e-3)}
CFG_P = {"wpo": dict(lr=3e-3, plr=30.0), "tfno_r": dict(lr=1e-2)}
JOBS = [f"F:{k}" for k in CFG_F] + [f"P:{k}" for k in CFG_P]


def locked_update(fn):
    OUT.parent.mkdir(exist_ok=True)
    with open(f"/tmp/{OUT.stem}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
        fn(out)
        OUT.write_text(json.dumps(out, indent=1, default=float))
        return out


def done(part, kind, seed):
    return any(r["part"] == part and r["model"] == kind and r["seed"] == seed
               for r in locked_update(lambda o: None)["runs"])


def run_f(kind, device):
    data = M.make_data()
    data["high"] = X36.high_set()
    t = np.linspace(0, M.T1, M.NT1)
    pfk, pdt = M.probe_inputs()
    for seed in NEW_SEEDS:
        if done("F", kind, seed):
            continue
        t1 = time.time()
        m = X36.build(kind, seed, t, device)
        F.train(m, data, device, steps=F.FINAL_STEPS, **CFG_F[kind])
        r = X36.evaluate(m, kind, seed, CFG_F[kind], data, t, pfk, pdt, device, t1)
        r["part"] = "F"
        print("F", kind, seed, f"test {r['test']:.4f} high/test {r['high'] / r['test']:.2f} oob {r['oob']:.3f}",
              f"{r['seconds']} s", flush=True)
        locked_update(lambda o: o["runs"].append(r))
        del m
        torch.cuda.empty_cache()


def run_p(kind, device):
    d = X37.load()
    t = d["t"].numpy()
    S = X37.splits(d)
    n_t = S["train"].shape[2]
    full_test = d["full"][X37.NTR:].to(torch.float64)
    rank = X37.tfno_rank(n_real_params(X37.WPO(t)))
    for seed in NEW_SEEDS:
        if done("P", kind, seed):
            continue
        t1 = time.time()
        m = X37.build(kind, seed, t, device, rank)
        X37.train(m, S["train"], device, steps=X37.FINAL_STEPS, **CFG_P[kind])
        m.eval()
        r = dict(part="P", model=kind, seed=seed, cfg=CFG_P[kind], real_params=n_real_params(m),
                 test=X37.nrmse(X37.predict(m, S["test"], device), S["test"]),
                 test_1024=X37.nrmse(X37.predict(m, full_test, device), full_test),
                 probe=X37.model_probe(m, t, 256, n_t, device), seconds=round(time.time() - t1, 1))
        print("P", kind, seed, f"test {r['test']:.4f} nu' {r['probe']['nu_hat']:.4f}", f"{r['seconds']} s", flush=True)
        locked_update(lambda o: o["runs"].append(r))
        del m
        torch.cuda.empty_cache()


def collect():
    out = json.loads(OUT.read_text())
    a36 = json.loads((HERE / "results" / "a36_index_poles.json").read_text())
    a37 = json.loads((HERE / "results" / "a37_pdebench_burgers.json").read_text())
    new = lambda p, k: [r for r in out["runs"] if r["part"] == p and r["model"] == k]
    F5 = {"wpo": a36["ref"] + new("F", "wpo")}
    F5.update({k: [r for r in a36["final"] if r["model"] == k] + new("F", k) for k in ("wpo_index", "tfno_r")})
    P5 = {"wpo": a37["rerun"] + new("P", "wpo"),
          "tfno_r": [r for r in a37["final"] if r["model"] == "tfno_r"] + new("P", "tfno_r")}
    return F5, P5


def ci(xs):
    xs = np.asarray(xs, float)
    m, s = xs.mean(), xs.std(ddof=1)
    h = T975_4 * s / np.sqrt(len(xs)) if len(xs) == 5 else float("nan")
    return m, s, m - h, m + h


def fmt(xs, f):
    m, s, lo, hi = ci(xs)
    return f"{m:{f}} ± {s:{f}} [{lo:{f}}, {hi:{f}}]"


def summarize():
    F5, P5 = collect()
    lines = ["# (A) Five seeds with 95 % t-intervals for the headline claims", "",
             "Mean ± standard deviation [95 % t-interval] over seeds 0–4 (seeds 0–2 from a36 / a37).", "",
             "## F: the form claim, forced Burgers (a36 setup)", "",
             "| model | n | test | (5, 10] / test | probe outside the band | unseen / test |", "|---|---|---|---|---|---|"]
    for k, rs in F5.items():
        lines.append(f"| {k} | {len(rs)} | {fmt([r['test'] for r in rs], '.4f')} | "
                     f"{fmt([r['ext_late'] / r['test'] for r in rs], '.2f')} | {fmt([r['oob'] for r in rs], '.3f')} | "
                     f"{fmt([r['high'] / r['test'] for r in rs], '.2f')} |")
    lines += ["", "## P: PDEBench 1D Burgers (a37 setup)", "",
              "| model | n | nRMSE 256 | probe ν' (truth 0.1257) | seeds with ν' within 5 % | dispersion error |",
              "|---|---|---|---|---|---|"]
    for k, rs in P5.items():
        nu = [r["probe"]["nu_hat"] for r in rs]
        within = sum(abs(v / X37.NU_P - 1) <= 0.05 for v in nu)
        lines.append(f"| {k} | {len(rs)} | {fmt([r['test'] for r in rs], '.4f')} | {fmt(nu, '.4f')} | {within} of {len(rs)} | "
                     f"{fmt([r['probe']['disp_err'] for r in rs], '.3f')} |")
    if all(len(rs) == 5 for rs in P5.values()):
        ratio = np.mean([r["test"] for r in P5["wpo"]]) / np.mean([r["test"] for r in P5["tfno_r"]])
        lines += ["", f"Ratio of mean nRMSE, WPO / TFNO: {ratio:.2f}."]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    jobs = [a for a in sys.argv[1:] if a in JOBS]
    if "--summary" not in sys.argv:
        device = torch.device("cuda")
        for j in jobs or JOBS:
            part, kind = j.split(":")
            (run_f if part == "F" else run_p)(kind, device)
    summarize()


if __name__ == "__main__":
    main()
