"""(A) Plan-review follow-up to a43: how much of the WPO's cost is precision, and does float32 keep its results?

a43 measured the WPO's training step at 4.7x (F) and 4.4x (P) the TFNO's, but the WPO ran in float64 and the TFNO
in float32, so method and precision were confounded. `lfo.models` now has a float32 path for the 1D deep WPO
(`WavenumberPoleOperator1d(dtype=torch.float32)`): activations, pointwise layers and the pole layers' channel-sized
work (spatial FFT, sums over channel pairs, temporal convolution) in float32 / complex64. The pole and residue
functions, the FOH coefficients and the kernels' FFT (N x K x n_t) stay complex128 and are cast once, and the pole
parameters stay float64, so the real parameter count is unchanged. The default float64 path is bit-identical to
before (checked on both setups before this commit).

`wpo32` is the WPO with that path, everything else unchanged: same constructor and budget (F 59,529, P 61,130
real), same selected configuration (lr 3e-3, pole lr x30, 12000 steps), same protocols as a42.

**Part 1, cost** (a43's procedure: alone on the GPU, RTX 4090, batch 25, median of 5 x 20 steps after 10 warm-up
steps, inference batch 50): F and P, `wpo` (float64), `wpo32`, `tfno_r`, all re-measured in the same session.

**Part 2, accuracy**: `wpo32` with seeds 0-4 against a42's five-seed float64 WPO.
- F, forced Burgers (a36 / a42 evaluation): test, (5, 10] extrapolation, probe outside the band, unseen modes.
- P, PDEBench 1D Burgers (a37 / a42 metrics): nRMSE, 1024-point zero-shot, probe nu'.

**Predictions** (written before running).

1. In both setups `wpo32`'s training step takes <= 0.5x the float64 WPO's.
2. In both setups `wpo32`'s training step is still >= 1.3x the TFNO's (a method cost remains: the FOH convolution
   runs over all n_t samples per mode and channel pair).
3. In both setups `wpo32`'s peak training memory is <= 0.7x the float64 WPO's.
4. F: `wpo32` keeps the form claim: its 95 % interval of the unseen-mode ratio lies below a42's `wpo_index` and
   `tfno_r` intervals (no overlap), and its mean test error is within 5 % of the float64 WPO's (0.0239).
5. P: `wpo32`'s mean nRMSE is within 5 % of the float64 WPO's (0.0450).

    python experiments/A/deep/a44_float32.py cost            # alone on the GPU, first
    python experiments/A/deep/a44_float32.py F|P [seed ...]   # training runs (parallel processes)
    python experiments/A/deep/a44_float32.py --summary        # results/a44_float32.md
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
import a42_five_seeds as X42  # noqa: E402
import a43_cost as X43  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

OUT = HERE / "results" / "a44_float32.json"
CFG = dict(lr=3e-3, plr=30.0)
SEEDS = (0, 1, 2, 3, 4)


class WPO32P(X37.WPO):
    """a37's WPO (initial field -> trajectory) with the float32 path."""

    def __init__(self, t):
        torch.nn.Module.__init__(self)
        self.net = WavenumberPoleOperator1d(t, ic=True, dtype=torch.float32, **X37.WPO_KW)


def wpo32_f(seed, t, device):
    torch.manual_seed(seed)
    return WavenumberPoleOperator1d(t, dtype=torch.float32, **F.WPO_KW).to(device)


def wpo32_p(seed, t, device):
    torch.manual_seed(seed)
    return WPO32P(t).to(device)


def locked_update(fn):
    OUT.parent.mkdir(exist_ok=True)
    with open(f"/tmp/{OUT.stem}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(OUT.read_text()) if OUT.exists() else dict(cost=[], runs=[])
        fn(out)
        OUT.write_text(json.dumps(out, indent=1, default=float))
        return out


def cost(device):
    rows = []
    for part, setup in (("F", X43.setup_f), ("P", X43.setup_p)):
        models, batch, call = setup(device)
        if part == "F":
            t = np.linspace(0, M.T1, M.NT1)
            models = {"wpo": models["wpo"], "wpo32": lambda: wpo32_f(0, t, device), "tfno_r": models["tfno_r"]}
        else:
            t = X37.load()["t"].numpy()
            models = {"wpo": models["wpo"], "wpo32": lambda: wpo32_p(0, t, device), "tfno_r": models["tfno_r"]}
        for name, make in models.items():
            torch.manual_seed(0)
            m = make()
            tr, inf, peak = X43.measure(m, batch, call)
            r = dict(part=part, model=name, real_params=n_real_params(m), dtype=str(next(m.parameters()).dtype)[6:],
                     train_ms=tr, infer_ms_per_sample=inf, peak_train_mb=peak)
            rows.append(r)
            print(r, flush=True)
            del m
            torch.cuda.empty_cache()
    locked_update(lambda o: o.update(cost=rows, gpu=torch.cuda.get_device_name(), torch=torch.__version__))


def done(part, seed):
    return any(r["part"] == part and r["seed"] == seed for r in locked_update(lambda o: None)["runs"])


def run_f(seeds, device):
    data = M.make_data()
    data["high"] = X36.high_set()
    t = np.linspace(0, M.T1, M.NT1)
    pfk, pdt = M.probe_inputs()
    for seed in seeds:
        if done("F", seed):
            continue
        t1 = time.time()
        m = wpo32_f(seed, t, device)
        F.train(m, data, device, steps=F.FINAL_STEPS, **CFG)
        r = X36.evaluate(m, "wpo32", seed, CFG, data, t, pfk, pdt, device, t1)
        r["part"] = "F"
        print("F", seed, f"test {r['test']:.4f} high/test {r['high'] / r['test']:.2f} oob {r['oob']:.3f}",
              f"{r['seconds']} s", flush=True)
        locked_update(lambda o: o["runs"].append(r))
        del m
        torch.cuda.empty_cache()


def run_p(seeds, device):
    d = X37.load()
    t = d["t"].numpy()
    S = X37.splits(d)
    n_t = S["train"].shape[2]
    full_test = d["full"][X37.NTR:].to(torch.float64)
    for seed in seeds:
        if done("P", seed):
            continue
        t1 = time.time()
        m = wpo32_p(seed, t, device)
        X37.train(m, S["train"], device, steps=X37.FINAL_STEPS, **CFG)
        m.eval()
        r = dict(part="P", model="wpo32", seed=seed, cfg=CFG, real_params=n_real_params(m),
                 test=X37.nrmse(X37.predict(m, S["test"], device), S["test"]),
                 test_1024=X37.nrmse(X37.predict(m, full_test, device), full_test),
                 probe=X37.model_probe(m, t, 256, n_t, device), seconds=round(time.time() - t1, 1))
        print("P", seed, f"test {r['test']:.4f} nu' {r['probe']['nu_hat']:.4f}", f"{r['seconds']} s", flush=True)
        locked_update(lambda o: o["runs"].append(r))
        del m
        torch.cuda.empty_cache()


def summarize():
    out = json.loads(OUT.read_text())
    F5, P5 = X42.collect()
    fmt = X42.fmt
    lines = ["# (A) The WPO in float32: cost and accuracy", ""]
    if out.get("cost"):
        lines += [f"## Cost, measured alone on the GPU ({out['gpu']}, torch {out['torch']})", "",
                  "| setup | model | real params | dtype | train step (ms) | ratio to tfno_r | ratio to wpo (float64) | "
                  "inference (ms / sample) | peak train memory (MB) |", "|---|---|---|---|---|---|---|---|---|"]
        for part in ("F", "P"):
            rs = {r["model"]: r for r in out["cost"] if r["part"] == part}
            for r in rs.values():
                lines.append(f"| {part} | {r['model']} | {r['real_params']} | {r['dtype']} | {r['train_ms']:.1f} | "
                             f"{r['train_ms'] / rs['tfno_r']['train_ms']:.2f} | {r['train_ms'] / rs['wpo']['train_ms']:.2f} | "
                             f"{r['infer_ms_per_sample']:.3f} | {r['peak_train_mb']:.0f} |")
        lines.append("")
    fr = sorted([r for r in out["runs"] if r["part"] == "F"], key=lambda r: r["seed"])
    pr = sorted([r for r in out["runs"] if r["part"] == "P"], key=lambda r: r["seed"])
    lines += ["## F: forced Burgers (a42 setup)", "",
              "Mean ± standard deviation [95 % t-interval] over seeds 0–4; float64 rows from a42.", "",
              "| model | n | test | (5, 10] / test | probe outside the band | unseen / test |", "|---|---|---|---|---|---|"]
    for k, rs in list(F5.items()) + [("wpo32", fr)]:
        if rs:
            lines.append(f"| {k} | {len(rs)} | {fmt([r['test'] for r in rs], '.4f')} | "
                         f"{fmt([r['ext_late'] / r['test'] for r in rs], '.2f')} | {fmt([r['oob'] for r in rs], '.3f')} | "
                         f"{fmt([r['high'] / r['test'] for r in rs], '.2f')} |")
    lines += ["", "## P: PDEBench 1D Burgers (a42 setup)", "",
              "| model | n | nRMSE 256 | nRMSE 1024 / 256 | probe ν' (truth 0.1257) | dispersion error |",
              "|---|---|---|---|---|---|"]
    for k, rs in list(P5.items()) + [("wpo32", pr)]:
        if rs:
            lines.append(f"| {k} | {len(rs)} | {fmt([r['test'] for r in rs], '.4f')} | "
                         f"{fmt([r['test_1024'] / r['test'] for r in rs], '.2f')} | "
                         f"{fmt([r['probe']['nu_hat'] for r in rs], '.4f')} | {fmt([r['probe']['disp_err'] for r in rs], '.3f')} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = sys.argv[1:]
    device = torch.device("cuda")
    seeds = [int(a) for a in args if a.isdigit()] or list(SEEDS)
    if "cost" in args:
        cost(device)
    if "F" in args:
        run_f(seeds, device)
    if "P" in args:
        run_p(seeds, device)
    summarize()


if __name__ == "__main__":
    main()
