"""(A) Plan-review item: computational cost across models, at the budgets of the headline comparisons.

No (A) result so far reports cost. The wall times stored with the runs are not comparable: runs shared the GPU
with up to five other processes. This pass measures each model alone on the GPU (RTX 4090) under identical
conditions:

- training-step time: forward, relative-L2 loss, backward and Adam step at batch 25;
- inference time per sample: batch 50, no gradient;
- peak GPU memory of a training step.

Warm-up of 10 steps, then the median of 5 repeats of 20 steps. The precision is that of each model's code: the WPO
and LNO run in float64 (the FOH recursion and the pole fits), the neuraloperator FNOs in float32. A float32 WPO is
not part of the package and is not measured.

Setups and models (the same constructors and budgets as the runs):
- **F, forced Burgers** (a36: 64 × 201, forcing → trajectory): `wpo`, `wpo_index`, `tfno_r`, `lno`.
- **P, PDEBench 1D Burgers** (a37: 256 × 41, initial field → trajectory): `wpo`, `tfno_r`, plus a40's `wpo_mlp`
  and `fno_x`.

**Predictions** (written before running).

1. P: the WPO's training step takes 1.3–3× the TFNO's.
2. F: the WPO's training step takes ≥ 3× the TFNO's. Its FOH recursion runs over 201 time samples per mode;
   the TFNO keeps 21 temporal modes in float32.
3. In both setups the WPO's peak training memory is at least the TFNO's.

    python experiments/A/deep/a43_cost.py      # results/a43_cost.md / .json
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
import a36_index_poles as X36  # noqa: E402
import a37_pdebench_burgers as X37  # noqa: E402
import a40_gap_source as X40  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402

OUT = HERE / "results" / "a43_cost.json"
BATCH, INF_BATCH, WARM, REPS, STEPS = 25, 50, 10, 5, 20


def sync():
    torch.cuda.synchronize()


def measure(model, make_batch, call):
    """(train ms / step, inference ms / sample, peak train MB)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    def step():
        x, y = make_batch(BATCH)
        opt.zero_grad()
        p = call(model, x, y)
        loss = ((p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean()
        loss.backward()
        opt.step()

    model.train()
    for _ in range(WARM):
        step()
    sync()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(REPS):
        sync()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step()
        sync()
        times.append((time.perf_counter() - t0) / STEPS * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    model.eval()
    x, y = make_batch(INF_BATCH)
    inf = []
    with torch.no_grad():
        for _ in range(WARM):
            call(model, x, y)
        for _ in range(REPS):
            sync()
            t0 = time.perf_counter()
            for _ in range(STEPS):
                call(model, x, y)
            sync()
            inf.append((time.perf_counter() - t0) / STEPS / INF_BATCH * 1e3)
    return float(np.median(times)), float(np.median(inf)), float(peak)


def setup_f(device):
    data = M.make_data()
    xtr, ytr = data["train"]
    t = np.linspace(0, M.T1, M.NT1)
    g = torch.Generator().manual_seed(0)

    def batch(n):
        i = torch.randint(0, xtr.shape[0], (n,), generator=g)
        return xtr[i].to(device), ytr[i].to(device)

    models = {k: (lambda k=k: X36.build(k, 0, t, device)) for k in ("wpo", "wpo_index", "tfno_r", "lno")}
    return models, batch, lambda m, x, y: m(x)


def setup_p(device):
    d = X37.load()
    t = d["t"].numpy()
    u = X37.splits(d)["train"]
    g = torch.Generator().manual_seed(0)

    def batch(n):
        y = u[torch.randint(0, u.shape[0], (n,), generator=g)].to(device)
        return y[:, :, 0], y

    target = n_real_params(X37.WPO(t))
    rank = X37.tfno_rank(target)
    models = {"wpo": lambda: X37.build("wpo", 0, t, device),
              "tfno_r": lambda: X37.build("tfno_r", 0, t, device, rank)}
    for v in ("wpo_mlp", "fno_x"):
        make, _ = X40.builder(v, t, target)
        models[v] = lambda make=make: make().to(device)
    return models, batch, lambda m, x, y: m(x, y.shape[2])


def main():
    device = torch.device("cuda")
    out = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, runs=[])
    for part, setup in (("F", setup_f), ("P", setup_p)):
        models, batch, call = setup(device)
        for name, make in models.items():
            torch.manual_seed(0)
            m = make()
            dtype = str(next(m.parameters()).dtype).replace("torch.", "")
            tr, inf, peak = measure(m, batch, call)
            r = dict(part=part, model=name, real_params=n_real_params(m), dtype=dtype,
                     train_ms=tr, infer_ms_per_sample=inf, peak_train_mb=peak)
            out["runs"].append(r)
            print(r, flush=True)
            del m
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps(out, indent=1))
    lines = ["# (A) Cost per model, measured alone on the GPU", "",
             f"{out['gpu']}, torch {out['torch']}. Median of {REPS} × {STEPS} steps after {WARM} warm-up steps; "
             f"training batch {BATCH}, inference batch {INF_BATCH}.", "",
             "| setup | model | real params | dtype | train step (ms) | ratio to tfno_r | inference (ms / sample) | peak train memory (MB) |",
             "|---|---|---|---|---|---|---|---|"]
    for part in ("F", "P"):
        rs = [r for r in out["runs"] if r["part"] == part]
        ref = next(r for r in rs if r["model"] == "tfno_r")["train_ms"]
        lines += [f"| {part} | {r['model']} | {r['real_params']} | {r['dtype']} | {r['train_ms']:.1f} | "
                  f"{r['train_ms'] / ref:.2f} | {r['infer_ms_per_sample']:.3f} | {r['peak_train_mb']:.0f} |" for r in rs]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
