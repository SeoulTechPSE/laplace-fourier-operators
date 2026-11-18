"""(A) Deep operator, third pass: a fair WPO-FNO comparison on forced Burgers (tuned learning rates, longer training).

a10 (`a10_polelr.py`) reached a test error of 0.035 with the WPO (stable, pole learning rate x30), against 0.105 for
the a9 FNO. The comparison was not fair: only the WPO's learning rate was tuned, and both models were short of
convergence (a9: train ~ test error). Here both get the same tuning protocol and a longer budget.

**Protocol** (a9 data, models and probe; WPO with `stable=True`, the a10 default for dissipative problems):

1. **Sweep** (data seed fixed, model seed 0, 4000 steps): selection by the validation error.
   - FNO: lr in {1e-3, 3e-3, 1e-2};
   - WPO: lr in {1e-3, 3e-3} x pole ratio in {10, 30}.
2. **Final:** the selected configuration per model, 12000 steps (3x a9), cosine decay, batch 25, model seeds 0-2.
   Reported: test error, time extrapolation (T = 10, over (5, 10]), the linear probe, and the train error.

**Predictions** (written before running).

1. The WPO's mean test error is below the FNO's.
2. Extrapolation: the WPO's (5, 10] error is <= 2.5x its own test error in every seed; the FNO's is > 2.5x in every
   seed.
3. The probe bias of a10 persists with longer training: the WPO's nu_hat lies in [0.105, 0.13] in every seed.
4. Both models still have a train error within 0.8x of their test error (the gap is from the data, not
   overfitting).

    python experiments/A/deep/a11_fair.py      # results/a11_fair.md / .json; checkpoints in runs/A/a11
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
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

SWEEP_STEPS, FINAL_STEPS = 4000, 12000
SWEEP = {"fno": [dict(lr=lr) for lr in (1e-3, 3e-3, 1e-2)],
         "wpo": [dict(lr=lr, plr=p) for lr in (1e-3, 3e-3) for p in (10.0, 30.0)]}
WPO_KW = dict(M.WPO_KW, stable=True)
OUT = HERE / "results" / "a11_fair.json"
CKPT = M.ROOT / "runs" / "A" / "a11"


def build(kind, seed, t, device):
    if kind == "wpo":
        torch.manual_seed(seed)
        return WavenumberPoleOperator1d(t, **WPO_KW).to(device)
    return M.build("fno", seed, t, device)


def train(model, data, device, lr, steps, plr=1.0):
    xtr, ytr = data["train"]
    is_pole = lambda n: n.endswith("mu_re") or n.endswith("mu_im")
    groups = [dict(params=[p for n, p in model.named_parameters() if not is_pole(n)], lr=lr)]
    pole = [p for n, p in model.named_parameters() if is_pole(n)]
    if pole:
        groups.append(dict(params=pole, lr=lr * plr))
    opt = torch.optim.Adam(groups)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(0)
    for _ in range(steps):
        idx = torch.randint(0, xtr.shape[0], (M.BATCH,), generator=g)
        opt.zero_grad()
        loss = M.A.rel(model(xtr[idx].to(device)), ytr[idx].to(device))
        if not torch.isfinite(loss):
            return float("nan")
        loss.backward()
        opt.step()
        sch.step()
    return float(loss)


def err(model, data, split, device):
    x, y = data[split]
    return float(M.A.rel(M.batched(model, x.to(device)).cpu().to(torch.float64), y))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    data = M.make_data()
    t = np.linspace(0, M.T1, M.NT1)
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(sweep=[], final=[])
    save = lambda: OUT.write_text(json.dumps(out, indent=1, default=float))
    for kind, cfgs in SWEEP.items():
        for cfg in cfgs:
            if any(r["model"] == kind and r["cfg"] == cfg for r in out["sweep"]):
                continue
            t1 = time.time()
            model = build(kind, 0, t, device)
            last = train(model, data, device, steps=SWEEP_STEPS, **cfg)
            model.eval()
            vali = err(model, data, "vali", device) if np.isfinite(last) else float("inf")
            out["sweep"].append(dict(model=kind, cfg=cfg, vali=vali, seconds=round(time.time() - t1, 1)))
            print("sweep", kind, cfg, f"vali {vali:.3e}", f"{time.time() - t1:.0f} s", flush=True)
            save()
            del model
            torch.cuda.empty_cache()
    pfk, pdt = M.probe_inputs()
    n_ext = M.NT1 - 1
    for kind in SWEEP:
        best = min((r for r in out["sweep"] if r["model"] == kind), key=lambda r: r["vali"])["cfg"]
        for seed in M.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out["final"]):
                continue
            t1 = time.time()
            model = build(kind, seed, t, device)
            train(model, data, device, steps=FINAL_STEPS, **best)
            torch.save(model.state_dict(), CKPT / f"{kind}_{seed}.pt")
            model.eval()
            xe, ye = data["ext"]
            pe = M.batched(model, xe.to(device)).cpu().to(torch.float64)
            r = dict(model=kind, seed=seed, cfg=best, params=M.n_params(model), train=err(model, data, "train", device),
                     vali=err(model, data, "vali", device), test=err(model, data, "test", device),
                     ext_early=float(M.A.rel(pe[..., :n_ext + 1], ye[..., :n_ext + 1])),
                     ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])),
                     probe=M.model_probe(model, pfk, pdt, device), seconds=round(time.time() - t1, 1))
            out["final"].append(r)
            p = r["probe"]
            print("final", kind, seed, best, f"train {r['train']:.3e} test {r['test']:.3e}",
                  f"ext (5,10] {r['ext_late']:.3e}", f"probe nu {p['nu_hat']:.5f} disp {p['disp_err']:.2e} kept {p['kept']}",
                  f"{r['seconds']} s", flush=True)
            save()
            del model
            torch.cuda.empty_cache()
    summarize(out)


def summarize(out):
    lines = ["# (A) Fair WPO–FNO comparison on forced Burgers", "",
             f"Sweep: model seed 0, {SWEEP_STEPS} steps, selection by validation. Final: {FINAL_STEPS} steps, seeds 0–2. "
             "WPO stable.", "", "| sweep | config | validation |", "|---|---|---|"]
    for r in out["sweep"]:
        lines.append(f"| {r['model']} | {', '.join(f'{k} {v:g}' for k, v in r['cfg'].items())} | {r['vali']:.3e} |")
    lines += ["", "| model | seed | params | train | test (T = 5) | T = 10: (0, 5] / (5, 10] | (5, 10] / test | probe ν̂ | "
              "probe dispersion error |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out["final"]:
        p = r["probe"]
        lines.append(f"| {r['model']} | {r['seed']} | {r['params']} | {r['train']:.3e} | {r['test']:.3e} | "
                     f"{r['ext_early']:.3e} / {r['ext_late']:.3e} | {r['ext_late'] / r['test']:.2f} | {p['nu_hat']:.5f} | "
                     f"{p['disp_err']:.1e} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
