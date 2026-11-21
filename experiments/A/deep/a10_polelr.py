"""(A) Deep operator, second pass: a separate pole learning rate for the WPO on forced Burgers.

a9 (`a9_burgers.py`) found that the WPO's poles barely moved from their initialization at the shared Adam learning
rate (1e-3). The diffusion pole -nu k^2 needs a k^2 coefficient of -nu k_ref^2 = -25.6 in k / k_ref units, beyond
what 4000 Adam steps at 1e-3 can reach. Some poles stayed at Re mu > 0, and the T = 10 extrapolation blew up.

**Change.** The WPO's pole coefficients (`mu_re`, `mu_im` of every layer) get 30x the learning rate (3e-2, same
cosine decay); everything else stays as in a9 (data, budget, 4000 steps, batch 25, seeds 0-2, probe). Two
variants:

- `plr30`: the a9 layer (Re cap 0.5 with leak 0.05);
- `plr30_stable`: `stable=True`, i.e. Re mu = -softplus(.) <= 0. This is a prior that the operator is dissipative,
  which Burgers is, and is the plan's stability option (Section 4.3).

The a9 FNO runs are the reference; they are not rerun.

**Predictions** (written before running).

1. Probe: for each variant, nu_hat within 10 % and dispersion error <= 20 % in at least 2 of 3 seeds.
2. Accuracy: for each variant, the mean test error is <= 0.7x a9's WPO (0.117) and below a9's FNO (0.105).
3. Extrapolation: `plr30_stable` keeps its error over (5, 10] <= 2x its T = 5 test error in every seed. `plr30`
   exceeds 2x in at least one seed (the cap allows growing modes).

    python experiments/A/deep/a10_polelr.py      # results/a10_polelr.md / .json; checkpoints in runs/A/a10
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

PLR = 30.0
VARIANTS = {"plr30": dict(M.WPO_KW), "plr30_stable": dict(M.WPO_KW, stable=True)}
OUT = HERE / "results" / "a10_polelr.json"
CKPT = M.ROOT / "runs" / "A" / "a10"


def train(model, data, device):
    xtr, ytr = data["train"]
    pole = [p for n, p in model.named_parameters() if n.endswith("mu_re") or n.endswith("mu_im")]
    rest = [p for n, p in model.named_parameters() if not (n.endswith("mu_re") or n.endswith("mu_im"))]
    opt = torch.optim.Adam([dict(params=rest, lr=M.LR), dict(params=pole, lr=M.LR * PLR)])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, M.STEPS)
    g = torch.Generator().manual_seed(0)
    for _ in range(M.STEPS):
        idx = torch.randint(0, xtr.shape[0], (M.BATCH,), generator=g)
        opt.zero_grad()
        loss = M.A.rel(model(xtr[idx].to(device)), ytr[idx].to(device))
        loss.backward()
        opt.step()
        sch.step()
    return float(loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    data = M.make_data()
    t = np.linspace(0, M.T1, M.NT1)
    pfk, pdt = M.probe_inputs()
    a9 = json.loads(M.OUT.read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(runs=[])
    done = {(r["variant"], r["seed"]) for r in out["runs"]}
    n_ext = M.NT1 - 1
    for name, kw in VARIANTS.items():
        for seed in M.SEEDS:
            if (name, seed) in done:
                continue
            t1 = time.time()
            torch.manual_seed(seed)
            model = WavenumberPoleOperator1d(t, **kw).to(device)
            last = train(model, data, device)
            torch.save(model.state_dict(), CKPT / f"{name}_{seed}.pt")
            model.eval()
            xte, yte = data["test"]
            test = float(M.A.rel(M.batched(model, xte.to(device)).cpu().to(torch.float64), yte))
            xe, ye = data["ext"]
            pe = M.batched(model, xe.to(device)).cpu().to(torch.float64)
            r = dict(variant=name, seed=seed, train_loss=last, test=test,
                     ext_early=float(M.A.rel(pe[..., :n_ext + 1], ye[..., :n_ext + 1])),
                     ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])),
                     probe=M.model_probe(model, pfk, pdt, device), first_layer=M.first_layer_poles(model, device),
                     seconds=round(time.time() - t1, 1))
            out["runs"].append(r)
            p = r["probe"]
            print(name, seed, f"test {test:.3e}", f"ext (0,5] {r['ext_early']:.3e} (5,10] {r['ext_late']:.3e}",
                  f"probe nu {p['nu_hat']:.5f} disp {p['disp_err']:.2e} kept {p['kept']}",
                  "first layer k=1,4,8,16", [round(r["first_layer"]["rel_err"][j], 2) for j in (0, 3, 7, 15)],
                  f"{r['seconds']} s", flush=True)
            OUT.write_text(json.dumps(out, indent=1, default=float))
            del model
            torch.cuda.empty_cache()
    out["a9"] = [dict(model=r["model"], seed=r["seed"], test=r["test"], ext_late=r["ext_late"],
                      nu_hat=r["probe"]["nu_hat"], disp_err=r["probe"]["disp_err"]) for r in a9["runs"]]
    OUT.write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out):
    lines = ["# (A) Deep operator on forced Burgers: pole learning rate ×30", "",
             "a9 setup; WPO pole coefficients at 30× the learning rate. a9 rows for reference.", "",
             "| model | seed | test (T = 5) | T = 10: (0, 5] / (5, 10] | probe ν̂ (0.1) | probe dispersion error | "
             "first layer, nearest pole error at k = 1, 4, 8, 16 |", "|---|---|---|---|---|---|---|"]
    for r in out["runs"]:
        p, e = r["probe"], r["first_layer"]["rel_err"]
        lines.append(f"| {r['variant']} | {r['seed']} | {r['test']:.3e} | {r['ext_early']:.3e} / {r['ext_late']:.3e} | "
                     f"{p['nu_hat']:.5f} | {p['disp_err']:.1e} | {e[0]:.2f}, {e[3]:.2f}, {e[7]:.2f}, {e[15]:.2f} |")
    for r in out.get("a9", []):
        lines.append(f"| a9 {r['model']} | {r['seed']} | {r['test']:.3e} | – / {r['ext_late']:.3e} | {r['nu_hat']:.5f} | "
                     f"{r['disp_err']:.1e} | – |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
