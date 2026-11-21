"""(A) Deep operator, fourth pass: band-covering FNO baselines on forced Burgers.

a11 (`a11_fair.py`) compared the tuned WPO (test 0.024) against an FNO with n_modes (12, 12). In neuralop 2.0 that
keeps |k| <= 6 and 7 temporal frequencies, below the forcing band (k <= 8) and far below what steps and PRBS need.
Its error floor of ~0.1 came from the truncation. This pass replaces it with baselines whose modes cover the band:

- `tfno`: Tucker-factorized FNO (neuraloperator's TFNO), n_modes (32, 64) (|k| <= 16, 33 temporal frequencies),
  width 24, rank 0.02: 59,065 parameters, the WPO's budget (59,529).
- `fno_big`: full-rank FNO, n_modes (32, 64), width 16: 1,084,849 parameters (18x), as a reference outside the
  budget.

Both: domain padding 0.25 in t, float32. Same protocol as a11: lr in {1e-3, 3e-3, 1e-2} at 4000 steps (model seed
0, selection by validation), then 12000 steps with seeds 0-2. The WPO results are a11's (not rerun).

**Predictions** (written before running).

1. The matched-budget TFNO's mean test error is above the WPO's (0.024).
2. The 18x FNO's mean test error is below the WPO's.
3. Both FNO variants have an extrapolation ratio ((5, 10] error / test error) > 2.5 in every seed.
4. Probe: both FNO variants have a dispersion error above the WPO's (0.010-0.034) in every seed.

    python experiments/A/deep/a13_baselines.py      # results/a13_baselines.md / .json; checkpoints in runs/A/a13
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

MODELS = {"tfno": dict(n_modes=(32, 64), hidden_channels=24, factorization="tucker", rank=0.02),
          "fno_big": dict(n_modes=(32, 64), hidden_channels=16)}
LRS = (1e-3, 3e-3, 1e-2)
OUT = HERE / "results" / "a13_baselines.json"
CKPT = M.ROOT / "runs" / "A" / "a13"


class Net(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(in_channels=1, out_channels=1, n_layers=4, domain_padding=[0.0, 0.25], **kw)

    def forward(self, f):
        return self.net(f.float()[:, None])[:, 0].to(f.dtype)


def build(kind, seed, device):
    torch.manual_seed(seed)
    return Net(**MODELS[kind]).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    data = M.make_data()
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(sweep=[], final=[])
    save = lambda: OUT.write_text(json.dumps(out, indent=1, default=float))
    for kind in MODELS:
        for lr in LRS:
            if any(r["model"] == kind and r["lr"] == lr for r in out["sweep"]):
                continue
            t1 = time.time()
            model = build(kind, 0, device)
            last = F.train(model, data, device, lr=lr, steps=F.SWEEP_STEPS)
            model.eval()
            vali = F.err(model, data, "vali", device) if np.isfinite(last) else float("inf")
            out["sweep"].append(dict(model=kind, lr=lr, vali=vali, params=M.n_params(model), seconds=round(time.time() - t1, 1)))
            print("sweep", kind, lr, f"vali {vali:.3e}", f"{time.time() - t1:.0f} s", flush=True)
            save()
            del model
            torch.cuda.empty_cache()
    pfk, pdt = M.probe_inputs()
    n_ext = M.NT1 - 1
    for kind in MODELS:
        lr = min((r for r in out["sweep"] if r["model"] == kind), key=lambda r: r["vali"])["lr"]
        for seed in M.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out["final"]):
                continue
            t1 = time.time()
            model = build(kind, seed, device)
            F.train(model, data, device, lr=lr, steps=F.FINAL_STEPS)
            torch.save(model.state_dict(), CKPT / f"{kind}_{seed}.pt")
            model.eval()
            xe, ye = data["ext"]
            pe = M.batched(model, xe.to(device)).cpu().to(torch.float64)
            r = dict(model=kind, seed=seed, lr=lr, params=M.n_params(model), train=F.err(model, data, "train", device),
                     test=F.err(model, data, "test", device),
                     ext_early=float(M.A.rel(pe[..., :n_ext + 1], ye[..., :n_ext + 1])),
                     ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])),
                     probe=M.model_probe(model, pfk, pdt, device), seconds=round(time.time() - t1, 1))
            out["final"].append(r)
            p = r["probe"]
            print("final", kind, seed, lr, f"train {r['train']:.3e} test {r['test']:.3e}", f"ext (5,10] {r['ext_late']:.3e}",
                  f"probe nu {p['nu_hat']:.5f} disp {p['disp_err']:.2e} kept {p['kept']}", f"{r['seconds']} s", flush=True)
            save()
            del model
            torch.cuda.empty_cache()
    out["wpo_a11"] = [r for r in json.loads(F.OUT.read_text())["final"] if r["model"] == "wpo"]
    save()
    summarize(out)


def summarize(out):
    lines = ["# (A) Band-covering FNO baselines on forced Burgers (WPO from a11)", "",
             "| sweep | lr | params | validation |", "|---|---|---|---|"]
    for r in out["sweep"]:
        lines.append(f"| {r['model']} | {r['lr']:g} | {r['params']} | {r['vali']:.3e} |")
    lines += ["", "| model | seed | params | train | test (T = 5) | (5, 10] | (5, 10] / test | probe ν̂ | probe dispersion error |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in out["final"] + out.get("wpo_a11", []):
        p = r["probe"]
        lines.append(f"| {r['model']} | {r['seed']} | {r['params']} | {r['train']:.3e} | {r['test']:.3e} | {r['ext_late']:.3e} | "
                     f"{r['ext_late'] / r['test']:.2f} | {p['nu_hat']:.5f} | {p['disp_err']:.1e} |")
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
