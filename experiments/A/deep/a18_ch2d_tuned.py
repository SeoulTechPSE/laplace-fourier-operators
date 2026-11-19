"""(A) 2D Cahn-Hilliard: a tuned, budget-matched TFNO against the a17 WPO-2D.

In a17 (`a17_ch2d.py`) the TFNO had 20 % fewer parameters than the WPO-2D (15.0 k against 18.7 k: rank steps are
coarse at width 16) and no learning-rate sweep. It underfit (train 0.055-0.076) at 3.6x the WPO's test error. This
pass gives it a fair chance:

- architecture: the two Tucker configurations within 10 % of the WPO's budget: width 12 with rank 0.003
  (20,421 parameters) and width 20 with rank 0.001 (20,549);
- sweep: both architectures x lr in {1e-3, 3e-3, 1e-2}, 2000 steps, model seed 0, selection by the validation
  error;
- final: the selected setting, 6000 steps (as a17), seeds 0-2, a17 data and probe.

The WPO-2D results are a17's (not rerun; it had no sweep either, a11's setting).

**Predictions** (written before running).

1. The tuned TFNO's mean test error is below a17's TFNO (0.116) but above the WPO-2D's (0.032).
2. Its probe k_c is within 10 % of 5 in at least 2 of 3 seeds, and its median in-band per-mode error stays above
   the WPO-2D's.

    python experiments/A/deep/a18_ch2d_tuned.py      # results/a18_ch2d_tuned.md / .json
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
import a17_ch2d as S  # noqa: E402

ARCHS = {"w12_r003": dict(hidden_channels=12, rank=0.003), "w20_r001": dict(hidden_channels=20, rank=0.001)}
LRS, SWEEP_STEPS = (1e-3, 3e-3, 1e-2), 2000
OUT = HERE / "results" / "a18_ch2d_tuned.json"


class TFNO(torch.nn.Module):
    def __init__(self, hidden_channels, rank):
        super().__init__()
        from neuralop.models import FNO
        self.net = FNO(n_modes=(16, 16, 64), in_channels=1, out_channels=1, hidden_channels=hidden_channels, n_layers=4,
                       domain_padding=[0.0, 0.0, 0.25], factorization="tucker", rank=rank)

    def forward(self, f):
        return self.net(f.float()[:, None])[:, 0].to(f.dtype)


def build(arch, seed, device):
    torch.manual_seed(seed)
    return TFNO(**ARCHS[arch]).to(device)


def median_inband_err(p):
    kept = {tuple(k) for k in p["kept"]}
    e = [abs(m["mu"][0] - m["truth"]) / abs(m["truth"]) for m in p["modes"] if tuple(m["k"]) in kept and abs(m["truth"]) > 0.05]
    return float(np.median(e))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = S.make_data()
    pfk, pdt = S.probe_inputs()
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(sweep=[], final=[])
    save = lambda: OUT.write_text(json.dumps(out, indent=1, default=float))
    for arch in ARCHS:
        for lr in LRS:
            if any(r["arch"] == arch and r["lr"] == lr for r in out["sweep"]):
                continue
            t1 = time.time()
            m = build(arch, 0, device)
            S.train(m, data, device, SWEEP_STEPS, lr)
            m.eval()
            vali = float(S.A.rel(S.batched(m, data["vali"][0], device), data["vali"][1]))
            out["sweep"].append(dict(arch=arch, lr=lr, params=S.F.M.n_params(m), vali=vali, seconds=round(time.time() - t1, 1)))
            print("sweep", arch, lr, f"vali {vali:.3e}", flush=True)
            save()
    best = min(out["sweep"], key=lambda r: r["vali"])
    for seed in S.SEEDS:
        if any(r["seed"] == seed for r in out["final"]):
            continue
        t1 = time.time()
        m = build(best["arch"], seed, device)
        S.train(m, data, device, S.STEPS, best["lr"])
        m.eval()
        ev = lambda s: float(S.A.rel(S.batched(m, data[s][0], device), data[s][1]))

        def G(spec):
            f = torch.tensor(np.fft.irfft2(spec[..., ::S.FINE], s=(S.N, S.N), axes=(1, 2)))
            return f, S.batched(m, f, device)

        p = S.probe(G, pfk, device)
        r = dict(seed=seed, arch=best["arch"], lr=best["lr"], params=S.F.M.n_params(m), train=ev("train"), test=ev("test"),
                 probe=p, median_err=median_inband_err(p), seconds=round(time.time() - t1, 1))
        out["final"].append(r)
        print("final", seed, f"train {r['train']:.3e} test {r['test']:.3e} k_c {p['k_c']:.3f} median err {r['median_err']:.3f}", flush=True)
        save()
    a17 = json.loads(S.OUT.read_text())["runs"]
    out["a17"] = [dict(model=r["model"], seed=r["seed"], test=r["test"], k_c=r["probe"]["k_c"],
                       median_err=median_inband_err(r["probe"])) for r in a17]
    save()
    lines = ["# (A) 2D Cahn–Hilliard: tuned, budget-matched TFNO", "", "| sweep | lr | params | validation |", "|---|---|---|---|"]
    lines += [f"| {r['arch']} | {r['lr']:g} | {r['params']} | {r['vali']:.3e} |" for r in out["sweep"]]
    lines += ["", "| model | seed | train | test | probe k_c | median in-band per-mode error |", "|---|---|---|---|---|---|"]
    lines += [f"| TFNO tuned ({r['arch']}, lr {r['lr']:g}) | {r['seed']} | {r['train']:.3e} | {r['test']:.3e} | {r['probe']['k_c']:.3f} | "
              f"{r['median_err']:.3f} |" for r in out["final"]]
    lines += [f"| a17 {r['model']} | {r['seed']} | – | {r['test']:.3e} | {r['k_c']:.3f} | {r['median_err']:.3f} |" for r in out["a17"]]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
