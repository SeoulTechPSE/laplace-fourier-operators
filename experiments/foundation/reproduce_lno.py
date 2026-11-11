"""Reproduce the original LNO results with the ported model, plus FNO/TFNO baselines.

Usage:
    python experiments/foundation/reproduce_lno.py --case duffing_c0 --model lno --seeds 0 1 2
    python experiments/foundation/reproduce_lno.py --summarize

Training protocol for every model follows the original LNO scripts for the case
(epochs, batch size, learning rate, StepLR schedule, ComplexAdam with weight decay
1e-4, relative-L2 training loss). FNO/TFNO come from `neuraloperator`; their
hidden width and number of Fourier modes are chosen to match the LNO parameter
count as closely as possible (see `match_budget`).

Outputs: checkpoints under runs/foundation/ (git-ignored); one small JSON per run
under experiments/foundation/results/ (committed).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn

from lfo.metrics import relative_l2
from lfo.models import LNO1d, LNO2d
from lfo.training import ComplexAdam, count_params, relative_l2_loss, seed_everything

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
DATA_ROOTS = [Path("/srv/data/lno_drive"), ROOT / "runs" / "data" / "lno_drive"]


@dataclass
class Case:
    folder: str
    dim: int
    batch_size: int
    lr: float = 0.002
    epochs: int = 1000
    step_size: int = 100
    gamma: float = 0.5
    weight_decay: float = 1e-4
    lno_kwargs: dict = None
    fno_modes: tuple = ()          # candidate Fourier mode counts per axis for FNO/TFNO
    paper_note: str = ""


CASES = {
    "duffing_c0": Case("1D_Duffing_c0", dim=1, batch_size=20,
                       lno_kwargs=dict(width=4, modes=16, use_grid=False, activation="sin", transient_sign=-1.0),
                       fno_modes=(8, 12, 16, 24, 32)),
    "diffusion": Case("2D_Diffusion", dim=2, batch_size=50,
                      lno_kwargs=dict(width=16, modes1=4, modes2=4, transient_sign=1.0),
                      fno_modes=(4, 6, 8, 12, 16)),
}


# ------------------------------------------------------------------ data
def data_dir(folder: str, override: str | None) -> Path:
    roots = [Path(override)] if override else DATA_ROOTS
    for r in roots:
        if (r / folder / "Data" / "data.mat").exists():
            return r / folder / "Data"
    raise FileNotFoundError(f"{folder}/Data/data.mat not found under {roots}")


def load(case: Case, root: str | None):
    m = scipy.io.loadmat(data_dir(case.folder, root) / "data.mat")
    f32 = lambda k: torch.tensor(m[k], dtype=torch.float32)
    split = {s: (f32(f"f_{s}")[..., None], f32(f"u_{s}")) for s in ("train", "vali", "test")}
    grids = dict(t=f32("x_train")) if case.dim == 1 else dict(x=f32("x"), t=f32("t"))
    return split, grids


# ----------------------------------------------------------------- models
class ChannelFirst(nn.Module):
    """Adapt a neuraloperator model (b, c, *grid) to the LNO layout (b, *grid, c)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = torch.movedim(x, -1, 1)
        return torch.movedim(self.net(x), 1, -1)


def _neuralop(name: str, dim: int, modes: int, hidden: int, **extra) -> nn.Module:
    from neuralop.models import FNO, TFNO
    cls = {"fno": FNO, "tfno": TFNO}[name]
    return ChannelFirst(cls(n_modes=(modes,) * dim, in_channels=1, out_channels=1,
                            hidden_channels=hidden, n_layers=4, **extra))


def match_budget(name: str, case: Case, budget: int, **extra):
    """(modes, hidden) of a 4-layer FNO/TFNO whose parameter count is closest to `budget`."""
    best = None
    for modes in case.fno_modes:
        for hidden in range(2, 65):
            n = count_params(_neuralop(name, case.dim, modes, hidden, **extra))
            if best is None or abs(n - budget) < abs(best[2] - budget):
                best = (modes, hidden, n)
            if n > 2 * budget:
                break
    return best


def build(model: str, case: Case, grids: dict, fno_extra: dict | None = None, budget: int | None = None):
    """Build the model. For FNO/TFNO, `fno_extra` is passed to the neuraloperator
    constructor (e.g. positional_embedding=None, domain_padding=0.25) and `budget`
    overrides the target parameter count (default: the LNO's)."""
    if case.dim == 1:
        lno = LNO1d(t=grids["t"], **case.lno_kwargs)
    else:
        lno = LNO2d(x=grids["x"], t=grids["t"], **case.lno_kwargs)
    if model == "lno":
        return lno, dict(case.lno_kwargs)
    extra = dict(fno_extra or {})
    modes, hidden, _ = match_budget(model, case, budget or count_params(lno), **extra)
    return (_neuralop(model, case.dim, modes, hidden, **extra),
            dict(n_modes=modes, hidden_channels=hidden, n_layers=4, **extra))


# ------------------------------------------------------------------ train
def per_sample_error(model, x, y, device, bs=50):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            preds.append(model(x[i:i + bs].to(device)).reshape(y[i:i + bs].shape).cpu())
    pred = torch.cat(preds)
    return relative_l2(pred, y, axis=tuple(range(1, y.ndim)))


def run(case_name: str, model_name: str, seed: int, args) -> dict:
    case = CASES[case_name]
    device = torch.device(args.device)
    split, grids = load(case, args.data_root)
    seed_everything(seed)
    model, hparams = build(model_name, case, grids, json.loads(args.fno_kwargs), args.budget)
    model = model.to(device)
    n_params = count_params(model)

    (xtr, ytr), (xva, yva), (xte, yte) = split["train"], split["vali"], split["test"]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtr, ytr),
                                         batch_size=case.batch_size, shuffle=True)
    opt = ComplexAdam(model.parameters(), lr=case.lr, weight_decay=case.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=case.step_size, gamma=case.gamma)
    epochs = args.epochs or case.epochs

    t0 = time.time()
    history = []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = relative_l2_loss(model(x).reshape(y.shape), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        sched.step()
        if (ep + 1) % args.log_every == 0 or ep == epochs - 1:
            vali = float(np.mean(per_sample_error(model, xva, yva, device)))
            history.append(dict(epoch=ep + 1, train_l2=tot / n, vali_l2=vali))
            print(f"[{case_name}/{model_name}/s{seed}] ep {ep + 1:4d}  train {tot / n:.4e}  vali {vali:.4e}", flush=True)
    train_time = time.time() - t0

    test = per_sample_error(model, xte, yte, device)
    result = dict(case=case_name, model=model_name, seed=seed, epochs=epochs, n_params=n_params,
                  hparams=hparams, train_time_s=round(train_time, 1),
                  test_rel_l2=float(np.mean(test)), test_rel_l2_std_over_samples=float(np.std(test)),
                  vali_rel_l2=history[-1]["vali_l2"], history=history, tag=args.tag,
                  test_rel_l2_per_sample=[round(float(e), 6) for e in test],
                  torch=torch.__version__, device=torch.cuda.get_device_name() if device.type == "cuda" else "cpu")

    out = RESULTS / args.tag if args.tag else RESULTS
    ckpt = ROOT / "runs" / "foundation" / (args.tag or "") / case_name / model_name / f"seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{case_name}_{model_name}_seed{seed}.json").write_text(json.dumps(result, indent=1))
    print(f"[{case_name}/{model_name}/s{seed}] test rel L2 = {result['test_rel_l2']:.4e} "
          f"({n_params} params, {train_time:.0f} s)", flush=True)
    return result


def summarize(tag: str = "") -> str:
    results = RESULTS / tag if tag else RESULTS
    rows = {}
    for f in sorted(results.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        rows.setdefault((r["case"], r["model"]), []).append(r)
    lines = ["| case | model | params | seeds | test rel. L2 (mean ± std) | train time (s) |",
             "|---|---|---|---|---|---|"]
    for (c, m), rs in sorted(rows.items()):
        e = np.array([r["test_rel_l2"] for r in rs])
        tt = np.mean([r["train_time_s"] for r in rs])
        lines.append(f"| {c} | {m} | {rs[0]['n_params']} | {len(rs)} | {e.mean():.3e} ± {e.std(ddof=1) if len(e) > 1 else 0:.1e} | {tt:.0f} |")
    table = "\n".join(lines)
    (results / "summary.md").write_text(
        "# LNO reproduction summary\n\nGenerated by `reproduce_lno.py --summarize`. "
        "Std is the sample std over seeds (ddof=1).\n\n" + table + "\n")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=CASES, nargs="+", default=list(CASES))
    p.add_argument("--model", choices=["lno", "fno", "tfno"], nargs="+", default=["lno"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=None, help="override the case's epoch count (for smoke tests)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--summarize", action="store_true", help="only print/write the summary table")
    p.add_argument("--fno-kwargs", default="{}",
                   help='JSON passed to the FNO/TFNO constructor, e.g. \'{"positional_embedding": null}\'')
    p.add_argument("--budget", type=int, default=None, help="FNO/TFNO target parameter count (default: LNO's)")
    p.add_argument("--tag", default="", help="write results to results/<tag>/ (for ablations)")
    args = p.parse_args()
    if not args.summarize:
        for c in args.case:
            for m in args.model:
                for s in args.seeds:
                    run(c, m, s, args)
    print(summarize(args.tag))


if __name__ == "__main__":
    main()
