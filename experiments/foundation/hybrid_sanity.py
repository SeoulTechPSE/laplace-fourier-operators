"""Sanity check: LFNO under the LFNO papers' training protocols and both pole inits.

Duffing c=0, iid split, 1,000 epochs, seed 0. Reproduces the table in
paper_comparison.md, section "LFNO". Not a reported result (1 seed).

    python experiments/foundation/hybrid_sanity.py
"""
import numpy as np, torch
from lfo.data import lno_bench
from lfo.models import LFNO1d
from lfo.metrics import relative_l2
from lfo.training import relative_l2_loss, seed_everything
d, g = lno_bench.load_case("1D_Duffing_c0", "iid")
(xtr, ytr), _, (xte, yte) = d["train"], d["vali"], d["test"]; t = g["t"]
protos = {"workshop": dict(lr=0.002, wd=0.01, bs=64, gamma=0.8), "arxiv": dict(lr=0.0025, wd=0.02, bs=64, gamma=0.85)}
for pname, P in protos.items():
    for init in ("lno", "stable"):
        seed_everything(0); m = LFNO1d(4, 16, t, pole_init=init).cuda()
        opt = torch.optim.Adam(m.parameters(), lr=P["lr"], weight_decay=P["wd"])
        sch = torch.optim.lr_scheduler.StepLR(opt, 100, P["gamma"])
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtr, ytr), batch_size=P["bs"], shuffle=True)
        for ep in range(1000):
            m.train()
            for x, y in dl:
                x, y = x.cuda(), y.cuda(); opt.zero_grad()
                l = relative_l2_loss(m(x).reshape(y.shape), y); l.backward(); opt.step()
            sch.step()
        m.eval()
        with torch.no_grad(): p = m(xte.cuda()).reshape(yte.shape).cpu()
        mu = torch.cat([a.flatten() for a, _ in m.poles()])
        print(f"{pname:8s} init={init:6s} test {np.mean(relative_l2(p, yte, axis=(1,))):.3e} train {l.item():.3e} "
              f"|mu| median {mu.abs().median():.3f} Re>0 frac {(mu.real > 0).float().mean():.2f}", flush=True)
