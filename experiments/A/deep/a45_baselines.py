"""(A) Paper-1 reinforcement, step 1: the planned baselines that the deep comparison lacked.

The plan review (2026-11-22) listed as missing: an SSM-based operator (the proposal's main novelty risk) and LNO /
ALNO / LFNO in the deep comparison. The per-index WPO of a36 stands in for S4D / SGNO-type per-mode generators
inside our architecture; this pass runs the others as separate models, at the WPO's real-parameter budget
(`lfo.metrics.n_real_params`), with 5 seeds, on the two protocols of a42.

**Models.**

- `lno`: the original LNO v2 (`LNO2dV2`, as a36: transient + steady branches, sin, stride 2; steady modes (9, 9) on
  F), the width closest to the WPO's budget. F: seeds 3, 4 added to a36's 0-2 with a36's selected lr. P: new.
- `alno`: ALNO per layer on (x, t) (`lfo.models.ALNO2d`): separable pole-residue transient (4 x 4 poles per channel
  pair, stable initialization) + free FNO multiplier + 1x1 convolution, sin; 4 layers; grid channels as LNO.
- `lfno`: LFNO on (x, t) (`lfo.models.LFNO2d`): transient from the poles, R-mixed steady coefficients of the same
  poles in the first 2 of 4 layers, relu.
- `s4d`: an SSM operator: per layer v <- GELU(S4D_t(v) + SpectralConv_x(v) + W v) (no GELU after the last layer).
  S4D_t is the authors' S4D layer (state-spaces/s4, `models/s4/s4d.py`, vendored as `lfo.models.s4d`; d_state 64,
  its GELU and GLU output mixing included), applied along time for every grid point and channel: causal, poles per
  channel, not functions of k. SpectralConv_x is neuraloperator's dense `SpectralConv` over x per time sample, with
  the TFNO's x band (F: n_modes 32, |k| <= 16; P: 64, |m| <= 32). The S4D kernel parameters (log dt, A) get
  lr min(lr, 1e-3), as in the authors' example.
- Mode counts of the pole models' FNO parts: F (9, 9) as a36's LNO (|k| <= 8 forced band, 9 temporal modes); P
  (16, 6) (|m| <= 15, 6 temporal modes): dense 2D multipliers do not fit more into the budget.

Inputs as the references: F the forcing field (b, n_x, n_t); P the initial field repeated over t (as the TFNO).
Pole and grid buffers are re-set for time extrapolation (F, T = 10) and for P's 1024-point resolution test.

**Protocol.** a36 / a37: relative-L2 (F) or nRMSE (P) loss, Adam, cosine decay, batch 25; sweep at 4000 steps
with model seed 0, selection by validation; final 12000 steps, seeds 0-4. Sweeps: lr 1e-3, 3e-3, 1e-2; for
`alno` / `lfno` also pole lr x1 or x30 (6 configurations). References are a42's 5-seed WPO, per-index WPO and TFNO
(not rerun).

**Metrics.** a36's on F (test, (5, 10] / test, probe outside the band k = 9-16, unseen modes k = 9-12 / test);
a37's on P (nRMSE at 256 and 1024 points, probe nu', dispersion error). 95 % t-intervals over 5 seeds.

**Predictions** (written before running; WPO from a42: F test 0.024, unseen / test 1.3 [0.9, 1.6], probe outside
0.10 [0.06, 0.14], (5, 10] / test 1.85; P nRMSE 0.0450 [0.0440, 0.0459]; TFNO P 0.0254).

1. F, unseen modes: every new baseline's 95 % interval of unseen / test lies above the WPO's (no overlap).
2. F, probe outside the band: every new baseline's mean >= 0.35.
3. F, accuracy: `s4d` mean test within [0.8, 1.6]x the WPO's; `lno`, `alno`, `lfno` >= 1.5x.
4. F, time extrapolation: `s4d` mean (5, 10] / test <= 3 (causal, sample-time kernel); `alno` and `lfno` > 3
   (their FNO parts mix the whole window).
5. P, accuracy: `s4d` mean nRMSE below the WPO's 0.0450; `lno`, `alno`, `lfno` above it.
6. P, probe: no new baseline has nu' within 5 % of 0.1257 in more than 2 of 5 seeds.

    python experiments/A/deep/a45_baselines.py [F:lno F:alno F:lfno F:s4d P:lno P:alno P:lfno P:s4d] [--summary]
"""
from __future__ import annotations

import fcntl
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a9_burgers as M  # noqa: E402
import a11_fair as F  # noqa: E402
import a36_index_poles as X36  # noqa: E402
import a37_pdebench_burgers as X37  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import ALNO2d, LFNO2d, LNO2dV2, WavenumberPoleOperator1d  # noqa: E402

OUT = HERE / "results" / "a45_baselines.json"
SEEDS = (0, 1, 2, 3, 4)
T975_4 = 2.776
KINDS = ("lno", "alno", "lfno", "s4d")
JOBS = [f"{p}:{k}" for p in "FP" for k in KINDS]
MODES = {"F": (9, 9), "P": (16, 6)}
XMODES = {"F": 32, "P": 64}
LRS = (1e-3, 3e-3, 1e-2)
SWEEP = {"lno": [dict(lr=lr) for lr in LRS], "s4d": [dict(lr=lr) for lr in LRS],
         "alno": [dict(lr=lr, plr=p) for lr in LRS for p in (1.0, 30.0)],
         "lfno": [dict(lr=lr, plr=p) for lr in LRS for p in (1.0, 30.0)]}
POLE = ("mu1", "mu2")
SSM = ("log_dt", "log_A_real", "A_imag")


# ---------------------------------------------------------------- models
class Wrap2d(nn.Module):
    """(b, n_x, n_t) float64 forcing -> (b, n_x, n_t), through a float32 (b, n_x, n_t, 1) model."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, f):
        return self.net(f.float()[..., None])[..., 0].to(f.dtype)


class S4DOperator(nn.Module):
    """v <- GELU(S4D over t + SpectralConv over x + W v); input and output (b, n_x, n_t)."""

    def __init__(self, width, x_modes, n_layers=4, d_state=64):
        super().__init__()
        from neuralop.layers.spectral_convolution import SpectralConv
        from lfo.models.s4d import S4D
        self.lift = nn.Linear(1, width)
        self.ssm = nn.ModuleList(S4D(width, d_state=d_state) for _ in range(n_layers))
        self.spec = nn.ModuleList(SpectralConv(width, width, n_modes=(x_modes,)) for _ in range(n_layers))
        self.w = nn.ModuleList(nn.Linear(width, width) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, f):
        b, nx, nt = f.shape
        v = self.lift(f.float()[..., None])                                       # (b, x, t, w)
        for k, (ssm, spec, w) in enumerate(zip(self.ssm, self.spec, self.w)):
            a, _ = ssm(v.reshape(b * nx, nt, -1).transpose(1, 2))                   # (b x, w, t)
            a = a.transpose(1, 2).reshape(b, nx, nt, -1)
            s = spec(v.permute(0, 2, 3, 1).reshape(b * nt, -1, nx))                  # (b t, w, x)
            s = s.reshape(b, nt, -1, nx).permute(0, 3, 1, 2)
            v = a + s + w(v)
            if k < len(self.ssm) - 1:
                v = nn.functional.gelu(v)
        return self.proj(v)[..., 0].to(f.dtype)


def make(kind, part, width, x, t):
    m1, m2 = MODES[part]
    if kind == "lno":
        return Wrap2d(LNO2dV2(width, width, width, 4, 4, m1, m2, x, t))
    if kind == "alno":
        return Wrap2d(ALNO2d(width, 4, 4, m1, m2, x, t))
    if kind == "lfno":
        return Wrap2d(LFNO2d(width, 4, 4, m1, m2, x, t))
    return S4DOperator(width, XMODES[part])


def width_for(kind, part, x, t, target):
    return min(range(3, 40), key=lambda w: abs(n_real_params(make(kind, part, w, x, t)) - target))


def set_grids(model, x=None, t=None):
    for m in model.modules():
        bufs = dict(m.named_buffers(recurse=False))
        if t is not None and "t" in bufs:
            m.t = torch.as_tensor(t, dtype=m.t.dtype, device=m.t.device)
        if x is not None and "x" in bufs:
            m.x = torch.as_tensor(x, dtype=m.x.dtype, device=m.x.device)


class PWrap(nn.Module):
    """P interface: model(u0 (b, n_x), n_t) -> (b, n_x, n_t)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, u0, n_t):
        return self.net(u0[..., None].expand(-1, -1, n_t).contiguous())


# ---------------------------------------------------------------- training
def groups(model, lr, plr=1.0):
    pole = [p for n, p in model.named_parameters() if n.split(".")[-1] in POLE]
    ssm = [p for n, p in model.named_parameters() if n.split(".")[-1] in SSM]
    rest = [p for n, p in model.named_parameters() if n.split(".")[-1] not in POLE + SSM]
    g = [dict(params=rest, lr=lr)]
    if pole:
        g.append(dict(params=pole, lr=lr * plr))
    if ssm:
        g.append(dict(params=ssm, lr=min(lr, 1e-3), weight_decay=0.0))
    return g


def train(model, batch_fn, steps, lr, plr=1.0):
    opt = torch.optim.Adam(groups(model, lr, plr))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(0)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        opt.zero_grad()
        loss = batch_fn(model, gen)
        if not torch.isfinite(loss):
            return float("nan")
        loss.backward()
        opt.step()
        sch.step()
    return float(loss)


# ---------------------------------------------------------------- storage
def locked(fn):
    OUT.parent.mkdir(exist_ok=True)
    with open(f"/tmp/{OUT.stem}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(OUT.read_text()) if OUT.exists() else dict(budget={}, sweep=[], runs=[])
        fn(out)
        OUT.write_text(json.dumps(out, indent=1, default=float))
        return out


def best_cfg(part, kind, out):
    rs = [r for r in out["sweep"] if r["part"] == part and r["model"] == kind]
    return min(rs, key=lambda r: r["vali"])["cfg"]


# ---------------------------------------------------------------- F: forced Burgers
def run_f(kind, device):
    data = M.make_data()
    data["high"] = X36.high_set()
    t = np.linspace(0, M.T1, M.NT1)
    t2 = np.linspace(0, M.T2, M.NT2)
    x = np.linspace(0, 2 * np.pi, M.NX, endpoint=False)
    pfk, pdt = M.probe_inputs()
    xtr, ytr = data["train"]
    target = n_real_params(WavenumberPoleOperator1d(t, **F.WPO_KW))

    def batch(model, gen):
        idx = torch.randint(0, xtr.shape[0], (M.BATCH,), generator=gen)
        return M.A.rel(model(xtr[idx].to(device)), ytr[idx].to(device))

    def evaluate(model, seed, cfg, t0):
        model.eval()
        xe, ye = data["ext"]
        set_grids(model, t=t2)
        pe = M.batched(model, xe.to(device)).cpu().to(torch.float64)
        set_grids(model, t=t)
        probe = M.model_probe(model, pfk, pdt, device)
        n_ext = M.NT1 - 1
        return dict(part="F", model=kind, seed=seed, cfg=cfg, real_params=n_real_params(model),
                    test=F.err(model, data, "test", device), vali=F.err(model, data, "vali", device),
                    ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])),
                    high=F.err(model, data, "high", device), probe=probe, oob=X36.oob(probe),
                    seconds=round(time.time() - t0, 1))

    if kind == "lno":                                    # a36's widths and selected lr; add seeds 3, 4
        a36 = json.loads(X36.OUT.read_text())
        cfg = next(r for r in a36["final"] if r["model"] == "lno")["cfg"]
        seeds = (3, 4)
        build = lambda s: X36.build("lno", s, t, device)
    else:
        w = width_for(kind, "F", x, t, target)
        locked(lambda o: o["budget"].__setitem__(f"F:{kind}", dict(width=w, real=n_real_params(make(kind, "F", w, x, t)),
                                                                   wpo=target)))

        def build(s):
            torch.manual_seed(s)
            return make(kind, "F", w, x, t).to(device)
        for c in SWEEP[kind]:
            if any(r["part"] == "F" and r["model"] == kind and r["cfg"] == c for r in locked(lambda o: None)["sweep"]):
                continue
            t0 = time.time()
            m = build(0)
            last = train(m, batch, F.SWEEP_STEPS, **c)
            m.eval()
            vali = F.err(m, data, "vali", device) if np.isfinite(last) else float("inf")
            rec = dict(part="F", model=kind, cfg=c, vali=vali, seconds=round(time.time() - t0, 1))
            locked(lambda o: o["sweep"].append(rec))
            print("sweep F", kind, c, f"vali {vali:.4f}", f"{rec['seconds']} s", flush=True)
            del m
            torch.cuda.empty_cache()
        cfg = best_cfg("F", kind, locked(lambda o: None))
        seeds = SEEDS
    for s in seeds:
        if any(r["part"] == "F" and r["model"] == kind and r["seed"] == s for r in locked(lambda o: None)["runs"]):
            continue
        t0 = time.time()
        m = build(s)
        last = train(m, batch, F.FINAL_STEPS, **cfg)
        r = evaluate(m, s, cfg, t0) if np.isfinite(last) else dict(part="F", model=kind, seed=s, cfg=cfg, diverged=True)
        locked(lambda o: o["runs"].append(r))
        print("F", kind, s, {k: r.get(k) for k in ("test", "oob", "high", "ext_late")}, flush=True)
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- P: PDEBench 1D Burgers
def run_p(kind, device):
    d = X37.load()
    tt = d["t"].numpy()
    S = X37.splits(d)
    n_t = S["train"].shape[2]
    t = tt                                               # the 41-sample grid on [0, 2]
    x = np.arange(256) / 256
    x1024 = np.arange(1024) / 1024
    full_test = d["full"][X37.NTR:].to(torch.float64)
    target = n_real_params(X37.WPO(tt))
    u = S["train"]
    w = width_for(kind, "P", x, t, target)
    locked(lambda o: o["budget"].__setitem__(f"P:{kind}", dict(width=w, real=n_real_params(make(kind, "P", w, x, t)),
                                                               wpo=target)))

    def build(s):
        torch.manual_seed(s)
        return PWrap(make(kind, "P", w, x, t)).to(device)

    def batch(model, gen):
        idx = torch.randint(0, u.shape[0], (X37.BATCH,), generator=gen)
        y = u[idx].to(device)
        p = model(y[:, :, 0], y.shape[2])
        return ((p - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).mean()

    for c in SWEEP[kind]:
        if any(r["part"] == "P" and r["model"] == kind and r["cfg"] == c for r in locked(lambda o: None)["sweep"]):
            continue
        t0 = time.time()
        m = build(0)
        last = train(m, batch, X37.SWEEP_STEPS, **c)
        m.eval()
        vali = X37.nrmse(X37.predict(m, S["vali"], device), S["vali"]) if np.isfinite(last) else float("inf")
        rec = dict(part="P", model=kind, cfg=c, vali=vali, seconds=round(time.time() - t0, 1))
        locked(lambda o: o["sweep"].append(rec))
        print("sweep P", kind, c, f"vali {vali:.4f}", f"{rec['seconds']} s", flush=True)
        del m
        torch.cuda.empty_cache()
    cfg = best_cfg("P", kind, locked(lambda o: None))
    for s in SEEDS:
        if any(r["part"] == "P" and r["model"] == kind and r["seed"] == s for r in locked(lambda o: None)["runs"]):
            continue
        t0 = time.time()
        m = build(s)
        last = train(m, batch, X37.FINAL_STEPS, **cfg)
        m.eval()
        if not np.isfinite(last):
            r = dict(part="P", model=kind, seed=s, cfg=cfg, diverged=True)
        else:
            test = X37.nrmse(X37.predict(m, S["test"], device), S["test"])
            probe = X37.model_probe(m, tt, 256, n_t, device)
            set_grids(m, x=x1024)
            t1024 = X37.nrmse(X37.predict(m, full_test, device), full_test)
            set_grids(m, x=x)
            r = dict(part="P", model=kind, seed=s, cfg=cfg, real_params=n_real_params(m), test=test,
                     test_1024=t1024, probe=probe, seconds=round(time.time() - t0, 1))
        locked(lambda o: o["runs"].append(r))
        print("P", kind, s, {k: r.get(k) for k in ("test", "test_1024")}, flush=True)
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- summary
def ci(xs):
    xs = np.asarray(xs, float)
    m, s = xs.mean(), xs.std(ddof=1) if len(xs) > 1 else float("nan")
    h = T975_4 * s / np.sqrt(len(xs)) if len(xs) == 5 else float("nan")
    return m, s, m - h, m + h


def fmt(xs, f):
    if not len(xs):
        return "–"
    m, s, lo, hi = ci(xs)
    return f"{m:{f}} [{lo:{f}}, {hi:{f}}]"


def summarize():
    out = locked(lambda o: None)
    a36 = json.loads(X36.OUT.read_text())
    a42 = json.loads((HERE / "results" / "a42_five_seeds.json").read_text())
    import a42_five_seeds as X42
    F5, P5 = X42.collect()
    runs = lambda p, k: [r for r in out["runs"] if r["part"] == p and r["model"] == k and not r.get("diverged")]
    F5["lno"] = [r for r in a36["final"] if r["model"] == "lno"] + runs("F", "lno")
    for k in ("alno", "lfno", "s4d"):
        F5[k] = runs("F", k)
    for k in KINDS:
        P5[k] = runs("P", k)
    div = [(r["part"], r["model"], r["seed"]) for r in out["runs"] if r.get("diverged")]
    lines = ["# (A) a45: SSM, LNO, ALNO and LFNO baselines at the WPO's budget", "",
             "Mean [95 % t-interval] over 5 seeds. References (wpo, wpo_index, tfno_r) from a42.", "",
             f"Budgets: {out['budget']}", f"Diverged runs: {div or 'none'}", "",
             "## F: forced Burgers", "",
             "| model | n | test | (5, 10] / test | probe outside the band | unseen / test |", "|---|---|---|---|---|---|"]
    for k, rs in F5.items():
        lines.append(f"| {k} | {len(rs)} | {fmt([r['test'] for r in rs], '.4f')} | "
                     f"{fmt([r['ext_late'] / r['test'] for r in rs], '.2f')} | {fmt([r['oob'] for r in rs], '.3f')} | "
                     f"{fmt([r['high'] / r['test'] for r in rs], '.2f')} |")
    lines += ["", "## P: PDEBench 1D Burgers", "",
              "| model | n | nRMSE 256 | nRMSE 1024 | probe ν' | ν' within 5 % | dispersion error |",
              "|---|---|---|---|---|---|---|"]
    for k, rs in P5.items():
        nu = [r["probe"]["nu_hat"] for r in rs]
        within = sum(abs(v / X37.NU_P - 1) <= 0.05 for v in nu)
        lines.append(f"| {k} | {len(rs)} | {fmt([r['test'] for r in rs], '.4f')} | "
                     f"{fmt([r['test_1024'] for r in rs], '.4f')} | {fmt(nu, '.4f')} | {within} of {len(rs)} | "
                     f"{fmt([r['probe']['disp_err'] for r in rs], '.3f')} |")
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
