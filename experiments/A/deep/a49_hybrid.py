"""(A) Paper-1 reinforcement, step 3: a hybrid of the WPO and a whole-window FNO against the PDEBench gap.

The gap (a37 / a42): on PDEBench 1D Burgers the Tucker FNO is 1.8x more accurate than the WPO at an equal real
budget (nRMSE 0.0254 against 0.0450). a40 traced it to two parts: the FNO block (half) and the FNO's non-causal
mixing over the time window (the rest). A plain sum WPO + FNO would use both, but the FNO branch would then also
carry part of the linear response, and with it the out-of-band behavior that the pole form fixes (a36).

**The hybrid.** H(f) = W(f) + E(f), with W a WPO and E the even part of a Tucker FNO T, less its value at zero:

    E(f) = (T(f) + T(-f)) / 2 - T(0).

E is even in the input, so it has no linear term. The odd part of H, which the linear probe reads
((H(eps u) - H(-eps u)) / 2), is exactly W's. So by construction the hybrid's linearization at rest, its poles and
their continuation outside the band are the WPO's, while the FNO branch adds the even-order terms (quadratic,
quartic, ...) with whole-window temporal mixing. Burgers' leading nonlinear correction is quadratic.

**Models**, each at the WPO's real budget (F 59,529, P 61,130; `lfo.metrics.n_real_params`), split about evenly:
T is the Tucker FNO of a13 / a37 (hidden 24; F: n_modes (32, 64), a13; P: (64, 21), a37) with the Tucker rank nearest half the
budget; W is the WPO (F: a11's, P: a37's configuration) with the width that brings the total nearest the budget.
W runs in float32 (a44: float32 keeps the WPO's results within the 5-seed intervals, at 1.0-1.3x the TFNO's cost).
- `hyb_even`: W + E as above;
- `hyb_sum`: W + T, the plain sum (ablation: what the even construction buys).

Budgets (`budget()`): F rank 0.004 (29,897 real) + WPO width 17 (30,339) = 60,236; P rank 0.006 (30,769) + width 16
(30,057) = 60,826.

**Check before training.** With random TFNO weights, |odd(H) - odd(W)| / |odd(W)| at eps = 0.01 inputs is 1.4e-4
for `hyb_even` (float32 roundoff) and 0.54 for `hyb_sum`. It was 2.9e-2 with PyTorch's default TF32 convolutions
(cuDNN), which round T's O(1) output at about 1e-3; TF32 is therefore off in this script.

**Protocol.** F: a36 / a42's forced Burgers; P: a37 / a42's PDEBench. Sweep lr 1e-3 / 3e-3 / 1e-2 with pole lr x30
(the WPO's selected ratio), seed 0, 4000 steps, selection by validation; final 12000 steps, seeds 0-4. Metrics:
a42's (F: test, (5, 10] / test, probe outside the band, unseen / test; P: nRMSE at 256 and 1024 points, probe nu',
dispersion error).

**References** (not rerun): F WPO float64 test 0.0239, unseen / test 1.26 [0.92, 1.61], probe outside 0.098;
WPO float32 (a44) 0.0238, 1.33, 0.121; TFNO 0.0287, 30.5, 0.874. P WPO 0.0450 (float32 0.0459), dispersion error
0.25 (0.30); TFNO 0.0254, dispersion error 0.69.

**Predictions** (written before running).

1. P, accuracy: `hyb_even`'s mean nRMSE is below 0.040 (it closes at least a quarter of the gap).
2. P, accuracy: `hyb_sum`'s mean nRMSE is below 0.035 and at or below `hyb_even`'s.
3. P, probe: `hyb_even`'s mean dispersion error is <= 0.35 (the WPO's level); `hyb_sum`'s is > 0.5 (the TFNO's
   level).
4. F, form: `hyb_even`'s mean probe error outside the band is <= 0.2 and its mean unseen / test <= 5.
5. F, ablation: `hyb_sum`'s mean probe error outside the band is >= 0.35 and its mean unseen / test > 10.
6. F, accuracy: both hybrids' mean test error is within +-20 % of the WPO's (0.0238-0.0239).

    python experiments/A/deep/a49_hybrid.py [F:hyb_even F:hyb_sum P:hyb_even P:hyb_sum] [--summary]
"""
from __future__ import annotations

import fcntl
import json
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
import a13_baselines as B13  # noqa: E402
import a36_index_poles as X36  # noqa: E402
import a37_pdebench_burgers as X37  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

torch.backends.cudnn.allow_tf32 = False             # TF32 rounding breaks the exact evenness of E (see above)
torch.backends.cuda.matmul.allow_tf32 = False
OUT = HERE / "results" / "a49_hybrid.json"
SEEDS = (0, 1, 2, 3, 4)
T975_4 = 2.776
KINDS = ("hyb_even", "hyb_sum")
JOBS = [f"{p}:{k}" for p in "FP" for k in KINDS]
SWEEP = [dict(lr=lr, plr=30.0) for lr in (1e-3, 3e-3, 1e-2)]
RANKS = [r / 1000 for r in range(1, 40)]


# ---------------------------------------------------------------- models
def tfno(part, rank):
    if part == "F":
        return B13.Net(**dict(B13.MODELS["tfno"], rank=rank))
    return X37.TFNO(rank)


def wpo(part, width, t):
    if part == "F":
        return WavenumberPoleOperator1d(t, dtype=torch.float32, **dict(F.WPO_KW, width=width))
    return WavenumberPoleOperator1d(t, ic=True, dtype=torch.float32, **dict(X37.WPO_KW, width=width))


class Hybrid(nn.Module):
    """F: model(f (b, n_x, n_t)); P: model(u0 (b, n_x), n_t). Output float64."""

    def __init__(self, part, kind, width, rank, t):
        super().__init__()
        self.part, self.even = part, kind == "hyb_even"
        self.w = wpo(part, width, t)
        self.T = tfno(part, rank)

    def _w(self, *a):
        if self.part == "F":
            return self.w(a[0]).to(torch.float64)
        u0, n_t = a
        return self.w(torch.zeros(*u0.shape, n_t, dtype=torch.float64, device=u0.device), u0).to(torch.float64)

    def _T(self, z, *rest):
        return self.T(z, *rest).to(torch.float64)

    def forward(self, z, *rest):
        out = self._w(z, *rest)
        if not self.even:
            return out + self._T(z, *rest)
        both = self._T(torch.cat([z, -z, torch.zeros_like(z[:1])]), *rest)
        b = z.shape[0]
        return out + 0.5 * (both[:b] + both[b:2 * b]) - both[2 * b:]


def budget(part, t):
    target = n_real_params(WavenumberPoleOperator1d(t, **F.WPO_KW)) if part == "F" else \
        n_real_params(X37.WPO(t))
    rank = min(RANKS, key=lambda r: abs(n_real_params(tfno(part, r)) - target / 2))
    nt = n_real_params(tfno(part, rank))
    width = min(range(6, 24), key=lambda w: abs(n_real_params(wpo(part, w, t)) + nt - target))
    return dict(target=target, rank=rank, tfno=nt, width=width, wpo=n_real_params(wpo(part, width, t)))


def build(part, kind, b, t, seed, device):
    torch.manual_seed(seed)
    return Hybrid(part, kind, b["width"], b["rank"], t).to(device)


# ---------------------------------------------------------------- storage
def locked(fn):
    OUT.parent.mkdir(exist_ok=True)
    with open(f"/tmp/{OUT.stem}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(OUT.read_text()) if OUT.exists() else dict(budget={}, sweep=[], runs=[])
        fn(out)
        OUT.write_text(json.dumps(out, indent=1, default=float))
        return out


def best(part, kind):
    rs = [r for r in locked(lambda o: None)["sweep"] if r["part"] == part and r["model"] == kind]
    return min(rs, key=lambda r: r["vali"])["cfg"]


def have(part, kind, key, val):
    return any(r["part"] == part and r["model"] == kind and r.get(key) == val
               for r in locked(lambda o: None)["sweep" if key == "cfg" else "runs"])


# ---------------------------------------------------------------- runs
def run(part, kind, device):
    if part == "F":
        data = M.make_data()
        data["high"] = X36.high_set()
        t = np.linspace(0, M.T1, M.NT1)
        pfk, pdt = M.probe_inputs()
        fit = lambda m, steps, c: F.train(m, data, device, lr=c["lr"], steps=steps, plr=c["plr"])
        vali = lambda m: F.err(m, data, "vali", device)
    else:
        d = X37.load()
        t = d["t"].numpy()
        S = X37.splits(d)
        n_t = S["train"].shape[2]
        full_test = d["full"][X37.NTR:].to(torch.float64)
        fit = lambda m, steps, c: X37.train(m, S["train"], device, lr=c["lr"], steps=steps, plr=c["plr"])
        vali = lambda m: X37.nrmse(X37.predict(m, S["vali"], device), S["vali"])
    b = budget(part, t)
    locked(lambda o: o["budget"].__setitem__(part, b))
    for c in SWEEP:
        if have(part, kind, "cfg", c):
            continue
        t0 = time.time()
        m = build(part, kind, b, t, 0, device)
        last = fit(m, F.SWEEP_STEPS, c)
        m.eval()
        v = vali(m) if np.isfinite(last) else float("inf")
        rec = dict(part=part, model=kind, cfg=c, vali=v, seconds=round(time.time() - t0, 1))
        locked(lambda o: o["sweep"].append(rec))
        print("sweep", part, kind, c, f"vali {v:.4f}", f"{rec['seconds']} s", flush=True)
        del m
        torch.cuda.empty_cache()
    cfg = best(part, kind)
    for s in SEEDS:
        if have(part, kind, "seed", s):
            continue
        t0 = time.time()
        m = build(part, kind, b, t, s, device)
        last = fit(m, F.FINAL_STEPS, cfg)
        m.eval()
        r = dict(part=part, model=kind, seed=s, cfg=cfg, real_params=n_real_params(m))
        if not np.isfinite(last):
            r["diverged"] = True
        elif part == "F":
            xe, ye = data["ext"]
            pe = M.batched(m, xe.to(device)).cpu().to(torch.float64)
            n_ext = M.NT1 - 1
            pr = M.model_probe(m, pfk, pdt, device)
            r.update(test=F.err(m, data, "test", device), vali=vali(m),
                     ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])),
                     high=F.err(m, data, "high", device), probe=pr, oob=X36.oob(pr))
        else:
            r.update(test=X37.nrmse(X37.predict(m, S["test"], device), S["test"]),
                     test_1024=X37.nrmse(X37.predict(m, full_test, device), full_test),
                     probe=X37.model_probe(m, t, 256, n_t, device))
        r["seconds"] = round(time.time() - t0, 1)
        locked(lambda o: o["runs"].append(r))
        print(part, kind, s, {k: r.get(k) for k in ("test", "test_1024", "ext_late", "high", "oob")},
              {k: r.get("probe", {}).get(k) for k in ("nu_hat", "disp_err")}, flush=True)
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- summary
def fmt(xs, f):
    xs = np.asarray(xs, float)
    m = xs.mean()
    if xs.size != 5:
        return f"{m:{f}} (n = {xs.size})"
    h = T975_4 * xs.std(ddof=1) / np.sqrt(5)
    return f"{m:{f}} [{m - h:{f}}, {m + h:{f}}]"


def summarize():
    out = locked(lambda o: None)
    rs = lambda p, k: [r for r in out["runs"] if r["part"] == p and r["model"] == k and not r.get("diverged")]
    lines = ["# (A) a49: WPO + even FNO hybrid against the PDEBench gap", "",
             "Mean [95 % t-interval] over 5 seeds. References in the script's docstring (a42, a44).", "",
             f"Budgets: {out['budget']}",
             f"Diverged: {[(r['part'], r['model'], r['seed']) for r in out['runs'] if r.get('diverged')] or 'none'}", "",
             "## F: forced Burgers", "",
             "| model | cfg | test | (5, 10] / test | probe outside the band | unseen / test |", "|---|---|---|---|---|---|"]
    for k in KINDS:
        R = rs("F", k)
        if R:
            lines.append(f"| {k} | {R[0]['cfg']} | {fmt([r['test'] for r in R], '.4f')} | "
                         f"{fmt([r['ext_late'] / r['test'] for r in R], '.2f')} | {fmt([r['oob'] for r in R], '.3f')} | "
                         f"{fmt([r['high'] / r['test'] for r in R], '.2f')} |")
    lines += ["", "## P: PDEBench 1D Burgers", "",
              "| model | cfg | nRMSE 256 | nRMSE 1024 | probe ν' (truth 0.1257) | dispersion error |",
              "|---|---|---|---|---|---|"]
    for k in KINDS:
        R = rs("P", k)
        if R:
            lines.append(f"| {k} | {R[0]['cfg']} | {fmt([r['test'] for r in R], '.4f')} | "
                         f"{fmt([r['test_1024'] for r in R], '.4f')} | {fmt([r['probe']['nu_hat'] for r in R], '.4f')} | "
                         f"{fmt([r['probe']['disp_err'] for r in R], '.3f')} |")
    lines += ["", "Sweep (seed 0, 4000 steps, validation):", ""]
    lines += [f"- {r['part']} {r['model']} {r['cfg']}: {r['vali']:.4f}" for r in out["sweep"]]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    jobs = [a for a in sys.argv[1:] if a in JOBS]
    if "--summary" not in sys.argv:
        device = torch.device("cuda")
        for j in jobs or JOBS:
            run(*j.split(":"), device)
    summarize()


if __name__ == "__main__":
    main()
