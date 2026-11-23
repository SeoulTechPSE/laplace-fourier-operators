"""(A) Paper-1 reinforcement, step 2: a second nonlinear system with dispersive (complex) poles, forced KdV-Burgers.

Burgers' linearization has real poles only (-nu k^2), so the deep results so far test the real part of mu_n(k)
alone. Forced KdV-Burgers,

    u_t + u u_x = nu u_xx - delta u_xxx + f,      x in [0, 2 pi) periodic, u(x, 0) = 0,

linearizes at u = 0 to lambda(k) = -nu k^2 + i delta k^3: damped, dispersive modes. The WPO's default pole form
(Re in k^2, Im in odd powers k, k^3) contains it; a per-index table, an SSM or an FNO has no such constraint.
nu = 0.1 (as a9), delta = 0.01: at k = 8, lambda = -6.4 + 5.1 i, so each forced mode turns by about one radian
while it decays. Solver: `lfo.data.burgers.solve(..., delta=)` (exact linear part in the integrating factor).

**Protocol.** a36 / a42's forced-Burgers protocol unchanged except for the equation: same forcing law (4 modes in
k = 1..8, amplitude ~ s / k, s log-uniform in [0.03, 1]), grid 64 x 201 on T = 5, splits 200 / 50 / 100, the T = 10
extrapolation set, the unseen-mode set (forced only at k = 9..12) and the probe inputs (eps = 0.01, one mode
k = 1..16). Sweep at 4000 steps with model seed 0, selection by validation; final 12000 steps, seeds 0-4.

**Models** at the WPO's real budget (59,529, `lfo.metrics.n_real_params`):
- `wpo`: a11's WPO (width 24, 4 layers, 6 poles, modes 22, stable, k_ref 16). Sweep lr 1e-3 / 3e-3 x pole lr x10 /
  x30 (a11's grid);
- `wpo_index`: a36's per-index poles in the same architecture (the S4D / SGNO-type generator). lr 3e-3 x10 / x30;
- `tfno_r`: a36's Tucker FNO at the real budget (rank 0.01). lr 1e-3 / 3e-3 / 1e-2;
- `s4d`: a45's S4D-type SSM operator (the strongest a45 baseline). lr 1e-3 / 3e-3 / 1e-2, SSM kernel lr <= 1e-3.

**Metrics.** a36's: test; (5, 10] / test; unseen modes / test; out-of-band probe error, now against the complex
truth: mean over k = 9..16 of |mu_hat(k) - lambda(k)| / |lambda(k)|. The probe (a9's odd-part per-mode
identification; its pole grid widened to Im mu in [-60, 60], as lambda reaches 41 i at k = 16) reports nu_hat from
Re mu on {1, k^2} and delta_hat from Im mu on {k, k^3} over the kept modes, and the in-band dispersion error
|mu_hat - lambda| / |lambda| over the kept modes.

**Measured before training** (`--check`, stored under "check" in the results file):
- nonlinear part of u (not explained by the linear response): 0.8 % / 8.5 % / 49 % at the 10th / 50th / 90th
  percentile, up to 174 % (a9's Burgers: 0.5 / 6.5 / 39 %, up to 91 %);
- the solver's own probe (method check): nu_hat 0.10047 (0.5 %), delta_hat 0.010085 (0.85 %), in-band dispersion
  error 1.7 % over the kept modes k = 2..8, out-of-band error 0.024. This is the floor for the models' probes.
- a 30-step smoke run of all four models completes (sweep, final, extrapolation, probe).

**Predictions** (written before running; Burgers references from a42 / a45: test WPO 0.0239, per index 0.0230,
TFNO 0.0287, S4D 0.0390; unseen / test 1.26 against 29-181).

1. Accuracy: the WPO's mean test error is below the TFNO's and the S4D operator's.
2. In-band, the form does not matter: `wpo_index`'s mean test error is within +-15 % of the WPO's.
3. Unseen modes: the WPO's 95 % interval of unseen / test lies below those of all three others (no overlap).
4. Out of the band: the WPO's mean probe error (complex truth, k = 9..16) is <= 0.2; each other model's >= 0.35.
5. Probe in the band: the WPO's nu_hat is within 5 % of 0.1 and its delta_hat within 10 % of 0.01 in at least
   4 of 5 seeds.
6. Time extrapolation: the WPO's mean (5, 10] / test is <= 2.5; the TFNO's is > 5.

    python experiments/A/deep/a47_kdv_burgers.py --check
    python experiments/A/deep/a47_kdv_burgers.py [wpo wpo_index tfno_r s4d] [--summary]
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
import a45_baselines as X45  # noqa: E402
from lfo.data import burgers  # noqa: E402
from lfo.metrics import n_real_params  # noqa: E402
from lfo.models import WavenumberPoleOperator1d  # noqa: E402

NU, DELTA = 0.1, 0.01
OUT = HERE / "results" / "a47_kdv_burgers.json"
SEEDS = (0, 1, 2, 3, 4)
T975_4 = 2.776
KINDS = ("wpo", "wpo_index", "tfno_r", "s4d")
SWEEP = {"wpo": [dict(lr=lr, plr=p) for lr in (1e-3, 3e-3) for p in (10.0, 30.0)],
         "wpo_index": [dict(lr=3e-3, plr=p) for p in (10.0, 30.0)],
         "tfno_r": [dict(lr=lr) for lr in (1e-3, 3e-3, 1e-2)],
         "s4d": [dict(lr=lr) for lr in (1e-3, 3e-3, 1e-2)]}
M.B.GRID_IM = np.linspace(-60, 60, 481)


def lam(k):
    k = np.asarray(k, float)
    return -NU * k ** 2 + 1j * DELTA * k ** 3


def fields(fk, dtf):
    uk = burgers.solve(fk, dtf, NU, M.NX, stride=M.FINE, delta=DELTA)
    f = np.fft.irfft(fk[..., ::M.FINE], n=M.NX, axis=1)
    u = np.fft.irfft(uk, n=M.NX, axis=1)
    return torch.tensor(f), torch.tensor(u)


M.fields = fields                     # a9's data, a36's unseen-mode set and a9's probe inputs now use KdV-Burgers


def make_data():
    data = M.make_data()
    data["high"] = X36.high_set()
    return data


def probe(G, fk, device):
    f_plus, g_plus = G(fk)
    _, g_minus = G(-fk)
    dt = M.T1 / (M.NT1 - 1)
    modes = M.B.per_mode({"train": (f_plus, 0.5 * (g_plus - g_minus))}, dt, device)
    kept = [m for m in modes if abs(complex(*m["mu"])) * dt <= M.CUT and max(map(abs, m["mu"])) * M.T1 >= M.OBS]
    res = dict(modes=modes, kept=[m["k"] for m in kept])
    if len(kept) < 2:
        return dict(res, nu_hat=float("nan"), delta_hat=float("nan"), disp_err=float("inf"))
    ks = np.array(res["kept"], float)
    mu = np.array([complex(*m["mu"]) for m in kept])
    sw = 1 / np.array([max(m["resid"], 1e-6) for m in kept])
    re, *_ = np.linalg.lstsq(np.stack([ks ** 0, ks ** 2], 1) * sw[:, None], mu.real * sw, rcond=None)
    im, *_ = np.linalg.lstsq(np.stack([ks, ks ** 3], 1) * sw[:, None], mu.imag * sw, rcond=None)
    return dict(res, nu_hat=float(-re[1]), delta_hat=float(im[1]),
                disp_err=float(np.linalg.norm(mu - lam(ks)) / np.linalg.norm(lam(ks))),
                resid_median=float(np.median([m["resid"] for m in kept])))


def oob(pr):
    e = [abs(complex(*m["mu"]) - lam(m["k"])) / abs(lam(m["k"])) for m in pr["modes"] if 9 <= m["k"] <= 16]
    return float(np.mean(e))


def model_probe(model, fk, device):
    def G(spec):
        f = torch.tensor(np.fft.irfft(spec[..., ::M.FINE], n=M.NX, axis=1))
        return f, M.batched(model, f.to(device)).cpu().to(torch.float64)
    return probe(G, fk, device)


# ---------------------------------------------------------------- storage
def locked(fn):
    OUT.parent.mkdir(exist_ok=True)
    with open(f"/tmp/{OUT.stem}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        out = json.loads(OUT.read_text()) if OUT.exists() else dict(check={}, budget={}, sweep=[], runs=[])
        fn(out)
        OUT.write_text(json.dumps(out, indent=1, default=float))
        return out


# ---------------------------------------------------------------- check (before training)
def check(device):
    t0 = time.time()
    data = make_data()
    f, u = data["train"]
    rng = np.random.default_rng(0)                         # same draw as the train split: linear response by scaling
    fk, dtf = M.forcing(rng, sum(M.SPLIT), M.T1, M.NT1)
    eps = 1e-4
    _, ulin = fields(eps * fk[:M.SPLIT[0]], dtf)
    nl = ((u - ulin / eps).flatten(1).norm(dim=1) / u.flatten(1).norm(dim=1)).numpy()
    pfk, _ = M.probe_inputs()
    sol = probe(lambda spec: fields(spec, dtf), pfk, device)
    sol["oob"] = oob(sol)
    rec = dict(nonlinear_part_p10_p50_p90_max=[float(np.percentile(nl, q)) for q in (10, 50, 90, 100)],
               solver_probe={k: sol[k] for k in ("kept", "nu_hat", "delta_hat", "disp_err", "oob")},
               solver_modes=sol["modes"], seconds=round(time.time() - t0, 1))
    locked(lambda o: o.__setitem__("check", rec))
    print(json.dumps({k: v for k, v in rec.items() if k != "solver_modes"}, indent=1, default=float))


# ---------------------------------------------------------------- models and training
def build(kind, seed, t, device):
    if kind in ("wpo", "wpo_index", "tfno_r"):
        return X36.build(kind, seed, t, device)
    x = np.linspace(0, 2 * np.pi, M.NX, endpoint=False)
    target = n_real_params(WavenumberPoleOperator1d(t, **F.WPO_KW))
    w = X45.width_for("s4d", "F", x, t, target)
    torch.manual_seed(seed)
    return X45.make("s4d", "F", w, x, t).to(device)


def train(kind, model, data, device, steps, lr, plr=1.0):
    if kind != "s4d":
        return F.train(model, data, device, lr=lr, steps=steps, plr=plr)
    xtr, ytr = data["train"]

    def batch(m, gen):
        idx = torch.randint(0, xtr.shape[0], (M.BATCH,), generator=gen)
        return M.A.rel(m(xtr[idx].to(device)), ytr[idx].to(device))
    return X45.train(model, batch, steps, lr)


def run(kind, device):
    data = make_data()
    t = np.linspace(0, M.T1, M.NT1)
    t2 = np.linspace(0, M.T2, M.NT2)
    pfk, _ = M.probe_inputs()
    m0 = build(kind, 0, t, device)
    locked(lambda o: o["budget"].__setitem__(kind, n_real_params(m0)))
    del m0
    for c in SWEEP[kind]:
        if any(r["model"] == kind and r["cfg"] == c for r in locked(lambda o: None)["sweep"]):
            continue
        t0 = time.time()
        m = build(kind, 0, t, device)
        last = train(kind, m, data, device, F.SWEEP_STEPS, **c)
        m.eval()
        vali = F.err(m, data, "vali", device) if np.isfinite(last) else float("inf")
        rec = dict(model=kind, cfg=c, vali=vali, seconds=round(time.time() - t0, 1))
        locked(lambda o: o["sweep"].append(rec))
        print("sweep", kind, c, f"vali {vali:.4f}", f"{rec['seconds']} s", flush=True)
        del m
        torch.cuda.empty_cache()
    cfg = min((r for r in locked(lambda o: None)["sweep"] if r["model"] == kind), key=lambda r: r["vali"])["cfg"]
    for s in SEEDS:
        if any(r["model"] == kind and r["seed"] == s for r in locked(lambda o: None)["runs"]):
            continue
        t0 = time.time()
        m = build(kind, s, t, device)
        last = train(kind, m, data, device, F.FINAL_STEPS, **cfg)
        if not np.isfinite(last):
            r = dict(model=kind, seed=s, cfg=cfg, diverged=True)
        else:
            m.eval()
            xe, ye = data["ext"]
            X45.set_grids(m, t=t2)
            pe = M.batched(m, xe.to(device)).cpu().to(torch.float64)
            X45.set_grids(m, t=t)
            pr = model_probe(m, pfk, device)
            n_ext = M.NT1 - 1
            r = dict(model=kind, seed=s, cfg=cfg, real_params=n_real_params(m),
                     test=F.err(m, data, "test", device), vali=F.err(m, data, "vali", device),
                     ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])),
                     high=F.err(m, data, "high", device), probe=pr, oob=oob(pr), seconds=round(time.time() - t0, 1))
        locked(lambda o: o["runs"].append(r))
        print(kind, s, {k: r.get(k) for k in ("test", "ext_late", "high", "oob")},
              {k: r.get("probe", {}).get(k) for k in ("nu_hat", "delta_hat", "disp_err")}, flush=True)
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- summary
def ci(xs):
    xs = np.asarray(xs, float)
    m = xs.mean()
    if xs.size < 2:
        return m, float("nan"), float("nan")
    h = T975_4 * xs.std(ddof=1) / np.sqrt(xs.size) if xs.size == 5 else float("nan")
    return m, m - h, m + h


def fmt(xs, f):
    m, lo, hi = ci(xs)
    return f"{m:{f}} [{lo:{f}}, {hi:{f}}]"


def summarize():
    out = locked(lambda o: None)
    c = out.get("check", {})
    lines = ["# (A) a47: forced KdV–Burgers (ν = 0.1, δ = 0.01), complex poles −νk² + iδk³", "",
             "Mean [95 % t-interval] over 5 seeds.", "",
             f"Check before training: {json.dumps({k: v for k, v in c.items() if k != 'solver_modes'}, default=float)}",
             "", f"Budgets (real): {out['budget']}",
             f"Diverged runs: {[(r['model'], r['seed']) for r in out['runs'] if r.get('diverged')] or 'none'}", "",
             "| model | n | cfg | test | (5, 10] / test | unseen / test | probe outside the band | ν̂ | δ̂ | "
             "in-band dispersion error |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in KINDS:
        rs = [r for r in out["runs"] if r["model"] == k and not r.get("diverged")]
        if not rs:
            continue
        p = lambda key: [r["probe"][key] for r in rs]
        lines.append(f"| {k} | {len(rs)} | {rs[0]['cfg']} | {fmt([r['test'] for r in rs], '.4f')} | "
                     f"{fmt([r['ext_late'] / r['test'] for r in rs], '.2f')} | "
                     f"{fmt([r['high'] / r['test'] for r in rs], '.2f')} | {fmt([r['oob'] for r in rs], '.3f')} | "
                     f"{fmt(p('nu_hat'), '.4f')} | {fmt(p('delta_hat'), '.5f')} | {fmt(p('disp_err'), '.3f')} |")
    lines += ["", "Sweep (seed 0, 4000 steps, validation):", ""]
    lines += [f"- {r['model']} {r['cfg']}: {r['vali']:.4f}" for r in out["sweep"]]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if "--check" in sys.argv:
        check(torch.device("cuda"))
        return
    jobs = [a for a in sys.argv[1:] if a in KINDS]
    if "--summary" not in sys.argv:
        device = torch.device("cuda")
        for k in jobs or KINDS:
            run(k, device)
    summarize()


if __name__ == "__main__":
    main()
