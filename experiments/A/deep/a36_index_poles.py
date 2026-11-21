"""(A) Next-level step 1: is it the physical-wavenumber form of the poles that pays, and how does the LNO compare?

Literature re-check (2026-11-22, `experiments/A/literature.md`): recent operators also learn a linear generator per
Fourier mode. SGNO (arXiv 2602.18801) learns one free multiplier per Fourier *index* with an exponential (phi_1)
update; diagonal state-space models (S4D-type) likewise keep free per-mode, per-channel poles. LGN-KM (arXiv
2603.29264) ties its damping to |k|^p. The (A) claim is narrower than "learned per-mode poles": the poles are
smooth functions mu_n(k) of the *physical* wavenumber. This script isolates that claim on a11's forced Burgers
and adds the planned LNO baseline (proposal A, Section 7.2), which the deep comparison lacked.

**Models** (a11 protocol: relative L2, Adam with cosine decay, batch 25; sweep at 4000 steps with model seed 0,
selection by validation, then 12000 steps with seeds 0-2):
- `wpo_index`: a11's WPO (width 24, 4 layers, 6 poles, modes 22, stable) with each layer's poles replaced by a free
  table mu_n[k] over the 22 retained mode indices (real part through the same -softplus, so stable). Residues stay
  polynomial in k^2. Initialised at the polynomial's initial values, so the two start from the same poles.
  60,465 parameters (+1.6 %). Sweep: lr 3e-3 with pole lr x10 or x30 (a11 chose lr 3e-3, x30 for the WPO).
  This is the S4D / SGNO-type per-index generator inside the same architecture.
- `lno`: the original LNO v2 (Burgers example; `lfo.models.LNO2dV2`, transient + steady branches, sin activations,
  input stride 2), with its steady modes raised from (4, 4) to (9, 9) so that the steady branch covers the forced
  band k <= 8 (as a13 raised the FNO's), widths equal, the width whose real parameter count (complex weights
  counted twice) is closest to the WPO's 59,529. Sweep: lr 1e-3, 3e-3, 1e-2. Time extrapolation: the grid buffers
  are replaced by the T = 10 grid.
- References: `wpo` = a11's checkpoints (not retrained); `tfno` = a13's configuration retrained with lr 3e-3
  (a13's checkpoints are gone; its numbers are the reproduction check).
- `tfno_r`: found while setting up this script: a13's "matched" TFNO (59,065 by `numel`) has 110,569 real degrees
  of freedom, because its Tucker factors are complex (`numel` counts a complex weight once); the WPO's 59,529 are
  all real. So every TFNO / FNO budget in (A) and (D) so far was about 1.9x the WPO's in real terms. `tfno_r` is
  a13's TFNO with rank 0.01 (54,649 real, -8 %; rank steps are coarse), lr 3e-3. Budgets from now on use
  `lfo.metrics.n_real_params` (complex = 2).

**Metrics** (a11 / a13's, plus two):
- test (T = 5), time extrapolation (5, 10] / test, probe nu_hat and in-band dispersion error (kept modes);
- out-of-band probe error: mean over k = 9..16 of |mu_hat(k) - (-nu k^2)| / (nu k^2), from the probe's per-mode
  poles (forcing band k <= 8). Computed before this script from the stored probes: WPO 0.091 / 0.114 / 0.142,
  TFNO 0.92 / 0.93 / 0.71 (a11, a13);
- unseen modes: test error on 100 new samples forced only at k = 9..12 (same amplitude law, data seed 3), and its
  ratio to the test error.

**Predictions** (written before running).

1. `wpo_index`: mean test error within +-15 % of the WPO's (0.0237): in-band accuracy does not need the form.
2. `wpo_index`: mean out-of-band probe error >= 3x the WPO's (>= 0.35), and above the WPO's largest (0.142) in
   every seed.
3. Unseen modes: mean ratio (k = 9..12 error / test error) of `wpo_index` >= 2x the WPO's; the same for `tfno`.
4. `wpo_index`: time-extrapolation ratio <= 2 in every seed (the FOH recursion, not the form, carries it).
5. `lno`: mean test error >= 1.5x the WPO's, extrapolation ratio > 2.5 in every seed, mean out-of-band probe error
   >= 3x the WPO's.
6. `tfno_r`: mean test error <= 1.2x `tfno`'s (the 2x real budget is not what a13's comparison rests on).

    python experiments/A/deep/a36_index_poles.py      # results/a36_index_poles.md / .json; checkpoints runs/A/a36
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a9_burgers as M  # noqa: E402
import a11_fair as F  # noqa: E402
import a13_baselines as B13  # noqa: E402
from lfo.metrics import n_real_params as real_params  # noqa: E402
from lfo.models import LNO2dV2, WavenumberPoleOperator1d  # noqa: E402
from lfo.models.wavenumber import WavenumberPoleConv1d  # noqa: E402

OUT = HERE / "results" / "a36_index_poles.json"
CKPT = M.ROOT / "runs" / "A" / "a36"
SWEEP = {"wpo_index": [dict(lr=3e-3, plr=10.0), dict(lr=3e-3, plr=30.0)],
         "lno": [dict(lr=lr) for lr in (1e-3, 3e-3, 1e-2)],
         "tfno": [dict(lr=3e-3)], "tfno_r": [dict(lr=3e-3)]}
HIGH = (9, 12)
OOB = (9, 16)


class IndexPoleConv1d(WavenumberPoleConv1d):
    """Poles as a free table over the retained mode indices (no function of k)."""

    def mu(self, k):
        if k.shape[0] != self.tab_mu_re.shape[1]:
            raise ValueError("per-index poles are defined only on the training grid's modes")
        re = -Fn.softplus(self.tab_mu_re) if self.stable else self.tab_mu_re
        return torch.complex(re, self.tab_mu_im)


def to_index(layer: WavenumberPoleConv1d, n_x: int):
    """Turn a polynomial layer into an IndexPoleConv1d at the polynomial's current values."""
    k = layer.wavenumbers(n_x)[:layer.modes] / layer.k_ref
    with torch.no_grad():
        raw = sum(layer.mu_re[:, j, None] * (k ** 2) ** j for j in range(layer.mu_re.shape[1]))
        im = sum(layer.mu_im[:, j, None] * k ** p for j, p in enumerate(layer.im_powers))
    del layer.mu_re, layer.mu_im
    layer.tab_mu_re = nn.Parameter(raw.clone())            # names end in mu_re / mu_im: a11's pole lr applies
    layer.tab_mu_im = nn.Parameter(im.clone())
    layer.__class__ = IndexPoleConv1d


class LNOWrap(nn.Module):
    def __init__(self, width, x, t):
        super().__init__()
        self.net = LNO2dV2(width, width, width, 4, 4, 9, 9, x, t)

    def forward(self, f):
        return self.net(f.float()[..., None])[..., 0].to(f.dtype)


def set_time_grid(model, t):
    for m in model.modules():
        if "t" in dict(m.named_buffers(recurse=False)):
            m.t = torch.as_tensor(t, dtype=m.t.dtype, device=m.t.device)


def build(kind, seed, t, device):
    x = np.linspace(0, 2 * np.pi, M.NX, endpoint=False)
    if kind == "wpo":
        torch.manual_seed(seed)
        return WavenumberPoleOperator1d(t, **F.WPO_KW).to(device)
    if kind == "wpo_index":
        torch.manual_seed(seed)
        m = WavenumberPoleOperator1d(t, **F.WPO_KW)
        for layer in m.K:
            to_index(layer, M.NX)
        return m.to(device)
    if kind == "lno":
        target = real_params(WavenumberPoleOperator1d(t, **F.WPO_KW))
        w = min(range(3, 16), key=lambda w: abs(real_params(LNOWrap(w, x, t)) - target))
        torch.manual_seed(seed)
        return LNOWrap(w, x, t).to(device)
    torch.manual_seed(seed)
    if kind == "tfno_r":
        return B13.Net(**dict(B13.MODELS["tfno"], rank=0.01)).to(device)
    return B13.Net(**B13.MODELS["tfno"]).to(device)


def high_set():
    rng = np.random.default_rng(3)
    tf = np.linspace(0, M.T1, (M.NT1 - 1) * M.FINE + 1)
    n = 100
    fk = np.zeros((n, M.NX // 2 + 1, tf.size), complex)
    s = np.exp(rng.uniform(np.log(M.S_RANGE[0]), np.log(M.S_RANGE[1]), size=n))
    for _ in range(4):
        prof = M.A.profiles(tf, rng, n)
        km = rng.integers(HIGH[0], HIGH[1] + 1, size=n)
        amp = (rng.normal(size=n) + 1j * rng.normal(size=n)) / km * M.NX / 2 * s
        fk[np.arange(n), km] += amp[:, None] * prof
    return M.fields(fk, tf[1] - tf[0])


def oob(probe):
    e = [abs(complex(*m["mu"]) + M.NU * m["k"] ** 2) / (M.NU * m["k"] ** 2) for m in probe["modes"]
         if OOB[0] <= m["k"] <= OOB[1]]
    return float(np.mean(e))


def evaluate(model, kind, seed, cfg, data, t, pfk, pdt, device, t0):
    n_ext = M.NT1 - 1
    model.eval()
    xe, ye = data["ext"]
    if kind == "lno":
        set_time_grid(model, np.linspace(0, M.T2, M.NT2))
    pe = M.batched(model, xe.to(device)).cpu().to(torch.float64)
    if kind == "lno":
        set_time_grid(model, t)
    probe = M.model_probe(model, pfk, pdt, device)
    r = dict(model=kind, seed=seed, cfg=cfg, params=M.n_params(model), real_params=real_params(model),
             train=F.err(model, data, "train", device), vali=F.err(model, data, "vali", device),
             test=F.err(model, data, "test", device), ext_late=float(M.A.rel(pe[..., n_ext + 1:], ye[..., n_ext + 1:])),
             high=F.err(model, data, "high", device), probe=probe, oob=oob(probe))
    if t0 is not None:
        r["seconds"] = round(time.time() - t0, 1)
    return r


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(parents=True, exist_ok=True)
    data = M.make_data()
    data["high"] = high_set()
    t = np.linspace(0, M.T1, M.NT1)
    pfk, pdt = M.probe_inputs()
    out = json.loads(OUT.read_text()) if OUT.exists() else dict(sweep=[], final=[], ref=[])
    save = lambda: OUT.write_text(json.dumps(out, indent=1, default=float))
    if not out["ref"]:
        a11 = {r["seed"]: r for r in json.loads(F.OUT.read_text())["final"] if r["model"] == "wpo"}
        for seed in M.SEEDS:
            m = build("wpo", seed, t, device)
            m.load_state_dict(torch.load(F.CKPT / f"wpo_{seed}.pt", map_location=device))
            r = evaluate(m, "wpo", seed, a11[seed]["cfg"], data, t, pfk, pdt, device, None)
            r["test_a11"] = a11[seed]["test"]
            out["ref"].append(r)
            print("ref wpo", seed, f"test {r['test']:.4f} (a11 {r['test_a11']:.4f}) high {r['high']:.4f} oob {r['oob']:.3f}",
                  flush=True)
            save()
    for kind, cfgs in SWEEP.items():
        if len(cfgs) == 1:
            continue
        for cfg in cfgs:
            if any(r["model"] == kind and r["cfg"] == cfg for r in out["sweep"]):
                continue
            t1 = time.time()
            model = build(kind, 0, t, device)
            last = F.train(model, data, device, steps=F.SWEEP_STEPS, **cfg)
            model.eval()
            vali = F.err(model, data, "vali", device) if np.isfinite(last) else float("inf")
            out["sweep"].append(dict(model=kind, cfg=cfg, vali=vali, seconds=round(time.time() - t1, 1)))
            print("sweep", kind, cfg, f"vali {vali:.3e}", f"{time.time() - t1:.0f} s", flush=True)
            save()
            del model
            torch.cuda.empty_cache()
    for kind, cfgs in SWEEP.items():
        best = cfgs[0] if len(cfgs) == 1 else min((r for r in out["sweep"] if r["model"] == kind),
                                                  key=lambda r: r["vali"])["cfg"]
        for seed in M.SEEDS:
            if any(r["model"] == kind and r["seed"] == seed for r in out["final"]):
                continue
            t1 = time.time()
            model = build(kind, seed, t, device)
            F.train(model, data, device, steps=F.FINAL_STEPS, **best)
            torch.save(model.state_dict(), CKPT / f"{kind}_{seed}.pt")
            r = evaluate(model, kind, seed, best, data, t, pfk, pdt, device, t1)
            out["final"].append(r)
            print("final", kind, seed, best, f"test {r['test']:.4f} ext/test {r['ext_late'] / r['test']:.2f}",
                  f"high {r['high']:.4f} oob {r['oob']:.3f} disp {r['probe']['disp_err']:.3f}", f"{r['seconds']} s",
                  flush=True)
            save()
            del model
            torch.cuda.empty_cache()
    summarize(out)


def summarize(out):
    rows = [("wpo (a11)", out["ref"])] + [(k, [r for r in out["final"] if r["model"] == k]) for k in SWEEP]
    rg = lambda xs, f: f"{np.mean(xs):{f}} ({min(xs):{f}}–{max(xs):{f}})"
    lines = ["# (A) Forced Burgers: poles per mode index vs per physical wavenumber, and the LNO (a11 setup)", "",
             "Means over 3 seeds (range). Out-of-band: probe pole error over k = 9..16 (forcing band k <= 8). "
             "Unseen modes: forcing only at k = 9..12.", "",
             "| model | params (real) | test | (5, 10] / test | probe ν̂ | in-band dispersion | out-of-band | unseen modes | unseen / test |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, rs in rows:
        if not rs:
            continue
        lines.append(f"| {name} | {rs[0]['real_params']} | {rg([r['test'] for r in rs], '.4f')} | "
                     f"{rg([r['ext_late'] / r['test'] for r in rs], '.2f')} | {rg([r['probe']['nu_hat'] for r in rs], '.4f')} | "
                     f"{rg([r['probe']['disp_err'] for r in rs], '.3f')} | {rg([r['oob'] for r in rs], '.3f')} | "
                     f"{rg([r['high'] for r in rs], '.4f')} | {rg([r['high'] / r['test'] for r in rs], '.2f')} |")
    if out["sweep"]:
        lines += ["", "| sweep | cfg | validation |", "|---|---|---|"]
        lines += [f"| {r['model']} | {r['cfg']} | {r['vali']:.3e} |" for r in out["sweep"]]
    (OUT.with_suffix(".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
