"""(C) E2 phase 2: does training recover the poles where phase 1 says the data identifies them?

Phase 1 (``e2_richness.py``) predicts, per data configuration, which poles are practically
identifiable (profiled-error ratio >= 10). Here the E1 single tied layer is trained on
exactly those data and the recovered poles are compared with the prediction.

Fixed across conditions (only the data change):
  model     E1 ``Layer1("tied")`` (LNO v1 layer), n_poles = number of true poles
  init      E1 ``band`` init with its T = 20 ranges (Re in [-0.5, -0.05], Im in [0, 5.03] rad/s),
            so the initial pole distribution is the same in rad/s for every window T
  protocol  E1 ``common`` with pole lr x 10 (``common+plr10``), 1000 epochs
Data: ``e2_richness.make`` (dt = 20/1024): train = the 200 samples analysed in phase 1
(data seed 0), validation 50 (seed 1), test 100 (seed 2), same family / K / T.

    python experiments/C/theory/e2_train.py --cond p4_e1_T20 p4_ms16_T20 --seeds 0 1 2
    python experiments/C/theory/e2_train.py --cond p4_ms16_T20 --variant real_min warm --seeds 0 1 2
    python experiments/C/theory/e2_train.py --summarize

Variants (controls for the real poles; ``base`` is the setting above):
  base        n_poles = number of true poles (conjugates counted), all slots complex, band init
  warm        as base, E1 ``warm`` init (truth, Re and Im each x (1 + 0.2 z))
  free_min    n_poles = number of folded true poles (one slot per pair or real pole), all complex
  real_spare  as base, but as many slots as there are real true poles have Im mu fixed to 0
  real_min    free_min with the real-pole slots constrained real: the minimal form of phase 1
Remedies (E2 -> E3/E4 candidates):
  wide        n_poles = (complex pairs + 1) + 2 * (real poles) slots, all band init (slot-count control)
  mixed       same slots as wide; (pairs + 1) band slots, 2 * (real poles) slots started on the real
              axis with Re = -geomspace(0.05, 5) * exp(0.2 z) (stratified, so they cannot start
              together) and constrained real (Im mask as in real_spare). Left free, such slots
              leave the axis at once: d/d(Im mu) = -t Im(beta) e^{Re mu t} != 0 for a complex
              residue, and Adam turns any small gradient into a step of about lr.
  mixed_off   mixed with the real grid shifted by half a log step (k = 4: 0.11, 0.50, 2.3, 10.7),
              a control: the unshifted grid (0.05, 0.23, 1.08, 5) lies close to p4's -0.2 and -1
  *_ms        multi-start: 8 candidate inits (seeds 100 * seed + c) trained 100 epochs each,
              the best on validation continues with the full protocol (1000 epochs, fresh optimiser)
The real constraint is a parametrisation of ``pr.mu`` (Im multiplied by a 0/1 mask), applied
after the init, so the constrained slots start on the real axis at the band init's Re.

Outputs: results/e2_train/<cond>_s<seed>.json (committed), results/e2_train.md,
checkpoints in runs/C/e2/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.parametrize  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2_richness as R  # noqa: E402  (also puts e1_audit on the path)

E = R.E
from lfo.metrics import pole_error  # noqa: E402
from lfo.training import count_params, seed_everything  # noqa: E402

OUT = R.RESULTS / "e2_train"
RUNS = E.ROOT / "runs" / "C" / "e2"
PROTOCOL, PLR = "common", 10.0
RECOVERED = 0.1
VARIANTS = ("base", "warm", "free_min", "real_spare", "real_min", "wide", "mixed", "base_ms", "mixed_ms", "mixed_off", "mixed_off_ms")
MS_CANDIDATES, MS_EPOCHS = 8, 100
E.POLE_PARAMS = E.POLE_PARAMS + ("mu.original",)     # parametrised pr.mu still gets the pole lr


class ImagMask(torch.nn.Module):
    """mu -> Re mu + i Im mu * mask: slots with mask 0 are real poles."""

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, mu):
        return torch.complex(mu.real, mu.imag * self.mask)

# name -> (system, family, T, K)
CONDITIONS = {
    "p4_e1_T20": ("p4", "e1", 20.0, 0),            # E1 data: real poles predicted non-identifiable
    "p4_ramp_T20": ("p4", "ramp", 20.0, 0),        # negative control: nothing identifiable
    "p4_prbs_T20": ("p4", "prbs", 20.0, 0),
    "p4_ms16_T20": ("p4", "multisine", 20.0, 16),
    "p4_e1_T40": ("p4", "e1", 40.0, 0),
    "p4_ms16_T40": ("p4", "multisine", 40.0, 16),
    "p6_e1_T20": ("p6", "e1", 20.0, 0),
    "p6_ms16_T80": ("p6", "multisine", 80.0, 16),  # richest p6 data: pairs identifiable, real poles not
}


def make_splits(system, family, T, K):
    f = lambda a: a.to(torch.float32)
    splits = {}
    for name, seed, n in (("train", 0, 200), ("vali", 1, 50), ("test", 2, 100)):
        t, u, y, _, _ = R.make(system, family, T, K, seed=seed)
        splits[name] = (f(u[:n]), f(y[:n]))
    return splits, t.numpy()


def predicted(system, family, T, K):
    """Phase-1 ratios for this configuration (Re mu, per true pole), if computed."""
    path = R.RESULTS / "e2_richness.json"
    if not path.exists():
        return None
    for r in json.loads(path.read_text()):
        if (r["system"], r["family"], r["T"], r["K"]) == (system, family, T, K):
            return dict(floor=r["floor"], cond=r["cond"],
                        ratio={f"{p['pole'][0]:+g}{p['pole'][1]:+g}i": p["Re mu"]["ratio"] for p in r["poles"]})
    return None


def build(variant, mu_true, beta_true, t, seed):
    variant = variant.removesuffix("_ms")
    n_real = int(np.sum(mu_true.imag == 0))
    n_folded = int(np.sum(mu_true.imag >= 0))
    n_pairs = n_folded - n_real
    n = {"free_min": n_folded, "real_min": n_folded,
         "wide": n_pairs + 1 + 2 * n_real, "mixed": n_pairs + 1 + 2 * n_real,
         "mixed_off": n_pairs + 1 + 2 * n_real}.get(variant, mu_true.size)
    model = E.Layer1("tied", n, t, "stable")
    rng = np.random.default_rng(seed)
    E.layer_init(model, "warm" if variant == "warm" else "band", mu_true, beta_true, rng)
    if variant in ("mixed", "mixed_off"):
        k = 2 * n_real
        shift = (5.0 / 0.05) ** (0.5 / (k - 1)) if variant == "mixed_off" else 1.0   # half a grid step
        re = -np.geomspace(0.05, 5.0, k) * shift * np.exp(0.2 * rng.normal(size=k))
        with torch.no_grad():
            model.pr.mu[..., n - k:] = torch.tensor(re, dtype=torch.cfloat)
        mask = torch.ones(n)
        mask[n - k:] = 0
        torch.nn.utils.parametrize.register_parametrization(model.pr, "mu", ImagMask(mask.reshape(1, 1, -1)))
        return model, dict(n_poles=n, real_slots=k)
    if variant in ("real_spare", "real_min"):
        mask = torch.ones(n)
        mask[n - n_real:] = 0
        torch.nn.utils.parametrize.register_parametrization(model.pr, "mu", ImagMask(mask.reshape(1, 1, -1)))
    return model, dict(n_poles=n, real_slots=n_real if variant.startswith("real") else 0)


def run_name(cond, variant, seed):
    return f"{cond}_s{seed}" if variant == "base" else f"{cond}_{variant}_s{seed}"


def run(cond: str, seed: int, device, variant: str = "base"):
    system, family, T, K = CONDITIONS[cond]
    data, t = make_splits(system, family, T, K)
    mu_true, beta_true = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    t0 = time.time()
    if variant.endswith("_ms"):
        screen = dict(E.PROTOCOLS[PROTOCOL], epochs=MS_EPOCHS)
        cands = []
        for c in range(MS_CANDIDATES):
            seed_everything(100 * seed + c)
            m, cfg = build(variant, mu_true, beta_true, t, 100 * seed + c)
            m.to(device)
            h = E.train(m, data, screen, device, PLR)
            cands.append((h[-1][2] if np.isfinite(h[-1][2]) else np.inf, c, m))
        val, best, model = min(cands, key=lambda z: z[0])
        cfg = dict(cfg, ms_candidates=MS_CANDIDATES, ms_epochs=MS_EPOCHS, ms_best=best,
                   ms_vals=sorted(float(z[0]) for z in cands))
        seed_everything(seed)
    else:
        seed_everything(seed)
        model, cfg = build(variant, mu_true, beta_true, t, seed)
        model.to(device)
    hist = E.train(model, data, E.PROTOCOLS[PROTOCOL], device, PLR)
    secs = time.time() - t0
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    test = E.evaluate(model, xte, yte)
    mu_l, beta_l = (a.detach().cpu().numpy() for a in model.pole_sets())
    mt, _ = E.true_folded(mu_true, beta_true)
    diverged = not (np.isfinite(mu_l).all() and np.isfinite(test))
    mf, _ = E.fold(mu_l, beta_l)
    pe = None if diverged else pole_error(mf, mt, reduce="none").tolist()
    res = dict(cond=cond, variant=variant, config=cfg, system=system, family=family, T=T, K=K, seed=seed, n_t=int(t.size),
               model="tied", init="warm" if variant == "warm" else "band", protocol=f"{PROTOCOL}+plr{PLR:g}", params=count_params(model),
               seconds=round(secs, 1), test_rel_l2=test, diverged=bool(diverged), history=hist,
               oracle_rel_l2=E.oracle_error(data, t, mu_true, beta_true),
               true_poles=[[float(z.real), float(z.imag)] for z in mt], pole_error=pe,
               pole_error_mean=None if pe is None else float(np.mean(pe)),
               learned_poles=[[float(z.real), float(z.imag)] for z in mu_l],
               predicted=predicted(system, family, T, K))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run_name(cond, variant, seed)}.json").write_text(json.dumps(res, indent=1))
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{run_name(cond, variant, seed)}.pt")
    print(f"{cond} {variant} s{seed}: test {test:.2e} pole_err {pe and [round(x, 3) for x in pe]} ({secs:.0f}s)", flush=True)


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    for r in rs:
        r.setdefault("variant", "base")
    lines = ["# E2 phase 2: training on the phase-1 data configurations", "",
             "Tied layer, band init, `common` + pole lr × 10, 1000 epochs. Per true pole: phase-1 ratio "
             f"(Re μ; **bold** ≥ {R.RATIO_OK:g} = predicted identifiable) / fraction of seeds recovering it "
             f"(relative pole error < {RECOVERED}) / mean ± std pole error. Diverged runs excluded.", ""]
    for system in ("p4", "p6"):
        conds = [(c, v) for c in CONDITIONS for v in VARIANTS
                 if CONDITIONS[c][0] == system and any(r["cond"] == c and r["variant"] == v for r in rs)]
        if not conds:
            continue
        poles = next(r["true_poles"] for r in rs if r["system"] == system)
        labels = [f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}" for z in poles]
        keys = [f"{z[0]:+g}{z[1]:+g}i" for z in poles]
        lines += [f"## {system}", "",
                  "| condition | variant | slots | n_t | seeds | test rel. L2 | oracle | " + " | ".join(labels) + " |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(poles)]
        for c, v in conds:
            g = [r for r in rs if r["cond"] == c and r["variant"] == v]
            ok = [r for r in g if not r["diverged"]]
            te = np.array([r["test_rel_l2"] for r in ok])
            pe = np.array([r["pole_error"] for r in ok])
            pred = g[0]["predicted"] or {"ratio": {}}
            cells = []
            for j, k in enumerate(keys):
                q = pred["ratio"].get(k, float("nan"))
                qs = f"**{q:.1f}**" if q >= R.RATIO_OK else f"{q:.1f}"
                cells.append(f"{qs} / {np.mean(pe[:, j] < RECOVERED):.0%} / {pe[:, j].mean():.2f} ± {pe[:, j].std():.2f}")
            div = f" ({len(g) - len(ok)} div.)" if len(g) > len(ok) else ""
            slots = len(g[0]["learned_poles"])
            lines.append(f"| {c} | {v} | {slots} | {g[0]['n_t']} | {len(g)}{div} | {te.mean():.1e} ± {te.std():.1e} | "
                         f"{g[0]['oracle_rel_l2']:.1e} | " + " | ".join(cells) + " |")
        lines.append("")
    (R.RESULTS / "e2_train.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--variant", nargs="+", default=["base"], choices=VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for v in a.variant:
            for s in a.seeds:
                run(c, s, device, v)


if __name__ == "__main__":
    main()
