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
    python experiments/C/theory/e2_train.py --summarize

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


def run(cond: str, seed: int, device):
    system, family, T, K = CONDITIONS[cond]
    data, t = make_splits(system, family, T, K)
    mu_true, beta_true = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    seed_everything(seed)
    model = E.Layer1("tied", mu_true.size, t, "stable")
    E.layer_init(model, "band", mu_true, beta_true, np.random.default_rng(seed))
    model.to(device)
    t0 = time.time()
    hist = E.train(model, data, E.PROTOCOLS[PROTOCOL], device, PLR)
    secs = time.time() - t0
    xte, yte = (a.to(device)[..., None] for a in data["test"])
    test = E.evaluate(model, xte, yte)
    mu_l, beta_l = (a.detach().cpu().numpy() for a in model.pole_sets())
    mt, _ = E.true_folded(mu_true, beta_true)
    diverged = not (np.isfinite(mu_l).all() and np.isfinite(test))
    mf, _ = E.fold(mu_l, beta_l)
    pe = None if diverged else pole_error(mf, mt, reduce="none").tolist()
    res = dict(cond=cond, system=system, family=family, T=T, K=K, seed=seed, n_t=int(t.size),
               model="tied", init="band", protocol=f"{PROTOCOL}+plr{PLR:g}", params=count_params(model),
               seconds=round(secs, 1), test_rel_l2=test, diverged=bool(diverged), history=hist,
               oracle_rel_l2=E.oracle_error(data, t, mu_true, beta_true),
               true_poles=[[float(z.real), float(z.imag)] for z in mt], pole_error=pe,
               pole_error_mean=None if pe is None else float(np.mean(pe)),
               learned_poles=[[float(z.real), float(z.imag)] for z in mu_l],
               predicted=predicted(system, family, T, K))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cond}_s{seed}.json").write_text(json.dumps(res, indent=1))
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RUNS / f"{cond}_s{seed}.pt")
    print(f"{cond} s{seed}: test {test:.2e} pole_err {pe and [round(x, 3) for x in pe]} ({secs:.0f}s)", flush=True)


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lines = ["# E2 phase 2: training on the phase-1 data configurations", "",
             "Tied layer, band init, `common` + pole lr × 10, 1000 epochs. Per true pole: phase-1 ratio "
             f"(Re μ; **bold** ≥ {R.RATIO_OK:g} = predicted identifiable) / fraction of seeds recovering it "
             f"(relative pole error < {RECOVERED}) / mean ± std pole error. Diverged runs excluded.", ""]
    for system in ("p4", "p6"):
        conds = [c for c in CONDITIONS if CONDITIONS[c][0] == system and any(r["cond"] == c for r in rs)]
        if not conds:
            continue
        poles = next(r["true_poles"] for r in rs if r["system"] == system)
        labels = [f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}" for z in poles]
        keys = [f"{z[0]:+g}{z[1]:+g}i" for z in poles]
        lines += [f"## {system}", "",
                  "| condition | n_t | seeds | test rel. L2 | oracle | " + " | ".join(labels) + " |",
                  "|---|---|---|---|---|" + "---|" * len(poles)]
        for c in conds:
            g = [r for r in rs if r["cond"] == c]
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
            lines.append(f"| {c} | {g[0]['n_t']} | {len(g)}{div} | {te.mean():.1e} ± {te.std():.1e} | "
                         f"{g[0]['oracle_rel_l2']:.1e} | " + " | ".join(cells) + " |")
        lines.append("")
    (R.RESULTS / "e2_train.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for s in a.seeds:
            run(c, s, device)


if __name__ == "__main__":
    main()
