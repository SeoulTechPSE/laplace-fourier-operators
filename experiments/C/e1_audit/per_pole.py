"""(C) E1: per-pole breakdown of the audit runs.

For every configuration (setting, system, model, init, protocol) and every true pole
(folded to Im >= 0, one per conjugate pair): median relative pole error over seeds,
fraction of seeds that recover the pole (relative error < 0.1), and the median signed
deviation of the matched pole in Re and Im (learned - true). Diverged runs are excluded.

Matching is re-done here with ``lfo.metrics.match_poles`` on the folded poles, exactly as
in ``e1_audit.py`` (layer: learned poles; deep: the stored closest pooled poles).

Pole travel. ``travel()`` rebuilds each run's initial model from its seed (same RNG
sequence as ``e1_audit.run``) and compares the initial poles with the checkpoint in
runs/C/e1_audit/. Under Adam a parameter moves at most about lr per step, so the
schedule bounds the total travel by sum(lr * steps): 4.0 for ``common`` and 3.6 for
``lfno_paper`` (see ``travel_budget``). Needs the checkpoints (not committed).

    python experiments/C/e1_audit/per_pole.py      # writes results/per_pole.md (+ .json)
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import torch

from lfo.metrics import match_poles
from lfo.training import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_audit as E  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RECOVERED = 0.1


def _c(pairs):
    return np.array([complex(*z) for z in pairs], dtype=complex)


def matched(r):
    """(true poles, matched learned pole per true pole; nan if unmatched)."""
    mt = _c(r["true_poles"])
    if r["setting"] == "layer":
        ml = _c(r["learned_poles"])
        ml = np.where(ml.imag < 0, ml.conj(), ml)
        ok = np.isfinite(ml)
        ml = ml[ok]
        out = np.full(mt.size, np.nan, complex)
        m = match_poles(ml, mt)
        out[m.true_idx] = ml[m.pred_idx]
    else:
        out = _c(r["closest_poles"])
    return mt, out


def travel_budget(protocol: str) -> float:
    """Upper bound on Adam travel: sum over epochs of lr(epoch) * optimizer steps per epoch."""
    base, _, plr = protocol.partition("+plr")
    p = E.PROTOCOLS[base]
    mult = float(plr) if plr else 1.0
    steps = -(-E.N_TRAIN // p["batch_size"])
    return mult * sum(p["lr"] * p["gamma"] ** (ep // p["step_size"]) * steps for ep in range(p["epochs"]))


def _initial_mu(r) -> np.ndarray:
    t = E.time_grid()
    seed_everything(r["seed"])
    if r["setting"] == "layer":
        _, _, mu, beta = E.make_data(r["system"])
        seed_everything(r["seed"])
        m = E.Layer1(r["model"], r["config"]["n_poles"], t, r["init"] if r["init"] in ("lno", "stable") else "stable")
        if r["init"] in ("band", "warm"):
            E.layer_init(m, r["init"], mu, beta, np.random.default_rng(r["seed"]))
        return m.pr.mu.detach().flatten().numpy()
    cfg = E.deep_config(r["model"], t)
    m = E._deep_model(r["model"], cfg["width"], cfg["k"], t, r["init"])
    return E.deep_poles(m)[0].numpy()


def travel(groups) -> list[str]:
    """Median / max pole travel |mu_final - mu_init| per configuration, against the budget."""
    lines = ["## pole travel", "",
             "|mu_final − mu_init| over all learned poles and seeds; budget = sum(lr × steps) "
             "(Adam moves a parameter by at most ≈ lr per step). ΔIm: median signed change.", "",
             "| setting | system | model | init | protocol | median travel | max | budget | frac ≥ 0.9 budget | median ΔRe | median ΔIm |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, g in sorted(groups.items()):
        if key[3] == "lno":
            continue
        d = []
        for r in g:
            ck = E.RUNS / f"{r['setting']}_{r['system']}_{r['model']}_{r['init']}_{r['protocol']}_s{r['seed']}.pt"
            if not ck.exists():
                continue
            sd = torch.load(ck, map_location="cpu")
            if r["setting"] == "layer":
                mu1 = sd["pr.mu"].flatten().numpy()
            elif r["model"] == "lno_v1":
                mu1 = sd["conv0.weights_pole"].flatten().numpy()
            else:
                mu1 = torch.cat([v.flatten() for k, v in sd.items() if k.endswith(".mu")]).numpy()
            d.append(mu1 - _initial_mu(r))
        if not d:
            continue
        d = np.concatenate(d)
        b = travel_budget(key[4])
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {np.median(np.abs(d)):.2f} | "
                     f"{np.abs(d).max():.2f} | {b:.2f} | {np.mean(np.abs(d) >= 0.9 * b):.0%} | "
                     f"{np.median(d.real):+.2f} | {np.median(d.imag):+.2f} |")
    return lines + [""]


def main():
    groups = {}
    for f in sorted(RESULTS.glob("*/*.json")):
        r = json.loads(f.read_text())
        if r.get("diverged"):
            continue
        key = (r["setting"], r["system"], r["model"], r["init"], r["protocol"])
        groups.setdefault(key, []).append(r)

    rows = []
    for key, g in sorted(groups.items()):
        mts, mls = zip(*(matched(r) for r in g))
        mt = mts[0]
        ml = np.stack(mls)                                   # (seeds, true poles)
        err = np.abs(ml - mt) / np.abs(mt)
        for j, z in enumerate(mt):
            e = err[:, j]
            d = ml[:, j] - z
            rows.append(dict(
                setting=key[0], system=key[1], model=key[2], init=key[3], protocol=key[4],
                true_pole=[z.real, z.imag], seeds=len(g),
                median_error=float(np.nanmedian(e)), recovered=float(np.nanmean(e < RECOVERED)),
                median_dre=float(np.nanmedian(d.real)), median_dim=float(np.nanmedian(d.imag)),
            ))
    (RESULTS / "per_pole.json").write_text(json.dumps(rows, indent=1))

    fmt = lambda z: f"{z[0]:+.1f}{z[1]:+.1f}i" if z[1] else f"{z[0]:+.1f}"
    lines = ["# E1 per-pole breakdown", "",
             f"Per true pole (Im ≥ 0): median relative error over seeds, fraction recovered "
             f"(error < {RECOVERED}), median signed deviation of the matched pole "
             "(ΔRe, ΔIm = learned − true). Diverged runs excluded. Deep: closest pooled poles.", ""]
    for setting in ("layer", "deep"):
        lines += [f"## {setting}", "", "| system | model | init | protocol | true pole | seeds | median err | recovered | ΔRe | ΔIm |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["setting"] != setting:
                continue
            lines.append(f"| {r['system']} | {r['model']} | {r['init']} | {r['protocol']} | {fmt(r['true_pole'])} | "
                         f"{r['seeds']} | {r['median_error']:.3f} | {r['recovered']:.0%} | "
                         f"{r['median_dre']:+.2f} | {r['median_dim']:+.2f} |")
        lines.append("")
    lines += travel(groups)
    (RESULTS / "per_pole.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
