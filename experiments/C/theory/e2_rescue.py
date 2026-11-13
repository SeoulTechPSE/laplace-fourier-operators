"""(C) E2: truth-free restarts for polished runs that stop above the floor.

After the L-BFGS polish (``e2_polish.py``) the remaining failures are local minima: real
slots coalesced at one decay, or a damped spare complex slot holding a fast decay. Here
each polished run is given single-slot moves, without using the truth:

  candidates  every slot j is moved, with residue 0 (so the output is unchanged at first),
              to each point of a grid: real slots to Re = -geomspace(0.05, 5, 8); complex
              slots to Im = geomspace(0.3, 5, 8), Re = -0.1 Im (inside every slot bound)
  screen      each candidate is polished for SCREEN_ITERS L-BFGS iterations
  accept      the best candidate, if it lowers the training loss by >= 1 %; it is then
              polished in full (as in e2_polish.py), and the next round starts
  stop        after ROUNDS rounds or when no candidate is accepted

Every run is treated the same, whether or not it is at the floor; only training loss is used.

    python experiments/C/theory/e2_rescue.py --cond p6_ms16_T80_N4096 --variant damp8_ms --seeds 10 11
    python experiments/C/theory/e2_rescue.py --summarize      # writes results/e2_rescue.md
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
import e2_polish as P  # noqa: E402

T, E = P.T, P.E
OUT = T.R.RESULTS / "e2_rescue"
ROUNDS, SCREEN_ITERS, ACCEPT = 3, 300, 0.99
REAL_GRID = -np.geomspace(0.05, 5.0, 8)
IMAG_GRID = np.geomspace(0.3, 5.0, 8)


def train_loss(model, data, device):
    xtr, ytr = (a.to(device)[..., None] for a in data["train"])
    with torch.no_grad():
        model.eval()
        return float(P.relative_l2_loss(model(xtr), ytr))


def set_slot(model, j, mu_new):
    """Write pole mu_new into slot j through the raw parameter of its parametrisation; residue 0."""
    par = model.pr.parametrizations.mu[0]
    raw = model.pr.parametrizations.mu.original
    real = float(par.mask.flatten()[j]) == 0
    with torch.no_grad():
        if real:
            v = complex(mu_new.real, 0.0)
        elif isinstance(par, T.DampFloor):
            im = abs(mu_new.imag)
            q = float(np.clip(abs(mu_new.real) / (par.c * im), 0.02, 0.98))
            v = complex(np.log(q / (1 - q)), im - par.w_min)
        elif isinstance(par, T.ImagFloor):
            v = complex(mu_new.real, abs(mu_new.imag) - par.w_min)
        else:                                               # ImagMask
            v = complex(mu_new.real, mu_new.imag)
        raw[..., j] = v
        model.pr.beta[..., j] = 0


def candidates(model):
    mask = model.pr.parametrizations.mu[0].mask.flatten().cpu().numpy()
    for j, m in enumerate(mask):
        for z in (REAL_GRID if m == 0 else -0.1 * IMAG_GRID + 1j * IMAG_GRID):
            yield j, complex(z)


def run(cond, variant, seed, device):
    name = T.run_name(cond, variant, seed)
    ck = T.RUNS / f"{name}_lbfgs.pt"
    if not ck.exists():
        print(f"skip {name}: no polished checkpoint", flush=True)
        return
    system, family, Tw, Kk, N = T.unpack(cond)
    data, t = T.make_splits(system, family, Tw, Kk, N)
    mu_true, beta_true = (np.array(a, dtype=complex) for a in E.SYSTEMS[system])
    model, _ = T.build(variant, mu_true, beta_true, t, seed)
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.to(device)
    t0 = time.time()
    before = P.score(model, data, mu_true, beta_true, device)
    loss = train_loss(model, data, device)
    loss0, moves = loss, []
    for rnd in range(ROUNDS):
        state = {k: v.clone() for k, v in model.state_dict().items()}
        best = (loss, None, None)
        for j, z in candidates(model):
            model.load_state_dict(state)
            set_slot(model, j, z)
            P.polish(model, data, device, maxiter=SCREEN_ITERS)
            lj = train_loss(model, data, device)
            if lj < best[0]:
                best = (lj, j, z, {k: v.clone() for k, v in model.state_dict().items()})
        if best[1] is None or best[0] > ACCEPT * loss:
            model.load_state_dict(state)
            break
        model.load_state_dict(best[3])
        P.polish(model, data, device)
        new = train_loss(model, data, device)
        moves.append(dict(round=rnd + 1, slot=int(best[1]), to=[best[2].real, best[2].imag], loss_before=loss,
                          loss_after=new))
        loss = new
    after = P.score(model, data, mu_true, beta_true, device)
    res = dict(cond=cond, variant=variant, seed=seed, system=system, n_per_20=N, seconds=round(time.time() - t0, 1),
               oracle_rel_l2=json.loads((P.OUT / f"{name}.json").read_text())["oracle_rel_l2"],
               true_poles=[[float(z.real), float(z.imag)] for z in E.true_folded(mu_true, beta_true)[0]],
               train_before=loss0, train_after=loss, moves=moves, before=before, after=after)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), T.RUNS / f"{name}_rescue.pt")
    r3 = lambda pe: pe and [round(x, 3) for x in pe]
    print(f"{name}: train/oracle {loss0 / res['oracle_rel_l2']:.2f} -> {loss / res['oracle_rel_l2']:.2f}, "
          f"moves {len(moves)}, pole_err {r3(before['pole_error'])} -> {r3(after['pole_error'])} ({res['seconds']:.0f}s)",
          flush=True)


CASES = [("p4_ms16_T20", "mixed_off_ms"), ("p4_ms16_T20", "damp8_ms"),
         ("p6_ms16_T80", "mixed_off_ms"), ("p6_ms16_T80", "damp8_ms"),
         ("p6_ms16_T80_N4096", "mixed_off_ms"), ("p6_ms16_T80_N4096", "damp8_ms")]


def summarize():
    rs = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))]
    lab = lambda z: f"{z[0]:+g}{z[1]:+g}i" if z[1] else f"{z[0]:+g}"
    ok = lambda s: max(s["pole_error"]) < T.RECOVERED
    lines = ["# E2: truth-free single-slot restarts after the polish", "",
             f"Every polished run (seeds 10-29) gets up to {ROUNDS} rounds of single-slot moves (residue 0, "
             f"{SCREEN_ITERS}-iteration screen, accept if training loss falls ≥ 1 %, then full polish). "
             "Before = after the polish (e2_polish.py). Best of 5: lowest validation loss in groups of 5 "
             "consecutive seeds.", "",
             "| condition | variant | runs | runs moved | train/oracle (median) | test rel. L2 (median) | all (single) | "
             "best of 5: all | per pole recovered | s per run |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for cond, variant in CASES:
        g = sorted((r for r in rs if r["cond"] == cond and r["variant"] == variant), key=lambda r: r["seed"])
        if not g:
            continue
        by = {r["seed"]: r for r in g}
        row = dict(cond=cond, variant=variant, runs=len(g), moved=sum(bool(r["moves"]) for r in g))
        for when, tr in (("before", "train_before"), ("after", "train_after")):
            groups = [ok(min((by[s] for s in range(g0, g0 + 5)), key=lambda r: r[when]["vali"])[when])
                      for g0 in range(10, 30, 5) if all(s in by for s in range(g0, g0 + 5))]
            row[when] = dict(ratio=float(np.median([r[tr] / r["oracle_rel_l2"] for r in g])),
                             test=float(np.median([r[when]["test"] for r in g])),
                             single=float(np.mean([ok(r[when]) for r in g])), best=[int(sum(groups)), len(groups)],
                             per_pole=(np.array([r[when]["pole_error"] for r in g]) < T.RECOVERED).mean(0).tolist())
        b, a = row["before"], row["after"]
        pp = ", ".join(f"{lab(z)} {x:.0%} → {y:.0%}" for z, x, y in zip(g[0]["true_poles"], b["per_pole"], a["per_pole"]))
        lines.append(f"| {cond} | {variant} | {len(g)} | {row['moved']} | {b['ratio']:.2f} → {a['ratio']:.2f} | "
                     f"{b['test']:.1e} → {a['test']:.1e} | {b['single']:.0%} → {a['single']:.0%} | "
                     f"{b['best'][0]}/{b['best'][1]} → {a['best'][0]}/{a['best'][1]} | {pp} | "
                     f"{np.median([r['seconds'] for r in g]):.0f} |")
        out.append(row)
    lines.append("")
    (T.R.RESULTS / "e2_rescue.json").write_text(json.dumps(out, indent=1))
    (T.R.RESULTS / "e2_rescue.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+")
    ap.add_argument("--variant", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 30)))
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.summarize:
        return summarize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in a.cond:
        for v in a.variant:
            for s in a.seeds:
                run(c, v, s, device)


if __name__ == "__main__":
    main()
